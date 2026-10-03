//! The ingest pipeline: download -> normalize -> transcribe (GPU) -> align -> enrich.
//!
//! Runs on its own thread and reports back over a channel, so the UI stays
//! responsive. Everything is cached in `library::root()`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Sender;

use serde_json::json;

use crate::{library, onnx_llm};

#[derive(Debug, Clone)]
pub enum Event {
    Stage(String),
    Progress(f32),
    Log(String),
    Done(Box<library::Item>),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Options {
    /// audio URL, or a path to a local audio file
    pub url: String,
    /// known German article text: when set, the transcript is force-aligned to it
    pub text: Option<String>,
    /// where that text came from (a path, or `pasted`), recorded in `meta.json`
    pub text_source: String,
    pub language: Option<String>,
    pub target: String,
    /// sherpa-onnx ASR model label, e.g. `nemo-de`
    pub asr_model: String,
    pub translate: bool,
    /// label of the local ONNX model, e.g. `qwen2.5:3b`
    pub llm_model: Option<String>,
    /// run everything on the GPU when possible
    pub use_gpu: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            url: String::new(),
            text: None,
            text_source: String::new(),
            language: None,
            target: "en".to_string(),
            asr_model: crate::asr::DEFAULT_MODEL.to_string(),
            translate: true,
            llm_model: Some(onnx_llm::DEFAULT_MODEL.to_string()),
            use_gpu: true,
        }
    }
}

pub fn spawn(options: Options, tx: Sender<Event>) {
    std::thread::Builder::new()
        .name("ingest".into())
        .spawn(move || {
            if let Err(error) = run(&options, &tx) {
                let _ = tx.send(Event::Failed(error));
            }
        })
        .expect("spawn ingest thread");
}

fn run(options: &Options, tx: &Sender<Event>) -> Result<(), String> {
    let dir = library::directory(&library::name_from_url(&options.url));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;

    let mut meta = library::Meta {
        id: dir
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default(),
        source_url: options.url.clone(),
        article: options.text_source.clone(),
        language: options.language.clone().unwrap_or_default(),
        target: options.target.clone(),
        model: options.asr_model.clone(),
        created: chrono::Local::now().timestamp(),
        status: "running".to_string(),
        ..Default::default()
    };
    let _ = library::write_meta(&dir, &meta);

    match ingest(options, tx, &dir, &mut meta) {
        Ok(item) => {
            library::write_meta(&item.dir, &meta)?;
            let _ = tx.send(Event::Done(Box::new(item)));
            Ok(())
        }
        Err(error) => {
            meta.status = "failed".to_string();
            meta.error = Some(error.clone());
            let _ = library::write_meta(&dir, &meta);
            Err(error)
        }
    }
}

fn ingest(
    options: &Options,
    tx: &Sender<Event>,
    dir: &Path,
    meta: &mut library::Meta,
) -> Result<library::Item, String> {
    // ---------------------------------------------------------------- download
    if let Some(source) = local_audio(&options.url) {
        copy_local(&source, dir, tx)?;
    } else if looks_like_path(&options.url) {
        return Err(format!("no such audio file: {}", options.url.trim()));
    } else if looks_like_media(&options.url) {
        stage(tx, "Downloading audio", 0.0);
        let target = dir.join(format!("audio.{}", media_extension(&options.url)));
        download(&options.url, &target, tx).or_else(|err| {
            log(tx, &format!("direct download failed ({err}), trying yt-dlp"));
            let _ = std::fs::remove_file(&target);
            yt_dlp(&options.url, dir, tx)
        })?;
    } else {
        stage(tx, "Downloading with yt-dlp", 0.0);
        yt_dlp(&options.url, dir, tx)?;
    }

    // keep whatever came in: symphonia (recognition) and rodio (playback) both
    // decode mp3, m4a, ogg, opus, flac, wav, mp4, mkv and webm by themselves
    let downloaded = find_audio(dir).ok_or_else(|| "no audio found after download".to_string())?;
    let extension = downloaded
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("bin")
        .to_string();
    let audio = dir.join(format!("audio.{extension}"));
    if downloaded != audio {
        std::fs::rename(&downloaded, &audio).map_err(|err| err.to_string())?;
    }
    meta.audio = format!("audio.{extension}");
    log(tx, &format!("audio: {} ({} KB)", meta.audio, std::fs::metadata(&audio).map(|m| m.len() / 1024).unwrap_or(0)));

    // -------------------------------------------------------------- transcribe
    stage(tx, "Transcribing", 0.3);
    let language = options.language.clone().unwrap_or_default();
    let result = transcribe(&options.asr_model, &language, &audio, tx)?;

    meta.language = result["language"].as_str().unwrap_or("unknown").to_string();
    meta.device = result["device"].as_str().unwrap_or_default().to_string();
    meta.duration = result["duration"].as_f64().unwrap_or(meta.duration as f64) as f32;
    log(
        tx,
        &format!(
            "{:.1}s of audio ({} recognition)",
            meta.duration,
            result["model"].as_str().unwrap_or_default()
        ),
    );

    let mut words = result["words"].as_array().cloned().unwrap_or_default();
    let mut phrases = result["phrases"].as_array().cloned().unwrap_or_default();

    // a known German article is force-aligned to the recognized word timings:
    // the text is the source of truth, the recognizer only supplies the clock
    if let Some(article) = &options.text {
        let duration = result["duration"].as_f64().unwrap_or(0.0) as f32;
        let asr_words = words
            .iter()
            .filter_map(seg_from_json)
            .collect::<Vec<library::Seg>>();
        let aligned = crate::align::align(article, &asr_words, duration);
        log(
            tx,
            &format!(
                "article: {} words / {} sentences aligned to {} recognized words",
                aligned.words.len(),
                aligned.sentences.len(),
                asr_words.len()
            ),
        );
        words = aligned.words.iter().map(seg_to_json).collect();
        phrases = aligned.sentences.iter().map(seg_to_json).collect();
    }

    meta.words = words.len();
    meta.sentences = phrases.len();
    log(
        tx,
        &format!(
            "{} words in {} sentences ({})",
            meta.words,
            meta.sentences,
            result["elapsed"].as_f64().map(|s| format!("{s:.1}s")).unwrap_or_default()
        ),
    );

    write_transcription(dir, &result, &meta.audio, &words, &phrases)?;

    // ------------------------------------------------------------------ enrich
    if let Some(label) = &options.llm_model {
        let sentences = phrases
            .iter()
            .filter_map(|phrase| phrase["text"].as_str().map(str::to_string))
            .collect::<Vec<_>>();
        let text = sentences.join(" ");

        // the recognizer releases its GPU memory asynchronously (VAD and models)
        wait_for_free_gpu(tx);
        stage(tx, &format!("Loading {label}"), 0.82);
        // no-op when already called at startup, keeps the CLI path working too
        let _ = onnx_llm::prepare_runtime();
        let tx_for_log = tx.clone();
        let mut log_model = move |message: String| {
            if std::env::var_os("TRANSCRIBE_LOG").is_some() {
                eprintln!("[llm] {message}");
            }
            let _ = tx_for_log.send(Event::Log(message));
        };
        log_model(format!(
            "gpu before language model: {} MiB used of {} MiB",
            gpu_memory_used().unwrap_or(0),
            onnx_llm::vram_total_mb()
        ));
        log_model(format!(
            "onnxruntime environment: {}",
            if ort::environment::Environment::current().is_ok() {
                "present"
            } else {
                "MISSING"
            }
        ));

        if let Some(runtime) = onnx_llm::runtime_path() {
            log(
                tx,
                &format!(
                    "onnxruntime: {} [{}]",
                    runtime.display(),
                    onnx_llm::init_note().unwrap_or("unknown")
                ),
            );
        }

        match onnx_llm::Llm::load(label, &models_dir(), options.use_gpu, &mut log_model) {
            Ok(mut model) => {
                log(tx, &format!("language model: {}", model.summary()));

                stage(tx, "Writing headline and summary", 0.85);
                match model.title_and_description(&text, &meta.language) {
                    Ok((title, description)) => {
                        log(tx, &format!("title: {title}"));
                        meta.title = title;
                        meta.description = description;
                    }
                    Err(err) => log(tx, &format!("summary failed: {err}")),
                }

                if options.translate && !sentences.is_empty() {
                    stage(tx, &format!("Translating to {}", options.target), 0.92);
                    let mut progress = |fraction: f32| {
                        let _ = tx.send(Event::Progress(0.92 + 0.07 * fraction));
                    };

                    match model.translate(&sentences, &options.target, &mut progress) {
                        Ok(translations) => {
                            let doc = json!({ "target": options.target, "sentences": translations });
                            std::fs::write(
                                dir.join(library::TRANSLATION),
                                serde_json::to_string_pretty(&doc).unwrap_or_default(),
                            )
                            .map_err(|err| err.to_string())?;
                            log(tx, &format!("{} sentences translated", translations.len()));
                        }
                        Err(err) => log(tx, &format!("translation failed: {err}")),
                    }
                }
            }
            Err(err) => log(tx, &format!("no language model: {err}")),
        }
    } else {
        log(tx, "no language model selected: skipping title, description and translation");
    }

    if meta.title.is_empty() {
        meta.title = fallback_title(&phrases, &options.url);
        meta.description = phrases
            .iter()
            .take(2)
            .filter_map(|phrase| phrase["text"].as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }

    // the folder name should follow the generated title
    let final_dir = rename_dir(dir, &meta.title, meta.created);
    if final_dir != dir && std::fs::rename(dir, &final_dir).is_ok() {
        meta.id = final_dir
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
    } else if !final_dir.exists() {
        meta.id = dir
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
    }
    let final_dir = if final_dir.exists() { final_dir } else { dir.to_path_buf() };

    meta.status = "done".to_string();
    stage(tx, "Done", 1.0);

    Ok(library::Item {
        dir: final_dir,
        meta: meta.clone(),
    })
}

// ------------------------------------------------------------------- download

fn looks_like_media(url: &str) -> bool {
    const MEDIA: &[&str] = &[
        ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac", ".mp4", ".m4b", ".webm",
        ".mkv", ".mov",
    ];
    let path = url.split(['?', '#']).next().unwrap_or(url).to_ascii_lowercase();
    MEDIA.iter().any(|extension| path.ends_with(extension))
}

/// A local audio file the user wants ingested instead of downloaded.
fn local_audio(source: &str) -> Option<PathBuf> {
    let path = expand_path(source.trim());
    path.is_file().then_some(path)
}

/// `true` when the input clearly addresses a file, not a URL or a search term.
fn looks_like_path(source: &str) -> bool {
    let source = source.trim();
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with("~/")
        || source.starts_with("file://")
}

/// Expands `~/` and strips a `file://` prefix.
pub fn expand_path(source: &str) -> PathBuf {
    let source = source.strip_prefix("file://").unwrap_or(source);
    if let Some(rest) = source.strip_prefix("~/") {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        return home.join(rest);
    }
    PathBuf::from(source)
}

/// Copies a local audio file into the article folder as `audio.<ext>`.
fn copy_local(source: &Path, dir: &Path, tx: &Sender<Event>) -> Result<(), String> {
    stage(tx, "Using local audio", 0.0);

    let extension = source
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| !value.is_empty() && value.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or_else(|| "bin".to_string());
    let target = dir.join(format!("audio.{extension}"));

    std::fs::copy(source, &target)
        .map_err(|err| format!("cannot copy {}: {err}", source.display()))?;
    log(tx, &format!("local audio: {}", source.display()));
    Ok(())
}

fn download(url: &str, target: &Path, tx: &Sender<Event>) -> Result<(), String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(None)
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) TranscriptPlayer/1.0")
        .build()
        .new_agent();

    let mut response = agent.get(url).call().map_err(|err| err.to_string())?;
    let total = response.body().content_length();
    let mut reader = response.body_mut().as_reader();
    let mut file = std::fs::File::create(target).map_err(|err| err.to_string())?;

    let mut buffer = vec![0u8; 64 * 1024];
    let mut written: u64 = 0;
    let mut last_report = std::time::Instant::now();

    loop {
        let read = reader.read(&mut buffer).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read]).map_err(|err| err.to_string())?;
        written += read as u64;

        if last_report.elapsed().as_millis() > 200 {
            last_report = std::time::Instant::now();
            let fraction = total
                .map(|total| written as f32 / total as f32)
                .unwrap_or(0.0);
            let _ = tx.send(Event::Progress(0.25 * fraction.min(1.0)));
        }
    }

    file.flush().map_err(|err| err.to_string())?;
    log(tx, &format!("{} MB downloaded", written / 1_048_576));

    if written < 1024 {
        return Err(format!("download produced only {written} bytes"));
    }

    Ok(())
}

fn yt_dlp(url: &str, dir: &Path, tx: &Sender<Event>) -> Result<(), String> {
    let output = Command::new("yt-dlp")
        .args([
            "-x",
            "--audio-format",
            "mp3",
            "--no-playlist",
            "--no-progress",
            "-o",
        ])
        .arg(dir.join("source.%(ext)s"))
        .arg(url)
        .output()
        .map_err(|err| format!("yt-dlp: {err}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let lines: Vec<&str> = stderr.lines().filter(|line| !line.trim().is_empty()).collect();
    for line in lines.iter().rev().take(3).rev() {
        log(tx, line);
    }

    if !output.status.success() {
        let tail: String = stderr
            .chars()
            .rev()
            .take(300)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(format!("yt-dlp failed: {tail}"));
    }

    Ok(())
}

/// Audio extension from a URL path (`.../episode.m4a?token=…` -> `m4a`).
fn media_extension(url: &str) -> String {
    let path = url.split(['?', '#']).next().unwrap_or(url).to_ascii_lowercase();
    match path.rsplit_once('.') {
        Some((_, extension)) if extension.len() <= 4 && extension.chars().all(|c| c.is_ascii_alphanumeric()) => {
            extension.to_string()
        }
        _ => "bin".to_string(),
    }
}

fn find_audio(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(u64, PathBuf)> = None;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name()?.to_string_lossy().to_string();
        // `audio.<ext>` for direct downloads, `source.<ext>` for yt-dlp
        if !name.starts_with("source") && !name.starts_with("audio") {
            continue;
        }
        if name.ends_with(".part") || name.ends_with(".ytdl") {
            continue;
        }
        let size = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        if best.as_ref().map(|(best, _)| size > *best).unwrap_or(true) {
            best = Some((size, path));
        }
    }

    best.map(|(_, path)| path)
}

// ------------------------------------------------------------------- helpers

/// A recognized word from the ASR result JSON.
fn seg_from_json(value: &serde_json::Value) -> Option<library::Seg> {
    Some(library::Seg {
        start: value["start"].as_f64()? as f32,
        end: value["end"].as_f64()? as f32,
        text: value["text"].as_str()?.to_string(),
    })
}

/// A word/sentence segment for `write_transcription`.
fn seg_to_json(seg: &library::Seg) -> serde_json::Value {
    json!({ "start": seg.start, "end": seg.end, "text": seg.text })
}

fn stage(tx: &Sender<Event>, message: &str, progress: f32) {
    let _ = tx.send(Event::Stage(message.to_string()));
    let _ = tx.send(Event::Progress(progress));
    let _ = tx.send(Event::Log(message.to_string()));
}

fn log(tx: &Sender<Event>, message: &str) {
    if std::env::var_os("TRANSCRIBE_LOG").is_some() {
        eprintln!("[pipeline] {message}");
    }
    let _ = tx.send(Event::Log(message.to_string()));
}

/// Waits until the GPU is mostly free: the transcription process may still be
/// tearing down its CUDA context, and a 4 GB card then has no room for the
/// language model (cuBLAS reports `resource allocation failed`).
fn wait_for_free_gpu(tx: &Sender<Event>) {
    let mut reported = false;

    for _ in 0..40 {
        let used = gpu_memory_used().unwrap_or(0);
        if used < 384 {
            if reported {
                log(tx, &format!("GPU free again ({used} MiB in use)"));
            }
            // let the driver settle before the next context is created
            std::thread::sleep(std::time::Duration::from_millis(300));
            return;
        }
        if !reported {
            log(tx, &format!("waiting for the GPU ({used} MiB still in use)"));
            reported = true;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    log(tx, "GPU still busy, trying anyway");
}

fn gpu_memory_used() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Where the ONNX language models are cached.
pub fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("TRANSCRIBE_MODELS") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join("transcribe-models/onnx")
}

/// Runs the sherpa-onnx recognizer over the audio file.
fn transcribe(
    model: &str,
    language: &str,
    audio: &Path,
    tx: &Sender<Event>,
) -> Result<serde_json::Value, String> {
    let tx_for_log = tx.clone();
    let mut log_model = move |message: String| {
        let _ = tx_for_log.send(Event::Log(message));
    };

    let mut asr = crate::asr::Asr::load(
        model,
        &crate::asr::models_dir(),
        language,
        true,
        &mut log_model,
    )?;

    let mut progress = |fraction: f32, _: &str| {
        let _ = tx.send(Event::Progress(0.3 + 0.5 * fraction));
    };

    let transcript = asr.transcribe(audio, &mut progress)?;

    // free the recognition model (and its CUDA context) before the language
    // model is loaded: a 4 GB card cannot hold both at once
    drop(asr);
    log(tx, "recognizer unloaded");
    log(
        tx,
        &format!(
            "{:.1}x realtime on {}, {} sentences, {} words",
            transcript.speed,
            transcript.device,
            transcript.phrases.len(),
            transcript.words.len()
        ),
    );

    let words = transcript
        .words
        .iter()
        .map(|word| json!({ "start": word.start, "end": word.end, "text": word.text }))
        .collect::<Vec<_>>();
    let phrases = transcript
        .phrases
        .iter()
        .map(|phrase| json!({ "start": phrase.start, "end": phrase.end, "text": phrase.text }))
        .collect::<Vec<_>>();

    Ok(json!({
        "language": transcript.language,
        "device": transcript.device,
        "duration": transcript.duration,
        "model": transcript.model,
        "elapsed": transcript.duration / transcript.speed.max(0.001),
        "words": words,
        "phrases": phrases,
    }))
}

/// Writes `transcription.json` (word level) and `transcript.json` (sentences).
fn write_transcription(
    dir: &Path,
    result: &serde_json::Value,
    audio: &str,
    words: &[serde_json::Value],
    phrases: &[serde_json::Value],
) -> Result<(), String> {
    let language = result["language"].as_str().unwrap_or("unknown");
    let text = words
        .iter()
        .filter_map(|word| word["text"].as_str())
        .collect::<String>()
        .trim()
        .to_string();

    let word_segments = words
        .iter()
        .map(|word| {
            json!({
                "start": word["start"],
                "end": word["end"],
                "speaker_id": null,
                "text": word["text"],
                "type": "transcription_segment",
            })
        })
        .collect::<Vec<_>>();

    let transcription = json!({
        "language": language,
        "model": result["model"],
        "segments": word_segments,
        "text": text,
        "type": "transcription.done",
        "usage": {
            "prompt_audio_seconds": result["duration"].as_f64().unwrap_or_default().round(),
            "request_count": 1,
            "device": result["device"],
            "elapsed_seconds": result["elapsed"],
        },
    });

    let phrase_segments = phrases
        .iter()
        .map(|phrase| {
            json!({
                "start": phrase["start"],
                "end": phrase["end"],
                "text": phrase["text"],
            })
        })
        .collect::<Vec<_>>();

    let transcript = json!({
        "file": audio,
        "model": result["model"],
        "language": language,
        "duration": result["duration"],
        "segments": phrase_segments,
    });

    let write = |name: &str, value: &serde_json::Value| -> Result<(), String> {
        let raw = serde_json::to_string_pretty(value).map_err(|err| err.to_string())?;
        std::fs::write(dir.join(name), raw).map_err(|err| err.to_string())
    };

    write(library::WORDS, &transcription)?;
    write(library::PHRASES, &transcript)
}

fn fallback_title(phrases: &[serde_json::Value], url: &str) -> String {
    let first = phrases
        .first()
        .and_then(|phrase| phrase["text"].as_str())
        .unwrap_or_default();

    if first.is_empty() {
        library::name_from_url(url)
    } else {
        first.chars().take(70).collect()
    }
}

/// Renames `20261003-1200-flydubai` to `20261003-1200-flydubai-co-pilot`.
fn rename_dir(dir: &Path, title: &str, created: i64) -> PathBuf {
    let Some(parent) = dir.parent() else {
        return dir.to_path_buf();
    };
    let stamp = chrono::DateTime::from_timestamp(created, 0)
        .map(|utc| {
            let local: chrono::DateTime<chrono::Local> = utc.into();
            local.format("%Y%m%d-%H%M").to_string()
        })
        .unwrap_or_else(|| "article".to_string());

    let candidate = parent.join(format!("{stamp}-{}", library::slug(title)));
    if candidate == dir || candidate.exists() {
        dir.to_path_buf()
    } else {
        candidate
    }
}

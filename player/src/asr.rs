//! Speech recognition through sherpa-onnx — the same ONNX Runtime family as the
//! language model, no Python and no C++ build (the crate uses prebuilt libs).
//!
//! Pipeline: symphonia decodes the file to 16 kHz mono, Silero VAD finds the
//! speech segments (those become the sentences), every segment is recognized,
//! and token timestamps turn into word timings when the model provides them
//! (Whisper exports do not, so words are interpolated inside their segment).

use std::path::{Path, PathBuf};
use std::time::Instant;

use sherpa_onnx::{
    LinearResampler, OfflineCanaryModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineRecognizerResult, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{Decoder, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatReader;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

use crate::library::Seg;

pub const DEFAULT_MODEL: &str = "whisper-turbo";

/// One prebuilt sherpa-onnx model.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub label: &'static str,
    pub url: &'static str,
    pub dir: &'static str,
    pub download_mb: u32,
    pub kind: Kind,
    /// language hints (empty = let the model detect / not needed)
    pub language: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// NVIDIA NeMo FastConformer transducer (token timestamps)
    NemoTransducer,
    /// NVIDIA Canary (speech recognition + translation)
    Canary,
    /// OpenAI Whisper exports (no token timestamps)
    Whisper,
}

impl ModelSpec {
    pub fn get(label: &str) -> Option<Self> {
        let spec = match label {
            "nemo-de" => ModelSpec {
                label: "nemo-de",
                url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-stt_de_fastconformer_hybrid_large_pc.tar.bz2",
                dir: "sherpa-onnx-nemo-transducer-stt_de_fastconformer_hybrid_large_pc",
                download_mb: 449,
                kind: Kind::NemoTransducer,
                language: "de",
            },
            "whisper-turbo" => ModelSpec {
                label: "whisper-turbo",
                url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
                dir: "sherpa-onnx-whisper-turbo",
                download_mb: 564,
                kind: Kind::Whisper,
                language: "",
            },
            "whisper-large-v3" => ModelSpec {
                label: "whisper-large-v3",
                url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-large-v3.tar.bz2",
                dir: "sherpa-onnx-whisper-large-v3",
                download_mb: 3112,
                kind: Kind::Whisper,
                language: "",
            },
            "canary-180m" => ModelSpec {
                label: "canary-180m",
                url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2",
                dir: "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
                download_mb: 154,
                kind: Kind::Canary,
                language: "",
            },
            _ => return None,
        };

        Some(spec)
    }
}

/// Everything a finished transcription needs.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub language: String,
    pub duration: f32,
    pub device: String,
    pub model: String,
    pub phrases: Vec<Seg>,
    pub words: Vec<Seg>,
    /// seconds per audio second (1.0 = realtime)
    pub speed: f32,
}

pub struct Asr {
    recognizer: OfflineRecognizer,
    vad: VoiceActivityDetector,
    spec: ModelSpec,
    device: String,
    language: String,
}

impl Asr {
    pub fn load(
        label: &str,
        models_dir: &Path,
        language: &str,
        use_gpu: bool,
        log: &mut dyn FnMut(String),
    ) -> Result<Self, String> {
        let spec = ModelSpec::get(label).ok_or_else(|| format!("unknown ASR model {label}"))?;
        ensure_model(&spec, models_dir, log)?;

        let dir = models_dir.join(spec.dir);
        let silero = models_dir.join(SILERO);
        if !silero.is_file() {
            log(format!("downloading {SILERO}"));
            let url = format!(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{SILERO}"
            );
            fetch(&url, &silero)?;
        }

        let language = if language.is_empty() || language == "auto" {
            spec.language.to_string()
        } else {
            language.to_string()
        };

        let provider = if use_gpu { "cuda" } else { "cpu" };
        let mut config = config_for(&spec, &dir, &language)?;
        config.model_config.provider = Some(provider.to_string());
        config.model_config.num_threads = 4;

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| format!("cannot create recognizer for {label}"))?;

        let mut vad_config = VadModelConfig::default();
        vad_config.silero_vad = SileroVadModelConfig {
            model: Some(silero.to_string_lossy().to_string()),
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.6,
            window_size: 512,
            max_speech_duration: 20.0,
            ..Default::default()
        };
        vad_config.sample_rate = 16000;
        vad_config.num_threads = 1;

        let vad = VoiceActivityDetector::create(&vad_config, 120.0)
            .ok_or_else(|| "cannot create the voice activity detector".to_string())?;

        log(format!(
            "asr: {label} on {provider}, language {}",
            if language.is_empty() { "auto" } else { &language }
        ));

        Ok(Self {
            recognizer,
            vad,
            spec,
            device: provider.to_string(),
            language,
        })
    }

    /// Transcribes a media file; `progress` gets the decoded fraction.
    ///
    /// Speech segments come from Silero VAD, but each one is recognized with a
    /// little padding on both sides: a hard cut at a segment boundary truncates
    /// the word that straddles it ("Sicherheitsris" instead of
    /// "Sicherheitsrisiko.").
    pub fn transcribe(
        &mut self,
        path: &Path,
        progress: &mut dyn FnMut(f32, &str),
    ) -> Result<Transcript, String> {
        const PAD: usize = 5_600; // 0.35 s at 16 kHz
        const RETENTION: usize = 16_000 * 40;

        let (mut reader, mut decoder, track_id, _channels, container_rate, time_base) =
            open_audio(path)?;

        // Container metadata and the decoder's own spec can both be wrong
        // (HE-AAC/SBR streams decode twice as many samples). The packet
        // timeline is reliable, so the effective sample rate is derived from
        // the first two decoded frames.
        let mut source_rate = container_rate;
        let mut resampler = (source_rate != 16000)
            .then(|| LinearResampler::create(source_rate as i32, 16000).expect("resampler"));
        let mut pending: Vec<f32> = Vec::new();
        let mut last_ts: Option<u64> = None;
        let mut rate_known = false;
        let mut source_decoded = 0usize;
        let mut fed = 0usize;

        let total_samples = reader
            .default_track()
            .and_then(|track| track.codec_params.n_frames)
            .map(|frames| frames as usize);

        let started = Instant::now();
        self.vad.reset();

        // rolling window of decoded 16 kHz samples: `window[0]` is absolute sample `base`
        let mut window: Vec<f32> = Vec::new();
        let mut base = 0usize;
        let mut total = 0usize;

        // speech ranges reported by the VAD (absolute sample indices)
        let mut ready: std::collections::VecDeque<(usize, usize)> = Default::default();
        let mut phrases: Vec<(f32, f32, String, Vec<Seg>)> = Vec::new();

        let feed = |window: &mut Vec<f32>, total: &mut usize, samples: &[f32]| {
            window.extend_from_slice(samples);
            *total += samples.len();
        };

        /// Feeds whole 512-sample windows to the VAD and trims old audio,
        /// never dropping samples the VAD or a queued segment still needs.
        fn advance(
            vad: &mut VoiceActivityDetector,
            window: &mut Vec<f32>,
            base: &mut usize,
            fed: &mut usize,
            total: usize,
            retention: usize,
        ) {
            while total - *fed >= 512 {
                let start = *fed - *base;
                let chunk = window[start..start + 512].to_vec();
                vad.accept_waveform(&chunk);
                *fed += 512;
            }

            if window.len() > retention {
                let drop = (window.len() - retention).min(fed.saturating_sub(*base));
                if drop > 0 {
                    window.drain(..drop);
                    *base += drop;
                }
            }
        }

        loop {
            let packet = match reader.next_packet() {
                Ok(packet) => packet,
                Err(_) => break,
            };
            if packet.track_id() != track_id {
                continue;
            }
            let Ok(frame) = decoder.decode(&packet) else {
                continue;
            };

            let channels = frame.spec().channels.count();
            let mono = to_mono(&frame, channels);
            source_decoded += mono.len();

            if !rate_known {
                if let Some(previous) = last_ts {
                    let seconds =
                        packet.ts().saturating_sub(previous) as f64 * time_base.0 / time_base.1;
                    let estimate = frame.frames() as f64 / seconds;
                    if estimate.is_finite() && (8000.0..=192_000.0).contains(&estimate) {
                        source_rate = estimate.round() as u32;
                        resampler = (source_rate != 16000).then(|| {
                            LinearResampler::create(source_rate as i32, 16000).expect("resampler")
                        });
                        rate_known = true;

                        let held = std::mem::take(&mut pending);
                        let pcm = match &resampler {
                            Some(resampler) if !held.is_empty() => resampler.resample(&held, false),
                            _ => held,
                        };
                        feed(&mut window, &mut total, &pcm);
                    }
                }
                last_ts = Some(packet.ts());

                if !rate_known {
                    pending.extend_from_slice(&mono);
                    continue;
                }
            }

            let pcm = match &resampler {
                Some(resampler) => resampler.resample(&mono, false),
                None => mono,
            };
            feed(&mut window, &mut total, &pcm);

            advance(
                &mut self.vad,
                &mut window,
                &mut base,
                &mut fed,
                total,
                RETENTION,
            );
            self.queue_segments(&mut ready);
            self.recognize_ready(&window, base, total, PAD, &mut ready, &mut phrases);

            if let Some(frames) = total_samples.filter(|frames| *frames > 0) {
                progress((source_decoded as f32 / frames as f32).min(1.0), "");
            }
        }

        // flush: whatever is left in the last partial window
        if total > fed && fed >= base {
            let mut tail: Vec<f32> = window[fed - base..].to_vec();
            while tail.len() % 512 != 0 {
                tail.push(0.0);
            }
            for chunk in tail.chunks(512) {
                self.vad.accept_waveform(chunk);
            }
        }
        self.vad.flush();
        self.queue_segments(&mut ready);
        self.recognize_ready(&window, base, total, 0, &mut ready, &mut phrases);

        let duration = total as f32 / 16000.0;
        let mut words = Vec::new();
        let mut sentences = Vec::new();
        for (start, end, text, token_words) in phrases {
            let segment_words = if token_words.is_empty() {
                interpolate(&text, start, end)
            } else {
                token_words
            };

            for (sentence, from, to) in split_sentences(&text, &segment_words, start, end) {
                sentences.push(Seg {
                    start: from,
                    end: to,
                    text: sentence,
                    block: None,
                });
            }

            words.extend(segment_words);
        }

        let elapsed = started.elapsed().as_secs_f32().max(0.001);

        let language = match self.describe_language() {
            // Whisper with `auto` reports nothing: guess from the transcript so
            // the summary and the translation prompt use the right language
            detected if detected == "auto" => {
                let sample = sentences
                    .iter()
                    .take(40)
                    .map(|sentence| sentence.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                guess_language(&sample).to_string()
            }
            other => other,
        };

        Ok(Transcript {
            language,
            duration,
            device: self.device.clone(),
            model: self.spec.label.to_string(),
            phrases: sentences,
            words,
            speed: duration / elapsed,
        })
    }

    fn describe_language(&self) -> String {
        if !self.language.is_empty() {
            self.language.clone()
        } else if self.spec.language.is_empty() {
            "auto".to_string()
        } else {
            self.spec.language.to_string()
        }
    }

    /// Moves finished VAD segments into the queue as absolute sample ranges.
    fn queue_segments(&mut self, ready: &mut std::collections::VecDeque<(usize, usize)>) {
        while let Some(segment) = self.vad.front() {
            let start = segment.start() as usize;
            let end = start + segment.samples().len();
            ready.push_back((start, end));
            self.vad.pop();
        }
    }

    /// Recognizes every queued range that has `pad` samples of follow-up audio.
    fn recognize_ready(
        &mut self,
        window: &[f32],
        base: usize,
        total: usize,
        pad: usize,
        ready: &mut std::collections::VecDeque<(usize, usize)>,
        out: &mut Vec<(f32, f32, String, Vec<Seg>)>,
    ) {
        while let Some(&(start, end)) = ready.front() {
            if end + pad > total {
                break;
            }
            ready.pop_front();

            let from = start.saturating_sub(pad).max(base);
            let to = (end + pad).min(total);
            if to <= from || from < base || to > base + window.len() {
                continue;
            }

            let samples = &window[from - base..to - base];
            let offset = from as f32 / 16000.0;

            let stream = self.recognizer.create_stream();
            stream.accept_waveform(16000, samples);
            self.recognizer.decode(&stream);

            if let Some(result) = stream.get_result() {
                let text = clean_text(&result.text);
                if !text.is_empty() {
                    out.push((offset, offset + samples.len() as f32 / 16000.0, text, token_words(&result, offset)));
                }
            }
        }
    }
}

/// Very small language guess used only when the model reports nothing
/// (Whisper with `auto`). German vs English is enough for our content.
fn guess_language(text: &str) -> &'static str {
    const GERMAN: &[&str] = &[
        "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "sich", "dass", "für",
        "auf", "den", "dem", "des", "werden", "wurde", "auch", "noch", "über", "bei", "aus",
    ];
    const ENGLISH: &[&str] = &[
        "the", "and", "of", "to", "in", "that", "is", "for", "with", "was", "it", "on", "as",
        "at", "be", "this", "have", "from",
    ];

    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != 'ß')
        .filter(|word| !word.is_empty())
        .collect();

    let mut german = words.iter().filter(|word| GERMAN.contains(word)).count() * 2;
    let english = words.iter().filter(|word| ENGLISH.contains(word)).count() * 2;

    // ä/ö/ü/ß are a strong hint
    german += text
        .chars()
        .filter(|c| matches!(c, 'ä' | 'ö' | 'ü' | 'ß' | 'Ä' | 'Ö' | 'Ü'))
        .count();

    if german >= english {
        "de"
    } else {
        "en"
    }
}

/// Drops Whisper's sound annotations (`*music*`, `[applause]`) and keeps the rest.
fn clean_text(text: &str) -> String {
    const TAGS: &[&str] = &[
        "music", "musik", "applause", "silence", "noise", "laughter", "inaudible",
    ];

    let mut kept = String::new();
    for chunk in text.split(['*', '[', ']', '(', ')']) {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        if TAGS.contains(&trimmed.to_ascii_lowercase().as_str()) {
            continue;
        }
        if !kept.is_empty() {
            kept.push(' ');
        }
        kept.push_str(trimmed);
    }

    kept.trim().to_string()
}

const SILERO: &str = "silero_vad.onnx";

// ------------------------------------------------------------------- configs

fn config_for(spec: &ModelSpec, dir: &Path, language: &str) -> Result<OfflineRecognizerConfig, String> {
    let _file = |name: &str| -> Result<String, String> {
        let path = dir.join(name);
        if path.is_file() {
            Ok(path.to_string_lossy().to_string())
        } else {
            Err(format!("{} is missing", path.display()))
        }
    };

    let find = |suffix: &str| -> Result<String, String> {
        let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|err| err.to_string())?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().ends_with(suffix))
                    .unwrap_or(false)
            })
            .collect();
        matches.sort();
        matches
            .into_iter()
            .next()
            .map(|path| path.to_string_lossy().to_string())
            .ok_or_else(|| format!("no *{suffix} in {}", dir.display()))
    };

    let mut config = OfflineRecognizerConfig::default();
    config.model_config.tokens = Some(find("tokens.txt")?);

    match spec.kind {
        Kind::NemoTransducer => {
            config.model_config.transducer = OfflineTransducerModelConfig {
                encoder: Some(find("encoder.onnx")?),
                decoder: Some(find("decoder.onnx")?),
                joiner: Some(find("joiner.onnx")?),
            };
            config.model_config.model_type = Some("nemo_transducer".into());
        }
        Kind::Canary => {
            config.model_config.canary = OfflineCanaryModelConfig {
                encoder: Some(find("encoder.int8.onnx")?),
                decoder: Some(find("decoder.int8.onnx")?),
                src_lang: Some(if language.is_empty() { "de".into() } else { language.into() }),
                tgt_lang: Some(if language.is_empty() { "de".into() } else { language.into() }),
                use_pnc: true,
                ..Default::default()
            };
        }
        Kind::Whisper => {
            config.model_config.whisper = OfflineWhisperModelConfig {
                encoder: Some(find("-encoder.int8.onnx")?),
                decoder: Some(find("-decoder.int8.onnx")?),
                language: Some(language.to_string()).filter(|value| !value.is_empty()),
                task: Some("transcribe".into()),
                // the published exports have no cross-attention outputs, so
                // token timestamps cannot be produced (sherpa-onnx would only
                // warn); words are interpolated inside each segment instead
                enable_token_timestamps: false,
                enable_segment_timestamps: false,
                ..Default::default()
            };
            config.model_config.model_type = Some("whisper".into());
        }
    }

    Ok(config)
}

// -------------------------------------------------------------------- words

/// Per-token timestamps -> words (sherpa-onnx returns raw token strings, which
/// carry the word boundary as a leading space).
fn token_words(result: &OfflineRecognizerResult, offset: f32) -> Vec<Seg> {
    let Some(timestamps) = result.timestamps.as_ref().filter(|t| !t.is_empty()) else {
        return Vec::new();
    };

    let mut words: Vec<Seg> = Vec::new();
    for (index, token) in result.tokens.iter().enumerate() {
        let text = token.replace('\u{2581}', " ");
        if text.trim().is_empty() {
            continue;
        }

        let start = offset + timestamps.get(index).copied().unwrap_or_default();
        let end = offset
            + timestamps
                .get(index + 1)
                .copied()
                .unwrap_or_else(|| start - offset + 0.2);

        let boundary = text.starts_with(' ') || words.is_empty();
        let piece = text.trim_start().to_string();

        match (boundary, words.last_mut()) {
            (true, _) => words.push(Seg {
                start,
                end,
                text: piece,
                block: None,
            }),
            (false, Some(last)) => {
                last.text.push_str(&piece);
                last.end = end;
            }
            (false, None) => words.push(Seg {
                start,
                end,
                text: piece,
                block: None,
            }),
        }
    }

    words
}

/// Fallback: spread the segment duration over its words by character count.
fn interpolate(text: &str, start: f32, end: f32) -> Vec<Seg> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let total: usize = words.iter().map(|word| word.chars().count().max(1)).sum();
    let span = (end - start).max(0.05);
    let mut cursor = start;
    let mut out = Vec::with_capacity(words.len());

    for word in words {
        let share = word.chars().count().max(1) as f32 / total as f32;
        let length = span * share;
        out.push(Seg {
            start: cursor,
            end: (cursor + length).min(end),
            text: word.to_string(),
            block: None,
        });
        cursor += length;
    }

    out
}

/// Splits a segment into sentences at `.`/`!`/`?`. When word timings are
/// available they define the sentence boundaries exactly, otherwise the words
/// are only used to find the split points in the text.
fn split_sentences(
    text: &str,
    words: &[Seg],
    start: f32,
    end: f32,
) -> Vec<(String, f32, f32)> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for (index, chunk) in text.split_inclusive(['.', '!', '?']).enumerate() {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(chunk);
        let _ = index;

        // a sentence ends here; keep very short leftovers together
        if current.chars().count() >= 25 {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }
    if sentences.is_empty() {
        sentences.push(text.trim().to_string());
    }

    if words.is_empty() {
        // no word level data: divide the segment evenly
        let span = (end - start).max(0.05) / sentences.len() as f32;
        return sentences
            .into_iter()
            .enumerate()
            .map(|(index, sentence)| {
                let from = start + span * index as f32;
                (sentence, from, from + span)
            })
            .collect();
    }

    // walk the words and cut where the sentence text ends
    let mut out = Vec::with_capacity(sentences.len());
    let mut word_index = 0usize;
    for sentence in sentences {
        let target: String = normalise(&sentence);
        let mut collected = String::new();
        let first = word_index.min(words.len().saturating_sub(1));
        while word_index < words.len() {
            let word = &words[word_index];
            collected.push_str(&normalise(&word.text));
            word_index += 1;
            if collected.len() >= target.len() {
                break;
            }
        }
        let last = word_index.saturating_sub(1).max(first);
        let from = words.get(first).map(|w| w.start).unwrap_or(start);
        let to = words.get(last).map(|w| w.end).unwrap_or(end);
        out.push((sentence, from.min(to), to));
    }

    out
}

fn normalise(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

// --------------------------------------------------------------------- audio

type AudioFile = (
    Box<dyn FormatReader>,
    Box<dyn Decoder>,
    u32,
    usize,
    u32,
    (f64, f64),
);

fn open_audio(path: &Path) -> Result<AudioFile, String> {
    let file = std::fs::File::open(path).map_err(|err| err.to_string())?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &Default::default(), &Default::default())
        .map_err(|err| format!("unsupported audio format: {err}"))?;

    let format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no audio track found")?;

    let channel_count = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    let track_id = track.id;

    let decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(|err| format!("cannot decode this audio: {err}"))?;

    let time_base = track
        .codec_params
        .time_base
        .map(|base| {
            (
                base.numer as f64 / base.denom as f64,
                1.0f64,
            )
        })
        .unwrap_or((1.0 / sample_rate.max(1) as f64, 1.0));

    Ok((
        format,
        decoder,
        track_id,
        channel_count,
        sample_rate,
        time_base,
    ))
}

fn to_mono(frame: &AudioBufferRef, channels: usize) -> Vec<f32> {
    let mut buffer = SampleBuffer::<f32>::new(frame.capacity() as u64, *frame.spec());
    buffer.copy_interleaved_ref(frame.clone());
    let samples = buffer.samples();

    if channels <= 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

// ----------------------------------------------------------------- downloads

fn ensure_model(
    spec: &ModelSpec,
    models_dir: &Path,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    let dir = models_dir.join(spec.dir);
    if dir.is_dir() && dir.read_dir().map(|mut e| e.next().is_some()).unwrap_or(false) {
        return Ok(());
    }

    std::fs::create_dir_all(models_dir).map_err(|err| err.to_string())?;
    log(format!(
        "downloading {} ({} MB)",
        spec.label, spec.download_mb
    ));

    let archive = models_dir.join(format!("{}.tar.bz2", spec.dir));
    if !archive.is_file() {
        fetch(spec.url, &archive)?;
    }

    let status = std::process::Command::new("tar")
        .arg("xjf")
        .arg(&archive)
        .current_dir(models_dir)
        .status()
        .map_err(|err| format!("tar: {err}"))?;
    if !status.success() {
        return Err(format!("cannot extract {}", archive.display()));
    }
    let _ = std::fs::remove_file(&archive);

    Ok(())
}

fn fetch(url: &str, path: &Path) -> Result<(), String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(None)
        .user_agent("Mozilla/5.0 TranscriptPlayer/1.0")
        .build()
        .new_agent();

    let mut response = agent.get(url).call().map_err(|err| err.to_string())?;
    let partial = PathBuf::from(format!("{}.part", path.display()));
    let mut file = std::fs::File::create(&partial).map_err(|err| err.to_string())?;

    {
        use std::io::{Read, Write};
        let mut reader = response.body_mut().as_reader();
        let mut buffer = vec![0u8; 256 * 1024];
        loop {
            let read = reader.read(&mut buffer).map_err(|err| err.to_string())?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read]).map_err(|err| err.to_string())?;
        }
        file.flush().map_err(|err| err.to_string())?;
    }

    std::fs::rename(&partial, path).map_err(|err| err.to_string())
}

/// Where the ASR models live (`$TRANSCRIBE_MODELS/sherpa`).
pub fn models_dir() -> PathBuf {
    let base = std::env::var_os("TRANSCRIBE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
            home.join("transcribe-models")
        });
    base.join("sherpa")
}

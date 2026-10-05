//! Small local language model for headlines, summaries and translation.
//!
//! Runs Qwen2.5 as an ONNX decoder through ONNX Runtime (`ort`) — no Python,
//! no Ollama, no external daemon. The model is downloaded once into
//! `~/transcribe-models/onnx/` and cached there.
//!
//! The generation loop keeps its own KV cache: the exported graph takes
//! `past_key_values.N.{key,value}` and returns `present.N.{key,value}`, so each
//! step only feeds the newly generated token.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::execution_providers::cuda::CUDA;
use ort::execution_providers::ArenaExtendStrategy;
use ort::execution_providers::cpu::CPU;
use ort::session::Session;
use ort::value::{Tensor, Value};
use tokenizers::Tokenizer;

pub const DEFAULT_MODEL: &str = "qwen2.5:1.5b";
/// The vocabulary/pair pass model. Its int4 export only fits the CPU, where it
/// runs while the GPU keeps the speech recognizer and the headline model.
pub const VOCABULARY_MODEL: &str = "qwen2.5:3b";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// fp16 weights: fastest, but 3 GB of weights alone need a big GPU.
    Fp16,
    /// int4 weights + fp16 activations: small enough for a 4 GB card.
    Q4,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub label: &'static str,
    pub repo: &'static str,
    pub onnx: &'static str,
    /// external weight file next to `onnx`, when the export uses one
    pub onnx_data: Option<&'static str>,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub download_mb: u32,
}

impl ModelSpec {
    pub fn get(label: &str, precision: Precision) -> Option<Self> {
        let spec = match (label, precision) {
            ("qwen2.5:1.5b", Precision::Fp16) => ModelSpec {
                label: "qwen2.5:1.5b",
                repo: "onnx-community/Qwen2.5-1.5B-Instruct",
                onnx: "onnx/model_fp16.onnx",
                onnx_data: Some("onnx/model_fp16.onnx_data"),
                layers: 28,
                kv_heads: 2,
                head_dim: 128,
                download_mb: 3105,
            },
            ("qwen2.5:1.5b", Precision::Q4) => ModelSpec {
                label: "qwen2.5:1.5b",
                repo: "onnx-community/Qwen2.5-1.5B-Instruct",
                onnx: "onnx/model_q4f16.onnx",
                onnx_data: None,
                layers: 28,
                kv_heads: 2,
                head_dim: 128,
                download_mb: 1222,
            },
            ("qwen2.5:3b", Precision::Q4) => ModelSpec {
                label: "qwen2.5:3b",
                repo: "keisuke-miyako/Qwen2.5-3B-Instruct-onnx-int4",
                onnx: "model.onnx",
                onnx_data: Some("model.onnx.data"),
                layers: 36,
                kv_heads: 2,
                head_dim: 128,
                download_mb: 3191,
            },
            // no fp16 export is practical for 3B on a 4 GB GPU
            ("qwen2.5:3b", Precision::Fp16) => return None,
            _ => return None,
        };

        Some(spec)
    }

    /// One directory per model variant, so the graph finds its own files.
    fn dir(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(format!(
            "{}-{}",
            self.label.replace([':', '.'], "-"),
            if self.onnx_data.is_some() { "fp16" } else { "q4f16" }
        ))
    }

    fn file(&self, models_dir: &Path) -> PathBuf {
        self.dir(models_dir).join(local_name(self.onnx))
    }

    fn data_file(&self, models_dir: &Path) -> Option<PathBuf> {
        self.onnx_data
            .map(|remote| self.dir(models_dir).join(local_name(remote)))
    }

    fn tokenizer_file(&self, models_dir: &Path) -> PathBuf {
        self.dir(models_dir).join("tokenizer.json")
    }
}

/// the file name the ONNX graph itself refers to for external weights
fn local_name(remote: &str) -> String {
    remote
        .rsplit('/')
        .next()
        .unwrap_or(remote)
        .to_string()
}

/// int4 is the safe default: it is GPU accelerated (MatMulNBits has a CUDA
/// kernel) and still fits a 4 GB card. fp16 weights are only worth it when the
/// GPU has room for them.
pub fn preferred_precision(vram_mb: u64) -> Precision {
    if vram_mb >= 6000 {
        Precision::Fp16
    } else {
        Precision::Q4
    }
}

/// Total VRAM of the first GPU, in MiB (0 when unknown).
pub fn vram_total_mb() -> u64 {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output();
    let Ok(output) = output else { return 0 };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .and_then(|line| line.trim().parse().ok())
        .unwrap_or(0)
}

pub struct Llm {
    session: Session,
    tokenizer: Tokenizer,
    spec: ModelSpec,
    device: String,
    pub model: String,
    precision: Precision,
}

impl Llm {
    /// Downloads the model if needed and creates the CUDA session.
    pub fn load(
        label: &str,
        models_dir: &Path,
        use_gpu: bool,
        log: &mut dyn FnMut(String),
    ) -> Result<Self, String> {
        // the CPU only ever needs the int4 export, whatever the GPU could hold
        let precision = if use_gpu {
            preferred_precision(vram_total_mb())
        } else {
            Precision::Q4
        };
        let spec = ModelSpec::get(label, precision)
            .or_else(|| ModelSpec::get(label, Precision::Q4))
            .ok_or_else(|| format!("unknown model {label}"))?;

        std::fs::create_dir_all(models_dir).map_err(|err| err.to_string())?;
        download(&spec, models_dir, log)?;

        let session = create_session(&spec, models_dir, use_gpu).map_err(|err| {
            let hint = if err.contains("DefaultLogger") {
                " (ONNX Runtime's default logger is gone — something released the ORT \
                 environment; restart the app)"
            } else {
                ""
            };
            format!("language model session failed: {err}{hint}")
        })?;

        let mut model = Self {
            session,
            tokenizer: Tokenizer::from_file(spec.tokenizer_file(models_dir))
                .map_err(|err| err.to_string())?,
            spec,
            device: if use_gpu { "cuda" } else { "cpu" }.to_string(),
            model: label.to_string(),
            precision,
        };

        // a session that builds can still fail on the very first run
        model
            .generate_once("Hallo", 4, 1.0)
            .map_err(|err| format!("the session cannot run the model: {err}"))?;

        Ok(model)

    }

    /// Short description for the job log: model, device, quantisation.
    pub fn summary(&self) -> String {
        format!(
            "{} on {} ({})",
            self.model,
            self.device,
            match self.precision {
                Precision::Fp16 => "fp16",
                Precision::Q4 => "int4",
            }
        )
    }

    /// One assistant turn. A failed first run is retried once: when the
    /// transcription just released the GPU, cuBLAS can still report a
    /// transient `resource allocation failed`.
    pub fn generate(&mut self, prompt: &str, max_new: usize) -> Result<String, String> {
        self.generate_with(prompt, max_new, 1.0)
    }

    /// [`generate`] with a `repetition_penalty`: greedy decoding otherwise
    /// degenerates into repeating one phrase on long, list-like answers.
    pub fn generate_with(
        &mut self,
        prompt: &str,
        max_new: usize,
        repetition_penalty: f32,
    ) -> Result<String, String> {
        match self.generate_once(prompt, max_new, repetition_penalty) {
            Ok(text) => Ok(text),
            Err(first) => {
                std::thread::sleep(std::time::Duration::from_millis(1500));
                self.generate_once(prompt, max_new, repetition_penalty)
                    .map_err(|second| format!("{second} (first attempt: {first})"))
            }
        }
    }

    fn generate_once(
        &mut self,
        prompt: &str,
        max_new: usize,
        repetition_penalty: f32,
    ) -> Result<String, String> {
        let im_end = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .ok_or("tokenizer has no <|im_end|>")?;

        let chat = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let encoding = self.tokenizer.encode(chat, false).map_err(|err| err.to_string())?;
        let mut tokens: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
        if tokens.is_empty() {
            return Err("empty prompt".to_string());
        }

        let layers = self.spec.layers;
        let (kv_heads, head_dim) = (self.spec.kv_heads, self.spec.head_dim);
        let mut keys: Vec<Vec<f32>> = vec![Vec::new(); layers];
        let mut values: Vec<Vec<f32>> = vec![Vec::new(); layers];
        let mut past_len = 0usize;
        let mut generated: Vec<u32> = Vec::new();
        // which token ids are words, for the repetition penalty below
        let mut word_tokens: HashMap<u32, bool> = HashMap::new();

        for step in 0..max_new {
            let new_tokens: Vec<i64> = if step == 0 {
                tokens.clone()
            } else {
                vec![*tokens.last().unwrap()]
            };
            let seq = new_tokens.len();
            let total = past_len + seq;

            let mut inputs: HashMap<String, Value> = HashMap::new();
            inputs.insert(
                "input_ids".into(),
                Tensor::from_array((vec![1i64, seq as i64], new_tokens.clone()))
                    .map_err(|err| err.to_string())?
                    .into(),
            );
            inputs.insert(
                "attention_mask".into(),
                Tensor::from_array((vec![1i64, total as i64], vec![1i64; total]))
                    .map_err(|err| err.to_string())?
                    .into(),
            );
            // not every export takes `position_ids` (the int4 Qwen 2.5 3B
            // does not), so feed it only when the graph declares it
            if self
                .session
                .inputs()
                .iter()
                .any(|outlet| outlet.name() == "position_ids")
            {
                let positions: Vec<i64> = (past_len..total).map(|p| p as i64).collect();
                inputs.insert(
                    "position_ids".into(),
                    Tensor::from_array((vec![1i64, seq as i64], positions))
                        .map_err(|err| err.to_string())?
                        .into(),
                );
            }
            for layer in 0..layers {
                let shape = vec![1i64, kv_heads as i64, past_len as i64, head_dim as i64];
                inputs.insert(
                    format!("past_key_values.{layer}.key"),
                    Tensor::from_array((shape.clone(), keys[layer].clone()))
                        .map_err(|err| err.to_string())?
                        .into(),
                );
                inputs.insert(
                    format!("past_key_values.{layer}.value"),
                    Tensor::from_array((shape, values[layer].clone()))
                        .map_err(|err| err.to_string())?
                        .into(),
                );
            }

            let outputs = self.session.run(inputs).map_err(|err| err.to_string())?;

            let next = {
                let (shape, logits) = outputs["logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(|err| err.to_string())?;
                let vocab = *shape.last().unwrap_or(&0) as usize;
                if vocab == 0 || logits.len() < vocab {
                    return Err("unexpected logits shape".to_string());
                }
                let mut scores = logits[logits.len() - vocab..].to_vec();
                if repetition_penalty > 1.0 {
                    // only words: the `=` and the line breaks of a list have to
                    // stay available even though they repeat
                    for &token in &generated {
                        let is_word = *word_tokens.entry(token).or_insert_with(|| {
                            self.tokenizer
                                .decode(&[token], false)
                                .map(|text| text.chars().any(char::is_alphanumeric))
                                .unwrap_or(false)
                        });
                        let index = token as usize;
                        if !is_word || index >= scores.len() {
                            continue;
                        }
                        if scores[index] > 0.0 {
                            scores[index] /= repetition_penalty;
                        } else {
                            scores[index] *= repetition_penalty;
                        }
                    }
                    ban_repeat_pair(&mut scores, &generated);
                }
                argmax(&scores) as u32
            };

            for layer in 0..layers {
                let key = outputs
                    .get(format!("present.{layer}.key"))
                    .ok_or_else(|| format!("missing present.{layer}.key"))?
                    .try_extract_tensor::<f32>()
                    .map_err(|err| err.to_string())?;
                let value = outputs
                    .get(format!("present.{layer}.value"))
                    .ok_or_else(|| format!("missing present.{layer}.value"))?
                    .try_extract_tensor::<f32>()
                    .map_err(|err| err.to_string())?;
                keys[layer].clear();
                keys[layer].extend_from_slice(key.1);
                values[layer].clear();
                values[layer].extend_from_slice(value.1);
            }

            past_len = total;
            if next == im_end {
                break;
            }
            generated.push(next);
            tokens.push(next as i64);
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|err| err.to_string())
    }

    /// German (or source-language) headline + summary for a transcript.
    pub fn title_and_description(
        &mut self,
        text: &str,
        language: &str,
    ) -> Result<(String, String), String> {
        let raw = self.generate(&summary_prompt(language, text), 320)?;
        let value = extract_json(&raw)
            .ok_or_else(|| format!("model did not answer with JSON: {}", first_chars(&raw, 160)))?;

        let title = tidy_title(value["title"].as_str().unwrap_or_default());
        let description = first_sentences(value["description"].as_str().unwrap_or_default(), 2);
        if title.is_empty() {
            return Err("model returned no title".to_string());
        }

        Ok((title, description))
    }

    /// Translates sentence by sentence in small batches (same 1:1 mapping).
    pub fn translate(
        &mut self,
        sentences: &[String],
        target: &str,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Vec<String>, String> {
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        let target_name = language_name(target);
        let batch_size = 6;
        let total_batches = sentences.len().div_ceil(batch_size);
        let mut out = Vec::with_capacity(sentences.len());

        for (index, batch) in sentences.chunks(batch_size).enumerate() {
            let lines = match self.batch(batch, &target_name) {
                Some(lines) if lines.len() == batch.len() => lines,
                _ => batch
                    .iter()
                    .map(|sentence| {
                        self.batch(std::slice::from_ref(sentence), &target_name)
                            .and_then(|lines| lines.into_iter().next())
                            .unwrap_or_default()
                    })
                    .collect(),
            };

            out.extend(lines);
            progress((index + 1) as f32 / total_batches as f32);
        }

        Ok(out)
    }
    /// The most relevant words and short phrases of an article, each as
    /// `(original, translation)` — a vocabulary list for a learner.
    ///
    /// When `translations` are given they are shown to the model, so it can copy
    /// the target phrase verbatim, and both sides are then snapped to the exact
    /// wording of the two texts.
    pub fn word_pairs(
        &mut self,
        sentences: &[&str],
        translations: &[&str],
        language: &str,
        target: &str,
        limit: usize,
    ) -> Result<Vec<(String, String)>, String> {
        let text = sentences.join(" ");
        let bilingual = !translations.is_empty();
        let body = if bilingual {
            vocabulary_body(
                sentences,
                translations,
                &language_name(language),
                &language_name(target),
            )
        } else {
            text.clone()
        };

        // a list is where greedy decoding degenerates most, so penalise hard
        let raw = self.generate_with(
            &vocabulary_prompt(language, target, limit, &body, bilingual),
            200 + 60 * limit,
            1.5,
        )?;
        if std::env::var_os("TRANSCRIBE_LOG").is_some() {
            eprintln!("[vocab] raw model answer: {raw}");
        }
        let pairs = parse_word_pairs(&raw).ok_or_else(|| {
            format!("model did not answer with word pairs: {}", first_chars(&raw, 160))
        })?;

        // Snap both sides to the exact wording of the same sentence and keep
        // only what really occurs there verbatim: `pairs.json` finds the word
        // indexes by matching those exact words, and a small model likes to
        // paraphrase, echo the prompt's example, return names, or inflect.
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<(String, String)> = Vec::new();

        for (original, translation) in pairs {
            // the sentence that actually contains the German phrase
            let mut located = None;
            for (index, sentence) in sentences.iter().enumerate() {
                if let Some(exact) = snap_to_text(sentence, &original) {
                    located = Some((index, exact));
                    break;
                }
            }
            let Some((sentence, original)) = located else {
                continue;
            };

            let translation = if bilingual {
                let line = translations.get(sentence).copied().unwrap_or("");
                match snap_to_text(line, &translation) {
                    Some(exact) => exact,
                    None => continue,
                }
            } else {
                translation
            };

            if !is_learnable_pair(&original, &translation) {
                continue;
            }
            if !seen.insert(original.to_lowercase()) {
                continue;
            }
            out.push((original, translation));
            if out.len() >= limit {
                break;
            }
        }

        if out.is_empty() {
            return Err("model returned no usable word pairs".to_string());
        }
        Ok(out)
    }

    /// Vocabulary for the sentences of an article, with the target sentences so
    /// both sides can be quoted verbatim.
    ///
    /// A long prompt makes the graph allocate a `[1, sequence, vocabulary]`
    /// logits buffer, which exhausts a 4 GB GPU, so long articles are asked in
    /// character-bounded chunks and the results are merged.
    pub fn vocabulary(
        &mut self,
        sentences: &[String],
        translations: &[String],
        language: &str,
        target: &str,
        limit: usize,
    ) -> Result<Vec<(String, String)>, String> {
        const MAX_CHARS: usize = 1200;

        // chunk by sentence, so each German chunk keeps its translations
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        let mut size = 0usize;
        for (index, sentence) in sentences.iter().enumerate() {
            let length = sentence.chars().count();
            if !current.is_empty() && size + length > MAX_CHARS {
                chunks.push(std::mem::take(&mut current));
                size = 0;
            }
            size += length + 1;
            current.push(index);
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        if chunks.is_empty() {
            return Err("no sentences to build vocabulary from".to_string());
        }

        let bilingual = !translations.is_empty();
        let mut out: Vec<(String, String)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut last_error = None;

        for chunk in &chunks {
            let german: Vec<&str> = chunk
                .iter()
                .map(|&index| sentences[index].as_str())
                .collect();
            let english: Vec<&str> = if bilingual {
                chunk
                    .iter()
                    .map(|&index| translations.get(index).map(String::as_str).unwrap_or(""))
                    .collect()
            } else {
                Vec::new()
            };

            let share = (limit / chunks.len()).max(2).min(limit);
            match self.word_pairs(&german, &english, language, target, share) {
                Ok(pairs) => {
                    for (de, en) in pairs {
                        if seen.insert(de.to_lowercase()) {
                            out.push((de, en));
                        }
                    }
                }
                Err(err) => last_error = Some(err),
            }
        }

        out.truncate(limit);
        if out.is_empty() {
            return Err(last_error.unwrap_or_else(|| "model returned no word pairs".to_string()));
        }
        Ok(out)
    }

    fn batch(&mut self, batch: &[String], target_name: &str) -> Option<Vec<String>> {
        let numbered = batch
            .iter()
            .enumerate()
            .map(|(index, line)| format!("{}. {}", index + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are a professional translator. Translate each numbered German line into \
             {target_name}. Keep the numbering and output exactly {} lines, nothing else. \
             Translate every line, never copy the German text, never merge or split lines.\n\n{numbered}",
            batch.len(),
        );

        // a line costs roughly 40 tokens plus the prompt
        let max_new = 200 + 140 * batch.len();
        let text = self.generate(&prompt, max_new).ok()?;
        let translations = parse_numbered(&text, batch.len())?;
        let untranslated = translations
            .iter()
            .zip(batch)
            .any(|(translation, source)| same_text(translation, source));

        (!untranslated).then_some(translations)
    }
}

// ------------------------------------------------------------------ downloads

fn download(
    spec: &ModelSpec,
    models_dir: &Path,
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    std::fs::create_dir_all(spec.dir(models_dir)).map_err(|err| err.to_string())?;

    let onnx = spec.file(models_dir);
    let tokenizer = spec.tokenizer_file(models_dir);

    let mut files: Vec<(PathBuf, String)> = vec![
        (onnx.clone(), spec.onnx.to_string()),
        (tokenizer.clone(), "tokenizer.json".to_string()),
    ];
    if let Some(data) = spec.onnx_data {
        files.push((
            spec.data_file(models_dir)
                .ok_or("missing external data path")?,
            data.to_string(),
        ));
    }

    for (path, remote) in files {
        if path.is_file() {
            continue;
        }
        log(format!("downloading {remote} ({} MB)", spec.download_mb));
        let url = format!("https://huggingface.co/{}/resolve/main/{remote}", spec.repo);
        fetch(&url, &path).map_err(|err| format!("{remote}: {err}"))?;
    }

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

// --------------------------------------------------------------- ort session

fn create_session(spec: &ModelSpec, models_dir: &Path, use_gpu: bool) -> Result<Session, String> {
    let mut last = String::new();

    // a session that does not fit can fail transiently while another process is
    // still releasing its GPU memory, so retry a couple of times
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(1200));
        }

        let builder = Session::builder().map_err(|err| err.to_string())?;
        let mut builder = builder
            .with_execution_providers(if use_gpu {
                vec![
                    // grow the arena on demand instead of reserving big chunks:
                    // on a 4 GB card that is the difference between loading and
                    // "Failed to allocate memory for requested buffer"
                    CUDA::default()
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .build(),
                    // the CPU EP stays second in line: ONNX Runtime hands it the
                    // few nodes the CUDA provider does not implement
                    CPU::default().build(),
                ]
            } else {
                vec![CPU::default().build()]
            })
            .map_err(|err| err.to_string())?;
        match builder.commit_from_file(spec.file(models_dir)) {
            Ok(session) => return Ok(session),
            Err(err) => last = err.to_string(),
        }
    }

    Err(last)
}

static RUNTIME: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static RUNTIME_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static INIT_NOTE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
/// The environment created at startup, kept for the whole process.
static ENVIRONMENT: std::sync::OnceLock<std::sync::Arc<ort::environment::Environment>> =
    std::sync::OnceLock::new();
/// What the ONNX Runtime initialisation did (for the job log).
pub fn init_note() -> Option<&'static str> {
    INIT_NOTE.get().map(String::as_str)
}

/// The `libonnxruntime.so` in use, once [`prepare_runtime`] has run.
pub fn runtime_path() -> Option<&'static Path> {
    RUNTIME.get().map(PathBuf::as_path)
}

/// Points ONNX Runtime at a CUDA enabled `libonnxruntime.so` and preloads the
/// CUDA runtime libraries it needs. Returns the library directory in use.
pub fn prepare_runtime() -> Option<PathBuf> {
    // idempotent: the second call (GUI boot) must not re-initialise ORT
    if let Some(dir) = RUNTIME_DIR.get() {
        return Some(dir.clone());
    }

    // One ONNX Runtime is already in the process (sherpa-onnx links it), so
    // adopt that instance instead of loading another copy by path — see
    // `adopt_loaded_runtime`.
    if let Some((lib, version)) = adopt_loaded_runtime() {
        let dir = lib.parent().map(Path::to_path_buf).unwrap_or_default();

        preload(&dir);
        // create ONNX Runtime's environment — and with it the process-wide
        // default logger every session needs — before the recognizer starts
        // creating and releasing handles of its own
        if let Ok(environment) = ort::environment::Environment::current() {
            let _ = ENVIRONMENT.set(environment);
        }
        let _ = INIT_NOTE.set(format!("{version}, already loaded"));
        let _ = RUNTIME.set(lib);
        let _ = RUNTIME_DIR.set(dir.clone());

        return Some(dir);
    }
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Some(dir) = std::env::var_os("TRANSCRIBE_ORT_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    dirs.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ort"));
    if let Some(home) = std::env::var_os("HOME") {
        // the sherpa-onnx GPU bundle ships a CUDA enabled ONNX Runtime; using it
        // for speech recognition and the language model keeps one ORT in the
        // process (same soname, same CUDA provider)
        dirs.push(PathBuf::from(&home).join("sherpa-onnx"));
    }
    dirs.extend(cuda_library_dirs());

    for dir in &dirs {
        let Some(lib) = find_library(dir, "libonnxruntime.so") else {
            continue;
        };

        std::env::set_var("ORT_DYLIB_PATH", &lib);
        preload(dir);

        // Load ONNX Runtime and create its environment *now*, once, on this
        // thread. Creating a session later without a registered default logger
        // fails with "Attempt to use DefaultLogger but none has been
        // registered" — which is what made the language model fall back to the
        // CPU when the recognizer had already touched ORT.
        let note = match ort::init_from(&lib) {
            Ok(builder) => {
                if builder.commit() {
                    "initialised".to_string()
                } else {
                    "already initialised".to_string()
                }
            }
            Err(err) => format!("init_from failed: {err}"),
        };
        let _ = INIT_NOTE.set(note);
        let _ = RUNTIME.set(lib);
        // Create ONNX Runtime's environment (and with it the process-wide
        // default logger) now, before sherpa's recognizer creates and releases
        // handles of its own. See `ENVIRONMENT`.
        if ENVIRONMENT.get().is_none() {
            if let Ok(environment) = ort::environment::Environment::current() {
                let _ = ENVIRONMENT.set(environment);
            }
        }
        let _ = RUNTIME_DIR.set(dir.clone());

        return Some(dir.clone());
    }

    // no CUDA build found: still initialise whatever ORT is on the system
    let _ = ort::init().commit();

    None
}

fn find_library(dir: &Path, name: &str) -> Option<PathBuf> {
    for candidate in [dir.join(name), dir.join("lib").join(name)] {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // onnxruntime-linux-x64-gpu_cudaXX-1.30.0/lib/libonnxruntime.so
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_library(&path, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// ONNX Runtime the process already has, as `(path, version)`.
///
/// sherpa-onnx links ONNX Runtime, so one instance is mapped before `main`
/// runs. `dlopen(NULL, …)` (see [`libloading::os::unix::Library::this`]) only
/// looks at what the process already loaded — it loads nothing — and
/// `OrtGetApiBase` found there belongs to that instance. Loading a second one by
/// path is what split ONNX Runtime's process-wide default logger in two (the
/// dynamic loader may pick a different copy than a path search: sherpa's build
/// script leaves copies next to the binary, and `LD_LIBRARY_PATH` pointing there
/// wins over the binary's runpath). Sessions are built through the API adopted
/// here, so recognizer and language model share one ONNX Runtime — and one
/// default logger, without which session creation fails with "Attempt to use
/// DefaultLogger but none has been registered".
fn adopt_loaded_runtime() -> Option<(PathBuf, String)> {
    use libloading::os::unix::{Library, Symbol};

    let this = Library::this();
    let base = unsafe {
        let get_base: Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
            this.get(b"OrtGetApiBase\0").ok()?;
        let base = get_base();
        if base.is_null() {
            return None;
        }

        let api = ((*base).GetApi)(ort::sys::ORT_API_VERSION);
        if api.is_null() {
            return None;
        }
        // hand the adopted API to `ort`: every later call goes to this instance
        ort::set_api(std::ptr::read(api));
        base
    };
    std::mem::forget(this);

    let version = unsafe {
        let text = ((*base).GetVersionString)();
        if text.is_null() {
            "unknown".to_string()
        } else {
            std::ffi::CStr::from_ptr(text).to_string_lossy().into_owned()
        }
    };
    let path = library_path(base.cast())?;

    Some((path, version))
}

/// File the loader mapped `address` from (used to report which library is in
/// use); `dladdr` only looks up an address, it loads nothing.
fn library_path(address: *const std::ffi::c_void) -> Option<PathBuf> {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
    if unsafe { libc::dladdr(address, info.as_mut_ptr()) } == 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy();

    Some(PathBuf::from(path.into_owned()))
}

/// Directories that may hold libcublas / libcudnn / libcudart.
fn cuda_library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(list) = std::env::var_os("TRANSCRIBE_CUDA_LIBS") {
        dirs.extend(std::env::split_paths(&list));
    }
    for dir in ["/usr/local/cuda/lib64", "/usr/lib64"] {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            dirs.push(dir);
        }
    }
    // versioned toolkits: /usr/local/cuda-12.9/targets/x86_64-linux/lib
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("cuda") {
                continue;
            }
            let dir = entry.path().join("targets/x86_64-linux/lib");
            if dir.is_dir() {
                dirs.push(dir);
            }
        }
    }

    dirs
}

/// `dlopen(..., RTLD_GLOBAL)` for every CUDA library we can find, so the
/// ONNX Runtime CUDA provider resolves them when it loads.
fn preload(dir: &Path) {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.extend(cuda_library_dirs());
    candidates.push(dir.to_path_buf());

    for dir in candidates {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if !name.ends_with(".so") && !name.contains(".so.") {
                continue;
            }
            if !["libcublas", "libcublasLt", "libcudnn", "libcudart", "libnvrtc"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
            {
                continue;
            }
            // RTLD_GLOBAL matters: ONNX Runtime's CUDA provider dlopens
            // `libcublas.so.12` / `libcudnn.so.9` by name, and that only resolves
            // to an already loaded library if its symbols are global.
            use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
            if let Ok(library) = unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
                // the handle must stay open, otherwise the library is unloaded
                std::mem::forget(library);
            }
        }
    }
}

// -------------------------------------------------------------------- prompts

/// One line, no quotes, at most ~70 characters.
fn tidy_title(raw: &str) -> String {
    let cleaned = raw
        .trim()
        .trim_matches(['"', '\'', '*', '#'])
        .replace(['\n', '\r'], " ");
    let cleaned = cleaned.trim();

    if cleaned.chars().count() <= 70 {
        return cleaned.to_string();
    }

    let mut title = String::new();
    for word in cleaned.split_whitespace() {
        if title.chars().count() + word.chars().count() + 1 > 70 {
            break;
        }
        if !title.is_empty() {
            title.push(' ');
        }
        title.push_str(word);
    }
    title
}

/// Keeps at most `count` sentences (models happily write a whole paragraph).
fn first_sentences(raw: &str, count: usize) -> String {
    let text = raw.trim().replace(['\n', '\r'], " ");
    let mut out = String::new();
    let mut sentences = 0;

    for chunk in text.split_inclusive(['.', '!', '?']) {
        out.push_str(chunk.trim());
        out.push(' ');
        sentences += 1;
        if sentences >= count {
            break;
        }
    }

    let trimmed = out.trim();
    if trimmed.is_empty() {
        return text.trim().to_string();
    }
    trimmed.to_string()
}

pub fn summary_prompt(language: &str, text: &str) -> String {
    let name = language_name(language);

    if name == "Deutsch" {
        format!(
            "Du bist Redakteur einer deutschen Nachrichtensendung. Schreibe eine deutsche Schlagzeile \
             (hoechstens 70 Zeichen, keine Anfuehrungszeichen) und eine deutsche Zusammenfassung \
             (ein bis zwei Saetze). Antworte nur mit JSON: {{\"title\": \"...\", \"description\": \"...\"}}\n\n\
             Beispiel: {{\"title\": \"Copilot soll Absturz geplant haben\", \"description\": \"Die Ermittler \
             werfen dem Mann vor, das Flugzeug zum Absturz bringen zu wollen.\"}}\n\nTranskript:\n{text}"
        )
    } else {
        format!(
            "You are an editor. Write a {name} headline (max 70 characters, no quotes) and a {name} \
             summary (one or two sentences) for the transcript below. Answer with JSON only: \
             {{\"title\": \"...\", \"description\": \"...\"}}. Write everything in {name}.\n\nTranscript:\n{text}"
        )
    }
}

/// Asks for `phrase = translation` lines, one per vocabulary item.
///
/// The goal is efficient reading: push the model towards false friends and
/// non-transparent {source} vocabulary, away from names, countries and
/// internationalisms that look the same in {goal}. With `bilingual` the body
/// carries each sentence and its translation, and the model is told to quote
/// both sides verbatim.
pub fn vocabulary_prompt(
    language: &str,
    target: &str,
    limit: usize,
    body: &str,
    bilingual: bool,
) -> String {
    let source = language_name(language);
    let goal = language_name(target);

    if bilingual {
        return format!(
            "You are a {source} teacher preparing a vocabulary list for an advanced {goal} learner. \
             Below are {source} sentences, each followed by its {goal} translation. Extract as many \
             distinct pairs as possible (up to {limit}): from every sentence list every {source} word \
             or short phrase worth learning, false friends as well as words whose {goal} meaning \
             cannot be guessed from the {source} form. Skip names of people, places and countries, \
             and skip words that look almost the same in {goal}. Quote both sides exactly as they \
             appear in the texts: the {source} phrase from the {source} sentence and the {goal} \
             phrase from that sentence's translation. Write one item per line and nothing else as \
             `{source} phrase = {goal} phrase`, one to four words per side, for example:\nbekommen \
             = to receive\ndas Gift = the poison\n\n{body}"
        );
    }

    format!(
        "You are a {source} teacher preparing a vocabulary list for an advanced {goal} learner. \
         From the transcript below extract as many distinct entries as possible (up to {limit}): \
         every {source} word and short expression that teaches something, above all {source} words \
         and short expressions that are easy to confuse with {goal}, or whose {goal} meaning cannot \
         be guessed from the {source} form. Skip names of people, places and countries, and skip \
         words that look almost the same in {goal}. Give the {goal} translation used in the \
         article, in lower case. Write one item per line and nothing else as `{source} phrase = \
         {goal} translation`, for example:\nbekommen = to receive\ndas Gift = the poison\n\n\
         Transcript:\n{body}"
    )
}

/// The bilingual prompt body: each sentence followed by its translation.
fn vocabulary_body(sentences: &[&str], translations: &[&str], source: &str, goal: &str) -> String {
    let mut body = String::new();
    for (index, sentence) in sentences.iter().enumerate() {
        body.push_str(&format!("{source}: {}\n", sentence.trim()));
        if let Some(translation) = translations.get(index) {
            body.push_str(&format!("{goal}: {}\n", translation.trim()));
        }
        body.push('\n');
    }
    body
}

// -------------------------------------------------------------------- parsing

/// Tolerant JSON extraction: models wrap answers in prose, add fences, or get
/// cut off by the token budget, so a truncated object is repaired.
pub fn extract_json(text: &str) -> Option<serde_json::Value> {
    let start = text.find('{')?;
    let mut candidate = text[start..].to_string();

    if let Some(end) = candidate.rfind('}') {
        candidate.truncate(end + 1);
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) {
        return Some(value);
    }

    // repair a truncated answer: close the string and the object
    let mut fixed = candidate.clone();
    if fixed.matches('"').count() % 2 == 1 {
        fixed.push('"');
    }
    if fixed.matches('{').count() > fixed.matches('}').count() {
        fixed.push('}');
    }
    serde_json::from_str(&fixed).ok()
}

/// Splits `1. ...\n2. ...` into the individual translations. Lines that do not
/// start with the next expected number are treated as a continuation.
pub fn parse_numbered(text: &str, expected: usize) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let (index, rest) = match line.split_once('.') {
            Some((head, tail)) => match head.trim().parse::<usize>() {
                Ok(number) => (Some(number), tail.trim()),
                Err(_) => (None, line),
            },
            None => (None, line),
        };

        match index {
            Some(number) if number == out.len() + 1 => out.push(rest.to_string()),
            _ => match out.last_mut() {
                Some(last) => {
                    last.push(' ');
                    last.push_str(rest);
                }
                None => out.push(rest.to_string()),
            },
        }
    }

    if out.len() > expected {
        out.truncate(expected);
    }

    (out.len() == expected && out.iter().all(|line| !line.trim().is_empty())).then_some(out)
}

/// Reads a vocabulary answer. A small model usually answers with the requested
/// `phrase = translation` lines; JSON is accepted too when it produces that.
pub fn parse_word_pairs(text: &str) -> Option<Vec<(String, String)>> {
    parse_word_pair_json(text).or_else(|| parse_pair_lines(text))
}

fn parse_word_pair_json(text: &str) -> Option<Vec<(String, String)>> {
    let value = extract_json(text).or_else(|| extract_json_array(text))?;
    let array = value
        .get("pairs")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.get("words").and_then(serde_json::Value::as_array))
        .or_else(|| value.as_array())?;

    let field = |item: &serde_json::Value, keys: &[&str]| -> String {
        keys.iter()
            .find_map(|key| item.get(*key).and_then(serde_json::Value::as_str))
            .unwrap_or_default()
            .trim()
            .to_string()
    };

    let mut out = Vec::new();
    for item in array {
        let original = field(item, &["de", "german", "source", "original"]);
        let translation = field(item, &["en", "english", "target", "translation"]);
        if !original.is_empty() && !translation.is_empty() {
            out.push((original, translation));
        }
    }

    (!out.is_empty()).then_some(out)
}

/// `German phrase = English translation`, with an optional `1.` / `-` prefix.
fn parse_pair_lines(text: &str) -> Option<Vec<(String, String)>> {
    let mut out = Vec::new();

    for raw in text.lines() {
        let line = raw.trim().trim_start_matches(|c: char| {
            c.is_ascii_digit() || c == '.' || c == ')' || c == '-' || c == '*' || c == ' '
        });
        let (original, translation) = match line.split_once('=') {
            Some(pair) => pair,
            None => match line.split_once('→') {
                Some(pair) => pair,
                None => match line.split_once("->") {
                    Some(pair) => pair,
                    None => continue,
                },
            },
        };

        let clean = |value: &str| -> String {
            value
                .trim()
                .trim_matches(['"', '\'', '*', ' ', ':'])
                .trim()
                .to_string()
        };
        let original = clean(original);
        let translation = clean(translation);
        let lower = original.to_lowercase();

        // skip an echoed prompt header like `Deutsch phrase = English translation`
        if original.is_empty()
            || translation.is_empty()
            || lower.contains("phrase")
            || translation.to_lowercase().contains("translation")
        {
            continue;
        }

        out.push((original, translation));
    }

    (!out.is_empty()).then_some(out)
}

/// [`extract_json`] for an answer that is a bare `[...]` instead of an object.
fn extract_json_array(text: &str) -> Option<serde_json::Value> {
    let start = text.find('[')?;
    let mut candidate = text[start..].to_string();

    if let Some(end) = candidate.rfind(']') {
        candidate.truncate(end + 1);
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) {
        return Some(value);
    }

    // repair a truncated answer: close the string, then the item, then the array
    let mut fixed = candidate;
    if fixed.matches('"').count() % 2 == 1 {
        fixed.push('"');
    }
    if fixed.matches('{').count() > fixed.matches('}').count() {
        fixed.push('}');
    }
    if fixed.matches('[').count() > fixed.matches(']').count() {
        fixed.push(']');
    }
    serde_json::from_str(&fixed).ok()
}


/// Articles a model may add before a noun; ignored when comparing spellings.
const ARTICLES: &[&str] = &[
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "eines", "the",
    "a", "an", "to",
];

/// Function words: a phrase made only of these teaches nothing.
const STOPWORDS: &[&str] = &[
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "eines", "und",
    "oder", "aber", "doch", "jedoch", "auch", "nicht", "kein", "keine", "keinen", "keinem", "mit",
    "ohne", "für", "von", "vom", "zu", "zum", "zur", "bei", "beim", "nach", "aus", "in", "im",
    "an", "am", "auf", "über", "unter", "vor", "hinter", "neben", "zwischen", "durch", "gegen",
    "um", "als", "wie", "wenn", "dass", "weil", "damit", "ob", "sie", "er", "es", "ich", "wir",
    "ihr", "man", "sich", "sein", "seine", "seinen", "ihre", "ihren", "ihrem", "habe", "hat",
    "hatte", "haben", "wird", "werden", "wurde", "wurden", "ist", "sind", "war", "waren", "soll",
    "sollte", "kann", "könnte", "muss", "darf", "will", "wollte", "mag", "lässt", "lassen",
    "macht", "machen", "geht", "gehen", "kommt", "kommen", "gibt", "geben", "nimmt", "nehmen",
    "jede", "jeder", "jedes", "alle", "allen", "allem", "dieser", "diese", "dieses", "noch",
    "schon", "nur", "so", "da", "dort", "hier", "mehr", "sehr", "bereits", "zudem", "deshalb",
    "trotzdem", "zwar", "denn", "dann", "also",
];

/// `false` when the phrase is only function words (`und`, `in die`, `soll jedoch`).
fn is_content_phrase(text: &str) -> bool {
    text.split_whitespace().any(|word| {
        let word: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        !word.is_empty() && !STOPWORDS.contains(&word.as_str())
    })
}

/// `true` for a name-like phrase: two or more words, each starting upper case
/// ("Abu Dhabi", "Vereinigte Arabische Emirate").
fn is_proper_noun_phrase(text: &str) -> bool {
    let mut words = 0usize;
    for word in text.split_whitespace() {
        let Some(first) = word.chars().find(|c| c.is_alphabetic()) else {
            continue;
        };
        words += 1;
        if !first.is_uppercase() {
            return false;
        }
    }
    words >= 2
}

/// Keeps only entries worth learning: no proper noun, no internationalism that
/// is nearly the same in the target language.
fn is_learnable_pair(original: &str, translation: &str) -> bool {
    // names and countries come back fully title cased ("Netanyahu", "Egypt",
    // "United Arab Emirates"); a mere sentence-case "President" is fine
    let titled = translation
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .all(|word| {
            word.chars()
                .find(|c| c.is_alphabetic())
                .map(char::is_uppercase)
                .unwrap_or(false)
        });
    if titled {
        return false;
    }

    let source = comparable(original);
    let target = comparable(translation);
    if source.is_empty()
        || target.is_empty()
        || source == target
        || !is_content_phrase(original)
        || is_proper_noun_phrase(original)
    {
        return false;
    }

    // a shared prefix ("telefonat"/"telephone", "massaker"/"massacre") or a
    // high edit similarity ("präsident"/"president", "golf"/"gulf") is a
    // cognate a reader can guess
    if common_prefix(&source, &target) >= 4 && source.len().min(target.len()) >= 6 {
        return false;
    }

    edit_similarity(&source, &target) < 0.75
}

/// Lowercase letters and digits only, with a leading article removed.
fn comparable(text: &str) -> String {
    let mut out = String::new();

    for (index, word) in text.split_whitespace().enumerate() {
        let word: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if word.is_empty() || (index == 0 && ARTICLES.contains(&word.as_str())) {
            continue;
        }
        out.push_str(&word);
    }

    out
}

/// The phrase exactly as it is written in `text`, or `None` when its words are
/// not a contiguous run there. Keeping the original wording is what lets
/// `pairs.json` find the word indexes later.
fn snap_to_text(text: &str, phrase: &str) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let needles: Vec<String> = phrase
        .split_whitespace()
        .map(normalize_token)
        .filter(|word| !word.is_empty())
        .collect();
    if needles.is_empty() {
        return None;
    }

    if let Some(start) = find_span(&words, &needles) {
        return Some(join_span(&words[start..start + needles.len()]));
    }

    // the model may add a leading article the sentence does not have
    let trimmed: Vec<String> = needles
        .iter()
        .skip_while(|word| ARTICLES.contains(&word.as_str()))
        .cloned()
        .collect();
    if trimmed.len() < needles.len() {
        if let Some(start) = find_span(&words, &trimmed) {
            return Some(join_span(&words[start..start + trimmed.len()]));
        }
    }

    None
}

fn find_span(words: &[&str], needles: &[String]) -> Option<usize> {
    if needles.is_empty() || needles.len() > words.len() {
        return None;
    }
    let normalized: Vec<String> = words.iter().map(|word| normalize_token(word)).collect();
    (0..=words.len() - needles.len())
        .find(|&start| normalized[start..start + needles.len()] == needles[..])
}

/// Joins the exact words of a span, dropping only the punctuation that clings
/// to its edges (`Hamas-Angriff:`, `Golf.`, `»groß`).
fn join_span(words: &[&str]) -> String {
    let mut out: Vec<String> = words.iter().map(|word| word.to_string()).collect();
    if let Some(first) = out.first_mut() {
        *first = first
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
    }
    if let Some(last) = out.last_mut() {
        *last = last
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
    }
    out.join(" ")
}

/// Lowercase letters and digits only, so punctuation does not block a match.
fn normalize_token(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// 1.0 for equal strings, 0.0 for completely different ones.
fn edit_similarity(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];

    for i in 1..=a.len() {
        current[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            current[j] = (previous[j] + 1)
                .min(current[j - 1] + 1)
                .min(previous[j - 1] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    let distance = previous[b.len()];
    1.0 - distance as f32 / a.len().max(b.len()) as f32
}

pub fn same_text(a: &str, b: &str) -> bool {
    let normalise = |text: &str| {
        text.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect::<String>()
    };
    let (a, b) = (normalise(a), normalise(b));
    !a.is_empty() && a == b
}

/// Bans a token that would repeat the pair generated just before, so an
/// answer cannot lock into the same two-token cycle.
fn ban_repeat_pair(scores: &mut [f32], generated: &[u32]) {
    let n = 3;
    if generated.len() < n {
        return;
    }

    let prefix = &generated[generated.len() - (n - 1)..];
    for start in 0..generated.len().saturating_sub(n - 1) {
        if &generated[start..start + n - 1] == prefix {
            let banned = generated[start + n - 1] as usize;
            if banned < scores.len() {
                scores[banned] = f32::NEG_INFINITY;
            }
        }
    }
}
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}


fn first_chars(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

pub fn language_name(code: &str) -> String {
    match code.split(['-', '_']).next().unwrap_or(code) {
        "de" => "Deutsch".to_string(),
        "en" => "English".to_string(),
        "fr" => "Francais".to_string(),
        "es" => "Espanol".to_string(),
        "it" => "Italiano".to_string(),
        "nl" => "Nederlands".to_string(),
        "pl" => "Polski".to_string(),
        "pt" => "Portugues".to_string(),
        "tr" => "Turkce".to_string(),
        "ru" => "Russkiy".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod vocabulary_tests {
    use super::*;

    #[test]
    fn parses_numbered_and_arrowed_lines() {
        let answer = "1. Botschaft = embassy\n2. Angriff -> attack\n- Waffe = weapon\n";
        assert_eq!(
            parse_word_pairs(answer),
            Some(vec![
                ("Botschaft".to_string(), "embassy".to_string()),
                ("Angriff".to_string(), "attack".to_string()),
                ("Waffe".to_string(), "weapon".to_string()),
            ])
        );
    }

    #[test]
    fn drops_an_echoed_prompt_header() {
        let answer = "Deutsch phrase = English translation\nBotschaft = embassy\n";
        assert_eq!(
            parse_word_pairs(answer),
            Some(vec![("Botschaft".to_string(), "embassy".to_string())])
        );
    }

    #[test]
    fn parses_a_json_answer_too() {
        let answer = "```json\n{\"pairs\":[{\"de\":\"Waffe\",\"en\":\"weapon\"}]}\n```";
        assert_eq!(
            parse_word_pairs(answer),
            Some(vec![("Waffe".to_string(), "weapon".to_string())])
        );
    }


    #[test]
    fn snaps_a_phrase_to_the_original_wording() {
        let text = "Der israelische Premier bestreitet jede Kenntnis, soll jedoch aus den \
                    Emiraten gewarnt haben.";
        assert_eq!(snap_to_text(text, "kenntnis").as_deref(), Some("Kenntnis"));
        assert_eq!(
            snap_to_text(text, "aus den emiraten").as_deref(),
            Some("aus den Emiraten")
        );
        // a leading article the sentence does not have is dropped
        assert_eq!(snap_to_text(text, "der Premier").as_deref(), Some("Premier"));
        // anything that is not literally there is refused
        assert_eq!(snap_to_text(text, "Angriff"), None);
    }

    #[test]
    fn drops_names_countries_and_cognates() {
        assert!(!is_learnable_pair("Israel", "Israel"));
        assert!(!is_learnable_pair("Ägypten", "Egypt"));
        assert!(!is_learnable_pair(
            "Vereinigte Arabische Emirate",
            "United Arab Emirates"
        ));
        assert!(!is_learnable_pair("Präsident", "the president"));
        assert!(!is_learnable_pair("Golf", "gulf"));
        assert!(!is_learnable_pair("Telefonat", "telephone call"));
        assert!(!is_learnable_pair("Region", "region"));
        assert!(!is_learnable_pair("Fake News", "fake news"));
    }

    #[test]
    fn keeps_false_friends_and_unguessable_words() {
        assert!(is_learnable_pair("bekommen", "to receive"));
        assert!(is_learnable_pair("das Gift", "the poison"));
        assert!(is_learnable_pair("die Wirtschaft", "the economy"));
        assert!(is_learnable_pair("Verleumdungsklage", "defamation lawsuit"));
        assert!(!is_learnable_pair("und", "and"));
        assert!(!is_learnable_pair("in die", "into"));
        assert!(!is_learnable_pair("soll jedoch", "may be"));
        assert!(!is_learnable_pair("Abu Dhabi", "capital city"));
        assert!(is_learnable_pair("Druck auf", "pressure"));
        assert!(is_learnable_pair("vertrauliche Gespräche", "confidential talks"));
        assert!(is_learnable_pair(
            "Präsident der Emirate",
            "President of the Emirates"
        ));
        assert!(!is_learnable_pair("Netanyahu", "Benjamin Netanyahu"));
    }
}


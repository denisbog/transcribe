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

/// Models the user can pick from, named like the Ollama tags they replace.
pub const MODELS: &[&str] = &["qwen2.5:3b", "qwen2.5:1.5b"];
pub const DEFAULT_MODEL: &str = "qwen2.5:1.5b";

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
}

impl Llm {
    /// Downloads the model if needed and creates the CUDA session.
    pub fn load(
        label: &str,
        models_dir: &Path,
        use_gpu: bool,
        log: &mut dyn FnMut(String),
    ) -> Result<Self, String> {
        if !use_gpu {
            return Err("the language model runs on the GPU only".to_string());
        }

        let spec = ModelSpec::get(label, preferred_precision(vram_total_mb()))
            .or_else(|| ModelSpec::get(label, Precision::Q4))
            .ok_or_else(|| format!("unknown model {label}"))?;

        std::fs::create_dir_all(models_dir).map_err(|err| err.to_string())?;
        download(&spec, models_dir, log)?;

        // GPU only. The int4 export keeps fp16 activations, which the CPU
        // execution provider cannot run (`Mul ... GetElementType is not
        // implemented`), so a CPU fallback would only hide a broken GPU setup
        // behind a slow, wrong answer. Report the real error instead.
        let session = create_session(&spec, models_dir).map_err(|err| {
            let hint = if err.contains("DefaultLogger") {
                " (ONNX Runtime's default logger is gone — something released the ORT \
                 environment; restart the app)"
            } else {
                ""
            };
            format!("CUDA session failed: {err}{hint}")
        })?;

        let mut model = Self {
            session,
            tokenizer: Tokenizer::from_file(spec.tokenizer_file(models_dir))
                .map_err(|err| err.to_string())?,
            spec,
            device: "cuda".to_string(),
            model: label.to_string(),
        };

        // a session that builds can still fail on the very first run
        model
            .generate_once("Hallo", 4)
            .map_err(|err| format!("the CUDA session cannot run the model: {err}"))?;

        Ok(model)
    }

    /// Short description for the job log: model, device, quantisation.
    pub fn summary(&self) -> String {
        format!(
            "{} on {} ({})",
            self.model,
            self.device,
            if self.spec.onnx_data.is_some() {
                "fp16"
            } else {
                "int4"
            }
        )
    }

    /// One assistant turn. A failed first run is retried once: when the
    /// transcription just released the GPU, cuBLAS can still report a
    /// transient `resource allocation failed`.
    pub fn generate(&mut self, prompt: &str, max_new: usize) -> Result<String, String> {
        match self.generate_once(prompt, max_new) {
            Ok(text) => Ok(text),
            Err(first) => {
                std::thread::sleep(std::time::Duration::from_millis(1500));
                self.generate_once(prompt, max_new)
                    .map_err(|second| format!("{second} (first attempt: {first})"))
            }
        }
    }

    fn generate_once(&mut self, prompt: &str, max_new: usize) -> Result<String, String> {
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
            let positions: Vec<i64> = (past_len..total).map(|p| p as i64).collect();
            inputs.insert(
                "position_ids".into(),
                Tensor::from_array((vec![1i64, seq as i64], positions))
                    .map_err(|err| err.to_string())?
                    .into(),
            );
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
                argmax(&logits[logits.len() - vocab..]) as u32
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

fn create_session(spec: &ModelSpec, models_dir: &Path) -> Result<Session, String> {
    let mut last = String::new();

    // a session that does not fit can fail transiently while another process is
    // still releasing its GPU memory, so retry a couple of times
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(1200));
        }

        let builder = Session::builder().map_err(|err| err.to_string())?;
        let mut builder = builder
            .with_execution_providers([
                // grow the arena on demand instead of reserving big chunks:
                // on a 4 GB card that is the difference between loading and
                // "Failed to allocate memory for requested buffer"
                CUDA::default()
                    .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                    .build(),
                // the CPU EP stays second in line: ONNX Runtime hands it the
                // few nodes the CUDA provider does not implement
                CPU::default().build(),
            ])
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


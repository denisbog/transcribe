//! Vocabulary pairs from an external model through the `pi` CLI.
//!
//! The local ONNX model is small and runs on the CPU. When a stronger model is
//! wanted, the `--pairs-provider`/`--pairs-model` flags (or the Article screen
//! toggle) ask `pi` in print mode for the same `de = en` list instead.

use std::ffi::OsString;
use std::io::Write;
use std::process::{Command, Stdio};

use crate::onnx_llm;

/// Provider and model id passed to `pi --provider` / `pi --model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairsModel {
    pub provider: String,
    pub model: String,
}

impl Default for PairsModel {
    fn default() -> Self {
        Self {
            provider: "deepinfra".to_string(),
            // DeepInfra serves this model under the full `deepinfra/…`
            // reference; a bare id makes `pi` warn (benignly) and use it as a
            // custom model id
            model: "deepseek-ai/DeepSeek-V4.1-Flash".to_string(),
        }
    }
}



impl PairsModel {
    pub fn label(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }

    /// The defaults, overridden by `TRANSCRIBE_PAIRS_PROVIDER` and
    /// `TRANSCRIBE_PAIRS_MODEL`.
    pub fn from_env() -> Self {
        let mut model = Self::default();
        if let Ok(provider) = std::env::var("TRANSCRIBE_PAIRS_PROVIDER") {
            if !provider.trim().is_empty() {
                model.provider = provider.trim().to_string();
            }
        }
        if let Ok(name) = std::env::var("TRANSCRIBE_PAIRS_MODEL") {
            if !name.trim().is_empty() {
                model.model = name.trim().to_string();
            }
        }
        model.strip_provider_prefix();
        model
    }

    /// Accepts a `provider/` prefix in the model reference, so both
    /// `--pairs-model deepseek-ai/DeepSeek-V4.1-Flash` and
    /// `--pairs-model deepinfra/deepseek-ai/DeepSeek-V4.1-Flash` work. The
    /// prefix only wins when it names the selected provider.
    pub fn strip_provider_prefix(&mut self) {
        let prefix = format!("{}/", self.provider);
        if let Some(rest) = self.model.strip_prefix(&prefix) {
            if !rest.is_empty() {
                self.model = rest.to_string();
            }
        }
    }
}

/// `true` when the environment selects an external model.
pub fn enabled_from_env() -> bool {
    std::env::var_os("TRANSCRIBE_PAIRS_PROVIDER").is_some()
        || std::env::var_os("TRANSCRIBE_PAIRS_MODEL").is_some()
}

/// The `pi` binary: `$TRANSCRIBE_PI`, else `pi` from `PATH`.
fn pi_binary() -> OsString {
    std::env::var_os("TRANSCRIBE_PI").unwrap_or_else(|| OsString::from("pi"))
}

/// Asks `pi` for the vocabulary pairs of the bilingual sentences.
///
/// The prompt goes in over stdin, so even a long article stays below any
/// command-line limit, and `--no-tools` keeps the answer a single model reply.
pub fn vocabulary(
    pairs_model: &PairsModel,
    sentences: &[String],
    translations: &[String],
    language: &str,
    target: &str,
    limit: usize,
    log: &mut dyn FnMut(String),
) -> Result<Vec<(String, String)>, String> {
    if sentences.is_empty() {
        return Err("no sentences to build vocabulary from".to_string());
    }

    let body = bilingual_body(sentences, translations, language, target);
    let prompt = onnx_llm::vocabulary_prompt(language, target, limit, &body, true);

    log(format!(
        "asking pi for pairs ({}, up to {limit})",
        pairs_model.label()
    ));

    let mut child = Command::new(pi_binary())
        .arg("--print")
        .arg("--no-tools")
        // extraction needs no chain-of-thought; without this a reasoning model
        // spends a minute "thinking" over one short list
        .arg("--thinking")
        .arg("off")
        .arg("--provider")
        .arg(&pairs_model.provider)
        .arg("--model")
        .arg(&pairs_model.model)
        // keep session files out of the user's project
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("cannot run pi: {err}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .map_err(|err| format!("cannot send the prompt to pi: {err}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|err| format!("pi did not finish: {err}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(warning) = concise_error(&stderr) {
        log(format!("pi: {warning}"));
    }

    // The `pi` extension sometimes exits non-zero after printing a good answer
    // (unknown model pricing, session teardown), so trust the text when it
    // carries pairs and only report the exit status when there are none.
    let mut seen = std::collections::HashSet::new();
    let pairs = onnx_llm::parse_word_pairs(&stdout)
        .unwrap_or_default()
        .into_iter()
        .filter(|(de, en)| !de.is_empty() && !en.is_empty() && seen.insert(de.to_lowercase()))
        .take(limit)
        .collect::<Vec<_>>();
    if !pairs.is_empty() {
        return Ok(pairs);
    }

    if !output.status.success() {
        return Err(format!(
            "pi exited with {}: {}",
            output.status,
            concise_error(&stderr).unwrap_or_else(|| "no output".to_string())
        ));
    }

    let head = stdout.trim().chars().take(160).collect::<String>();
    Err(format!("pi answered without pairs: {head}"))
}

/// A short, useful line from `pi`'s stderr: the first `Error:`/`Warning:` line,
/// else the last short non-empty line. The bundled stack traces are a single
/// multi-megabyte line, so longer lines are skipped.
fn concise_error(stderr: &str) -> Option<String> {
    // `pi` recovers from an unknown model id by using it as a custom model id;
    // that warning is noise, so drop it before looking for something useful.
    let lines: Vec<&str> = stderr
        .lines()
        .map(str::trim)
        .filter(|line| !line.contains("Using custom model id"))
        .collect();
    let line = lines
        .iter()
        .copied()
        .find(|line| line.starts_with("Error:") || line.starts_with("Warning:"))
        .or_else(|| {
            lines
                .iter()
                .rev()
                .copied()
                .find(|line| !line.is_empty() && line.chars().count() < 200)
        })?;
    Some(line.chars().take(200).collect())
}

/// `German: ...\nEnglish: ...` per sentence, the body `vocabulary_prompt` expects.
fn bilingual_body(
    sentences: &[String],
    translations: &[String],
    language: &str,
    target: &str,
) -> String {
    let source = onnx_llm::language_name(language);
    let goal = onnx_llm::language_name(target);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_a_matching_provider_prefix() {
        let mut model = PairsModel {
            provider: "deepinfra".to_string(),
            model: "deepinfra/deepseek-ai/DeepSeek-V4.1-Flash".to_string(),
        };
        model.strip_provider_prefix();
        assert_eq!(model.model, "deepseek-ai/DeepSeek-V4.1-Flash");
        assert_eq!(model.label(), "deepinfra/deepseek-ai/DeepSeek-V4.1-Flash");

        // a bare id (or a prefix that is not the provider) is left alone
        let mut bare = PairsModel::default();
        bare.strip_provider_prefix();
        assert_eq!(bare.model, "deepseek-ai/DeepSeek-V4.1-Flash");
    }

    #[test]
    fn bilingual_body_labels_both_sides() {
        let body = bilingual_body(
            &["Hallo Welt.".to_string()],
            &["Hello world.".to_string()],
            "de",
            "en",
        );
        assert_eq!(body, "Deutsch: Hallo Welt.\nEnglish: Hello world.\n\n");
    }

    #[test]
    fn env_overrides_the_default_model() {
        // the default is the requested provider/model
        let default = PairsModel::default();
        assert_eq!(default.provider, "deepinfra");
        assert_eq!(default.model, "deepseek-ai/DeepSeek-V4.1-Flash");
        assert_eq!(default.label(), "deepinfra/deepseek-ai/DeepSeek-V4.1-Flash");
    }
}

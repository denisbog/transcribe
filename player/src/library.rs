//! On-disk cache: one folder per article.
//!
//! ```text
//! ~/transcribe-library/
//!   2026-10-03-flydubai-co-pilot/
//!     meta.json            title, description, url, language, model, ...
//!     audio.mp3            the downloaded audio
//!     audio.wav            16 kHz mono, input for the transcriber
//!     transcription.json   word level segments
//!     transcript.json      sentence level segments
//!     translation.json     {"target": "en", "sentences": [...]}
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const META: &str = "meta.json";
pub const WORDS: &str = "transcription.json";
pub const PHRASES: &str = "transcript.json";
pub const TRANSLATION: &str = "translation.json";
pub const VOCABULARY: &str = "vocabulary.json";
pub const PAIRS: &str = "pairs.json";
/// The output folder inside each crawler article (`<article>/transcribe/`).
pub const TRANSCRIPT_DIR: &str = "transcribe";

pub fn default_audio() -> String {
    "audio.mp3".to_string()
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            id: String::new(),
            audio: default_audio(),
            title: String::new(),
            description: String::new(),
            source_url: String::new(),
            article: String::new(),
            language: String::new(),
            target: String::new(),
            duration: 0.0,
            model: String::new(),
            device: String::new(),
            words: 0,
            sentences: 0,
            created: 0,
            status: String::new(),
            error: None,
        }
    }
}

/// A timed piece of text: one word or one sentence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Seg {
    pub start: f32,
    pub end: f32,
    pub text: String,
    /// The `article.json` block this sentence belongs to (`None` for words and
    /// for the spoken metadata around the article body).
    #[serde(default)]
    pub block: Option<usize>,
}

/// `translation.json`: the sentence-level translation, aligned 1:1 with the
/// sentence segments of `transcription.json`, plus the `article.json` paragraph
/// alignment used for 1:1 patching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Translation {
    #[serde(default)]
    pub target: String,
    /// target-language sentences, parallel to the source sentences
    #[serde(default)]
    pub sentences: Vec<String>,
    /// source-language sentences (article tree output only)
    #[serde(default)]
    pub source: Vec<String>,
    /// one entry per `article.json` paragraph
    #[serde(default)]
    pub blocks: Vec<TranslationBlock>,
}

/// One `article.json` paragraph inside `translation.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranslationBlock {
    /// index into the article's `blocks[]`
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub kind: String,
    /// first sentence of the paragraph in `source`/`sentences`
    #[serde(default)]
    pub first: usize,
    #[serde(default)]
    pub count: usize,
    /// the paragraph translation, sentences joined with a space
    #[serde(default)]
    pub translation: String,
}

/// One vocabulary entry: the original phrase and its translation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WordPair {
    #[serde(alias = "german", alias = "source", alias = "original")]
    pub de: String,
    #[serde(alias = "english", alias = "target", alias = "translation")]
    pub en: String,
}

/// `vocabulary.json`: the most relevant word pairs of an article.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Vocabulary {
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub pairs: Vec<WordPair>,
}

/// One vocabulary pairing: the word indexes of the original phrase and of its
/// translation inside one sentence of a block.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairRef {
    /// index of the sentence inside the block's `source`/`sentences` arrays
    pub sentence: usize,
    #[serde(default)]
    pub de: String,
    #[serde(default)]
    pub en: String,
    pub source: Vec<usize>,
    pub target: Vec<usize>,
}

/// One `article.json` block in `pairs.json`, mirroring `translation.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PairsBlock {
    /// index into the article's `blocks[]`
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub pairs: Vec<PairRef>,
}

/// `pairs.json`: the word pairings of original and translation, grouped the
/// same way as `article.json`'s `blocks[]` (images skipped).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Pairs {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub blocks: Vec<PairsBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub id: String,
    /// the audio file inside the article folder (kept in its original format)
    #[serde(default = "default_audio")]
    pub audio: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub source_url: String,

    /// known article text that was force-aligned to the audio (path or `pasted`)
    #[serde(default)]
    pub article: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub duration: f32,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub words: usize,
    #[serde(default)]
    pub sentences: usize,
    #[serde(default)]
    pub created: i64,
    /// `done` or `failed`
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
}

impl Meta {
    pub fn created_text(&self) -> String {
        use chrono::{DateTime, Local};
        DateTime::from_timestamp(self.created, 0)
            .map(|utc| {
                let local: DateTime<Local> = utc.into();
                local.format("%Y-%m-%d %H:%M").to_string()
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct Item {
    pub dir: PathBuf,
    pub meta: Meta,
}

impl Item {
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// The audio file of this article. Falls back to any `audio.*` in the
    /// folder, because older metadata (or an article that is still being
    /// processed) may not name it.
    pub fn audio_path(&self) -> Option<PathBuf> {
        if !self.meta.audio.is_empty() {
            let path = self.dir.join(&self.meta.audio);
            if path.is_file() {
                return Some(path);
            }
        }

        std::fs::read_dir(&self.dir)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.is_file()
                    && path
                        .file_name()
                        .map(|name| name.to_string_lossy().starts_with("audio."))
                        .unwrap_or(false)
            })
    }

    pub fn translation(&self) -> Translation {
        std::fs::read_to_string(self.path(TRANSLATION))
            .ok()
            .and_then(|raw| serde_json::from_str::<Translation>(&raw).ok())
            .unwrap_or_default()
    }

    /// The article's vocabulary pairs, empty when the pass has not run.
    pub fn vocabulary(&self) -> Vec<WordPair> {
        std::fs::read_to_string(self.path(VOCABULARY))
            .ok()
            .and_then(|raw| serde_json::from_str::<Vocabulary>(&raw).ok())
            .map(|doc| doc.pairs)
            .unwrap_or_default()
    }

    /// The article's word pairings, grouped by `article.json` block.
    pub fn pairs(&self) -> Vec<PairsBlock> {
        std::fs::read_to_string(self.path(PAIRS))
            .ok()
            .and_then(|raw| serde_json::from_str::<Pairs>(&raw).ok())
            .map(|doc| doc.blocks)
            .unwrap_or_default()
    }
}

/// Where articles are cached (`$TRANSCRIBE_LIBRARY`, else `~/transcribe-library`).
pub fn root() -> PathBuf {
    if let Some(dir) = std::env::var_os("TRANSCRIBE_LIBRARY") {
        return PathBuf::from(dir);
    }

    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join("transcribe-library")
}


pub fn read_meta(dir: &Path) -> Option<Meta> {
    let raw = std::fs::read_to_string(dir.join(META)).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn write_meta(dir: &Path, meta: &Meta) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    let raw = serde_json::to_string_pretty(meta).map_err(|err| err.to_string())?;
    std::fs::write(dir.join(META), raw).map_err(|err| err.to_string())
}


/// A unique, readable folder name for a new article.
pub fn directory(id_source: &str) -> PathBuf {
    let root = root();
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M").to_string();
    let base = slug(id_source);
    let mut candidate = root.join(format!("{stamp}-{base}"));
    let mut counter = 2;
    while candidate.exists() {
        candidate = root.join(format!("{stamp}-{base}-{counter}"));
        counter += 1;
    }
    candidate
}

/// Lowercase, ASCII, dash separated (used for folder names).
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;

    for ch in text.chars() {
        let mapped = match ch {
            'ä' | 'Ä' => 'a',
            'ö' | 'Ö' => 'o',
            'ü' | 'Ü' => 'u',
            'ß' => 's',
            'é' | 'è' | 'ê' | 'É' => 'e',
            'à' | 'â' | 'á' => 'a',
            'ç' => 'c',
            other => other,
        };

        if mapped.is_ascii_alphanumeric() {
            out.push(mapped.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }

        if out.len() >= 60 {
            break;
        }
    }

    let slug = out.trim_matches('-').to_string();
    if slug.is_empty() {
        "article".to_string()
    } else {
        slug
    }
}

/// Best effort file name from a URL (used when no title exists yet).
pub fn name_from_url(url: &str) -> String {
    let without_query = url.split(['?', '#']).next().unwrap_or(url);
    let last = without_query.rsplit('/').next().unwrap_or(without_query);
    let stem = last.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(last);
    if stem.is_empty() {
        "audio".to_string()
    } else {
        stem.to_string()
    }
}

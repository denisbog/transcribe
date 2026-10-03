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

    pub fn translation(&self) -> Vec<String> {
        #[derive(Deserialize)]
        struct Doc {
            #[serde(default)]
            sentences: Vec<String>,
        }

        std::fs::read_to_string(self.path(TRANSLATION))
            .ok()
            .and_then(|raw| serde_json::from_str::<Doc>(&raw).ok())
            .map(|doc| doc.sentences)
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

pub fn list() -> Vec<Item> {
    let root = root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut items = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        if let Some(meta) = read_meta(&dir) {
            items.push(Item { dir, meta });
        }
    }

    items.sort_by_key(|item| std::cmp::Reverse(item.meta.created));
    items
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

pub fn remove(item: &Item) -> Result<(), String> {
    std::fs::remove_dir_all(&item.dir).map_err(|err| err.to_string())
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

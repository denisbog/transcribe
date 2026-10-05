//! The crawler's `article.json`: the source of truth for the article-tree
//! pipeline.
//!
//! One folder per article holds `article.json`, the audio (usually under
//! `audio/`) named by `article.json`, and the `transcribe/` output folder.
//! The spoken text is rebuilt from the same fields the crawler writes into
//! `article.md`, so the ASR can be force-aligned to it while every sentence
//! keeps the index of the `blocks[]` entry it belongs to.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::align::Section;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Article {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub kicker: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub section: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub published: String,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub audio: Option<Audio>,
    #[serde(default)]
    pub images: Vec<Image>,
    #[serde(default)]
    pub blocks: Vec<Block>,
}

fn default_language() -> String {
    "de".to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Audio {
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub duration_text: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Image {
    #[serde(default)]
    pub alt: String,
    #[serde(default)]
    pub caption: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Block {
    Head {
        #[serde(default)]
        text: String,
    },
    Para {
        #[serde(default)]
        text: String,
    },
    Quote {
        #[serde(default)]
        text: String,
        #[serde(default)]
        source: String,
    },
    Item {
        #[serde(default)]
        text: String,
    },
    Image {
        #[serde(default)]
        index: usize,
    },
}

impl Block {
    /// The block kind as it appears in `article.json`.
    pub fn kind(&self) -> &'static str {
        match self {
            Block::Head { .. } => "head",
            Block::Para { .. } => "para",
            Block::Quote { .. } => "quote",
            Block::Item { .. } => "item",
            Block::Image { .. } => "image",
        }
    }

}

impl Article {
    /// Reads `<dir>/article.json`.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let path = dir.join("article.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
        serde_json::from_str(&raw).map_err(|err| format!("cannot parse {}: {err}", path.display()))
    }

    /// The audio file of the article, from `audio.file` (or any file under
    /// `audio/` as a fallback).
    pub fn audio_path(&self, dir: &Path) -> Option<PathBuf> {
        if let Some(file) = self.audio.as_ref().and_then(|audio| audio.file.as_deref()) {
            let path = dir.join(file);
            if path.is_file() {
                return Some(path);
            }
        }

        let audio_dir = dir.join("audio");
        std::fs::read_dir(&audio_dir)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.is_file()
                    && path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| {
                            matches!(
                                ext.to_ascii_lowercase().as_str(),
                                "mp3" | "m4a" | "aac" | "ogg" | "opus" | "wav" | "flac"
                            )
                        })
                        .unwrap_or(false)
            })
    }

    /// The spoken text in the order the auto-voiced audio reads it: title,
    /// kicker, description, byline, listen link, then the blocks. Every text
    /// block carries its index in `blocks[]`; titles and image captions do not.
    pub fn sections(&self) -> Vec<Section> {
        let mut out: Vec<Section> = Vec::new();

        push_meta(&mut out, self.title.clone());
        push_meta(&mut out, self.kicker.clone());
        push_meta(&mut out, self.description.clone());

        let mut byline: Vec<String> = Vec::new();
        if !self.authors.is_empty() {
            byline.push(format!("von {}", self.authors.join(", ")));
        }
        if !self.published.is_empty() {
            byline.push(self.published.clone());
        }
        if !self.section.is_empty() {
            byline.push(self.section.clone());
        }
        push_meta(&mut out, byline.join(" · "));

        if let Some(audio) = &self.audio {
            let len = if audio.duration_text.is_empty() {
                String::new()
            } else {
                format!(" ({})", audio.duration_text)
            };
            push_meta(&mut out, format!("🎧 Artikel anhören{len}"));
        }

        for (index, block) in self.blocks.iter().enumerate() {
            match block {
                Block::Head { text } | Block::Para { text } | Block::Item { text } => {
                    out.push(Section {
                        block: Some(index),
                        text: text.clone(),
                    });
                }
                Block::Quote { text, source } => {
                    out.push(Section {
                        block: Some(index),
                        text: text.clone(),
                    });
                    if !source.is_empty() {
                        push_meta(&mut out, format!("— {source}"));
                    }
                }
                Block::Image { index: image } => {
                    if let Some(image) = self.images.get(*image) {
                        let caption = if image.caption.is_empty() {
                            image.alt.clone()
                        } else {
                            image.caption.clone()
                        };
                        push_meta(&mut out, caption);
                    }
                }
            }
        }

        out
    }
}

/// Adds a spoken line that does not belong to an `article.json` block.
fn push_meta(out: &mut Vec<Section>, text: String) {
    if !text.trim().is_empty() {
        out.push(Section { block: None, text });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn article(json: &str) -> Article {
        serde_json::from_str(json).expect("article")
    }

    #[test]
    fn sections_keep_block_indexes_and_skip_images() {
        let article = article(
            r#"{
                "title": "Titel",
                "kicker": "DER SPIEGEL",
                "description": "Beschreibung.",
                "authors": ["A", "B"],
                "published": "2026-10-01T13:52:00+02:00",
                "section": "Politik",
                "audio": {"file": "audio/x.mp3", "duration_text": "2 Min"},
                "images": [{"alt": "Bild", "caption": "Ein Bild"}],
                "blocks": [
                    {"kind": "image", "index": 0},
                    {"kind": "para", "text": "Erster Absatz."},
                    {"kind": "para", "text": "Zweiter Absatz."}
                ]
            }"#,
        );

        let sections = article.sections();
        let blocks: Vec<Option<usize>> = sections.iter().map(|s| s.block).collect();
        assert_eq!(
            blocks,
            vec![None, None, None, None, None, None, Some(1), Some(2)]
        );
        assert_eq!(sections[5].text, "Ein Bild");
        assert_eq!(sections[6].text, "Erster Absatz.");
    }

    #[test]
    fn audio_path_follows_article_json() {
        let dir = std::env::temp_dir().join(format!("article-audio-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("audio")).unwrap();
        std::fs::write(dir.join("audio/episode.mp3"), b"x").unwrap();

        let article = article(r#"{"audio": {"file": "audio/episode.mp3"}}"#);
        assert!(article.audio_path(&dir).unwrap().ends_with("audio/episode.mp3"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}

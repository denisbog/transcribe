//! The user's `config.toml`, kept next to the OS config dir.
//!
//! The only setting so far is the folder the article tree lives in, so that the
//! selection survives a restart:
//!
//! ```toml
//! articles = "/home/me/llm/crawler/articles-ihre-artikel"
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// the folder whose `article.json` files make up the library
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub articles: Option<String>,
}

/// `$XDG_CONFIG_HOME/transcribe/config.toml`, else `~/.config/transcribe/config.toml`.
pub fn path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
        })
        .unwrap_or_else(|| PathBuf::from(".config"));
    base.join("transcribe").join("config.toml")
}

/// Reads the config, falling back to defaults when it is missing or unreadable.
pub fn load() -> Config {
    std::fs::read_to_string(path())
        .ok()
        .and_then(|raw| toml::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn save(config: &Config) -> Result<(), String> {
    let path = path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    }
    let raw = toml::to_string_pretty(config).map_err(|err| err.to_string())?;
    std::fs::write(&path, raw).map_err(|err| format!("cannot write {}: {err}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_the_articles_folder() {
        let config = Config {
            articles: Some("/tmp/articles".to_string()),
        };
        let raw = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&raw).unwrap();
        assert_eq!(parsed.articles.as_deref(), Some("/tmp/articles"));
    }
}

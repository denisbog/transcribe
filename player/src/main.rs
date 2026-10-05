//! Transcript player + article library.
//!
//! Screens:
//!   * Library – every cached article, click one to open it in the player
//!   * New     – submit an audio URL or a local file path plus an optional
//!     German article: with text, the transcript is force-aligned to it
//!     (otherwise transcribed freely); summary and translation still follow
//!   * Player  – audio playback with the current word highlighted
//!
//! Usage:
//!     transcript-player                     # open the library
//!     transcript-player <audio> [words] [sentences]   # play, or transcribe then play
//!     transcript-player <article-folder>    # play a cached or crawler article
//!     transcript-player ingest <url|file> [model]        # run the pipeline headless
//!     transcript-player align <audio> <article.txt>      # force-align German
//!     transcript-player transcribe-tree <folder>         # batch: audio tree -> transcribe/
//!     transcript-player articles <folder> [--force] [--pairs-provider P] [--pairs-model M]
//!     transcript-player vocabulary <folder> [model] [--force] [--external-pairs]
//!     transcript-player pairs <folder> [model] [--force] [--jobs N] [--pairs-provider P] [--pairs-model M]

mod align;
mod article;
mod config;
mod external;
mod asr;
mod audio;
mod library;
mod onnx_llm;
mod pairs;
mod pipeline;
mod theme;

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Duration;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use audio::{AudioHandle, Cmd};
use iced::widget::{
    button, column, container, progress_bar, row, scrollable, slider, text, toggler, Row, Space,
};
use iced::{keyboard, time, Alignment, Element, Length, Size, Subscription, Task, Theme};
use serde::Deserialize;


// ---------------------------------------------------------------- data model

#[derive(Debug, Deserialize)]
struct Doc {
    #[serde(default)]
    language: String,
    #[serde(default)]
    segments: Vec<library::Seg>,
    /// sentence-level segments of the article-tree `transcription.json`
    #[serde(default)]
    sentences: Vec<library::Seg>,
}

#[derive(Debug, Clone)]
struct Phrase {
    start: f32,
    end: f32,
    text: String,
    /// range into `Player::words`
    words: Range<usize>,
    /// the `article.json` block this sentence belongs to
    block: Option<usize>,
}

/// A paragraph as shown by the player: consecutive sentences that share an
/// `article.json` block, or a single spoken metadata line.
#[derive(Debug, Clone)]
struct Paragraph {
    start: f32,
    end: f32,
    text: String,
    translation: String,
    /// range into `Player::words`
    words: Range<usize>,
    /// range into `Player::phrases`
    sentences: Range<usize>,
    /// the `article.json` block, `None` for spoken metadata
    block: Option<usize>,
}

/// One article folder in the selected tree.
#[derive(Debug, Clone)]
struct ArticleEntry {
    dir: PathBuf,
    title: String,
    description: String,
    language: String,
    audio: bool,
    transcription: bool,
    translation: bool,
    vocabulary: bool,
    pairs: bool,
}

impl ArticleEntry {
    /// Reads `<dir>/article.json` and looks at what `<dir>/transcribe/` holds.
    fn found(dir: &Path) -> Option<Self> {
        let article = article::Article::read(dir).ok()?;
        let out = dir.join(library::TRANSCRIPT_DIR);
        let audio = article.audio_path(dir).is_some();
        Some(Self {
            dir: dir.to_path_buf(),
            title: if article.title.is_empty() {
                dir.file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_default()
            } else {
                article.title
            },
            description: article.description,
            language: article.language,
            audio,
            transcription: out.join(library::WORDS).is_file(),
            translation: out.join(library::TRANSLATION).is_file(),
            vocabulary: out.join(library::VOCABULARY).is_file(),
            pairs: out.join(library::PAIRS).is_file(),
        })
    }

    fn playable(&self) -> bool {
        self.audio && self.transcription
    }

    fn complete(&self) -> bool {
        self.playable() && self.translation && self.vocabulary && self.pairs
    }
}

/// Every article below `root`, sorted by path.
fn load_articles(root: &Path) -> Vec<ArticleEntry> {
    let mut dirs = Vec::new();
    collect_articles(root, &mut dirs);
    dirs.sort();
    dirs.iter().filter_map(|dir| ArticleEntry::found(dir)).collect()
}

/// The in-app folder picker.
#[derive(Debug, Clone)]
struct Browser {
    dir: PathBuf,
    dirs: Vec<PathBuf>,
}

impl Browser {
    fn new(dir: PathBuf) -> Self {
        let mut browser = Self {
            dir,
            dirs: Vec::new(),
        };
        browser.reload();
        browser
    }

    /// The visible subdirectories of the current folder, sorted.
    fn reload(&mut self) {
        self.dirs = std::fs::read_dir(&self.dir)
            .map(|entries| {
                let mut dirs = entries
                    .flatten()
                    .filter(|entry| {
                        entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false)
                    })
                    .map(|entry| entry.path())
                    .filter(|path| {
                        !path
                            .file_name()
                            .map(|name| name.to_string_lossy().starts_with('.'))
                            .unwrap_or(false)
                    })
                    .collect::<Vec<_>>();
                dirs.sort();
                dirs
            })
            .unwrap_or_default();
    }

    fn up(&mut self) {
        if let Some(parent) = self.dir.parent() {
            self.dir = parent.to_path_buf();
            self.reload();
        }
    }

    fn enter(&mut self, index: usize) {
        if let Some(dir) = self.dirs.get(index) {
            self.dir = dir.clone();
            self.reload();
        }
    }
}

// ---------------------------------------------------------------- messages

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    /// the articles found under the selected folder
    Library,
    /// one article: status + run the pipeline + play
    Article,
    /// in-app folder picker
    Browse,
    Player,
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    Screen(Screen),

    // article folder
    BrowseFolder,
    BrowseEnter(usize),
    BrowseUp,
    BrowseSelect,
    OpenArticle(usize),
    RunPipeline,
    ToggleExternalPairs(bool),
    OpenPlayer,
    CopyLog,

    // playback
    TogglePlay,
    ToggleFollow(bool),
    ToggleTranslation(bool),
    ToggleVocabulary,
    Drag(f32),
    CommitDrag,
    Volume(f32),
    SeekTo(f32),
    Key(keyboard::Event),
}

// ---------------------------------------------------------------- player

struct Player {
    item: Option<library::Item>,
    words: Vec<library::Seg>,
    phrases: Vec<Phrase>,
    paragraphs: Vec<Paragraph>,
    translation: library::Translation,
    show_translation: bool,
    vocabulary: Vec<library::WordPair>,
    show_vocabulary: bool,
    pairs: Vec<library::PairsBlock>,
    audio: AudioHandle,
    position: f32,
    duration: f32,
    drag: Option<f32>,
    volume: f32,
    playing: bool,
    follow: bool,
    current_word: usize,
    current_phrase: usize,
}

impl Player {
    /// Loads a cached article or plain local files.
    fn new(
        item: Option<library::Item>,
        audio_path: PathBuf,
        words_path: PathBuf,
        phrases_path: PathBuf,
        translation: library::Translation,
        vocabulary: Vec<library::WordPair>,
        pairs: Vec<library::PairsBlock>,
    ) -> Result<Self, String> {
        // The article-tree `transcription.json` carries the sentence segments
        // itself; a legacy `transcript.json` next to it still wins.
        let words_doc = load(&words_path)?;
        let phrase_segs = if phrases_path != words_path && phrases_path.is_file() {
            load(&phrases_path)?.segments
        } else {
            words_doc.sentences.clone()
        };
        let words = words_doc.segments;
        let phrases = group(&phrase_segs, &words);
        let paragraphs = paragraphs(&phrases, &translation);

        let duration = words
            .last()
            .map(|word| word.end)
            .unwrap_or_default()
            .max(phrase_segs.last().map(|phrase| phrase.end).unwrap_or_default());

        let audio = AudioHandle::spawn(audio_path);
        match std::env::var("START_AT").ok().and_then(|v| v.parse::<f32>().ok()) {
            Some(start) => audio.send(Cmd::Seek(start)),
            None if std::env::var_os("AUTOPLAY").is_some() => audio.send(Cmd::Toggle),
            None => {}
        }

        let show_translation = !translation.sentences.is_empty();
        let show_vocabulary = !vocabulary.is_empty();

        Ok(Self {
            item,
            words,
            phrases,
            translation,
            paragraphs,
            show_translation,
            vocabulary,
            show_vocabulary,
            pairs,
            audio,
            position: std::env::var("START_AT")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0),
            duration,
            drag: None,
            volume: 0.9,
            playing: false,
            follow: true,
            current_word: 0,
            current_phrase: 0,
        })
    }

    fn from_item(item: library::Item) -> Result<Self, String> {
        let translation = item.translation();
        let vocabulary = item.vocabulary();
        let pairs = item.pairs();
        let audio = item
            .audio_path()
            .ok_or_else(|| format!("{}: no audio file yet", item.dir.display()))?;
        let words = item.path(library::WORDS);
        let phrases = item.path(library::PHRASES);
        if !words.is_file() {
            return Err("this article has no transcript yet (job still running?)".to_string());
        }

        Self::new(Some(item), audio, words, phrases, translation, vocabulary, pairs)
    }

    /// Opens a crawler article folder directly: `article.json` + `transcribe/`.
    fn from_article(dir: &Path) -> Result<Self, String> {
        let article = article::Article::read(dir)?;
        let audio = article
            .audio_path(dir)
            .ok_or_else(|| format!("{}: no audio file", dir.display()))?;
        let out = dir.join(library::TRANSCRIPT_DIR);
        let words_path = out.join(library::WORDS);
        if !words_path.is_file() {
            return Err(format!("{} has no transcription yet", dir.display()));
        }

        let duration = load(&words_path)?
            .sentences
            .last()
            .map(|sentence| sentence.end)
            .unwrap_or_default();
        let meta = library::Meta {
            id: dir
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default(),
            audio: audio.to_string_lossy().to_string(),
            title: article.title.clone(),
            description: article.description.clone(),
            article: dir.join("article.json").display().to_string(),
            language: article.language.clone(),
            duration,
            created: chrono::Local::now().timestamp(),
            status: "done".to_string(),
            ..Default::default()
        };

        let mut item = library::Item { dir: out, meta };
        let translation = item.translation();
        item.meta.target = translation.target.clone();
        let vocabulary = item.vocabulary();
        let pairs = item.pairs();
        let words = item.path(library::WORDS);
        let phrases = item.path(library::PHRASES);

        Self::new(Some(item), audio, words, phrases, translation, vocabulary, pairs)
    }

    fn sync(&mut self) {
        self.playing = self.audio.is_playing();
        if self.drag.is_none() {
            self.position = self.audio.position().min(self.duration);
        }
        self.current_word = word_at(&self.words, self.position);
        self.current_phrase = self
            .phrases
            .partition_point(|phrase| phrase.words.end <= self.current_word)
            .min(self.phrases.len().saturating_sub(1));
    }

    fn seek(&mut self, seconds: f32) {
        let seconds = seconds.clamp(0.0, self.duration);
        self.position = seconds;
        self.drag = None;
        self.audio.send(Cmd::Seek(seconds));
        self.sync();
    }

    fn shown_position(&self) -> f32 {
        self.drag.unwrap_or(self.position)
    }


    /// The paragraph that contains the sentence being spoken.
    fn current_paragraph(&self) -> usize {
        self.paragraphs
            .partition_point(|paragraph| paragraph.sentences.end <= self.current_phrase)
            .min(self.paragraphs.len().saturating_sub(1))
    }

    /// The translation shown for a paragraph.
    fn paragraph_translation(&self, index: usize) -> Option<&str> {
        self.paragraphs
            .get(index)
            .map(|paragraph| paragraph.translation.as_str())
            .filter(|line| !line.is_empty())
    }
}

// ---------------------------------------------------------------- job state

struct Job {
    stage: String,
    progress: f32,
    log: Vec<String>,
    done: Option<library::Item>,
    /// article-tree job that finished (the article folder)
    done_dir: Option<PathBuf>,
    failed: Option<String>,
}

impl Job {
    fn new() -> Self {
        Self {
            stage: "Starting".to_string(),
            progress: 0.0,
            log: Vec::new(),
            done: None,
            done_dir: None,
            failed: None,
        }
    }

    fn running(&self) -> bool {
        self.done.is_none() && self.done_dir.is_none() && self.failed.is_none()
    }

    /// The whole log as text, for the clipboard.
    fn text(&self) -> String {
        let mut out = String::new();
        if let Some(done) = &self.done {
            out.push_str(&format!(
                "{} — {}\n{}\n\n",
                done.meta.title, done.meta.status, done.meta.description
            ));
        }
        if let Some(error) = &self.failed {
            out.push_str(&format!("failed: {error}\n\n"));
        }
        out.push_str(&format!("stage: {}\n\n", self.stage));
        for line in &self.log {
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    fn push(&mut self, line: String) {
        if self.log.last() != Some(&line) {
            self.log.push(line);
        }
        if self.log.len() > 200 {
            self.log.drain(..self.log.len() - 200);
        }
    }
}

// ---------------------------------------------------------------- app

struct App {
    fonts: theme::Fonts,
    screen: Screen,
    /// the folder the user picked (persisted in `config.toml`)
    root: Option<PathBuf>,
    /// every `article.json` found under `root`
    articles: Vec<ArticleEntry>,
    /// the open folder picker
    browser: Option<Browser>,
    /// the article folder currently open
    article_dir: Option<PathBuf>,
    player: Option<Player>,
    /// translation target of a new pipeline run
    target: String,
    /// extract the pairs with the external `pi` model instead of the local one
    external_pairs: bool,
    /// local language model, e.g. `qwen2.5:1.5b`
    llm_model: String,
    /// where ONNX Runtime was found (GPU build or system one)
    ort_dir: Option<PathBuf>,
    job: Option<Job>,
    rx: Option<Receiver<pipeline::Event>>,
    /// the article folder the running job belongs to
    job_dir: Option<PathBuf>,
    status: Option<String>,
}

impl App {
    fn new(fonts: theme::Fonts) -> Self {
        let mut args = std::env::args().skip(1);
        let first = args.next();

        // `transcript-player <audio> [words] [sentences]` plays local files
        let (player, screen, root, article_dir) = match first.as_deref() {
            // a folder: an article (`article.json`), a legacy cache, or a tree root
            Some(path) if !path.starts_with("--") && PathBuf::from(path).is_dir() => {
                let dir = PathBuf::from(path);
                if dir.join("article.json").is_file() {
                    let player = Player::from_article(&dir).ok();
                    (player, Screen::Article, None, Some(dir))
                } else if let Some(meta) = library::read_meta(&dir) {
                    let item = library::Item { dir, meta };
                    let player = Player::from_item(item).ok();
                    (player, Screen::Player, None, None)
                } else {
                    (None, Screen::Library, Some(dir), None)
                }
            }
            Some(path) if !path.starts_with("--") => {
                let audio = PathBuf::from(path);
                let words_arg = args.next().map(PathBuf::from);
                let phrases_arg = args.next().map(PathBuf::from);
                let words = words_arg.unwrap_or_else(|| audio.with_file_name(library::WORDS));
                let phrases = phrases_arg.unwrap_or_else(|| audio.with_file_name(library::PHRASES));
                let player = Player::new(
                    None,
                    audio,
                    words,
                    phrases,
                    library::Translation::default(),
                    Vec::new(),
                    Vec::new(),
                )
                .ok();
                (player, Screen::Player, None, None)
            }
            _ => (None, Screen::Library, None, None),
        };

        let mut app = Self::empty(fonts);
        app.player = player;
        app.screen = screen;
        app.article_dir = article_dir;
        // a folder opened directly on the command line may live outside `root`
        if let Some(dir) = app.article_dir.clone() {
            if !app.articles.iter().any(|entry| entry.dir == dir) {
                if let Some(entry) = ArticleEntry::found(&dir) {
                    app.articles.push(entry);
                }
            }
        }
        if let Some(root) = root {
            app.select_root(root);
        }
        app
    }

    fn empty(fonts: theme::Fonts) -> Self {
        let config = config::load();
        let root = config
            .articles
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir());
        let articles = root.as_deref().map(load_articles).unwrap_or_default();

        Self {
            fonts,
            screen: Screen::Library,
            root,
            articles,
            browser: None,
            article_dir: None,
            player: None,
            target: "en".to_string(),
            external_pairs: external::enabled_from_env(),
            llm_model: onnx_llm::DEFAULT_MODEL.to_string(),
            ort_dir: onnx_llm::prepare_runtime(),
            job: None,
            rx: None,
            job_dir: None,
            status: None,
        }
    }

    /// Picks the article tree, loads it and remembers it for the next start.
    fn select_root(&mut self, root: PathBuf) {
        self.root = Some(root.clone());
        self.refresh_articles();
        self.status = None;
        if let Err(err) = config::save(&config::Config {
            articles: Some(root.to_string_lossy().to_string()),
        }) {
            self.status = Some(err);
        }
    }

    fn refresh_articles(&mut self) {
        self.articles = self.root.as_deref().map(load_articles).unwrap_or_default();
    }

    fn open_article(&mut self, index: usize) {
        let Some(entry) = self.articles.get(index) else {
            return;
        };
        self.article_dir = Some(entry.dir.clone());
        self.player = None;
        self.screen = Screen::Article;
        self.status = None;
    }

    fn open_player(&mut self) {
        let Some(dir) = self.article_dir.clone() else {
            self.status = Some("Open an article first".to_string());
            return;
        };
        match Player::from_article(&dir) {
            Ok(player) => {
                self.player = Some(player);
                self.screen = Screen::Player;
                self.status = None;
            }
            Err(err) => self.status = Some(err),
        }
    }

    /// Runs transcribe + translation + vocabulary + pairs for the open article.
    fn start_pipeline(&mut self) {
        if self.job.as_ref().map(Job::running).unwrap_or(false) {
            self.status = Some("a pipeline run is already in progress".to_string());
            return;
        }
        let Some(dir) = self.article_dir.clone() else {
            self.status = Some("Open an article first".to_string());
            return;
        };

        let options = pipeline::ArticleOptions {
            target: self.target.clone(),
            llm_model: Some(self.llm_model.clone()),
            pairs_model: self.external_pairs.then(external::PairsModel::from_env),
            ..Default::default()
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let thread_dir = dir.clone();
        let spawned = std::thread::Builder::new()
            .name("article".into())
            .spawn(move || match pipeline::run_article(&thread_dir, &options, &tx) {
                Ok(_) => {
                    let _ = tx.send(pipeline::Event::ArticleDone(thread_dir));
                }
                Err(error) => {
                    let _ = tx.send(pipeline::Event::Failed(error));
                }
            });
        if let Err(err) = spawned {
            self.status = Some(format!("cannot start the pipeline: {err}"));
            return;
        }

        self.rx = Some(rx);
        self.job = Some(Job::new());
        self.job_dir = Some(dir);
        self.status = None;
    }

    /// Drains pipeline events; called on every tick.
    fn poll_job(&mut self) {
        let Some(rx) = &self.rx else {
            return;
        };

        let mut finished = false;
        if let Some(job) = &mut self.job {
            while let Ok(event) = rx.try_recv() {
                match event {
                    pipeline::Event::Stage(stage) => job.stage = stage,
                    pipeline::Event::Progress(progress) => {
                        job.progress = job.progress.max(progress);
                    }
                    pipeline::Event::Log(line) => job.push(line),
                    pipeline::Event::Done(item) => {
                        job.done = Some(*item);
                        job.stage = "Done".to_string();
                        job.progress = 1.0;
                        finished = true;
                    }
                    pipeline::Event::ArticleDone(dir) => {
                        job.done_dir = Some(dir);
                        job.stage = "Done".to_string();
                        job.progress = 1.0;
                        finished = true;
                    }
                    pipeline::Event::Failed(error) => {
                        job.push(format!("error: {error}"));
                        job.failed = Some(error);
                        finished = true;
                    }
                }
            }
        }

        if finished {
            self.rx = None;
            self.refresh_articles();
            // reload the open player when its article just became ready
            if let Some(dir) = self.job_dir.clone() {
                if self.screen == Screen::Player && self.article_dir.as_deref() == Some(&dir) {
                    if let Ok(player) = Player::from_article(&dir) {
                        self.player = Some(player);
                    }
                }
            }
        }
    }

    fn player_mut(&mut self) -> Option<&mut Player> {
        self.player.as_mut()
    }
}

// ---------------------------------------------------------------- update

fn update(app: &mut App, message: Message) -> Task<Message> {
    match message {
        Message::Tick => {
            app.poll_job();
            if let Some(player) = app.player_mut() {
                player.sync();
            }
        }
        Message::Screen(screen) => {
            if screen == Screen::Library {
                app.refresh_articles();
            }
            if screen == Screen::Player && app.player.is_none() {
                app.status = Some("No article open".to_string());
                return Task::none();
            }
            app.screen = screen;
        }
        Message::BrowseFolder => {
            let start = app
                .root
                .clone()
                .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("/"));
            app.browser = Some(Browser::new(start));
            app.screen = Screen::Browse;
            app.status = None;
        }
        Message::BrowseEnter(index) => {
            if let Some(browser) = app.browser.as_mut() {
                browser.enter(index);
            }
        }
        Message::BrowseUp => {
            if let Some(browser) = app.browser.as_mut() {
                browser.up();
            }
        }
        Message::BrowseSelect => {
            if let Some(browser) = app.browser.take() {
                app.select_root(browser.dir);
                app.article_dir = None;
                app.player = None;
                app.screen = Screen::Library;
            }
        }
        Message::OpenArticle(index) => app.open_article(index),
        Message::RunPipeline => app.start_pipeline(),
        Message::ToggleExternalPairs(enabled) => app.external_pairs = enabled,
        Message::OpenPlayer => app.open_player(),
        Message::CopyLog => {
            if let Some(job) = &app.job {
                return iced::clipboard::write(job.text());
            }
        }
        Message::TogglePlay => {
            if let Some(player) = app.player_mut() {
                player.audio.send(Cmd::Toggle);
            }
        }
        Message::ToggleFollow(follow) => {
            if let Some(player) = app.player_mut() {
                player.follow = follow;
            }
        }
        Message::ToggleTranslation(show) => {
            if let Some(player) = app.player_mut() {
                player.show_translation = show;
            }
        }
        Message::ToggleVocabulary => {
            if let Some(player) = app.player_mut() {
                player.show_vocabulary = !player.show_vocabulary;
            }
        }
        Message::Drag(value) => {
            if let Some(player) = app.player_mut() {
                player.drag = Some(value);
            }
        }
        Message::CommitDrag => {
            if let Some(player) = app.player_mut() {
                if let Some(value) = player.drag {
                    player.seek(value);
                }
            }
        }
        Message::Volume(value) => {
            if let Some(player) = app.player_mut() {
                player.volume = value;
                player.audio.send(Cmd::Volume(value));
            }
        }
        Message::SeekTo(seconds) => {
            if let Some(player) = app.player_mut() {
                player.seek(seconds);
            }
        }
        Message::Key(event) => {
            let keyboard::Event::KeyPressed { key, .. } = event else {
                return Task::none();
            };
            let Some(player) = app.player_mut() else {
                return Task::none();
            };

            match key {
                keyboard::Key::Named(keyboard::key::Named::Space) => {
                    player.audio.send(Cmd::Toggle);
                }
                keyboard::Key::Named(keyboard::key::Named::ArrowRight) => {
                    player.seek(player.position + 5.0)
                }
                keyboard::Key::Named(keyboard::key::Named::ArrowLeft) => {
                    player.seek(player.position - 5.0)
                }
                keyboard::Key::Named(keyboard::key::Named::ArrowDown) => {
                    let current = player.current_paragraph();
                    let next = (current + 1).min(player.paragraphs.len().saturating_sub(1));
                    if let Some(paragraph) = player.paragraphs.get(next) {
                        let start = paragraph.start;
                        player.seek(start);
                    }
                }
                keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
                    let previous = player.current_paragraph().saturating_sub(1);
                    if let Some(paragraph) = player.paragraphs.get(previous) {
                        let start = paragraph.start;
                        player.seek(start);
                    }
                }
                keyboard::Key::Named(keyboard::key::Named::Home) => player.seek(0.0),
                _ => {}
            }
        }
    }

    Task::none()
}

fn subscription(_app: &App) -> Subscription<Message> {
    Subscription::batch([
        time::every(Duration::from_millis(40)).map(|_| Message::Tick),
        keyboard::listen().map(Message::Key),
    ])
}

// --------------------------------------------------------------- view: shell

fn view(app: &App) -> Element<'_, Message> {
    let body: Element<'_, Message> = match app.screen {
        Screen::Library => articles_view(app),
        Screen::Article => article_view(app),
        Screen::Browse => browse_view(app),
        Screen::Player => player_view(app),
    };

    container(column![top_bar(app), body].spacing(14).height(Length::Fill))
        .padding(18)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::root)
        .into()
}

fn top_bar(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;

    let tab = |label: &str, screen: Screen| {
        let selected = app.screen == screen;
        button(text(label.to_string()).size(13).font(fonts.body))
            .on_press(Message::Screen(screen))
            .padding([7, 14])
            .style(move |theme, status| theme::tab(theme, status, selected))
    };

    let llm_badge = format!(
        "{} · {}",
        app.llm_model,
        match &app.ort_dir {
            Some(dir) => format!("ort: {}", dir.display()),
            None => "onnxruntime not found".to_string(),
        }
    );

    let mut bar = row![
        text("Transcribe")
            .size(16)
            .font(fonts.display)
            .color(theme::TEXT),
        Space::new().width(10),
        tab("Articles", Screen::Library),
        Space::new().width(Length::Fill),
        container(
            text(llm_badge)
                .size(11)
                .font(fonts.mono)
                .color(theme::TEXT_DIM)
        )
        .padding([4, 10])
        .style(theme::badge),
    ]
    .spacing(6)
    .align_y(Alignment::Center);

    if app.screen == Screen::Player && app.player.is_some() {
        bar = bar.push(
            button(text("Player").size(13).font(fonts.body))
                .on_press(Message::Screen(Screen::Player))
                .padding([7, 14])
                .style(|theme, status| theme::tab(theme, status, true)),
        );
    }

    bar.into()
}

// ------------------------------------------------------------ view: articles

fn articles_view(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;

    let mut cards: Vec<Element<'_, Message>> = Vec::new();
    for (index, entry) in app.articles.iter().enumerate() {
        cards.push(article_card(app, index, entry));
    }

    let grid: Element<'_, Message> = if cards.is_empty() {
        articles_empty(app)
    } else {
        Row::with_children(cards)
            .spacing(12)
            .wrap()
            .vertical_spacing(12)
            .into()
    };

    let root = app
        .root
        .as_ref()
        .map(|dir| dir.display().to_string())
        .unwrap_or_else(|| "no folder selected".to_string());

    let header = row![
        column![
            text("Articles")
                .size(22)
                .font(fonts.display)
                .color(theme::TEXT),
            text(format!(
                "{} article{} in {}",
                app.articles.len(),
                if app.articles.len() == 1 { "" } else { "s" },
                root
            ))
            .size(11)
            .font(fonts.mono)
            .color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        button(text("Browse folder…").size(12).font(fonts.body))
            .on_press(Message::BrowseFolder)
            .padding([9, 16])
            .style(theme::primary),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let status = app.status.as_ref().map(|status| {
        container(
            text(status.clone())
                .size(11)
                .font(fonts.mono)
                .style(theme::danger_text),
        )
        .padding([6, 10])
        .style(theme::log_panel)
    });

    container(
        column![
            header,
            status
                .map(Element::from)
                .unwrap_or_else(|| Space::new().height(0).into()),
            scrollable(grid).height(Length::Fill),
        ]
        .spacing(12)
        .height(Length::Fill),
    )
    .padding(18)
    .width(Length::Fill)
    .height(Length::Fill)
    .style(theme::card)
    .into()
}

fn article_card<'a>(app: &'a App, index: usize, entry: &'a ArticleEntry) -> Element<'a, Message> {
    let fonts = app.fonts;

    let description: String = {
        let text = if entry.description.is_empty() {
            entry.dir.display().to_string()
        } else {
            entry.description.clone()
        };
        if text.chars().count() > 170 {
            format!("{}…", text.chars().take(170).collect::<String>())
        } else {
            text
        }
    };

    let mut footer = row![text(entry.language.to_uppercase())
        .size(10)
        .font(fonts.mono)
        .color(theme::TEXT_MUTED)]
    .spacing(8);

    for (label, done) in [
        ("TXT", entry.transcription),
        ("TRA", entry.translation),
        ("VOC", entry.vocabulary),
        ("PAIR", entry.pairs),
    ] {
        footer = footer.push(
            text(label)
                .size(10)
                .font(fonts.mono)
                .style(if done { theme::success_text } else { theme::danger_text }),
        );
    }

    let running = app.job_dir.as_deref() == Some(entry.dir.as_path())
        && app.job.as_ref().map(Job::running).unwrap_or(false);
    if running {
        footer = footer.push(
            text("RUNNING")
                .size(10)
                .font(fonts.mono)
                .color(theme::ACCENT_HI),
        );
    } else if entry.complete() {
        footer = footer.push(
            text("READY")
                .size(10)
                .font(fonts.mono)
                .style(theme::success_text),
        );
    }

    let content = column![
        text(entry.title.clone())
            .size(15)
            .font(fonts.display)
            .color(theme::TEXT),
        text(description)
            .size(12)
            .font(fonts.body)
            .color(theme::TEXT_DIM),
        Space::new().height(4),
        footer,
    ]
    .spacing(6)
    .width(Length::Fill);

    button(content)
        .on_press(Message::OpenArticle(index))
        .padding(14)
        .width(Length::Fixed(330.0))
        .style(theme::card_interactive)
        .into()
}

fn articles_empty(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;
    let (title, body) = if app.root.is_some() {
        (
            "No articles found",
            "The selected folder holds no article.json below it. Pick another folder.",
        )
    } else {
        (
            "Choose an article folder",
            "Browse for the folder that holds your article subfolders — each with an article.json. Then open an article and run the pipeline.",
        )
    };

    container(
        column![
            text(title)
                .size(18)
                .font(fonts.display)
                .color(theme::TEXT),
            text(body)
                .size(12)
                .font(fonts.body)
                .color(theme::TEXT_DIM),
            Space::new().height(6),
            button(text("Browse folder…").size(12).font(fonts.body))
                .on_press(Message::BrowseFolder)
                .padding([9, 16])
                .style(theme::primary),
        ]
        .spacing(8),
    )
    .padding(22)
    .width(Length::Fill)
    .style(theme::log_panel)
    .into()
}

// ------------------------------------------------------------- view: article

fn article_view(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;
    let entry = app
        .article_dir
        .as_deref()
        .and_then(|dir| app.articles.iter().find(|entry| entry.dir == dir));

    let Some(entry) = entry else {
        return articles_empty(app);
    };

    let running = app.job_dir.as_deref() == Some(entry.dir.as_path())
        && app.job.as_ref().map(Job::running).unwrap_or(false);

    let statuses = column(
        [
            ("Audio", "the mp3 named by article.json", entry.audio),
            (
                "Transcription",
                "word and sentence timings",
                entry.transcription,
            ),
            ("Translation", "sentence translation to English", entry.translation),
            ("Vocabulary", "the most relevant word pairs", entry.vocabulary),
            ("Pairs", "vocabulary placed in its sentence", entry.pairs),
        ]
        .into_iter()
        .map(|(label, detail, done)| status_row(fonts, label, detail, done))
        .collect::<Vec<Element<'_, Message>>>(),
    )
    .spacing(6);

    let header = row![
        button(text("← Articles").size(12).font(fonts.body))
            .on_press(Message::Screen(Screen::Library))
            .padding([8, 12])
            .style(theme::chip),
        column![
            text(entry.title.clone())
                .size(20)
                .font(fonts.display)
                .color(theme::TEXT),
            text(entry.dir.display().to_string())
                .size(10)
                .font(fonts.mono)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        Space::new().width(Length::Fill),
    ]
    .spacing(12)
    .align_y(Alignment::Center);

    let pairs_option = toggler(app.external_pairs)
        .label(format!(
            "Pairs via pi ({})",
            external::PairsModel::from_env().label()
        ))
        .on_toggle(Message::ToggleExternalPairs)
        .text_size(12)
        .font(fonts.body)
        .style(theme::toggle);

    let actions = row![
        button(text(if running { "Running…" } else { "Run pipeline" }).size(13).font(fonts.body))
            .on_press_maybe((!running).then_some(Message::RunPipeline))
            .padding([10, 18])
            .style(theme::primary),
        button(text("Play").size(13).font(fonts.body))
            .on_press_maybe(entry.playable().then_some(Message::OpenPlayer))
            .padding([10, 18])
            .style(theme::chip),
        Space::new().width(Length::Fill),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    let description: Element<'_, Message> = if entry.description.is_empty() {
        Space::new().height(0).into()
    } else {
        text(entry.description.clone())
            .size(12)
            .font(fonts.body)
            .color(theme::TEXT_DIM)
            .into()
    };

    let job: Element<'_, Message> = match &app.job {
        Some(job) if app.job_dir.as_deref() == Some(entry.dir.as_path()) => job_card(app, job),
        _ => Space::new().height(0).into(),
    };

    let status = app.status.as_ref().map(|status| {
        container(
            text(status.clone())
                .size(11)
                .font(fonts.mono)
                .style(theme::danger_text),
        )
        .padding([6, 10])
        .style(theme::log_panel)
    });

    container(
        column![
            header,
            description,
            statuses,
            pairs_option,
            actions,
            status
                .map(Element::from)
                .unwrap_or_else(|| Space::new().height(0).into()),
            job,
        ]
        .spacing(14)
        .height(Length::Fill),
    )
    .padding(18)
    .width(Length::Fill)
    .height(Length::Fill)
    .style(theme::card)
    .into()
}

fn status_row(
    fonts: theme::Fonts,
    label: &str,
    detail: &str,
    done: bool,
) -> Element<'static, Message> {
    row![
        text(if done { "✓" } else { "○" })
            .size(13)
            .font(fonts.mono)
            .style(if done {
                theme::success_text
            } else {
                theme::danger_text
            })
            .width(Length::Fixed(18.0)),
        text(label.to_string())
            .size(13)
            .font(fonts.body)
            .color(theme::TEXT)
            .width(Length::Fixed(130.0)),
        text(detail.to_string())
            .size(11)
            .font(fonts.body)
            .color(theme::TEXT_MUTED),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

// -------------------------------------------------------------- view: browse

fn browse_view(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;
    let Some(browser) = app.browser.as_ref() else {
        return articles_empty(app);
    };

    let mut rows: Vec<Element<'_, Message>> = Vec::new();
    rows.push(
        button(text("↑ Up").size(12).font(fonts.body))
            .on_press(Message::BrowseUp)
            .padding([8, 12])
            .style(theme::chip)
            .into(),
    );

    for (index, dir) in browser.dirs.iter().enumerate() {
        let name = dir
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        rows.push(
            button(
                row![
                    text("📁").size(13),
                    text(name).size(13).font(fonts.body).color(theme::TEXT),
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            )
            .on_press(Message::BrowseEnter(index))
            .padding([8, 12])
            .width(Length::Fill)
            .style(theme::card_interactive)
            .into(),
        );
    }
    if browser.dirs.is_empty() {
        rows.push(
            text("(no subfolders)")
                .size(11)
                .font(fonts.mono)
                .color(theme::TEXT_MUTED)
                .into(),
        );
    }

    let header = row![
        column![
            text("Choose a folder")
                .size(22)
                .font(fonts.display)
                .color(theme::TEXT),
            text(browser.dir.display().to_string())
                .size(11)
                .font(fonts.mono)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        button(text("Use this folder").size(12).font(fonts.body))
            .on_press(Message::BrowseSelect)
            .padding([9, 16])
            .style(theme::primary),
        button(text("Cancel").size(12).font(fonts.body))
            .on_press(Message::Screen(Screen::Library))
            .padding([9, 16])
            .style(theme::chip),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    container(
        column![header, scrollable(column(rows).spacing(4)).height(Length::Fill)]
            .spacing(12)
            .height(Length::Fill),
    )
    .padding(18)
    .width(Length::Fill)
    .height(Length::Fill)
    .style(theme::card)
    .into()
}

fn job_card<'a>(app: &'a App, job: &'a Job) -> Element<'a, Message> {
    let fonts = app.fonts;

    let stage: Element<'_, Message> = if let Some(error) = &job.failed {
        text(format!("Failed: {error}"))
            .size(12)
            .font(fonts.body)
            .style(theme::danger_text)
            .into()
    } else if job.done.is_some() || job.done_dir.is_some() {
        text("Finished")
            .size(12)
            .font(fonts.body)
            .style(theme::success_text)
            .into()
    } else {
        text(job.stage.clone())
            .size(12)
            .font(fonts.body)
            .color(theme::TEXT)
            .into()
    };

    let mut header = row![stage, Space::new().width(Length::Fill)]
        .spacing(10)
        .align_y(Alignment::Center);

    header = header.push(
        button(text("Copy log").size(11).font(fonts.body))
            .on_press(Message::CopyLog)
            .padding([6, 10])
            .style(theme::chip),
    );

    if job.done_dir.is_some() {
        header = header.push(
            button(text("Open article").size(12).font(fonts.body))
                .on_press(Message::OpenPlayer)
                .padding([8, 14])
                .style(theme::primary),
        );
    }

    // newest first, so the interesting lines are always visible
    let log = scrollable(
        column(
            job.log
                .iter()
                .rev()
                .take(10)
                .map(|line| {
                    text(line.clone())
                        .size(11)
                        .font(fonts.mono)
                        .color(theme::TEXT_DIM)
                        .into()
                })
                .collect::<Vec<Element<'_, Message>>>(),
        )
        .spacing(2),
    )
    .height(Length::Fixed(168.0))
    .direction(scrollable::Direction::Vertical(
        scrollable::Scrollbar::new().width(6.0).scroller_width(6.0).margin(2.0),
    ))
    .style(theme::scroll);

    container(
        column![
            header,
            progress_bar(0.0..=1.0, job.progress)
                .girth(6.0)
                .style(theme::progress),
            container(log).padding(10).style(theme::log_panel),
        ]
        .spacing(8),
    )
    .padding(14)
    .width(Length::Fill)
    .style(theme::card)
    .into()
}

// ------------------------------------------------------------- view: player

fn player_view(app: &App) -> Element<'_, Message> {
    let Some(player) = app.player.as_ref() else {
        return articles_empty(app);
    };
    let fonts = app.fonts;

    let (title, meta_line) = match &player.item {
        Some(item) => (
            if item.meta.title.is_empty() {
                item.meta.id.clone()
            } else {
                item.meta.title.clone()
            },
            format!(
                "{} → {} · {} · {} · {}",
                dash(&item.meta.language),
                item.meta.target,
                fmt_time(item.meta.duration),
                item.meta.created_text(),
                item.meta.device
            ),
        ),
        None => (
            "Local files".to_string(),
            format!(
                "{} paragraphs · {} words",
                player.paragraphs.len(),
                player.words.len()
            ),
        ),
    };

    let back = if app.article_dir.is_some() {
        Screen::Article
    } else {
        Screen::Library
    };
    let back_label = if app.article_dir.is_some() {
        "← Article"
    } else {
        "← Articles"
    };

    let mut header = row![
        button(text(back_label).size(12).font(fonts.body))
            .on_press(Message::Screen(back))
            .padding([8, 12])
            .style(theme::chip),
        column![
            text(title).size(18).font(fonts.display).color(theme::TEXT),
            text(meta_line)
                .size(10)
                .font(fonts.mono)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        Space::new().width(Length::Fill),
    ]
    .spacing(12)
    .align_y(Alignment::Center);

    if !player.translation.sentences.is_empty() {
        header = header.push(
            toggler(player.show_translation)
                .label("Translation")
                .on_toggle(Message::ToggleTranslation)
                .text_size(12)
                .font(fonts.body)
                .style(theme::toggle),
        );
    }

    if !player.vocabulary.is_empty() {
        header = header.push(
            toggler(player.show_vocabulary)
                .label("Vocabulary")
                .on_toggle(|_| Message::ToggleVocabulary)
                .text_size(12)
                .font(fonts.body)
                .style(theme::toggle),
        );
    }


    let description: Element<'_, Message> = match &player.item {
        Some(item) if !item.meta.description.is_empty() => text(item.meta.description.clone())
            .size(12)
            .font(fonts.body)
            .color(theme::TEXT_DIM)
            .into(),
        _ => Space::new().height(0).into(),
    };

    let mut content = column![
        header,
        description,
        transport(player, fonts),
        now_playing(player, fonts),
    ]
    .spacing(12)
    .height(Length::Fill);

    if player.show_vocabulary {
        content = content.push(vocabulary_panel(player, fonts));
    }
    content = content
        .push(transcript(player, fonts))
        .push(footer(player, fonts));

    container(content)
    .padding(18)
    .width(Length::Fill)
    .height(Length::Fill)
    .style(theme::card)
    .into()
}

/// Play/pause, ±5 s, big time readout, seek bar and volume.
fn transport(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {
    let position = player.shown_position();

    let buttons = row![
        button(text("−5 s").size(12).font(fonts.body))
            .on_press(Message::SeekTo(player.position - 5.0))
            .padding([9, 13])
            .style(theme::ghost),
        button(text(if player.playing { "❚❚" } else { "▶" }).size(15).font(fonts.body))
            .on_press(Message::TogglePlay)
            .padding([9, 19])
            .style(theme::play),
        button(text("+5 s").size(12).font(fonts.body))
            .on_press(Message::SeekTo(player.position + 5.0))
            .padding([9, 13])
            .style(theme::ghost),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    let readout = column![
        row![
            text(fmt_time(position))
                .size(28)
                .font(fonts.display)
                .color(theme::TEXT),
            text(format!("/ {}", fmt_time(player.duration)))
                .size(13)
                .font(fonts.mono)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(8)
        .align_y(Alignment::End),
        slider(0.0..=player.duration.max(0.01), position, Message::Drag)
            .step(0.05_f32)
            .on_release(Message::CommitDrag)
            .style(theme::seek)
            .width(Length::Fill),
    ]
    .spacing(6)
    .width(Length::Fill);

    let volume = column![
        text("VOLUME")
            .size(9)
            .font(fonts.display)
            .color(theme::TEXT_MUTED),
        slider(0.0..=1.0, player.volume, Message::Volume)
            .style(theme::volume)
            .width(Length::Fixed(84.0)),
    ]
    .spacing(4)
    .align_x(Alignment::End);

    row![buttons, readout, volume]
        .spacing(18)
        .align_y(Alignment::Center)
        .into()
}

/// The pairs stored for a paragraph's `article.json` block.
fn block_pairs(player: &Player, block: Option<usize>) -> &[library::PairRef] {
    let Some(block) = block else {
        return &[];
    };
    player
        .pairs
        .iter()
        .find(|entry| entry.index == block)
        .map(|entry| entry.pairs.as_slice())
        .unwrap_or(&[])
}

/// Running token offset of each sentence inside a joined paragraph.
fn sentence_offsets(counts: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut offset = 0;
    for count in counts {
        offsets.push(offset);
        offset += count;
    }
    offsets
}

/// Source-word indexes of a paragraph's vocabulary pairs.
fn paragraph_pair_sources(player: &Player, paragraph: usize) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    let Some(paragraph) = player.paragraphs.get(paragraph) else {
        return out;
    };
    let pairs = block_pairs(player, paragraph.block);
    if pairs.is_empty() {
        return out;
    }

    let offsets = sentence_offsets(paragraph.sentences.clone().map(|sentence| {
        player
            .phrases
            .get(sentence)
            .map(|phrase| phrase.text.split_whitespace().count())
            .unwrap_or(0)
    }));

    for pair in pairs {
        let Some(base) = offsets.get(pair.sentence) else {
            continue;
        };
        for index in &pair.source {
            out.insert(base + index);
        }
    }
    out
}

/// Target-word indexes of a paragraph's vocabulary pairs.
fn paragraph_pair_targets(player: &Player, paragraph: usize) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    let Some(paragraph) = player.paragraphs.get(paragraph) else {
        return out;
    };
    let pairs = block_pairs(player, paragraph.block);
    if pairs.is_empty() {
        return out;
    }

    let offsets = sentence_offsets(paragraph.sentences.clone().map(|sentence| {
        player
            .translation
            .sentences
            .get(sentence)
            .map(|line| line.split_whitespace().count())
            .unwrap_or(0)
    }));

    for pair in pairs {
        let Some(base) = offsets.get(pair.sentence) else {
            continue;
        };
        for index in &pair.target {
            out.insert(base + index);
        }
    }
    out
}

/// A paragraph's translation, with the words of a vocabulary pair marked.
fn paragraph_line(
    player: &Player,
    index: usize,
    fonts: theme::Fonts,
    base: iced::Color,
) -> Option<Element<'_, Message>> {
    let line = player.paragraph_translation(index)?;
    let targets = paragraph_pair_targets(player, index);

    let mut words: Vec<Element<'_, Message>> = Vec::new();
    for (position, word) in line.split_whitespace().enumerate() {
        let label = text(word.to_string()).size(13).font(fonts.body);
        if targets.contains(&position) {
            words.push(
                container(label.color(theme::ACCENT_HI))
                    .padding([0, 4])
                    .style(theme::pair_chip)
                    .into(),
            );
        } else {
            words.push(label.color(base).into());
        }
    }

    Some(
        Row::with_children(words)
            .spacing(3)
            .wrap()
            .vertical_spacing(3)
            .into(),
    )
}

/// The original sentence, with the words of a vocabulary pair marked.
fn pair_words_line(
    sentence: &str,
    sources: &std::collections::HashSet<usize>,
    fonts: theme::Fonts,
    color: iced::Color,
) -> Element<'static, Message> {
    let mut words: Vec<Element<'static, Message>> = Vec::new();
    for (position, word) in sentence.split_whitespace().enumerate() {
        let label = text(word.to_string()).size(15).font(fonts.body);
        if sources.contains(&position) {
            words.push(
                container(label.color(theme::ACCENT_HI))
                    .padding([0, 3])
                    .style(theme::pair_chip)
                    .into(),
            );
        } else {
            words.push(label.color(color).into());
        }
    }

    Row::with_children(words)
        .spacing(3)
        .wrap()
        .vertical_spacing(2)
        .into()
}

/// The current paragraph, word by word.
fn now_playing(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {
    let paragraph_index = player.current_paragraph();
    let paragraph = player.paragraphs.get(paragraph_index);
    let sources = paragraph_pair_sources(player, paragraph_index);

    let mut cells: Vec<Element<'_, Message>> = Vec::new();
    if let Some(paragraph) = paragraph {
        for index in paragraph.words.clone() {
            let Some(word) = player.words.get(index) else {
                continue;
            };
            let label = word.text.trim();
            if label.is_empty() {
                continue;
            }

            let active = index == player.current_word;
            let paired =
                index >= paragraph.words.start && sources.contains(&(index - paragraph.words.start));
            let color = if active {
                theme::WHITE
            } else if paired {
                theme::ACCENT_HI
            } else if index < player.current_word {
                theme::TEXT_MUTED
            } else {
                theme::TEXT
            };

            cells.push(
                button(text(label).size(20).font(fonts.body).color(color))
                    .on_press(Message::SeekTo(word.start))
                    .padding([2, 5])
                    .style(move |theme, status| {
                        if paired {
                            theme::word_pair(theme, status, active)
                        } else {
                            theme::word(theme, status, active)
                        }
                    })
                    .into(),
            );
        }
    }

    let words = Row::with_children(cells)
        .spacing(3)
        .wrap()
        .vertical_spacing(4);

    let current_translation = if player.show_translation {
        paragraph_line(player, paragraph_index, fonts, theme::TEXT_DIM)
    } else {
        None
    };

    let mut content = column![
        row![
            text("NOW PLAYING")
                .size(10)
                .font(fonts.display)
                .color(theme::TEXT_MUTED),
            container(
                text(format!("{} / {}", paragraph_index + 1, player.paragraphs.len()))
                    .size(10)
                    .font(fonts.mono)
                    .color(theme::TEXT_DIM),
            )
            .padding([3, 9])
            .style(theme::pill),
            Space::new().width(Length::Fill),
            text(
                paragraph
                    .map(|paragraph| {
                        format!("{} – {}", fmt_time(paragraph.start), fmt_time(paragraph.end))
                    })
                    .unwrap_or_default()
            )
            .size(11)
            .font(fonts.mono)
            .color(theme::TEXT_MUTED),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        Space::new().height(10),
        words,
    ]
    .spacing(6);

    if let Some(translation) = current_translation {
        content = content.push(translation);
    }

    container(content)
        .padding(16)
        .width(Length::Fill)
        .style(theme::highlight_card)
        .into()
}

fn transcript(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {

    let current = player.current_paragraph();
    let (first, last) = if player.follow {
        let first = current.saturating_sub(3);
        (first, (first + 7).min(player.paragraphs.len()))
    } else {
        (0, player.paragraphs.len())
    };

    let mut rows = column![].spacing(2).width(Length::Fill);

    for index in first..last {
        let paragraph = &player.paragraphs[index];
        let active = index == current;
        let spoken = index < current;

        let mut cells: Vec<Element<'_, Message>> = Vec::new();
        if active {
            cells.push(
                container(Space::new().width(3).height(18))
                    .style(theme::accent_bar)
                    .into(),
            );
        } else {
            cells.push(Space::new().width(3).height(0).into());
        }
        cells.push(
            text(format!("{} – {}", fmt_time(paragraph.start), fmt_time(paragraph.end)))
                .size(11)
                .font(fonts.mono)
                .color(if active { theme::ACCENT_HI } else { theme::TEXT_MUTED })
                .width(Length::Fixed(96.0))
                .into(),
        );

        let color = if active {
            theme::TEXT
        } else if spoken {
            theme::TEXT_MUTED
        } else {
            theme::TEXT_DIM
        };
        let sources = paragraph_pair_sources(player, index);
        let mut lines = column![pair_words_line(&paragraph.text, &sources, fonts, color)]
            .spacing(2)
            .width(Length::Fill);

        if player.show_translation {
            if let Some(line) = paragraph_line(
                player,
                index,
                fonts,
                if active { theme::ACCENT_HI } else { theme::TEXT_MUTED },
            ) {
                lines = lines.push(line);
            }
        }

        cells.push(lines.into());

        rows = rows.push(
            button(Row::with_children(cells).spacing(12).align_y(Alignment::Center))
                .on_press(Message::SeekTo(paragraph.start))
                .padding([9, 12])
                .width(Length::Fill)
                .style(move |theme, status| theme::sentence(theme, status, active)),
        );
    }
    let list: Element<'_, Message> = if player.follow {
        column![
            Space::new().height(Length::Fill),
            rows,
            Space::new().height(Length::Fill),
        ]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    } else {
        scrollable(rows)
            .height(Length::Fill)
            .direction(scrollable::Direction::Vertical(
                scrollable::Scrollbar::new().width(6.0).scroller_width(6.0).margin(2.0),
            ))
            .style(theme::scroll)
            .into()
    };

    let header = row![
        text("TRANSCRIPT")
            .size(10)
            .font(fonts.display)
            .color(theme::TEXT_MUTED),
        Space::new().width(Length::Fill),
        toggler(player.follow)
            .label("Follow")
            .on_toggle(Message::ToggleFollow)
            .text_size(12)
            .font(fonts.body)
            .style(theme::toggle),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    container(column![header, Space::new().height(8), list].height(Length::Fill))
        .padding(16)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::card)
        .into()
}

/// The article's most relevant word pairs: German original, English translation.
fn vocabulary_panel(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {
    let mut list = column![].spacing(4).width(Length::Fill);

    for pair in &player.vocabulary {
        list = list.push(
            row![
                text(pair.de.clone())
                    .size(14)
                    .font(fonts.body)
                    .color(theme::TEXT)
                    .width(Length::FillPortion(3)),
                text(pair.en.clone())
                    .size(13)
                    .font(fonts.body)
                    .color(theme::TEXT_DIM)
                    .width(Length::FillPortion(2)),
            ]
            .spacing(12),
        );
    }

    let header = row![
        text("VOCABULARY")
            .size(10)
            .font(fonts.display)
            .color(theme::TEXT_MUTED),
        Space::new().width(Length::Fill),
        text(format!("{} pairs", player.vocabulary.len()))
            .size(11)
            .font(fonts.mono)
            .color(theme::TEXT_MUTED),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let list: Element<'_, Message> = scrollable(list)
        .height(Length::Fixed(170.0))
        .direction(scrollable::Direction::Vertical(
            scrollable::Scrollbar::new().width(6.0).scroller_width(6.0).margin(2.0),
        ))
        .style(theme::scroll)
        .into();

    container(column![header, Space::new().height(8), list].spacing(4))
        .padding(16)
        .width(Length::Fill)
        .style(theme::highlight_card)
        .into()
}

fn footer(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {
    let hint = if player.audio.failed() {
        "Audio device unavailable — transcript only"
    } else {
        "Space play/pause · ←/→ 5 s · ↑/↓ paragraph · click a word or paragraph to jump"
    };

    row![
        text(hint).size(11).font(fonts.body).color(theme::TEXT_MUTED),
        Space::new().width(Length::Fill),
        text(format!(
            "{} paragraphs · {} words",
            player.paragraphs.len(),
            player.words.len()
        ))
        .size(11)
        .font(fonts.mono)
        .color(theme::TEXT_MUTED),
    ]
    .padding([0, 4])
    .into()
}

// ---------------------------------------------------------------- helpers

fn load(path: &PathBuf) -> Result<Doc, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("cannot parse {}: {err}", path.display()))
}

/// Assigns word indices to sentences, greedily by time.
fn group(phrases: &[library::Seg], words: &[library::Seg]) -> Vec<Phrase> {
    let mut out = Vec::with_capacity(phrases.len());
    let mut next = 0usize;

    for (index, phrase) in phrases.iter().enumerate() {
        let start = next;
        while next < words.len() && words[next].start < phrase.end - 0.08 {
            next += 1;
        }
        if index + 1 == phrases.len() {
            next = words.len();
        }
        if next == start && start < words.len() {
            next += 1;
        }
        out.push(Phrase {
            start: phrase.start,
            end: phrase.end,
            text: phrase.text.trim().to_string(),
            words: start..next,
            block: phrase.block,
        });
    }

    out
}

/// Groups consecutive sentences that share an `article.json` block into the
/// paragraphs the player shows; spoken metadata stays on its own line. The
/// paragraph translation is the stored one (from `translation.json`), falling
/// back to the sentence translations.
fn paragraphs(phrases: &[Phrase], translation: &library::Translation) -> Vec<Paragraph> {
    let mut out: Vec<Paragraph> = Vec::new();

    for (index, phrase) in phrases.iter().enumerate() {
        if let Some(last) = out.last_mut() {
            if phrase.block.is_some() && last.block == phrase.block {
                last.end = last.end.max(phrase.end);
                last.words.end = phrase.words.end;
                last.sentences.end = index + 1;
                last.text.push(' ');
                last.text.push_str(&phrase.text);
                continue;
            }
        }

        out.push(Paragraph {
            start: phrase.start,
            end: phrase.end,
            text: phrase.text.clone(),
            translation: String::new(),
            words: phrase.words.clone(),
            sentences: index..index + 1,
            block: phrase.block,
        });
    }

    for paragraph in &mut out {
        let fallback = translation
            .sentences
            .get(paragraph.sentences.clone())
            .map(|lines| lines.join(" "))
            .unwrap_or_default();
        paragraph.translation = match paragraph.block {
            Some(block) => translation
                .blocks
                .iter()
                .find(|entry| entry.index == block)
                .map(|entry| entry.translation.trim().to_string())
                .filter(|text| !text.is_empty())
                .unwrap_or(fallback),
            None => fallback,
        };
    }

    out
}

fn word_at(words: &[library::Seg], seconds: f32) -> usize {
    if words.is_empty() {
        return 0;
    }
    let index = words.partition_point(|word| word.start <= seconds);
    index.saturating_sub(1).min(words.len() - 1)
}

fn fmt_time(seconds: f32) -> String {
    let total = seconds.max(0.0) as u32;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn dash(value: &str) -> &str {
    if value.is_empty() {
        "??"
    } else {
        value
    }
}

/// `transcript-player ingest <url|file>`: run the pipeline without the UI.
/// `transcript-player asr-probe <model> <audio>`: measure one ASR model.
fn asr_probe(model: &str, audio: &str) -> i32 {
    let mut log = |message: String| println!("   {message}");
    let started = std::time::Instant::now();

    let mut asr = match asr::Asr::load(model, &asr::models_dir(), "", true, &mut log) {
        Ok(asr) => asr,
        Err(err) => {
            eprintln!("== failed: {err}");
            return 1;
        }
    };

    let mut last = 0.0f32;
    let mut progress = |fraction: f32, _: &str| {
        if fraction - last > 0.1 {
            last = fraction;
            println!("   {:>3.0}%", fraction * 100.0);
        }
    };

    match asr.transcribe(std::path::Path::new(audio), &mut progress) {
        Ok(transcript) => {
            println!(
                "== {} on {}: {:.1}s audio in {:.1}s ({:.1}x realtime)",
                transcript.model,
                transcript.device,
                transcript.duration,
                started.elapsed().as_secs_f32(),
                transcript.speed
            );
            println!(
                "   {} sentences, {} words, language {}",
                transcript.phrases.len(),
                transcript.words.len(),
                transcript.language
            );
            for phrase in transcript.phrases.iter().take(4) {
                println!(
                    "   [{:7.2}-{:7.2}] {}",
                    phrase.start, phrase.end, phrase.text
                );
            }
            println!(
                "   words: {:?}",
                transcript
                    .words
                    .iter()
                    .take(6)
                    .map(|word| format!("{}@{:.2}", word.text, word.start))
                    .collect::<Vec<_>>()
            );
            0
        }
        Err(err) => {
            eprintln!("== failed: {err}");
            1
        }
    }
}

fn headless(url: &str, model: Option<&str>) -> i32 {
    run_headless(pipeline::Options {
        url: url.to_string(),
        llm_model: Some(model.unwrap_or(onnx_llm::DEFAULT_MODEL).to_string()),
        use_gpu: true,
        ..Default::default()
    })
}

/// `transcript-player align <audio> <article.txt>`: force-align German text.
fn headless_align(audio: &str, article: &str) -> i32 {
    let text = match std::fs::read_to_string(article) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("== failed: cannot read {article}: {err}");
            return 1;
        }
    };

    run_headless(pipeline::Options {
        url: audio.to_string(),
        text: Some(text),
        text_source: article.to_string(),
        language: Some("de".to_string()),
        asr_model: "nemo-de".to_string(),
        llm_model: Some(onnx_llm::DEFAULT_MODEL.to_string()),
        use_gpu: true,
        ..Default::default()
    })
}

fn run_headless(options: pipeline::Options) -> i32 {
    let (tx, rx) = std::sync::mpsc::channel();
    pipeline::spawn(options, tx);

    for event in rx {
        match event {
            pipeline::Event::Stage(stage) => println!("== {stage}"),
            pipeline::Event::Progress(progress) => {
                println!("   {:>3.0}%", progress * 100.0);
            }
            pipeline::Event::Log(line) => println!("   {line}"),
            pipeline::Event::Done(item) => {
                println!("== done: {}", item.dir.display());
                println!("   title:       {}", item.meta.title);
                println!("   description: {}", item.meta.description);
                println!(
                    "   {} / {}: {} words, {} sentences, {:.0}s audio on {}",
                    item.meta.language,
                    item.meta.target,
                    item.meta.words,
                    item.meta.sentences,
                    item.meta.duration,
                    item.meta.device
                );
                return 0;
            }
            pipeline::Event::ArticleDone(_) => {}
            pipeline::Event::Failed(error) => {
                eprintln!("== failed: {error}");
                return 1;
            }
        }
    }

    0
}

/// `transcript-player transcribe-tree <folder>`: walks a folder tree and, for
/// every `*.mp3` whose `transcribe/` folder is still missing, runs the German
/// pipeline (force-aligned to `<article>/article.md` when present) and drops
/// the transcript + translation files into `<article>/transcribe/`.
fn transcribe_tree(root: &str) -> i32 {
    let root = PathBuf::from(root);
    if !root.is_dir() {
        eprintln!("== failed: {} is not a folder", root.display());
        return 1;
    }

    let mut audios = Vec::new();
    collect_audio(&root, &mut audios);
    audios.sort();

    println!("== {} audio file(s) under {}", audios.len(), root.display());

    let mut failed = 0;
    for audio in &audios {
        let Some(base) = article_base(audio) else {
            continue;
        };

        let target = base.join("transcribe");
        if target.is_dir() {
            println!("== skip {} (already has transcribe/)", audio.display());
            continue;
        }

        let article = base.join("article.md");
        let article = article.is_file().then_some(article);
        println!(
            "== {} (article: {})",
            audio.display(),
            article
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "none".to_string())
        );

        match transcribe_one(audio, article.as_deref()) {
            Ok(item_dir) => match publish(&item_dir, &target) {
                Ok(()) => {
                    // the crawler folder is the source of truth; the intermediate
                    // library copy (incl. the audio) is not needed any more
                    if let Err(err) = std::fs::remove_dir_all(&item_dir) {
                        eprintln!("   warning: cannot remove {}: {err}", item_dir.display());
                    }
                    println!("   wrote {}", target.display());
                }
                Err(err) => {
                    eprintln!("   cannot write {}: {err}", target.display());
                    failed += 1;
                }
            },
            Err(err) => {
                eprintln!("   failed: {err}");
                failed += 1;
            }
        }
    }

    if failed > 0 {
        1
    } else {
        0
    }
}

/// Command-line interface: subcommands for the headless passes, and bare
/// positional paths to open the GUI player directly.
#[derive(clap::Parser)]
#[command(
    name = "transcript-player",
    version,
    about = "Transcribe audio and build vocabulary pairs",
    subcommand_precedence_over_arg = true
)]
struct Cli {
    /// audio file/URL, article folder or library folder (opens the GUI)
    path: Option<String>,
    /// word-level JSON for a plain audio file (GUI player)
    words: Option<String>,
    /// sentence-level JSON for a plain audio file (GUI player)
    sentences: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Ingest a URL or local audio file, without the UI
    Ingest {
        url: String,
        /// local language model label
        model: Option<String>,
    },
    /// Force-align a known German article to an audio file
    Align { audio: String, article: String },
    /// Batch: every audio file in a tree -> <article>/transcribe/
    TranscribeTree { folder: String },
    /// Batch: every article.json in a tree -> <article>/transcribe/
    Articles {
        folder: String,
        /// redo articles that are already translated
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        pairs: PairsArgs,
    },
    /// Re-run the vocabulary + pairs pass on one article folder
    Vocabulary {
        folder: String,
        /// local vocabulary model label
        model: Option<String>,
        /// rewrite vocabulary.json and pairs.json even when they exist
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        pairs: PairsArgs,
    },
    /// Batch: every transcription.json in a tree -> vocabulary.json + pairs.json
    Pairs {
        folder: String,
        /// local vocabulary model label
        model: Option<String>,
        /// re-extract vocabulary.json and overwrite pairs.json
        #[arg(long)]
        force: bool,
        /// folders to process at once when an external model is used
        #[arg(short = 'j', long, default_value_t = 1)]
        jobs: usize,
        #[command(flatten)]
        pairs: PairsArgs,
    },
    /// Measure one ASR model on an audio file
    AsrProbe {
        model: Option<String>,
        audio: Option<String>,
    },
}

/// The external `pi` model selected by `--pairs-provider` / `--pairs-model`
/// (or by `TRANSCRIBE_PAIRS_PROVIDER` / `TRANSCRIBE_PAIRS_MODEL`).
#[derive(clap::Args, Clone)]
struct PairsArgs {
    /// use the default external `pi` model (or the env-selected one)
    #[arg(long)]
    external_pairs: bool,
    /// external `pi` provider, e.g. `deepinfra`
    #[arg(long)]
    pairs_provider: Option<String>,
    /// external `pi` model, e.g. `deepinfra/deepseek-ai/DeepSeek-V4.1-Flash`
    #[arg(long)]
    pairs_model: Option<String>,
}

impl PairsArgs {
    /// `Some` when a flag or environment variable selects an external model.
    fn resolve(&self) -> Option<external::PairsModel> {
        let mut model = external::PairsModel::from_env();
        let mut explicit = external::enabled_from_env();
        if let Some(provider) = &self.pairs_provider {
            model.provider = provider.clone();
            explicit = true;
        }
        if let Some(name) = &self.pairs_model {
            model.model = name.clone();
            explicit = true;
        }
        if self.external_pairs {
            explicit = true;
        }
        model.strip_provider_prefix();
        explicit.then_some(model)
    }
}

/// `transcript-player articles <root> [--force] [--pairs-provider P] [--pairs-model M]`:
/// walks the crawler's article tree and, for every `article.json`, transcribes the
/// audio it names and writes `transcription.json` + `translation.json` (+ vocabulary
/// and pairs) into the article's `transcribe/` folder. An article that already has
/// `transcribe/translation.json` is skipped unless `--force` is given. With
/// `--pairs-provider`/`--pairs-model` the pairs come from the external `pi` model.
fn article_tree(root: &str, force: bool, pairs_model: Option<external::PairsModel>) -> i32 {
    let root = PathBuf::from(root);
    if !root.is_dir() {
        eprintln!("== failed: {} is not a folder", root.display());
        return 1;
    }

    let mut articles = Vec::new();
    collect_articles(&root, &mut articles);
    articles.sort();

    println!("== {} article(s) under {}", articles.len(), root.display());

    let options = pipeline::ArticleOptions {
        force,
        pairs_model,
        ..Default::default()
    };

    let mut failed = 0;
    for dir in &articles {
        println!("== {}", dir.display());
        match article_one(dir, &options) {
            Ok(true) => println!("   wrote {}", dir.join(library::TRANSCRIPT_DIR).display()),
            Ok(false) => {}
            Err(err) => {
                eprintln!("   failed: {err}");
                failed += 1;
            }
        }
    }

    if failed > 0 {
        1
    } else {
        0
    }
}

/// Every folder that holds an `article.json` (hidden folders and `target/` are
/// skipped).
fn collect_articles(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }

        let path = entry.path();
        if file_type.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "target" || name == library::TRANSCRIPT_DIR {
                continue;
            }
            collect_articles(&path, out);
        } else if path
            .file_name()
            .map(|name| name == "article.json")
            .unwrap_or(false)
        {
            if let Some(parent) = path.parent() {
                out.push(parent.to_path_buf());
            }
        }
    }
}

/// Runs the article pipeline for one folder and prints its events.
fn article_one(dir: &Path, options: &pipeline::ArticleOptions) -> Result<bool, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();

    let dir = dir.to_path_buf();
    let options = options.clone();
    let handle = std::thread::Builder::new()
        .name("article".into())
        .spawn(move || {
            let outcome = pipeline::run_article(&dir, &options, &tx);
            let _ = result_tx.send(outcome);
        })
        .map_err(|err| err.to_string())?;

    let mut last = -1i32;
    for event in rx {
        match event {
            pipeline::Event::Stage(stage) => println!("   {stage}"),
            pipeline::Event::Progress(progress) => {
                let step = (progress * 10.0) as i32;
                if step > last {
                    last = step;
                    println!("   {:>3}%", step * 10);
                }
            }
            pipeline::Event::Log(line) => println!("   {line}"),
            pipeline::Event::Done(_) => {}
            pipeline::Event::ArticleDone(_) => {}
            pipeline::Event::Failed(error) => eprintln!("   failed: {error}"),
        }
    }
    let _ = handle.join();
    result_rx
        .recv()
        .unwrap_or_else(|_| Err("pipeline ended without a result".to_string()))
}

/// Every `.mp3` below `dir` (hidden folders and `target/` are skipped).
fn collect_audio(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }

        let path = entry.path();
        if file_type.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            collect_audio(&path, out);
        } else if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("mp3"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// The folder that holds `article.md` and where `transcribe/` belongs: the
/// parent of an `audio/` folder, otherwise the audio's own folder.
fn article_base(audio: &Path) -> Option<PathBuf> {
    let dir = audio.parent()?;
    if dir.file_name().and_then(|name| name.to_str()) == Some("audio") {
        dir.parent().map(Path::to_path_buf)
    } else {
        Some(dir.to_path_buf())
    }
}

/// Runs the pipeline for one audio file and returns its library folder.
fn transcribe_one(audio: &Path, article: Option<&Path>) -> Result<PathBuf, String> {
    let text = match article {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|err| format!("cannot read {}: {err}", path.display()))?,
        ),
        None => None,
    };

    let options = pipeline::Options {
        url: audio.to_string_lossy().to_string(),
        text,
        text_source: article.map(|path| path.display().to_string()).unwrap_or_default(),
        language: Some("de".to_string()),
        asr_model: "nemo-de".to_string(),
        llm_model: Some(onnx_llm::DEFAULT_MODEL.to_string()),
        use_gpu: true,
        ..Default::default()
    };

    let (tx, rx) = std::sync::mpsc::channel();
    pipeline::spawn(options, tx);

    let mut last = -1i32;
    for event in rx {
        match event {
            pipeline::Event::Stage(stage) => println!("   {stage}"),
            pipeline::Event::Progress(progress) => {
                let step = (progress * 10.0) as i32;
                if step > last {
                    last = step;
                    println!("   {:>3}%", step * 10);
                }
            }
            pipeline::Event::Log(line) => println!("   {line}"),
            pipeline::Event::Done(item) => return Ok(item.dir),
            pipeline::Event::ArticleDone(_) => {}
            pipeline::Event::Failed(error) => return Err(error),
        }
    }

    Err("pipeline ended without a result".to_string())
}

/// `transcript-player vocabulary <article-folder> [model]`: re-runs only the
/// vocabulary pass on an already cached article and rewrites `vocabulary.json`
/// together with the per-sentence `pairs.json`. The default model is
/// `qwen2.5:3b`, which always runs on the CPU.
fn vocabulary_pass(
    dir: &str,
    model: Option<&str>,
    pairs_model: Option<external::PairsModel>,
    force: bool,
) -> i32 {
    let dir = PathBuf::from(dir);
    // accept either the `transcribe/` folder or the article folder around it
    let dir = if dir.join(library::WORDS).is_file()
        || !dir.join(library::TRANSCRIPT_DIR).join(library::WORDS).is_file()
    {
        dir
    } else {
        dir.join(library::TRANSCRIPT_DIR)
    };
    if force {
        println!("   --force: rewriting vocabulary.json and pairs.json");
    }
    let transcript = match load(&dir.join(library::WORDS)) {
        Ok(doc) => doc,
        Err(err) => {
            eprintln!("== failed: {err}");
            return 1;
        }
    };

    let sentences: Vec<String> = transcript
        .sentences
        .iter()
        .map(|segment| segment.text.trim().to_string())
        .collect();
    if sentences.is_empty() {
        eprintln!("== failed: {} has no sentences", dir.display());
        return 1;
    }

    let translation: library::Translation =
        std::fs::read_to_string(dir.join(library::TRANSLATION))
            .ok()
            .and_then(|raw| serde_json::from_str::<library::Translation>(&raw).ok())
            .unwrap_or_default();
    let translations = translation.sentences.clone();
    // the extractor reads only the article body: the sentences of the
    // `article.json` `blocks[]`, not the spoken metadata around them
    let body = pairs::body_indexes(
        &transcript
            .sentences
            .iter()
            .map(|segment| segment.block)
            .collect::<Vec<_>>(),
    );
    let pair_sentences: Vec<String> = body.iter().map(|index| sentences[*index].clone()).collect();
    let pair_translations: Vec<String> = body
        .iter()
        .map(|index| translations.get(*index).cloned().unwrap_or_default())
        .collect();
    let meta = library::read_meta(&dir).unwrap_or_default();
    let target = if meta.target.is_empty() { "en" } else { meta.target.as_str() };
    let language = if meta.language.is_empty() { "de" } else { meta.language.as_str() };

    let Some((pairs, model_label)) = cli_pairs(
        &pair_sentences,
        &pair_translations,
        language,
        target,
        model,
        pairs_model.as_ref(),
    ) else {
        return 1;
    };

    let doc = library::Vocabulary {
        target: target.to_string(),
        pairs: pairs
            .iter()
            .map(|(de, en)| library::WordPair {
                de: de.clone(),
                en: en.clone(),
            })
            .collect(),
    };
    let path = dir.join(library::VOCABULARY);
    if let Err(err) = std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap_or_default())
    {
        eprintln!("== failed: cannot write {}: {err}", path.display());
        return 1;
    }
    for pair in &doc.pairs {
        println!("   {} — {}", pair.de, pair.en);
    }
    println!("== wrote {} ({} pairs)", path.display(), doc.pairs.len());

    // the same vocabulary locates the pairs inside the aligned sentences
    let refs = pairs::locate(&pairs, &sentences, &translations);
    if refs.is_empty() {
        eprintln!("   warning: no vocabulary pair could be placed in a sentence");
    } else {
        let ranges = translation
            .blocks
            .iter()
            .map(|block| (block.index, block.kind.as_str(), block.first, block.count))
            .collect::<Vec<_>>();
        let blocks = pairs::group(&refs, &ranges);
        match pairs::write(&dir, &model_label, target, &blocks) {
            Ok(()) => println!(
                "== wrote {} ({} pairs)",
                dir.join(library::PAIRS).display(),
                blocks.iter().map(|block| block.pairs.len()).sum::<usize>()
            ),
            Err(err) => {
                eprintln!("== failed: cannot write pairs.json: {err}");
                return 1;
            }
        }
    }
    0
}

/// Pairs for the CLI passes: the external `pi` model when selected, else the
/// local model. Prints the failure and returns `None` when neither produced
/// pairs.
fn cli_pairs(
    sentences: &[String],
    translations: &[String],
    language: &str,
    target: &str,
    local_model: Option<&str>,
    pairs_model: Option<&external::PairsModel>,
) -> Option<(Vec<(String, String)>, String)> {
    if let Some(pairs_model) = pairs_model {
        println!("   external model: {}", pairs_model.label());
        let mut log = |message: String| println!("   {message}");
        match external::vocabulary(
            pairs_model,
            sentences,
            translations,
            language,
            target,
            pipeline::VOCABULARY_LIMIT,
            &mut log,
        ) {
            Ok(pairs) => return Some((pairs, pairs_model.label())),
            Err(err) => eprintln!("   external pairs failed: {err}"),
        }
    }

    let label = local_model.unwrap_or(onnx_llm::VOCABULARY_MODEL);
    let mut log = |message: String| println!("   {message}");
    let mut llm = match onnx_llm::Llm::load(label, &pipeline::models_dir(), false, &mut log) {
        Ok(llm) => llm,
        Err(err) => {
            eprintln!("== failed: {err}");
            return None;
        }
    };
    println!("   vocabulary model: {}", llm.summary());
    match llm.vocabulary(sentences, translations, language, target, pipeline::VOCABULARY_LIMIT) {
        Ok(pairs) => Some((pairs, llm.model.clone())),
        Err(err) => {
            eprintln!("== failed: {err}");
            None
        }
    }
}
/// What one folder contributed to a `pairs` run.
#[derive(Default, Clone, Copy)]
struct PairRun {
    /// size of the vocabulary list used for this folder
    vocab: usize,
    /// pairings that could be placed in the sentences
    placed: usize,
}

/// Aggregated statistics of a whole `pairs` run.
#[derive(Default, Clone, Copy)]
struct PairTotals {
    written: usize,
    skipped: usize,
    failed: usize,
    vocab: usize,
    placed: usize,
}

/// The progress bar the `pairs` pass reports on, one tick per folder.
fn pairs_bar(total: usize) -> ProgressBar {
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} folders  {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    bar.set_message("starting");
    bar
}

/// Prints the final statistics of a `pairs` run.
fn print_pairs_summary(totals: PairTotals, model: &str, elapsed: std::time::Duration) {
    println!(
        "== {} folder(s): {} written, {} skipped, {} failed",
        totals.written + totals.skipped + totals.failed,
        totals.written,
        totals.skipped,
        totals.failed
    );
    println!(
        "   {} vocabulary pairs, {} placed in pairs.json, {:.1}s, model {model}",
        totals.vocab,
        totals.placed,
        elapsed.as_secs_f32()
    );
}

/// `transcript-player pairs <folder> [model] [--force] [--jobs N]`: walks a
/// folder tree and, for every `transcription.json` whose folder also holds a
/// `translation.json`, builds `vocabulary.json` (reused unless `--force`) and
/// the per-sentence `pairs.json` that maps each vocabulary pair to its word
/// indexes. A folder that already has `pairs.json` is skipped unless `--force`
/// is given. The model defaults to `qwen2.5:3b` on the CPU; with an external
/// `pi` model several folders run at once (`--jobs`).
fn pair_tree(
    root: &str,
    model: Option<&str>,
    pairs_model: Option<external::PairsModel>,
    force: bool,
    jobs: usize,
) -> i32 {
    let root = PathBuf::from(root);
    if !root.is_dir() {
        eprintln!("== failed: {} is not a folder", root.display());
        return 1;
    }

    let mut transcripts = Vec::new();
    collect_transcripts(&root, &mut transcripts);
    transcripts.sort();
    println!("== {} transcript(s) under {}", transcripts.len(), root.display());

    let model_label = match &pairs_model {
        Some(pairs_model) => pairs_model.label(),
        None => model.unwrap_or(onnx_llm::VOCABULARY_MODEL).to_string(),
    };
    if pairs_model.is_some() {
        println!("   external model: {model_label}");
    }

    let started = std::time::Instant::now();

    // Every external call shells out to its own `pi` process, so folders are
    // independent and several can run at once. The local model is a single CPU
    // session, so it always runs one folder at a time.
    let parallel = pairs_model.is_some() && jobs > 1;
    if parallel {
        println!("   {jobs} folders in parallel");
    } else if pairs_model.is_none() && jobs > 1 {
        println!(
            "   --jobs is ignored without an external model \
             (the local model runs one folder at a time)"
        );
    }

    let bar = pairs_bar(transcripts.len());
    if parallel {
        if let Some(pairs_model) = &pairs_model {
            let (code, totals) =
                pair_tree_parallel(&transcripts, pairs_model, force, jobs, &bar);
            bar.finish_and_clear();
            print_pairs_summary(totals, &model_label, started.elapsed());
            return code;
        }
    }

    let label = model.unwrap_or(onnx_llm::VOCABULARY_MODEL);
    let mut log = |message: String| println!("   {message}");
    let mut llm = match onnx_llm::Llm::load(label, &pipeline::models_dir(), false, &mut log) {
        Ok(llm) => llm,
        Err(err) => {
            bar.finish_and_clear();
            eprintln!("== failed: {err}");
            return 1;
        }
    };
    println!("   vocabulary model: {}", llm.summary());

    let mut totals = PairTotals::default();
    for transcript in &transcripts {
        let Some(dir) = transcript.parent() else { continue };
        let output = dir.join(library::PAIRS);
        if !force && output.is_file() {
            println!(
                "== skip {} (already has pairs.json, use --force)",
                dir.display()
            );
            totals.skipped += 1;
            bar.set_message("skipped");
            bar.inc(1);
            continue;
        }

        println!("== {}", transcript.display());
        match build_pairs(&mut llm, transcript, dir, pairs_model.as_ref(), force) {
            Ok(run) => {
                totals.written += 1;
                totals.vocab += run.vocab;
                totals.placed += run.placed;
                println!(
                    "   wrote {} ({} placed of {} pairs)",
                    output.display(),
                    run.placed,
                    run.vocab
                );
            }
            Err(err) => {
                eprintln!("   failed: {err}");
                totals.failed += 1;
            }
        }
        bar.set_message(format!("{} pairs, {} placed", totals.vocab, totals.placed));
        bar.inc(1);
    }

    bar.finish_and_clear();
    print_pairs_summary(totals, &model_label, started.elapsed());

    if totals.failed > 0 {
        1
    } else {
        0
    }
}

/// The `pairs` pass with an external model and `jobs > 1`: folders are handed
/// to the first free worker, each worker runs its own `pi` process. The local
/// model is never loaded, so there is no fallback here — a folder that fails is
/// reported and the run exits non-zero.
fn pair_tree_parallel(
    transcripts: &[PathBuf],
    pairs_model: &external::PairsModel,
    force: bool,
    jobs: usize,
    bar: &ProgressBar,
) -> (i32, PairTotals) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let next = AtomicUsize::new(0);
    let written = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let vocab = AtomicUsize::new(0);
    let placed = AtomicUsize::new(0);

    // no point in more workers than folders
    let workers = jobs.max(1).min(transcripts.len().max(1));

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                let Some(transcript) = transcripts.get(index) else {
                    break;
                };
                let Some(dir) = transcript.parent() else { continue };
                let output = dir.join(library::PAIRS);
                if !force && output.is_file() {
                    println!(
                        "== skip {} (already has pairs.json, use --force)",
                        dir.display()
                    );
                    skipped.fetch_add(1, Ordering::SeqCst);
                    bar.set_message("skipped");
                    bar.inc(1);
                    continue;
                }

                let tag = dir.display().to_string();
                let label = pairs_model.label();
                println!("== {}", transcript.display());
                match build_pairs_inner(
                    transcript,
                    dir,
                    force,
                    &label,
                    |sentences, translations, language, target| {
                        let mut log = |message: String| println!("   [{tag}] {message}");
                        external::vocabulary(
                            pairs_model,
                            sentences,
                            translations,
                            language,
                            target,
                            pipeline::VOCABULARY_LIMIT,
                            &mut log,
                        )
                        .map(|pairs| (pairs, label.clone()))
                    },
                ) {
                    Ok(run) => {
                        written.fetch_add(1, Ordering::SeqCst);
                        let total_vocab =
                            vocab.fetch_add(run.vocab, Ordering::SeqCst) + run.vocab;
                        let total_placed =
                            placed.fetch_add(run.placed, Ordering::SeqCst) + run.placed;
                        println!(
                            "   [{tag}] wrote {} ({} placed of {} pairs)",
                            output.display(),
                            run.placed,
                            run.vocab
                        );
                        bar.set_message(format!("{total_vocab} pairs, {total_placed} placed"));
                    }
                    Err(err) => {
                        eprintln!("   [{tag}] failed: {err}");
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                }
                bar.inc(1);
            });
        }
    });

    let totals = PairTotals {
        written: written.load(Ordering::SeqCst),
        skipped: skipped.load(Ordering::SeqCst),
        failed: failed.load(Ordering::SeqCst),
        vocab: vocab.load(Ordering::SeqCst),
        placed: placed.load(Ordering::SeqCst),
    };
    let code = if totals.failed > 0 { 1 } else { 0 };
    (code, totals)
}

/// Every file named `transcription.json` below `dir` (hidden folders and `target/`
/// are skipped), the same walk the transcript tree pass uses.
fn collect_transcripts(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }

        let path = entry.path();
        if file_type.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            collect_transcripts(&path, out);
        } else if path
            .file_name()
            .map(|name| name == library::WORDS)
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Extracts vocabulary pairs for the CLI passes with the external `pi` model
/// when one is selected, else with the already loaded local model. Returns the
/// pairs and the model label recorded in `pairs.json`.
fn extract_with(
    llm: &mut onnx_llm::Llm,
    pairs_model: Option<&external::PairsModel>,
    sentences: &[String],
    translations: &[String],
    language: &str,
    target: &str,
) -> (Vec<(String, String)>, String) {
    if let Some(pairs_model) = pairs_model {
        let mut log = |message: String| println!("   {message}");
        match external::vocabulary(
            pairs_model,
            sentences,
            translations,
            language,
            target,
            pipeline::VOCABULARY_LIMIT,
            &mut log,
        ) {
            Ok(pairs) => return (pairs, pairs_model.label()),
            Err(err) => eprintln!("   external pairs failed: {err}"),
        }
    }

    match llm.vocabulary(sentences, translations, language, target, pipeline::VOCABULARY_LIMIT) {
        Ok(pairs) => (pairs, llm.model.clone()),
        Err(err) => {
            eprintln!("   vocabulary failed: {err}");
            (Vec::new(), llm.model.clone())
        }
    }
}
/// Builds `pairs.json` for one folder from `transcription.json` and
/// `translation.json`; returns the number of pairings written.
fn build_pairs(
    llm: &mut onnx_llm::Llm,
    words_path: &Path,
    dir: &Path,
    pairs_model: Option<&external::PairsModel>,
    force: bool,
) -> Result<PairRun, String> {
    let fallback_label = llm.model.clone();
    build_pairs_inner(
        words_path,
        dir,
        force,
        &fallback_label,
        |sentences, translations, language, target| {
            Ok(extract_with(
                llm,
                pairs_model,
                sentences,
                translations,
                language,
                target,
            ))
        },
    )
}

/// Shared body of the `pairs` pass: reads `transcription.json` and
/// `translation.json`, reuses `vocabulary.json` unless `force` asks for a fresh
/// extraction, writes `vocabulary.json` and `pairs.json`, and returns the
/// vocabulary size and the number of placed pairings. `extract` builds the
/// vocabulary list; `fallback_label` is the model recorded when an existing
/// `vocabulary.json` is reused.
fn build_pairs_inner<F>(
    words_path: &Path,
    dir: &Path,
    force: bool,
    fallback_label: &str,
    extract: F,
) -> Result<PairRun, String>
where
    F: FnOnce(&[String], &[String], &str, &str) -> Result<(Vec<(String, String)>, String), String>,
{
    let transcript = load(&words_path.to_path_buf())?;
    let sentences: Vec<String> = transcript
        .sentences
        .iter()
        .map(|segment| segment.text.trim().to_string())
        .collect();
    if sentences.is_empty() {
        return Err("transcription.json has no sentences".to_string());
    }

    let translation_path = dir.join(library::TRANSLATION);
    let raw = std::fs::read_to_string(&translation_path)
        .map_err(|err| format!("cannot read {}: {err}", translation_path.display()))?;
    let translation: library::Translation = serde_json::from_str(&raw)
        .map_err(|err| format!("cannot parse {}: {err}", translation_path.display()))?;
    if translation.sentences.is_empty() {
        return Err("translation.json has no sentences".to_string());
    }

    // the two files are aligned 1:1; a mismatch means one was edited alone
    let count = sentences.len().min(translation.sentences.len());
    if sentences.len() != translation.sentences.len() {
        eprintln!(
            "   warning: {} sentences but {} translations, using the first {count}",
            sentences.len(),
            translation.sentences.len()
        );
    }

    let language = if transcript.language.is_empty() {
        "de".to_string()
    } else {
        transcript.language.clone()
    };
    let target = if translation.target.is_empty() {
        "en".to_string()
    } else {
        translation.target.clone()
    };

    // the extractor reads only the article body: the sentences of the
    // `article.json` `blocks[]`, not the spoken metadata around them
    let body = pairs::body_indexes(
        &transcript.sentences[..count]
            .iter()
            .map(|segment| segment.block)
            .collect::<Vec<_>>(),
    );
    let pair_sentences: Vec<String> = body.iter().map(|index| sentences[*index].clone()).collect();
    let pair_translations: Vec<String> = body
        .iter()
        .map(|index| translation.sentences[*index].clone())
        .collect();
    // The vocabulary pass writes both files; reuse vocabulary.json when it is
    // already there, unless `--force` asks for a fresh extraction.
    let vocabulary_path = dir.join(library::VOCABULARY);
    let reused = if force {
        None
    } else {
        std::fs::read_to_string(&vocabulary_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<library::Vocabulary>(&raw).ok())
            .filter(|doc| !doc.pairs.is_empty())
    };
    let (pairs, model_label, vocab): (Vec<(String, String)>, String, usize) = match reused {
        Some(doc) => {
            println!("   reusing {}", vocabulary_path.display());
            let pairs = doc
                .pairs
                .into_iter()
                .map(|pair| (pair.de, pair.en))
                .collect::<Vec<_>>();
            let vocab = pairs.len();
            (pairs, fallback_label.to_string(), vocab)
        }
        None => {
            let (found, label) =
                extract(&pair_sentences, &pair_translations, &language, &target)?;
            if found.is_empty() {
                return Err("no vocabulary pair could be extracted".to_string());
            }
            let doc = library::Vocabulary {
                target: target.clone(),
                pairs: found
                    .iter()
                    .map(|(de, en)| library::WordPair {
                        de: de.clone(),
                        en: en.clone(),
                    })
                    .collect(),
            };
            let raw = serde_json::to_string_pretty(&doc).map_err(|err| err.to_string())?;
            std::fs::write(&vocabulary_path, raw).map_err(|err| err.to_string())?;
            println!(
                "   wrote {} ({} pairs)",
                vocabulary_path.display(),
                doc.pairs.len()
            );
            let vocab = found.len();
            (found, label, vocab)
        }
    };

    let refs = pairs::locate(&pairs, &sentences[..count], &translation.sentences[..count]);
    if refs.is_empty() {
        return Err("no pairing could be located in the sentences".to_string());
    }
    let ranges = translation
        .blocks
        .iter()
        .map(|block| (block.index, block.kind.as_str(), block.first, block.count))
        .collect::<Vec<_>>();
    let blocks = pairs::group(&refs, &ranges);
    pairs::write(dir, &model_label, &target, &blocks)?;
    Ok(PairRun {
        vocab,
        placed: refs.len(),
    })
}


/// Copies the transcript files (and nothing else) into the `transcribe/` folder.
fn publish(item_dir: &Path, target: &Path) -> Result<(), String> {
    std::fs::create_dir_all(target).map_err(|err| err.to_string())?;

    for name in [
        library::WORDS,
        library::TRANSLATION,
        library::VOCABULARY,
        library::PAIRS,
    ] {
        let from = item_dir.join(name);
        if from.is_file() {
            std::fs::copy(&from, target.join(name)).map_err(|err| err.to_string())?;
        }
    }

    Ok(())
}

fn main() -> iced::Result {
    // ONNX Runtime is loaded dynamically: point it at the CUDA build and
    // preload the CUDA libraries (RTLD_GLOBAL) before anything uses `ort`.
    let _ = onnx_llm::prepare_runtime();

    let cli = Cli::parse();
    match &cli.command {
        Some(Command::Ingest { url, model }) => {
            std::process::exit(headless(url, model.as_deref()))
        }
        Some(Command::Align { audio, article }) => {
            std::process::exit(headless_align(audio, article))
        }
        Some(Command::TranscribeTree { folder }) => std::process::exit(transcribe_tree(folder)),
        Some(Command::Articles {
            folder,
            force,
            pairs,
        }) => std::process::exit(article_tree(folder, *force, pairs.resolve())),
        Some(Command::Vocabulary {
            folder,
            model,
            force,
            pairs,
        }) => std::process::exit(vocabulary_pass(
            folder,
            model.as_deref(),
            pairs.resolve(),
            *force,
        )),
        Some(Command::Pairs {
            folder,
            model,
            force,
            jobs,
            pairs,
        }) => std::process::exit(pair_tree(
            folder,
            model.as_deref(),
            pairs.resolve(),
            *force,
            *jobs,
        )),
        Some(Command::AsrProbe { model, audio }) => std::process::exit(asr_probe(
            model.as_deref().unwrap_or(asr::DEFAULT_MODEL),
            audio.as_deref().unwrap_or_default(),
        )),
        None => {}
    }

    let (fonts, font_bytes) = theme::load_fonts();
    let boot_fonts = fonts;

    let application = iced::application(move || App::new(boot_fonts), update, view)
        .title("Transcribe")
        .theme(Theme::Dark)
        .default_font(fonts.body)
        .subscription(subscription)
        .level(if std::env::var_os("ON_TOP").is_some() {
            iced::window::Level::AlwaysOnTop
        } else {
            iced::window::Level::Normal
        })
        .window_size(Size::new(1180.0, 900.0));

    font_bytes
        .into_iter()
        .fold(application, |application, bytes| application.font(bytes))
        .run()
}

#[cfg(test)]
mod tree_tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("transcribe-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn article_base_sits_next_to_the_audio_folder() {
        assert_eq!(
            article_base(Path::new("/a/b/audio/x.mp3")),
            Some(PathBuf::from("/a/b"))
        );
        assert_eq!(
            article_base(Path::new("/a/b/x.mp3")),
            Some(PathBuf::from("/a/b"))
        );
    }

    #[test]
    fn finds_mp3_files_in_depth() {
        let root = scratch("scan");
        std::fs::create_dir_all(root.join("a/audio")).unwrap();
        std::fs::create_dir_all(root.join(".hidden")).unwrap();
        std::fs::write(root.join("a/audio/one.mp3"), b"x").unwrap();
        std::fs::write(root.join("two.MP3"), b"x").unwrap();
        std::fs::write(root.join("a/audio/notes.txt"), b"x").unwrap();
        std::fs::write(root.join(".hidden/three.mp3"), b"x").unwrap();

        let mut found = Vec::new();
        collect_audio(&root, &mut found);
        found.sort();

        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found.iter().any(|path| path.ends_with("a/audio/one.mp3")));
        assert!(found.iter().any(|path| path.ends_with("two.MP3")));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn publish_copies_the_transcripts_only() {
        let root = scratch("publish");
        let item = root.join("item");
        std::fs::create_dir_all(&item).unwrap();
        for name in ["transcription.json", "translation.json", "meta.json"] {
            std::fs::write(item.join(name), b"{}").unwrap();
        }

        let target = root.join("transcribe");
        publish(&item, &target).unwrap();

        assert!(target.join("transcription.json").is_file());
        assert!(target.join("translation.json").is_file());
        assert!(!target.join("meta.json").exists());

        let _ = std::fs::remove_dir_all(&root);
    }


    #[test]
    fn collects_transcripts_in_depth() {
        let root = scratch("pairs");
        std::fs::create_dir_all(root.join("a/transcribe")).unwrap();
        std::fs::write(root.join("a/transcribe/transcription.json"), b"{}").unwrap();
        std::fs::write(root.join("a/transcribe/translation.json"), b"{}").unwrap();
        std::fs::write(root.join("a/transcribe/other.json"), b"{}").unwrap();

        let mut found = Vec::new();
        collect_transcripts(&root, &mut found);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].ends_with("a/transcribe/transcription.json"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn paragraphs_group_sentences_by_article_block() {
        let phrases = vec![
            Phrase {
                start: 0.0,
                end: 1.0,
                text: "Titel".to_string(),
                words: 0..1,
                block: None,
            },
            Phrase {
                start: 1.0,
                end: 3.0,
                text: "Erster Satz.".to_string(),
                words: 1..3,
                block: Some(1),
            },
            Phrase {
                start: 3.0,
                end: 5.0,
                text: "Zweiter Satz.".to_string(),
                words: 3..5,
                block: Some(1),
            },
            Phrase {
                start: 5.0,
                end: 6.0,
                text: "Dritter Satz.".to_string(),
                words: 5..7,
                block: Some(2),
            },
        ];
        let translation = library::Translation {
            target: "en".to_string(),
            sentences: vec![
                "Title".to_string(),
                "First.".to_string(),
                "Second.".to_string(),
                "Third.".to_string(),
            ],
            source: Vec::new(),
            blocks: vec![
                library::TranslationBlock {
                    index: 1,
                    kind: "para".to_string(),
                    first: 1,
                    count: 2,
                    translation: "First. Second.".to_string(),
                },
                library::TranslationBlock {
                    index: 2,
                    kind: "para".to_string(),
                    first: 3,
                    count: 1,
                    translation: "Third.".to_string(),
                },
            ],
        };

        let paragraphs = paragraphs(&phrases, &translation);

        assert_eq!(paragraphs.len(), 3);
        assert_eq!(paragraphs[0].text, "Titel");
        assert_eq!(paragraphs[0].translation, "Title");
        assert!(paragraphs[0].block.is_none());
        assert_eq!(paragraphs[1].text, "Erster Satz. Zweiter Satz.");
        assert_eq!(paragraphs[1].translation, "First. Second.");
        assert_eq!(paragraphs[1].words, 1..5);
        assert_eq!(paragraphs[1].sentences, 1..3);
        assert_eq!(paragraphs[2].text, "Dritter Satz.");
        assert_eq!(paragraphs[2].translation, "Third.");
    }
}

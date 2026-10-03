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
//!     transcript-player --ingest <url|file> # run the pipeline headless
//!     transcript-player --align <audio> <article.txt>   # force-align German
//!     transcript-player --transcribe-tree <folder>     # batch: audio tree -> transcribe/

mod align;
mod asr;
mod audio;
mod library;
mod onnx_llm;
mod pipeline;
mod theme;

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use audio::{AudioHandle, Cmd};
use iced::widget::{
    button, column, container, pick_list, progress_bar, row, scrollable, slider, text, text_input,
    toggler, Row, Space,
};
use iced::{keyboard, time, Alignment, Element, Length, Size, Subscription, Task, Theme};
use serde::Deserialize;

const LANGUAGES: &[&str] = &["auto", "de", "en", "fr", "es", "it", "nl", "pl", "pt", "ru", "tr"];

// ---------------------------------------------------------------- data model

#[derive(Debug, Deserialize)]
struct Doc {
    #[serde(default)]
    segments: Vec<library::Seg>,
}

#[derive(Debug, Clone)]
struct Phrase {
    start: f32,
    end: f32,
    text: String,
    /// range into `Player::words`
    words: Range<usize>,
}

// ---------------------------------------------------------------- messages

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Library,
    New,
    Player,
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    Screen(Screen),
    UrlChanged(String),
    ArticleChanged(String),
    TargetChanged(String),
    LanguageChanged(String),
    AsrModelChanged(String),
    LlmModelChanged(String),
    StartJob,
    OpenItem(usize),
    DeleteItem(usize),
    CopyLog,

    // playback
    TogglePlay,
    ToggleFollow(bool),
    ToggleTranslation(bool),
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
    translation: Vec<String>,
    show_translation: bool,
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
        translation: Vec<String>,
    ) -> Result<Self, String> {
        let words = load(&words_path)?.segments;
        let phrase_segs = load(&phrases_path)?.segments;
        let phrases = group(&phrase_segs, &words);

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

        let show_translation = !translation.is_empty();

        Ok(Self {
            item,
            words,
            phrases,
            translation,
            show_translation,
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
        let audio = item
            .audio_path()
            .ok_or_else(|| format!("{}: no audio file yet", item.dir.display()))?;
        let words = item.path(library::WORDS);
        let phrases = item.path(library::PHRASES);
        if !words.is_file() || !phrases.is_file() {
            return Err("this article has no transcript yet (job still running?)".to_string());
        }

        Self::new(Some(item), audio, words, phrases, translation)
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

    /// English (or whatever the target language is) line for a sentence.
    fn translation_of(&self, index: usize) -> Option<&str> {
        self.translation
            .get(index)
            .map(String::as_str)
            .filter(|line| !line.is_empty())
    }
}

// ---------------------------------------------------------------- job state

struct Job {
    stage: String,
    progress: f32,
    log: Vec<String>,
    done: Option<library::Item>,
    failed: Option<String>,
}

impl Job {
    fn new() -> Self {
        Self {
            stage: "Starting".to_string(),
            progress: 0.0,
            log: Vec::new(),
            done: None,
            failed: None,
        }
    }

    fn running(&self) -> bool {
        self.done.is_none() && self.failed.is_none()
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
    items: Vec<library::Item>,
    player: Option<Player>,
    /// selected local model, e.g. `qwen2.5:3b`
    llm_model: String,
    /// where ONNX Runtime was found (GPU build or system one)
    ort_dir: Option<PathBuf>,
    job: Option<Job>,
    rx: Option<Receiver<pipeline::Event>>,
    url: String,
    /// known German article (path or pasted text): when set, the job aligns it
    article: String,
    language: String,
    target: String,
    asr_model: String,
    status: Option<String>,
}

impl App {
    fn new(fonts: theme::Fonts) -> Self {
        let mut args = std::env::args().skip(1);
        let first = args.next();

        // `transcript-player <audio> [words] [sentences]` plays local files
        let (player, screen) = match first.as_deref() {
            // a URL as the first argument: ingest it right away
            Some(url) if url.starts_with("http://") || url.starts_with("https://") => {
                let mut app_state = Self::empty(fonts);
                app_state.url = url.to_string();
                app_state.start_job();
                app_state.screen = Screen::New;
                return app_state;
            }
            // a cached article folder opens in the player
            Some(path) if !path.starts_with("--") && PathBuf::from(path).is_dir() => {
                let dir = PathBuf::from(path);
                match library::read_meta(&dir)
                    .ok_or_else(|| format!("{}: no meta.json", dir.display()))
                    .and_then(|meta| {
                        Player::from_item(library::Item { dir, meta }).map(Some)
                    }) {
                    Ok(player) => (player, Screen::Player),
                    Err(err) => {
                        let mut app_state = Self::empty(fonts);
                        app_state.status = Some(err);
                        (None, Screen::Library)
                    }
                }
            }
            Some(path) if !path.starts_with("--") => {
                let audio = PathBuf::from(path);
                let words_arg = args.next().map(PathBuf::from);
                let phrases_arg = args.next().map(PathBuf::from);
                let has_words = words_arg.is_some();
                let words = words_arg.unwrap_or_else(|| audio.with_file_name(library::WORDS));
                let phrases = phrases_arg.unwrap_or_else(|| audio.with_file_name(library::PHRASES));

                match Player::new(None, audio.clone(), words, phrases, Vec::new()) {
                    Ok(player) => (Some(player), Screen::Player),
                    // a plain audio file without transcripts is transcribed first
                    Err(_) if !has_words => {
                        let mut app_state = Self::empty(fonts);
                        app_state.url = audio.to_string_lossy().to_string();
                        app_state.start_job();
                        (None, Screen::New)
                    }
                    Err(err) => {
                        let mut app_state = Self::empty(fonts);
                        app_state.status = Some(err);
                        return app_state;
                    }
                }
            }
            _ => (None, Screen::Library),
        };

        let mut app = Self::empty(fonts);
        app.player = player;
        app.screen = screen;
        app
    }

    fn empty(fonts: theme::Fonts) -> Self {
        Self {
            fonts,
            screen: Screen::Library,
            items: library::list(),
            player: None,
            llm_model: onnx_llm::DEFAULT_MODEL.to_string(),
            ort_dir: onnx_llm::prepare_runtime(),
            job: None,
            rx: None,
            url: String::new(),
            article: String::new(),
            language: "auto".to_string(),
            target: "en".to_string(),
            asr_model: asr::DEFAULT_MODEL.to_string(),
            status: None,
        }
    }

    fn refresh(&mut self) {
        self.items = library::list();
    }

    fn start_job(&mut self) {
        if self.job.as_ref().map(Job::running).unwrap_or(false) {
            return;
        }
        let source = self.url.trim().to_string();
        if source.is_empty() {
            self.status = Some("Enter an audio URL or a local file path first".to_string());
            return;
        }

        // a filled-in article switches the job to German force alignment
        let article = self.article.trim().to_string();
        let (text, text_source) = if article.is_empty() {
            (None, String::new())
        } else {
            let path = pipeline::expand_path(&article);
            if path.is_file() {
                match std::fs::read_to_string(&path) {
                    Ok(text) => (Some(text), path.display().to_string()),
                    Err(err) => {
                        self.status = Some(format!("cannot read {}: {err}", path.display()));
                        return;
                    }
                }
            } else {
                (Some(article.clone()), "pasted".to_string())
            }
        };
        let aligning = text.is_some();

        self.status = None;
        let options = pipeline::Options {
            url: source,
            text,
            text_source,
            language: if aligning {
                Some("de".to_string())
            } else {
                match self.language.as_str() {
                    "auto" => None,
                    other => Some(other.to_string()),
                }
            },
            target: self.target.clone(),
            asr_model: if aligning {
                "nemo-de".to_string()
            } else {
                self.asr_model.clone()
            },
            translate: true,
            llm_model: Some(self.llm_model.clone()),
            use_gpu: true,
        };

        let (tx, rx) = std::sync::mpsc::channel();
        pipeline::spawn(options, tx);
        self.rx = Some(rx);
        self.job = Some(Job::new());
        self.screen = Screen::New;
    }

    fn open_item(&mut self, index: usize) {
        let Some(item) = self.items.get(index).cloned() else {
            return;
        };

        match Player::from_item(item) {
            Ok(player) => {
                self.player = Some(player);
                self.screen = Screen::Player;
                self.status = None;
            }
            Err(err) => self.status = Some(err),
        }
    }

    fn delete_item(&mut self, index: usize) {
        if let Some(item) = self.items.get(index).cloned() {
            if let Err(err) = library::remove(&item) {
                self.status = Some(err);
            }
            self.refresh();
        }
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
            self.refresh();
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
                app.refresh();
            }
            if screen == Screen::Player && app.player.is_none() {
                app.status = Some("No article open".to_string());
                return Task::none();
            }
            app.screen = screen;
        }
        Message::UrlChanged(url) => app.url = url,
        Message::ArticleChanged(article) => app.article = article,
        Message::TargetChanged(target) => app.target = target,
        Message::LanguageChanged(language) => app.language = language,
        Message::AsrModelChanged(model) => app.asr_model = model,
        Message::LlmModelChanged(model) => app.llm_model = model,
        Message::StartJob => app.start_job(),
        Message::OpenItem(index) => app.open_item(index),
        Message::CopyLog => {
            if let Some(job) = &app.job {
                return iced::clipboard::write(job.text());
            }
        }
        Message::DeleteItem(index) => {
            app.delete_item(index);
            app.screen = Screen::Library;
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
                    let next = (player.current_phrase + 1).min(player.phrases.len().saturating_sub(1));
                    if let Some(phrase) = player.phrases.get(next) {
                        let start = phrase.start;
                        player.seek(start);
                    }
                }
                keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
                    let previous = player.current_phrase.saturating_sub(1);
                    if let Some(phrase) = player.phrases.get(previous) {
                        let start = phrase.start;
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
        Screen::Library => library_view(app),
        Screen::New => new_view(app),
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
        tab("Library", Screen::Library),
        tab("New", Screen::New),
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

// ------------------------------------------------------------ view: library

fn library_view(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;

    let mut cards: Vec<Element<'_, Message>> = Vec::new();
    for (index, item) in app.items.iter().enumerate() {
        cards.push(library_card(app, index, item));
    }

    let grid = if cards.is_empty() {
        Element::from(empty_state(app))
    } else {
        Row::with_children(cards)
            .spacing(12)
            .wrap()
            .vertical_spacing(12)
            .into()
    };

    let header = row![
        column![
            text("Library")
                .size(22)
                .font(fonts.display)
                .color(theme::TEXT),
            text(format!(
                "{} article{} cached in {}",
                app.items.len(),
                if app.items.len() == 1 { "" } else { "s" },
                library::root().display()
            ))
            .size(11)
            .font(fonts.mono)
            .color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        button(text("New article").size(12).font(fonts.body))
            .on_press(Message::Screen(Screen::New))
            .padding([9, 16])
            .style(theme::primary),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let status = app.status.as_ref().map(|status| {
        container(text(status.clone()).size(11).font(fonts.mono).style(theme::danger_text))
            .padding([6, 10])
            .style(theme::log_panel)
    });

    container(
        column![
            header,
            status.map(Element::from).unwrap_or_else(|| Space::new().height(0).into()),
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

fn library_card<'a>(app: &'a App, index: usize, item: &'a library::Item) -> Element<'a, Message> {
    let fonts = app.fonts;
    let meta = &item.meta;

    let title = if meta.title.is_empty() {
        meta.id.clone()
    } else {
        meta.title.clone()
    };

    let description: String = {
        let text = if meta.description.is_empty() {
            meta.error.clone().unwrap_or_default()
        } else {
            meta.description.clone()
        };
        if text.chars().count() > 170 {
            format!("{}…", text.chars().take(170).collect::<String>())
        } else {
            text
        }
    };

    let created = meta.created_text();
    let duration = if meta.duration > 0.0 {
        fmt_time(meta.duration)
    } else {
        "--:--".to_string()
    };

    let mut footer = row![
        text(format!("{} → {}", dash(&meta.language), meta.target))
            .size(10)
            .font(fonts.mono)
            .color(theme::TEXT_MUTED),
        text(format!("{duration} · {created}"))
            .size(10)
            .font(fonts.mono)
            .color(theme::TEXT_MUTED),
    ]
    .spacing(8);

    if meta.status != "done" {
        footer = footer.push(
            text(meta.status.to_uppercase())
                .size(10)
                .font(fonts.mono)
                .style(theme::danger_text),
        );
    }

    let content = column![
        text(title).size(15).font(fonts.display).color(theme::TEXT),
        text(description).size(12).font(fonts.body).color(theme::TEXT_DIM),
        Space::new().height(4),
        footer,
    ]
    .spacing(6)
    .width(Length::Fill);

    button(content)
        .on_press(Message::OpenItem(index))
        .padding(14)
        .width(Length::Fixed(330.0))
        .style(theme::card_interactive)
        .into()
}

fn empty_state(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;
    container(
        column![
            text("Nothing here yet")
                .size(18)
                .font(fonts.display)
                .color(theme::TEXT),
            text("Submit an audio URL — the app downloads it, transcribes it on the GPU,\nwrites a title, a description and a translation, and caches everything here.")
                .size(12)
                .font(fonts.body)
                .color(theme::TEXT_DIM),
            Space::new().height(6),
            text(format!("cache: {}", library::root().display()))
                .size(11)
                .font(fonts.mono)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(8),
    )
    .padding(22)
    .width(Length::Fill)
    .style(theme::log_panel)
    .into()
}

// ---------------------------------------------------------------- view: new

fn new_view(app: &App) -> Element<'_, Message> {
    let fonts = app.fonts;
    let running = app.job.as_ref().map(Job::running).unwrap_or(false);

    let form = column![
        row![
            text_input("https://example.com/episode.mp3  or  /path/to/audio.mp3", &app.url)
                .on_input(Message::UrlChanged)
                .on_submit(Message::StartJob)
                .padding(11)
                .size(13)
                .font(fonts.body)
                .style(theme::input)
                .width(Length::Fill),
            button(
                text(if running {
                    "Working…"
                } else if app.article.trim().is_empty() {
                    "Transcribe"
                } else {
                    "Align article"
                })
                    .size(13)
                    .font(fonts.body)
            )
            .on_press_maybe((!running).then_some(Message::StartJob))
            .padding([11, 18])
            .style(theme::primary),
        ]
        .spacing(8),
        text_input(
            "German article: /path/to/article.txt  or paste the text",
            &app.article,
        )
        .on_input(Message::ArticleChanged)
        .on_submit(Message::StartJob)
        .padding(11)
        .size(13)
        .font(fonts.body)
        .style(theme::input)
        .width(Length::Fill),
        row![
            field(fonts, "language", pick(app, LANGUAGES, &app.language, Message::LanguageChanged)),
            field(fonts, "translate to", pick(app, LANGUAGES, &app.target, Message::TargetChanged)),
            field(
                fonts,
                "speech model",
                pick(app, asr::MODELS, &app.asr_model, Message::AsrModelChanged),
            ),
            field(
                fonts,
                "language model",
                pick(app, onnx_llm::MODELS, &app.llm_model, Message::LlmModelChanged),
            ),
        ]
        .spacing(14),
        text("Audio: an audio URL (direct media streams in, web pages go through yt-dlp) or a local file path. Article: a German .txt path or pasted text — when set, the transcript is force-aligned to that text (nemo-de) instead of being recognized freely. Everything runs locally on the GPU through ONNX Runtime — no Python, no daemon.")
            .size(11)
            .font(fonts.body)
            .color(theme::TEXT_MUTED),
    ]
    .spacing(14);

    let job: Element<'_, Message> = match &app.job {
        Some(job) => job_card(app, job),
        None => Space::new().height(0).into(),
    };

    container(
        column![
            column![
                text("New article")
                    .size(22)
                    .font(fonts.display)
                    .color(theme::TEXT),
                text("Audio in; optionally a known German article to sync against; transcript + summary + translation out")
                    .size(12)
                    .font(fonts.body)
                    .color(theme::TEXT_MUTED),
            ]
            .spacing(2),
            form,
            job,
        ]
        .spacing(16)
        .height(Length::Fill),
    )
    .padding(18)
    .width(Length::Fill)
    .height(Length::Fill)
    .style(theme::card)
    .into()
}

fn pick(
    app: &App,
    options: &'static [&'static str],
    selected: &str,
    on_select: fn(String) -> Message,
) -> Element<'static, Message> {
    let options = options.iter().map(|option| option.to_string()).collect::<Vec<_>>();
    pick_list(options, Some(selected.to_string()), on_select)
        .text_size(12)
        .padding([7, 10])
        .font(app.fonts.body)
        .style(theme::picker)
        .into()
}

fn field(fonts: theme::Fonts, label: &str, widget: Element<'static, Message>) -> Element<'static, Message> {
    column![
        text(label.to_uppercase())
            .size(9)
            .font(fonts.display)
            .color(theme::TEXT_MUTED),
        widget,
    ]
    .spacing(4)
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
    } else if job.done.is_some() {
        text("Finished — article cached")
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

    if let Some(item) = &job.done {
        let index = app
            .items
            .iter()
            .position(|candidate| candidate.dir == item.dir)
            .unwrap_or(0);
        header = header.push(
            button(text("Open article").size(12).font(fonts.body))
                .on_press(Message::OpenItem(index))
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
        return empty_state(app);
    };
    let fonts = app.fonts;

    let (title, meta_line, index) = match &player.item {
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
            app.items.iter().position(|candidate| candidate.dir == item.dir),
        ),
        None => (
            "Local files".to_string(),
            format!("{} sentences · {} words", player.phrases.len(), player.words.len()),
            None,
        ),
    };

    let mut header = row![
        button(text("← Library").size(12).font(fonts.body))
            .on_press(Message::Screen(Screen::Library))
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

    if !player.translation.is_empty() {
        header = header.push(
            toggler(player.show_translation)
                .label("Translation")
                .on_toggle(Message::ToggleTranslation)
                .text_size(12)
                .font(fonts.body)
                .style(theme::toggle),
        );
    }

    if let Some(index) = index {
        header = header.push(
            button(text("Delete").size(11).font(fonts.body))
                .on_press(Message::DeleteItem(index))
                .padding([7, 11])
                .style(theme::ghost),
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

    container(
        column![
            header,
            description,
            transport(player, fonts),
            now_playing(player, fonts),
            transcript(player, fonts),
            footer(player, fonts),
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

/// The current sentence, word by word.
fn now_playing(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {
    let phrase = player.phrases.get(player.current_phrase);

    let mut cells: Vec<Element<'_, Message>> = Vec::new();
    if let Some(phrase) = phrase {
        for index in phrase.words.clone() {
            let Some(word) = player.words.get(index) else {
                continue;
            };
            let label = word.text.trim();
            if label.is_empty() {
                continue;
            }

            let active = index == player.current_word;
            let color = if active {
                theme::WHITE
            } else if index < player.current_word {
                theme::TEXT_MUTED
            } else {
                theme::TEXT
            };

            cells.push(
                button(text(label).size(20).font(fonts.body).color(color))
                    .on_press(Message::SeekTo(word.start))
                    .padding([2, 5])
                    .style(move |theme, status| theme::word(theme, status, active))
                    .into(),
            );
        }
    }

    let words = Row::with_children(cells)
        .spacing(3)
        .wrap()
        .vertical_spacing(4);

    let current_translation = player
        .translation_of(player.current_phrase)
        .filter(|_| player.show_translation)
        .map(|line| {
            text(line.to_string())
                .size(13)
                .font(fonts.body)
                .color(theme::TEXT_DIM)
        });

    let mut content = column![
        row![
            text("NOW PLAYING")
                .size(10)
                .font(fonts.display)
                .color(theme::TEXT_MUTED),
            container(
                text(format!("{} / {}", player.current_phrase + 1, player.phrases.len()))
                    .size(10)
                    .font(fonts.mono)
                    .color(theme::TEXT_DIM),
            )
            .padding([3, 9])
            .style(theme::pill),
            Space::new().width(Length::Fill),
            text(
                phrase
                    .map(|phrase| format!("{} – {}", fmt_time(phrase.start), fmt_time(phrase.end)))
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

    let (first, last) = if player.follow {
        let first = player.current_phrase.saturating_sub(3);
        (first, (first + 7).min(player.phrases.len()))
    } else {
        (0, player.phrases.len())
    };

    let mut rows = column![].spacing(2).width(Length::Fill);

    for index in first..last {
        let phrase = &player.phrases[index];
        let active = index == player.current_phrase;
        let spoken = index < player.current_phrase;

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
            text(format!("{} – {}", fmt_time(phrase.start), fmt_time(phrase.end)))
                .size(11)
                .font(fonts.mono)
                .color(if active { theme::ACCENT_HI } else { theme::TEXT_MUTED })
                .width(Length::Fixed(96.0))
                .into(),
        );

        let mut lines = column![text(phrase.text.as_str())
            .size(15)
            .font(fonts.body)
            .color(if active {
                theme::TEXT
            } else if spoken {
                theme::TEXT_MUTED
            } else {
                theme::TEXT_DIM
            })]
        .spacing(2)
        .width(Length::Fill);

        if player.show_translation {
            if let Some(line) = player.translation_of(index) {
                lines = lines.push(
                    text(line)
                        .size(13)
                        .font(fonts.body)
                        .color(if active {
                            theme::ACCENT_HI
                        } else {
                            theme::TEXT_MUTED
                        }),
                );
            }
        }

        cells.push(lines.into());

        rows = rows.push(
            button(Row::with_children(cells).spacing(12).align_y(Alignment::Center))
                .on_press(Message::SeekTo(phrase.start))
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

fn footer(player: &Player, fonts: theme::Fonts) -> Element<'_, Message> {
    let hint = if player.audio.failed() {
        "Audio device unavailable — transcript only"
    } else {
        "Space play/pause · ←/→ 5 s · ↑/↓ sentence · click a word or sentence to jump"
    };

    row![
        text(hint).size(11).font(fonts.body).color(theme::TEXT_MUTED),
        Space::new().width(Length::Fill),
        text(format!(
            "{} sentences · {} words",
            player.phrases.len(),
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
        });
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

/// `transcript-player --ingest <url|file>`: run the pipeline without the UI.
/// `transcript-player --asr-probe <model> <audio>`: measure one ASR model.
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

/// `transcript-player --align <audio> <article.txt>`: force-align German text.
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
            pipeline::Event::Failed(error) => {
                eprintln!("== failed: {error}");
                return 1;
            }
        }
    }

    0
}

/// `transcript-player --transcribe-tree <folder>`: walks a folder tree and, for
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
            pipeline::Event::Failed(error) => return Err(error),
        }
    }

    Err("pipeline ended without a result".to_string())
}

/// Copies the transcript files (and nothing else) into the `transcribe/` folder.
fn publish(item_dir: &Path, target: &Path) -> Result<(), String> {
    std::fs::create_dir_all(target).map_err(|err| err.to_string())?;

    for name in [library::WORDS, library::PHRASES, library::TRANSLATION] {
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

    let mut args = std::env::args().skip(1);
    if let Some(first) = args.next() {
        if first == "--asr-probe" {
            let model = args.next().unwrap_or_else(|| asr::DEFAULT_MODEL.to_string());
            let audio = args.next().unwrap_or_default();
            std::process::exit(asr_probe(&model, &audio));
        }
        if first == "--ingest" {
            let url = args.next().unwrap_or_default();
            let model = args.next();
            std::process::exit(headless(&url, model.as_deref()));
        }
        if first == "--align" {
            let audio = args.next().unwrap_or_default();
            let article = args.next().unwrap_or_default();
            std::process::exit(headless_align(&audio, &article));
        }
        if first == "--transcribe-tree" {
            let root = args.next().unwrap_or_default();
            std::process::exit(transcribe_tree(&root));
        }
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
        for name in ["transcription.json", "transcript.json", "translation.json", "meta.json"] {
            std::fs::write(item.join(name), b"{}").unwrap();
        }

        let target = root.join("transcribe");
        publish(&item, &target).unwrap();

        assert!(target.join("transcription.json").is_file());
        assert!(target.join("transcript.json").is_file());
        assert!(target.join("translation.json").is_file());
        assert!(!target.join("meta.json").exists());

        let _ = std::fs::remove_dir_all(&root);
    }
}

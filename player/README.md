# Transcribe

Rust (iced) desktop app that turns an audio URL or a local audio file into a cached article:

**submit a URL or a file path → download/copy → GPU transcription → title + description + translation → player.**

```
┌ Library ─┬─ New ─┐
│ cards of every cached article, click to open     │
└──────────────────────────────────────────────────┘
```

## What it does

* **Library** – a card per cached article (title, description, `de → en`, duration,
  date, status). Click a card to open it in the player; open an article to delete it.
* **New** – paste an audio URL, a web page URL, or a local file path (`/home/me/talk.mp3`, `~/talk.mp3`, `file:///…`). Optionally add a **German article** (a `.txt`/`.md` path or pasted text; YAML frontmatter and markdown scaffolding are stripped): the article then becomes the transcript and the audio is *force-aligned* to it, so the player follows the written text exactly. Press *Transcribe* (or *Align article*). The job
  card shows the stage, a progress bar and a live log. When it finishes, *Open article*
  jumps straight into the player.
* **Player** – plays the audio and keeps the transcript in sync: the current word is
  highlighted in the now-playing panel, the current sentence is highlighted in the
  transcript (with its translation underneath when available), plus play/pause, ±5 s,
  seek bar, volume, and a translation toggle. Keys: `Space`, `←/→` ±5 s,
  `↑/↓` previous/next sentence, `Home`.

## Pipeline

1. **Ingest** – a local file path is copied into the article folder; direct media URLs stream with `ureq` (progress reported); anything else is handed to `yt-dlp` (`-x --audio-format mp3`).
2. **Keep the audio as it came** – no transcoding: the file is stored as
   `audio.<ext>` and decoded on demand by symphonia (recognition) and rodio
   (playback); the sample rate is taken from the packet timeline, because
   containers lie about it (`HE-AAC`/mp4).
3. **Transcribe on the GPU (Whisper, in Rust)** – `sherpa-onnx` (the prebuilt CUDA build) decodes the
   file with symphonia, finds the speech segments with Silero VAD, recognizes each
   segment and turns token timestamps into word timings. Every segment is
   recognized with 0.35 s of padding on both sides, otherwise a hard cut at a VAD
   boundary truncates the word that straddles it.
   Models (all Whisper-family ONNX, no Python):
   * **`whisper-turbo`** (default, 564 MB) – multilingual, best on names and
     borrowed words. Its decoder has no cross-attention outputs, so word timings
     are interpolated inside each segment.
   * `whisper-large-v3` (3.1 GB) – same, larger.
   * `canary-180m` (154 MB) – en/es/de/fr, token timestamps.
   * `nemo-de` (449 MB) – NVIDIA's German FastConformer transducer: fastest of all
     (~37x realtime) and it *does* emit token timestamps, so the word highlight is
     exact; German only.
4. **Align to a known German article** – when the *New* screen carries an article,
   the recognizer is fixed to `nemo-de` and used only as a clock: the article is
   tokenized and matched to the recognized words with a global (Needleman–Wunsch)
   alignment (`src/align.rs`). The article's spelling and sentence boundaries win;
   words the recognizer missed are interpolated between their neighbours, so the
   player highlights the written text rather than a re-recognition of it.
5. **Small model, in process** – Qwen2.5 as an ONNX decoder through ONNX Runtime
   (`ort`), with its own KV-cache loop: headline + summary in the language of the
   audio, then a sentence-wise translation. Same ONNX Runtime as the recognizer
   (the one bundled with sherpa-onnx), no daemon.
6. **Cache** – everything is written to disk and re-used; nothing is downloaded twice.

## Cache layout

`~/transcribe-library/<timestamp>-<slug>/` (override with `TRANSCRIBE_LIBRARY`):

| file | content |
| --- | --- |
| `meta.json` | title, description, source URL or local path, aligned article path, language, target, duration, model, device, status, timestamps |
| `audio.<ext>` | the audio, exactly as downloaded or copied (name recorded in `meta.json`) |
| `transcription.json` | word-level segments (`start`/`end`/`text`, voxtral-style schema) |
| `transcript.json` | sentence-level segments (used by the player) |
| `translation.json` | `{"target": "en", "sentences": [...]}` |

The folder is renamed after the generated title once the job succeeds; a failed job
keeps its `meta.json` with `"status": "failed"` and the error message, and shows up in
the library with a red badge.

## Batch: `--transcribe-tree`

The crawler keeps one folder per German article:

```text
articles-ihre-artikel-new/<date>_<slug>/
  article.md                 # the authoritative text (YAML frontmatter)
  article.json
  audio/<slug>.mp3           # the auto-voiced audio
  images/
```

`--transcribe-tree <folder>` walks `<folder>` recursively, finds every `*.mp3`, and
for each one whose `<article>/transcribe/` folder is **missing** it runs the German
pipeline — force-aligned to `<article>/article.md` (`../article.md` next to the
`audio/` folder) — and writes only the text into the new folder:

```text
articles-ihre-artikel-new/<date>_<slug>/
  transcribe/
    transcription.json       # word-level timings, article text
    transcript.json          # sentence-level timings
    translation.json         # target-language sentences
```

* a folder that already has `transcribe/` is skipped, so re-running only fills gaps;
* the audio and `meta.json` are **not** copied — the temporary library copy
  (including the audio) is removed after the files are written;
* if `article.md` is missing, the audio is transcribed without alignment.

## Build & run

```bash
cd player
cargo run --release                      # opens the library
cargo run --release -- <url|file>        # ingest right away (GUI, progress)
cargo run --release -- --ingest <url|file>   # headless ingest: logs + exit code (CI friendly)
cargo run --release -- --align <audio> <article.txt>   # force-align German text (headless)
cargo run --release -- --transcribe-tree <folder>   # batch: every audio/*.mp3 -> <article>/transcribe/
cargo run --release -- <audio> [words.json] [sentences.json]   # play; transcribes first if needed
cargo run --release -- <article-folder>  # play a cached article
```

### System dependencies

| for | packages |
| --- | --- |
| GPU execution (ONNX Runtime CUDA provider) | `cuda-cudart-12-9`, `libcublas-12-9`, `libcudnn9-cuda-12` |
| extracting the model archives | `tar` (base system) |
| downloading web pages | `yt-dlp` (only for non-media URLs, e.g. podcast pages) |

```bash
sudo zypper install cuda-cudart-12-9 libcublas-12-9 libcudnn9-cuda-12
```

**No Python, no CUDA toolkit, no compiler, no CMake.** Both native libraries are
prebuilt: `sherpa-onnx` is fetched by its crate (or from `SHERPA_ONNX_LIB_DIR`),
and the CUDA-enabled `libonnxruntime.so` it ships is used by the recognizer *and*
the language model. Model weights are downloaded on demand into
`~/transcribe-models/{sherpa,onnx}`.

Environment variables:

| variable | meaning |
| --- | --- |
| `TRANSCRIBE_LIBRARY` | cache directory (default `~/transcribe-library`) |
| `TRANSCRIBE_MODELS` | where the models are cached (`~/transcribe-models`) |
| `TRANSCRIBE_ORT_DIR` | directory holding the CUDA build of `libonnxruntime.so` |
| `TRANSCRIBE_CUDA_LIBS` | extra directories with `libcublas`/`libcudnn`/`libcudart` |
| `AUTOPLAY=1`, `START_AT=<s>` | player: start playing / start at a position |
| `ON_TOP=1` | keep the window above others |

## Notes

* On a 4 GB GPU the language model and Whisper do not fit at the same time. The
  pipeline waits until the transcription has released the GPU before loading the
  language model, and the int4 export is used on small cards (fp16 above 6 GB).
  If the GPU still cannot fit the model, the session falls back to the CPU.
* The defaults are `whisper-turbo` for speech and `qwen2.5:1.5b` for the language
  model: both run on a 4 GB GPU. `qwen2.5:3b` has no fp16 export that fits and
  its int4 export ships as a 3.2 GB external-data file, so it ends up on the CPU.
* Speech models come from the sherpa-onnx `asr-models` releases: `nemo-de` (default),
  `whisper-turbo`, `whisper-large-v3` and `canary-180m` (en/es/de/fr). Pick one in the
  *New* screen; it is downloaded on first use.
* On a 4 GB GPU both models fit (nemo-de ~0.6 GB, Qwen2.5 1.5b int4 ~1.2 GB), but not
  at the same time: the pipeline waits for the recognizer to release the GPU before
  loading the language model.

## Layout

| file | role |
| --- | --- |
| `src/main.rs` | app state, messages, the three screens |
| `src/audio.rs` | rodio playback on a dedicated thread (stream is `!Send`) |
| `src/library.rs` | cache directory, `meta.json`, slugs |
| `src/asr.rs` | sherpa-onnx recognizer + Silero VAD + word timings |
| `src/align.rs` | force-aligns a known German article to ASR word timings |
| `src/pipeline.rs` | download → normalize → transcribe → enrich, as `Event`s |
| `src/theme.rs` | palette, system fonts, all widget styles |


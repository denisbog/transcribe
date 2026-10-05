# Transcribe

Rust (iced) desktop app that turns a folder of crawler articles (`article.json` +
audio) into timestamped transcripts, sentence/paragraph translations, vocabulary
and word pairs:

**browse a folder → see the articles → open one → Run pipeline → Play.**

```
┌ Articles ────────────────────────────────────────┐
│ cards of every article found under the folder     │
└───────────────────────────────────────────────────┘
```

## What it does

* **Articles** – on start the app loads the folder from the last session
  (`~/.config/transcribe/config.toml`, e.g. `~/llm/crawler/articles-ihre-artikel`)
  and shows a card per `article.json` found below it, with the state of
  `transcribe/` (transcription, translation, vocabulary, pairs). **Browse folder…**
  opens an in-app picker; the choice is written back to `config.toml`.
* **Article** – open a card to see its status and two actions: **Run pipeline**
  (transcribes the audio named by `article.json`, then translates sentence by
  sentence, then builds vocabulary + pairs, streaming stage, progress and log) and
  **Play**, which enables as soon as `transcribe/transcription.json` exists.
* **Player** – plays the audio and keeps the transcript in sync: the current word is
  highlighted in the now-playing panel, the current paragraph is highlighted in the
  transcript (with its translation underneath when available), plus play/pause, ±5 s,
  seek bar, volume, and a translation toggle; a vocabulary toggle shows the
  word pairs worth learning (false friends, non-transparent words); when `pairs.json`
  exists, the paired German words and their English translation are highlighted in the
  now-playing panel and the transcript. Keys: `Space`, `←/→` ±5 s,
  `↑/↓` previous/next paragraph, `Home`.

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
   (`ort`), with its own KV-cache loop: `qwen2.5:1.5b` on the GPU writes the headline
   + summary in the language of the audio and the sentence-wise translation; then
   `qwen2.5:3b` on the **CPU** runs the **vocabulary pass** that picks
   what a reader actually has to learn — false friends and words whose meaning
   cannot be guessed from the English form — and skips names, countries and
   internationalisms that look the same. The German sentence and its stored translation
   are shown to the model, and both sides of every pair are snapped back to the exact
   wording of that sentence, so the index pass can always locate them; the same pass
   writes `pairs.json`, which places every pair in its sentence and records the word
   indexes on both sides. Same ONNX Runtime as the recognizer (the one bundled with
   sherpa-onnx), no daemon.
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
| `vocabulary.json` | the article's word pairs: `{"target": "en", "pairs": [{"de": "...", "en": "..."}]}` |
| `pairs.json` | word pairings grouped like `article.json` blocks: `{"blocks": [{"index": 1, "kind": "para", "pairs": [{"sentence": 0, "de": "...", "en": "...", "source": [4], "target": [5]}]}]}` |

The folder is renamed after the generated title once the job succeeds; a failed job
keeps its `meta.json` with `"status": "failed"` and the error message, and shows up in
the library with a red badge.

## Batch: `transcribe-tree`

The crawler keeps one folder per German article:

```text
articles-ihre-artikel-new/<date>_<slug>/
  article.md                 # the authoritative text (YAML frontmatter)
  article.json
  audio/<slug>.mp3           # the auto-voiced audio
  images/
```

`transcript-player transcribe-tree <folder>` walks `<folder>` recursively, finds every `*.mp3`, and
for each one whose `<article>/transcribe/` folder is **missing** it runs the German
pipeline — force-aligned to `<article>/article.md` (`../article.md` next to the
`audio/` folder) — and writes only the text into the new folder:

```text
articles-ihre-artikel-new/<date>_<slug>/
  transcribe/
    transcription.json       # word-level timings, article text
    transcript.json          # sentence-level timings
    translation.json         # target-language sentences
    vocabulary.json           # German → English word pairs
    pairs.json                # vocabulary pairs with word indexes
```

* a folder that already has `transcribe/` is skipped, so re-running only fills gaps;
* the audio and `meta.json` are **not** copied — the temporary library copy
  (including the audio) is removed after the files are written;
* if `article.md` is missing, the audio is transcribed without alignment.


## Batch: `articles` (article.json driven)

`transcript-player articles <folder> [--force]` walks the crawler tree and, for every folder that
holds an `article.json`, transcribes the audio named by `audio.file` and writes the
results **into that article**:

```text
articles-ihre-artikel/<date>_<slug>/
  article.json               # source of truth (blocks[], audio.file, metadata)
  audio/<slug>.mp3
  transcribe/
    transcription.json       # words (+ sentence/block indexes) and sentence timings
    translation.json         # sentence-level translation, aligned to article.json
    vocabulary.json          # German → English word pairs
    pairs.json               # vocabulary pairs with word indexes
```

* the spoken text is rebuilt from `article.json` (title, kicker, description,
  byline, listen link, image captions, then `blocks[]`), so the audio is
  force-aligned to the article's exact wording; `nemo-de` supplies the clock;
* `translation.json.blocks` mirrors `article.json.blocks` one-to-one with the
  `image` blocks skipped: every text block appears once, in order, with its own
  `source` sentences, the translated `sentences` and the joined `translation`;
  `index` is the original `article.json` `blocks[]` index, so a paragraph can be
  patched 1:1:

  ```json
  {
    "target": "en", "language": "de", "model": "qwen2.5:1.5b on cuda (int4)",
    "source":    ["Der AfD-Verteidigungspolitiker …", "…"],
    "sentences": ["The AfD defense politician …", "…"],
    "blocks": [
      {
        "index": 1, "kind": "para",
        "source":      ["Der AfD-Verteidigungspolitiker …", "…"],
        "sentences":   ["The AfD defense politician …", "…"],
        "translation": "The AfD defense politician … …"
      }
    ]
  }
  ```
* `pairs.json.blocks` mirrors the same blocks (same `index`/`kind`, images
  skipped), each holding only its own `pairs`; a pair's `sentence` indexes the
  block's `source`/`sentences`, so original, translation and pairs align block by
  block.

* translation runs sentence by sentence, so the graph never allocates a logits
  buffer for a whole article (no OOM on a small card);
* a folder that already has `transcribe/translation.json` is skipped unless
  `--force` is given;
* the obsolete `transcript.json` is removed once `transcription.json` is written.
## External pairs via `pi`

The word pairs can come from a stronger external model instead of the local
`qwen2.5:3b`. The pipeline shells out to **`pi`** in print mode
(`pi --print --no-tools --thinking off --provider … --model …`, prompt on stdin, no tools), asks
for the same `de = en` list and snaps it into the blocks.

```bash
# article tree, external pairs (defaults: deepinfra / deepseek-ai/DeepSeek-V4.1-Flash)
cargo run --release -- articles <folder> --pairs-model deepinfra/deepseek-ai/DeepSeek-V4.1-Flash

# one article: re-run only the vocabulary + pairs pass
cargo run --release -- vocabulary <article-folder> --pairs-model deepinfra/deepseek-ai/DeepSeek-V4.1-Flash

# a tree of already transcribed folders, external pairs + full refresh
cargo run --release -- pairs <folder> --pairs-model deepinfra/deepseek-ai/DeepSeek-V4.1-Flash --force --jobs 4
```

* `--external-pairs` enables the external model with the defaults; or pass
  `--pairs-provider <name>` and/or `--pairs-model <id>` to choose another one.
  `--pairs-model` also accepts the provider as a prefix, e.g.
  `--pairs-model deepinfra/deepseek-ai/DeepSeek-V4.1-Flash`. Without any of
  them the local CPU model is used.
* `--force` re-extracts the vocabulary and overwrites both `vocabulary.json` and
  `pairs.json`; without it an existing `pairs.json` is skipped and an existing
  `vocabulary.json` is reused.
* `pairs --jobs N` processes `N` folders at once when an external model is
  used (each folder gets its own `pi` process); the default is `1`, and the flag
  is ignored for the local model, which runs one folder at a time.
* In the **Article** screen, the **Pairs via pi (…)** toggle does the same for the
  **Run pipeline** button.
* If `pi` fails or returns no pairs, the run logs the reason and falls back to the
  local model, so `vocabulary.json`/`pairs.json` are still written.
* `model` in `pairs.json` records the provider/model that produced them.
* Environment: `TRANSCRIBE_PAIRS_PROVIDER`, `TRANSCRIBE_PAIRS_MODEL` (defaults),
  and `TRANSCRIBE_PI` (path to the `pi` binary; default `pi` from `PATH`).

## Build & run

```bash
cd player
cargo run --release                      # opens the library
cargo run --release -- <url|file>        # ingest right away (GUI, progress)
cargo run --release -- ingest <url|file> [model]   # headless ingest: logs + exit code (CI friendly)
cargo run --release -- align <audio> <article.txt>   # force-align German text (headless)
cargo run --release -- transcribe-tree <folder>   # batch: every audio/*.mp3 -> <article>/transcribe/
cargo run --release -- articles <articles-folder> [--force] [--pairs-provider P] [--pairs-model M]   # article.json tree -> <article>/transcribe/
cargo run --release -- vocabulary <article-folder> [model] [--force] [--external-pairs]   # re-run only the vocabulary + pairs pass
cargo run --release -- pairs <folder> [model] [--force] [--jobs N] [--pairs-provider P] [--pairs-model M]   # batch: transcription tree -> vocabulary.json + pairs.json
cargo run --release -- <audio> [words.json] [sentences.json]   # play; transcribes first if needed
cargo run --release -- <article-folder>  # play a cached article or a crawler article folder
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
* The defaults are `whisper-turbo` for speech and `qwen2.5:1.5b` (int4) for the
  headline and the translation: those run on the GPU. The **vocabulary/pair pass**
  uses `qwen2.5:3b` (int4) on the **CPU**, so the GPU stays with the recognizer and
  the small model. The 3B export has no `position_ids` input; the decoder feeds it
  only when the graph declares it.
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


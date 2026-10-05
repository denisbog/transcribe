# Instruction template: correct a transcription against the original article

Reusable task description for turning a raw ASR transcript of an auto-voiced
article into an article-faithful transcript (text + timings + translation).

---

## Prompt template

> The timestamped transcription of an auto-voiced article is in
> `~/transcribe-library/<ARTICLE_ID>/transcript.json`.
> The original article text is `<PATH_TO_ARTICLE_MD>` (markdown with YAML frontmatter).
> Use the original article text to correct the transcription so its text matches the
> article, keeping the audio timings meaningful.
> Update the cache files as required in `~/transcribe-library/<ARTICLE_ID>/`.

## Inputs

| input | meaning |
| --- | --- |
| `~/transcribe-library/<ID>/transcript.json` | sentence-level ASR segments `{start,end,text}` |
| `~/transcribe-library/<ID>/transcription.json` | word-level ASR segments `{start,end,text,speaker_id,type}` |
| `~/transcribe-library/<ID>/translation.json` | `{target, sentences: [...]}` aligned to `transcript.json` |
| `~/transcribe-library/<ID>/meta.json` | title, description, counts, duration, model |
| `<ARTICLE>.md` | the authoritative article text (markdown + YAML frontmatter) |

## Outputs (all in `~/transcribe-library/<ID>/`)

| file | content |
| --- | --- |
| `transcript.json` | article sentences with timings: `{file, model, language, duration, segments:[{start,end,text}]}` |
| `transcription.json` | article words with timings: `{language, model, segments:[{start,end,speaker_id,text,type}], text, type, usage}` |
| `translation.json` | `{"target":"en","sentences":[...]}` aligned 1:1 with `transcript.json` segments |
| `meta.json` | refresh `description`, `words`, `sentences` counts |
| `vocabulary.json` | `{"target":"en","pairs":[{"de":..., "en":...}]}` — the article's most relevant German→English word pairs (optional pass) |
| `pairs.json` | `{"pairs":[{"sentence":n,"de":...,"en":...,"source":[i...],"target":[j...]}]}` — each vocabulary pair placed in its sentence, with the word indexes on both sides |

Keep the original `model`, `language`, `duration`, `usage` and the
`transcription_segment` / `transcription.done` type strings.

## Procedure

1. **Extract the spoken text** from the article markdown:
   - drop YAML frontmatter; drop markdown that is not read aloud:
     image lines `![...](...)`, italic captions `*...*` (incl. `*DER SPIEGEL*`),
     `<sub>…</sub>`, the byline line `von … · … · …`, the `🎧` listen link,
     and in-article ad/teaser blocks (e.g. "Wo beginnt Autismus?", "Lesen Sie unsere
     Titelgeschichte …").
   - keep: H1 (only the part inside `»…«` if that is all the TTS reads), the
     standfirst `>` blockquote, `###` subheads, and body paragraphs.
   - verify coverage against the ASR by aligning (step 3); drop/flag anything that
     does not align, and look for spoken material the ASR missed.
2. **Split into sentences.** Paragraph-isolated. Break after `[.!?]` optionally
   followed by `»«`, then whitespace, except:
   - next char is `– - — , ; :` → no break;
   - `<digit>. <Month>` (e.g. `7. Oktober`) → no break;
   - next word is lowercase → no break (initials like `Mohammed A.`, `Abed al G.`,
     `Borhan El-K.`, `Yousif C.`), **except** a short whitelist of real sentence
     starts (e.g. `Mohammed A. sitzt`, `Borhan El-K. lebt`, `El-K. tritt`) → break.
3. **Align reference words to ASR words** with `difflib.SequenceMatcher`.
   Normalize for matching: lowercase, `ß→ss`, strip edge punctuation, map German
   number words (`vier→4`, `drei→3`, …), `millimeter→mm`.
4. **Assign timings:** matched reference words take their ASR word times; unmatched
   words are linearly interpolated inside the surrounding gap (proportional to word
   count), with leading/trailing gaps clamped to the nearest anchor.
5. **Enforce monotonicity** at word level (`start[i] = max(start[i], end[i-1])`).
6. **Recover missing audio.** If the ASR stopped early but the file still has speech,
   find it (e.g. ffmpeg + per-0.1 s RMS) and distribute its span over the missing
   sentences. Short pauses (`RMS ≈ 0`) mark sentence boundaries.
7. **Build segments:** sentence timing = first-word start … last-word end.
   Keep non-article audio lines that the ASR did catch (e.g. the TTS intro
   "Automatisch vertonter Artikel") as their own segments, in order.
8. **Translate** the corrected sentences to the target language, one string per
   segment (same count/order as `transcript.json`).
9. **Update `meta.json`:** `description` = article standfirst, `words` = word count,
   `sentences` = segment count.
10. **Back up** the originals first (e.g. `/tmp/…/orig_*.json`).

## Verification (must pass)

- segment texts == `[title, tts-intro?] + article sentences` exactly;
- word tokens == whitespace-split segment texts;
- `transcript.json`, `transcription.json`, `translation.json` have matching counts;
- word and sentence start/end are monotonic;
- simulate the player's word→sentence grouping
  (`while words[next].start < phrase.end - 0.08`) and assert every sentence gets
  exactly its own word count → guarantees the word highlight stays in sync.

## Gotchas

- The player maps words to sentences **by time** (`group()` in `src/main.rs`), not by
  text; every word of a sentence must have `start < sentence.end - 0.08`.
- `transcription.json.text` is the concatenation of word texts **without spaces** —
  replicate that convention.
- `translation.json.sentences[i]` pairs with `transcript.json.segments[i]`; any
  re-segmentation requires retranslating.
- Article typos are part of "match the original article" unless told otherwise.
- ASR commonly mangles names (`Abed Alge/Al-Gey` → `Abed al G.`, `Bohan LK` →
  `Borhan El-K.`, `Basim Naim` → `Basem Naim`, `Josef C.` → `Yousif C.`) and may
  split mid-sentence or drop the audio tail entirely.

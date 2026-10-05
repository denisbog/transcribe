# Pairing evaluation with an external model

Reproducible way to build `vocabulary.json` + `pairs.json` for one article with an
external model (through `pi`) and to check/compare the result with the app's own
`qwen2.5:3b`-on-CPU pass.

Files:

| file | role |
| --- | --- |
| `pairing-eval-prompt.md` | the prompt to give the external model (has `<ARTICLE_DIR>`, `<OUT_DIR>`, `<model name>` placeholders) |
| `pairing-eval-check.py` | validates and compares the produced files against `transcript.json` / `translation.json` |

## 0. Pick the article

The article folder must contain `transcript.json` (German, sentence segments) and
`translation.json` (`{"target","sentences":[...]}` aligned 1:1):

```bash
ARTICLE=~/transcribe-library/20261003-1639-netanyahu-in-emiraten-vorwarnungen-gegen-hamas
OUT=/tmp/ext-vocab
mkdir -p "$OUT"
```

## 1. Run an external model with pi

Fill the placeholders into a copy of the prompt, then hand it to `pi` in
non-interactive mode:

```bash
cd /home/denis/llm/transcribe/player

sed -e "s|<ARTICLE_DIR>|$ARTICLE|g" \
    -e "s|<OUT_DIR>|$OUT|g" \
    -e "s|<model name>|deepseek-flash|g" \
    pairing-eval-prompt.md > /tmp/pairing-prompt.txt

# any pi provider/model works; the command that was tested:
timeout 1500 pi \
  --provider deepseek --model deepseek-flash \
  --print --no-session -nc -na \
  "$(cat /tmp/pairing-prompt.txt)"
```

Flag notes: `--print` is non-interactive, `--no-session` keeps no session,
`-nc` ignores `AGENTS.md`/`CLAUDE.md`, `-na` ignores project-local trust files.

When it finishes, `$OUT/vocabulary.json` and `$OUT/pairs.json` exist. It should
print a one-line summary with the pair counts.

## 2. Build the local baseline (for comparison)

The app's pass writes into the article folder itself, so copy the result aside:

```bash
cd /home/denis/llm/transcribe/player
cargo build --release
./target/release/transcript-player vocabulary "$ARTICLE"   # qwen2.5:3b on CPU, ~2 min

mkdir -p /tmp/app-vocab
cp "$ARTICLE"/vocabulary.json "$ARTICLE"/pairs.json /tmp/app-vocab/
```

## 3. Check the results

```bash
cd /home/denis/llm/transcribe/player
python3 pairing-eval-check.py "$ARTICLE" "$OUT" /tmp/app-vocab
```

It prints one line per run:

```
ext-vocab                vocab= 43  pairs= 43  invalid=0
app-vocab                vocab= 19  pairs= 17  invalid=0

overlap (exact de=en pairs):
  ext-vocab ∩ app-vocab: 7
```

and, for every bad entry, an `INVALID ...` line.

### What counts as invalid

For each entry of `pairs.json` the checker requires:

- `sentence` is a valid index into `transcript.json` segments (and `translation.json`);
- `source` and `target` are contiguous, in-range index runs;
- `de` reconstructs exactly the sentence words at `source` (case/punctuation-insensitive,
  so only edge punctuation such as `»groß` → `groß` may differ);
- `en` reconstructs exactly the translation words at `target`, and occurs in that
  translation sentence.

`invalid=0` means every pairing is verbatim and its indexes are correct. Compare
`vocab`/`pairs` counts and the overlap to judge a model: more entries at `invalid=0`
is better; entries with wrong indexes are worse than missing ones.

### See the entries themselves

```bash
```bash
# $OUT is expanded by the shell because the heredoc delimiter is unquoted
python3 - <<PY
import json
for x in json.load(open("$OUT/vocabulary.json"))["pairs"]:
    print(f'{x["de"]!r} = {x["en"]!r}')
PY
```

## Reference results (this article, `invalid=0` for all)

| run | device | vocabulary | placed |
| --- | --- | --- | --- |
| `qwen2.5:1.5b` int4 (app default, headline/translation only) | GPU | 9 | 8 |
| `qwen2.5:1.5b` int4 on CPU | CPU | 8 | 7 |
| `qwen2.5:1.5b` fp16 | GPU OOM / CPU degraded | 2 | 2 |
| **`qwen2.5:3b` int4 (app vocabulary pass)** | **CPU** | **19** | **17** |
| `deepseek-flash` | remote | 43 | 43 |

The reference article is
`~/transcribe-library/20261003-1639-netanyahu-in-emiraten-vorwarnungen-gegen-hamas`.

## Caveat

Segments and translations must stay aligned 1:1. Some articles have misaligned
segments (here 14–17); the same-sentence rule then drops pairs it cannot place,
which is correct — do not "fix" it by matching across sentences.

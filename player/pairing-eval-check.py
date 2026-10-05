#!/usr/bin/env python3
"""Validate and compare vocabulary/pairing output against transcript.json.

Usage:
    python3 pairing-eval-check.py <article-folder> <run-folder> [<run-folder> ...]

An "article folder" holds transcript.json (German) and translation.json
({"target","sentences":[...]}) aligned 1:1.
A "run folder" holds vocabulary.json and/or pairs.json produced by a pass
(an external model via pi, or the app itself).

For every run it checks, for each entry of pairs.json:
  * `sentence` is a valid segment index;
  * `source` / `target` are contiguous, in-range index runs;
  * the quoted `de`/`en` reconstruct the exact sentence words (case and
    punctuation-insensitively, so only edge punctuation may differ).
Then it prints the vocabulary sizes and the overlap between runs.
"""

import json
import os
import sys


def norm(text):
    return "".join(c.lower() for c in str(text) if c.isalnum())


def words(text):
    return str(text).split()


def contiguous(indexes):
    return bool(indexes) and indexes == list(range(indexes[0], indexes[0] + len(indexes)))


def load(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate(article, run_dir, label):
    transcript = load(os.path.join(article, "transcript.json"), {})
    translation = load(os.path.join(article, "translation.json"), {})
    segments = transcript.get("segments", [])
    sentences = translation.get("sentences", [])

    pairs_doc = load(os.path.join(run_dir, "pairs.json"))
    vocab_doc = load(os.path.join(run_dir, "vocabulary.json"))
    pairs = (pairs_doc or {}).get("pairs", [])
    vocab = (vocab_doc or {}).get("pairs", [])

    invalid = 0
    for entry in pairs:
        index = entry.get("sentence", -1)
        source, target = entry.get("source", []), entry.get("target", [])
        ok = 0 <= index < len(segments) and index < len(sentences)
        if ok:
            de_words, en_words = words(segments[index]["text"]), words(sentences[index])
            ok = (
                contiguous(source)
                and contiguous(target)
                and max(source) < len(de_words)
                and max(target) < len(en_words)
                and norm(entry.get("de", "")) == norm(" ".join(de_words[i] for i in source))
                and norm(entry.get("en", "")) == norm(" ".join(en_words[i] for i in target))
                and norm(entry.get("en", "")) in norm(sentences[index])
            )
        if not ok:
            invalid += 1
            print(f"  INVALID in {label}: {entry}")

    print(f"{label:<24} vocab={len(vocab):>3}  pairs={len(pairs):>3}  invalid={invalid}")
    return {
        "vocab_de": {entry["de"].lower() for entry in vocab},
        "vocab_pairs": {(entry["de"].lower(), entry["en"].lower()) for entry in vocab},
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    article = sys.argv[1]
    runs = sys.argv[2:]
    if not os.path.isfile(os.path.join(article, "transcript.json")):
        print(f"== {article}/transcript.json not found")
        return 1

    print(f"article: {article}\n")
    summary = {}
    for run in runs:
        label = os.path.basename(os.path.normpath(run)) or run
        result = validate(article, run, label)
        summary[label] = result

    if len(summary) > 1:
        print("\noverlap (exact de=en pairs):")
        labels = list(summary)
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                shared = summary[left]["vocab_pairs"] & summary[right]["vocab_pairs"]
                print(f"  {left} ∩ {right}: {len(shared)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

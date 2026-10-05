//! Places vocabulary pairs inside their sentence: `pairs.json`.
//!
//! The vocabulary pass returns `(original, translation)` phrases; this module
//! finds the sentence each phrase belongs to and the word indexes of the phrase
//! in that sentence and of its translation in the aligned target sentence.

use std::path::Path;

use crate::library;

/// Indexes of the article-body sentences: those that belong to an
/// `article.json` `blocks[]` entry. The title, kicker, description, byline and
/// image captions are spoken too but are not article prose, so the vocabulary
/// pass must not read them. A transcript without block indexes (an older file)
/// falls back to every sentence, so nothing is lost.
pub fn body_indexes(blocks: &[Option<usize>]) -> Vec<usize> {
    let body: Vec<usize> = blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.is_some())
        .map(|(index, _)| index)
        .collect();
    if body.is_empty() {
        (0..blocks.len()).collect()
    } else {
        body
    }
}
/// One entry per vocabulary pair that could be placed in a sentence.
pub fn locate(
    pairs: &[(String, String)],
    sentences: &[String],
    translations: &[String],
) -> Vec<library::PairRef> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (de, en) in pairs {
        let Some(sentence) = sentences.iter().position(|line| contains_phrase(line, de)) else {
            continue;
        };
        if !seen.insert((sentence, de.to_lowercase())) {
            continue;
        }
        let Some(source) = word_indexes(&sentences[sentence], de) else {
            continue;
        };
        let Some(target) = translations
            .get(sentence)
            .and_then(|line| target_indexes(line, en))
        else {
            continue;
        };

        out.push(library::PairRef {
            sentence,
            de: de.clone(),
            en: en.clone(),
            source,
            target,
        });
    }

    out
}

/// Distributes located pairs into `article.json`-mirrored blocks.
///
/// `ranges` is `(block index, kind, first, count)` per text block, exactly the
/// `article.json` `blocks[]` order with images skipped; `first`/`count` are
/// global sentence indexes, the same ones `locate` reports. Each returned block
/// mirrors the matching `translation.json` block and its pairs use a
/// block-local `sentence` index.
pub fn group(
    refs: &[library::PairRef],
    ranges: &[(usize, &str, usize, usize)],
) -> Vec<library::PairsBlock> {
    let mut blocks: Vec<library::PairsBlock> = ranges
        .iter()
        .map(|(index, kind, _, _)| library::PairsBlock {
            index: *index,
            kind: kind.to_string(),
            pairs: Vec::new(),
        })
        .collect();

    for pair in refs {
        let Some(position) = ranges
            .iter()
            .position(|(_, _, first, count)| pair.sentence >= *first && pair.sentence < first + count)
        else {
            continue;
        };
        let (_, _, first, _) = ranges[position];
        let mut local = pair.clone();
        local.sentence -= first;
        blocks[position].pairs.push(local);
    }

    blocks
}

/// Writes `pairs.json` next to `translation.json`.
pub fn write(
    dir: &Path,
    model: &str,
    target: &str,
    blocks: &[library::PairsBlock],
) -> Result<(), String> {
    let doc = library::Pairs {
        model: model.to_string(),
        target: target.to_string(),
        blocks: blocks.to_vec(),
    };
    let raw = serde_json::to_string_pretty(&doc).map_err(|err| err.to_string())?;
    std::fs::write(dir.join(library::PAIRS), raw).map_err(|err| err.to_string())
}

/// `true` when `phrase` occurs in `sentence`, ignoring case and punctuation.
fn contains_phrase(sentence: &str, phrase: &str) -> bool {
    let haystack = normalize_for_match(sentence);
    let needle = normalize_for_match(phrase);
    !needle.is_empty() && haystack.contains(&needle)
}

/// Indexes of `phrase`'s words inside `sentence`, matching on lowercase letters
/// and digits. `None` when the phrase is not a contiguous run of the sentence.
fn word_indexes(sentence: &str, phrase: &str) -> Option<Vec<usize>> {
    let words: Vec<String> = sentence.split_whitespace().map(normalize_word).collect();
    let needle: Vec<String> = phrase
        .split_whitespace()
        .map(normalize_word)
        .filter(|word| !word.is_empty())
        .collect();
    if needle.is_empty() || needle.len() > words.len() {
        return None;
    }

    for start in 0..=words.len() - needle.len() {
        if words[start..start + needle.len()] == needle[..] {
            return Some((start..start + needle.len()).collect());
        }
    }
    None
}

/// Function words that carry no index information on the target side.
const ENGLISH_STOPWORDS: &[&str] = &[
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "and", "or", "by", "with", "from",
    "that", "this", "his", "her", "its", "was", "were", "is", "are", "be", "as", "it", "he",
    "she", "they", "we", "you", "not", "no", "had", "has", "have",
];

/// Indexes of the translation words that match `phrase`. The model's wording can
/// differ from the stored translation ("Israeli attack" for "Hamas attacks"), so
/// the whole phrase is tried first and then its content words.
fn target_indexes(translation: &str, phrase: &str) -> Option<Vec<usize>> {
    if let Some(indexes) = word_indexes(translation, phrase) {
        return Some(indexes);
    }

    let words: Vec<String> = translation.split_whitespace().map(normalize_word).collect();
    let needles: Vec<String> = phrase
        .split_whitespace()
        .map(normalize_word)
        .filter(|word| !word.is_empty() && !ENGLISH_STOPWORDS.contains(&word.as_str()))
        .collect();

    let mut out = Vec::new();
    for (index, word) in words.iter().enumerate() {
        if needles.iter().any(|needle| words_match(word, needle)) {
            out.push(index);
        }
    }

    (!out.is_empty()).then_some(out)
}

/// Equality, or a shared stem long enough to survive an inflection.
fn words_match(a: &str, b: &str) -> bool {
    a == b || (a.len().min(b.len()) >= 4 && (a.starts_with(b) || b.starts_with(a)))
}

/// Lowercase letters and digits only, so punctuation does not block a match.
fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Lowercases and collapses everything that is not a letter or digit.
fn normalize_for_match(text: &str) -> String {
    let mut out = String::new();
    let mut space = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
            space = false;
        } else if !space && !out.is_empty() {
            out.push(' ');
            space = true;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_indexes_survive_a_paraphrase() {
        let translation = "Reports on warnings of Hamas attacks: Netanyahu visits";
        assert_eq!(word_indexes(translation, "Hamas attacks"), Some(vec![4, 5]));
        assert_eq!(target_indexes(translation, "Israeli attack"), Some(vec![5]));
    }

    #[test]
    fn locate_reports_sentence_and_indexes() {
        let sentences = vec![
            "Der israelische Premier bestreitet jede Kenntnis.".to_string(),
            "Nun flog er für vertrauliche Gespräche an den Golf.".to_string(),
        ];
        let translations = vec![
            "The Israeli Prime Minister denies any knowledge.".to_string(),
            "He flew to the Gulf for confidential conversations.".to_string(),
        ];
        let pairs = vec![
            ("vertrauliche Gespräche".to_string(), "confidential conversations".to_string()),
            ("Präsident".to_string(), "the president".to_string()),
        ];

        let refs = locate(&pairs, &sentences, &translations);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].sentence, 1);
        assert_eq!(refs[0].de, "vertrauliche Gespräche");
        assert_eq!(refs[0].source, vec![4, 5]);
        assert_eq!(refs[0].target, vec![6, 7]);
    }

    #[test]
    fn group_mirrors_blocks_with_local_sentence_indexes() {
        let refs = vec![
            library::PairRef {
                sentence: 1,
                de: "a".to_string(),
                en: "A".to_string(),
                source: vec![0],
                target: vec![1],
            },
            library::PairRef {
                sentence: 4,
                de: "b".to_string(),
                en: "B".to_string(),
                source: vec![2],
                target: vec![3],
            },
            // a metadata sentence outside every block is dropped
            library::PairRef {
                sentence: 9,
                de: "c".to_string(),
                en: "C".to_string(),
                source: vec![4],
                target: vec![5],
            },
        ];
        let ranges = vec![(1usize, "para", 0usize, 3usize), (2usize, "para", 3usize, 2usize)];

        let blocks = group(&refs, &ranges);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].index, 1);
        assert_eq!(blocks[0].kind, "para");
        assert_eq!(blocks[0].pairs.len(), 1);
        assert_eq!(blocks[0].pairs[0].sentence, 1);
        assert_eq!(blocks[0].pairs[0].de, "a");
        assert_eq!(blocks[1].index, 2);
        assert_eq!(blocks[1].pairs.len(), 1);
        assert_eq!(blocks[1].pairs[0].sentence, 1); // global 4 - block first 3
        assert_eq!(blocks[1].pairs[0].de, "b");
    }
    #[test]
    fn body_indexes_skip_the_spoken_metadata() {
        let blocks = vec![None, None, Some(1), Some(1), None, Some(2)];
        assert_eq!(body_indexes(&blocks), vec![2, 3, 5]);
        // an older transcript without block indexes keeps every sentence
        assert_eq!(body_indexes(&[None, None]), vec![0, 1]);
    }

}

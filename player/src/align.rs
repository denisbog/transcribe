//! Force-align a known (German) article text to ASR word timings.
//!
//! The recognizer supplies word-level timings for what it heard; the article
//! text is tokenized and matched to that word sequence with a global
//! (Needleman–Wunsch) alignment. Words the recognizer dropped are interpolated
//! between their aligned neighbours, so the whole article gets a timeline that
//! the player can follow.

use crate::library::Seg;

/// An article aligned to an audio file.
#[derive(Debug, Clone, Default)]
pub struct Aligned {
    /// One segment per sentence of the article, in the article's spelling.
    pub sentences: Vec<Seg>,
    /// One segment per word of the article, in order.
    pub words: Vec<Seg>,
}

/// One word of the article: how it is written, how it is compared, where it
/// belongs.
struct Token {
    text: String,
    key: String,
    sentence: usize,
}

/// Aligns `article` to `asr_words`; `duration` is the audio length in seconds.
pub fn align(article: &str, asr_words: &[Seg], duration: f32) -> Aligned {
    let sentences = split_sentences(&clean_markdown(article));
    let mut tokens: Vec<Token> = Vec::new();

    for (index, sentence) in sentences.iter().enumerate() {
        for word in sentence.split_whitespace() {
            let key = key_of(word);
            if key.is_empty() {
                continue;
            }
            tokens.push(Token {
                text: word.to_string(),
                key,
                sentence: index,
            });
        }
    }

    let keys = asr_words
        .iter()
        .map(|word| key_of(&word.text))
        .collect::<Vec<_>>();

    // times[i] = aligned (start, end) of article token i, if the recognizer
    // matched it
    let mut times: Vec<Option<(f32, f32)>> = vec![None; tokens.len()];
    for (i, j) in needleman_wunsch(&tokens, &keys) {
        times[i] = Some((asr_words[j].start, asr_words[j].end));
    }
    interpolate(&mut times, duration);

    let words = tokens
        .iter()
        .zip(&times)
        .filter_map(|(token, time)| {
            let (start, end) = (*time)?;
            Some(Seg {
                start,
                end: end.max(start + 0.02),
                text: token.text.clone(),
            })
        })
        .collect::<Vec<_>>();

    // token ranges per sentence (tokens are pushed in sentence order)
    let mut ranges = vec![None::<(usize, usize)>; sentences.len()];
    for (index, token) in tokens.iter().enumerate() {
        ranges[token.sentence] = match ranges[token.sentence] {
            None => Some((index, index)),
            Some((lo, _)) => Some((lo, index)),
        };
    }
    let covered = (0..sentences.len())
        .filter(|index| ranges[*index].is_some())
        .collect::<Vec<_>>();

    // The player groups words into sentences by time (`words[next].start <
    // phrase.end - 0.08`), so a sentence must end after its own last word
    // (min. 0.1 s) but before the next sentence's first word would be pulled in.
    let mut out_sentences = Vec::with_capacity(covered.len());
    for (position, &index) in covered.iter().enumerate() {
        let (first, last) = ranges[index].expect("covered sentences have tokens");
        let start = words[first].start;
        let mut end = words[last].end.max(words[last].start + 0.1);

        if let Some(&following) = covered.get(position + 1) {
            let next_start = words[ranges[following].expect("covered").0].start;
            end = end.min(next_start + 0.079).max(words[last].start + 0.09);
        }

        out_sentences.push(Seg {
            start,
            end,
            text: sentences[index].clone(),
        });
    }

    Aligned {
        sentences: out_sentences,
        words,
    }
}

/// The article as spoken prose: YAML frontmatter, heading/blockquote markers,
/// emphasis, links, images and inline HTML are removed; paragraphs stay on
/// their own lines.
pub fn clean_markdown(text: &str) -> String {
    let mut body = text.trim_start_matches('\u{feff}').trim_start();

    // YAML frontmatter: the file opens with `---`, the block ends at the next
    // `---` line
    if body.starts_with("---") {
        if let Some(end) = body[3..].find("\n---") {
            body = &body[3 + end + 4..];
        }
    }

    let mut out = String::with_capacity(body.len());
    for line in body.lines() {
        let trimmed = line.trim();
        // images, captions and layout HTML are not read aloud
        if trimmed.starts_with("![")
            || trimmed.starts_with("<sub")
            || trimmed.starts_with("<img")
            || trimmed.starts_with("</")
            || matches!(trimmed, "---" | "***" | "___" | "===")
        {
            continue;
        }

        let text = strip_inline(line);
        let text = text
            .trim()
            .trim_start_matches(['-', '•', '·'])
            .trim_start();
        if !text.is_empty() {
            out.push_str(text);
            out.push('\n');
        }
    }
    out
}

/// Removes inline markdown from one line: links keep their label, images and
/// HTML tags disappear, emphasis/heading markers are dropped.
fn strip_inline(line: &str) -> String {
    let chars = line.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '!' && chars.get(i + 1) == Some(&'[') {
            // ![alt](target): drop the whole image
            i += 2;
            while i < chars.len() && chars[i] != ']' {
                i += 1;
            }
            i += 1;
            if chars.get(i) == Some(&'(') {
                while i < chars.len() && chars[i] != ')' {
                    i += 1;
                }
                i += 1;
            }
            continue;
        }

        if ch == '[' {
            // [label](target) -> label
            i += 1;
            let mut label = String::new();
            while i < chars.len() && chars[i] != ']' {
                label.push(chars[i]);
                i += 1;
            }
            i += 1;
            if chars.get(i) == Some(&'(') {
                while i < chars.len() && chars[i] != ')' {
                    i += 1;
                }
                i += 1;
            }
            out.push_str(&label);
            continue;
        }

        if ch == '<' {
            while i < chars.len() && chars[i] != '>' {
                i += 1;
            }
            i += 1;
            continue;
        }

        if matches!(ch, '*' | '_' | '`' | '#' | '>') {
            i += 1;
            continue;
        }

        out.push(ch);
        i += 1;
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Splits prose into sentences: on line breaks and on `.!?…` that end a
/// sentence (a following uppercase letter or the end of the text), so German
/// abbreviations such as `z.B.` or `bzw.` stay inside their sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);

        let boundary = match ch {
            '\n' => true,
            '.' => match chars.get(i + 1) {
                // a letter or digit right after the dot is part of the word
                // (`z.B.`, `3.5`) and never ends a sentence
                Some(next) if !next.is_whitespace() => false,
                _ => match next_non_space(&chars, i + 1) {
                    None => true,
                    Some(next) => next.is_uppercase() || next == '„' || next == '"',
                },
            },
            '!' | '?' | '…' => match next_non_space(&chars, i + 1) {
                None => true,
                Some(next) => next.is_uppercase() || next == '„' || next == '"',
            },
            _ => false,
        };

        if boundary {
            let sentence = current.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            current.clear();
            while chars.get(i + 1).map(|c| c.is_whitespace()).unwrap_or(false) {
                i += 1;
            }
        }
        i += 1;
    }

    let tail = current.trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    sentences
}

/// Comparison form of a word: lowercase, umlauts folded, punctuation dropped.
fn key_of(word: &str) -> String {
    let mut out = String::new();
    for ch in word.chars().flat_map(|ch| ch.to_lowercase()) {
        match ch {
            'ä' => out.push('a'),
            'ö' => out.push('o'),
            'ü' => out.push('u'),
            'ß' => out.push_str("ss"),
            'é' | 'è' | 'ê' => out.push('e'),
            'à' | 'â' | 'á' => out.push('a'),
            ch if ch.is_alphanumeric() => out.push(ch),
            _ => {}
        }
    }
    out
}

fn next_non_space(chars: &[char], mut index: usize) -> Option<char> {
    while let Some(ch) = chars.get(index) {
        if !ch.is_whitespace() {
            return Some(*ch);
        }
        index += 1;
    }
    None
}

/// Global alignment of the article tokens against the recognized words; pairs
/// are `(article index, asr index)` for equal or similar words only.
fn needleman_wunsch(tokens: &[Token], keys: &[String]) -> Vec<(usize, usize)> {
    const GAP: i32 = -1;
    let (n, m) = (tokens.len(), keys.len());
    if n == 0 || m == 0 {
        return Vec::new();
    }

    let mut score = vec![vec![0i32; m + 1]; n + 1];
    for i in 1..=n {
        score[i][0] = score[i - 1][0] + GAP;
    }
    for j in 1..=m {
        score[0][j] = score[0][j - 1] + GAP;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = score[i - 1][j - 1] + pairwise(&tokens[i - 1].key, &keys[j - 1]);
            let del = score[i - 1][j] + GAP;
            let ins = score[i][j - 1] + GAP;
            score[i][j] = sub.max(del).max(ins);
        }
    }

    let mut pairs = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        let value = pairwise(&tokens[i - 1].key, &keys[j - 1]);
        if score[i][j] == score[i - 1][j - 1] + value {
            if value > 0 {
                pairs.push((i - 1, j - 1));
            }
            i -= 1;
            j -= 1;
        } else if score[i][j] == score[i - 1][j] + GAP {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

fn pairwise(left: &str, right: &str) -> i32 {
    if left == right {
        3
    } else if similar(left, right) {
        1
    } else {
        -1
    }
}

fn similar(left: &str, right: &str) -> bool {
    let common = left
        .chars()
        .zip(right.chars())
        .take_while(|(a, b)| a == b)
        .count();
    if common >= 3 {
        return true;
    }
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    short.len() >= 4 && long.contains(short)
}

/// Gives every unmatched word a slice of the silence between its neighbours.
fn interpolate(times: &mut [Option<(f32, f32)>], duration: f32) {
    let n = times.len();
    let mut index = 0;

    while index < n {
        if times[index].is_some() {
            index += 1;
            continue;
        }

        let start = index;
        while index < n && times[index].is_none() {
            index += 1;
        }
        let end = index;

        let left = if start == 0 {
            0.0
        } else {
            times[start - 1].map(|(_, end)| end).unwrap_or(0.0)
        };
        let mut right = if end < n {
            times[end].map(|(start, _)| start).unwrap_or(duration)
        } else {
            duration.max(left + 0.1)
        };
        if right < left {
            right = left;
        }

        let span = (right - left).max(0.0);
        let count = (end - start) as f32;
        for (offset, slot) in times[start..end].iter_mut().enumerate() {
            let from = left + span * offset as f32 / count;
            let to = left + span * (offset + 1) as f32 / count;
            *slot = Some((from, to));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, text: &str) -> Seg {
        Seg {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn splits_sentences_on_german_punctuation() {
        let out = split_sentences("Hallo Welt. Wie geht es dir? Gut!\nDanke.");
        assert_eq!(
            out,
            vec!["Hallo Welt.", "Wie geht es dir?", "Gut!", "Danke."]
        );
    }

    #[test]
    fn keeps_abbreviations_inside_a_sentence() {
        let out = split_sentences("Das ist z.B. gut. Der Hund bellt.");
        assert_eq!(out, vec!["Das ist z.B. gut.", "Der Hund bellt."]);
    }

    #[test]
    fn strips_markdown_scaffolding() {
        let article = "---\ntitle: Test\ncategory: Politik\n---\n\n# Die Überschrift\n\n> Der Standfirst.\n\n![Bild](/img.jpg)\n\nEin **fetter** Satz mit [Link](https://x.y) und <sub>klein</sub>.\n";
        let clean = clean_markdown(article);

        assert!(!clean.contains("title:"));
        assert!(!clean.contains('#'));
        assert!(!clean.contains('*'));
        assert!(!clean.contains("!["));
        assert!(!clean.contains("https://"));
        assert!(!clean.contains("sub>"));
        assert!(clean.contains("Die Überschrift"));
        assert!(clean.contains("Der Standfirst."));
        assert!(clean.contains("Ein fetter Satz mit Link und klein."));
    }

    #[test]
    fn aligning_markdown_ignores_the_scaffolding() {
        let asr = vec![seg(0.0, 0.5, "Der"), seg(0.5, 1.0, "Standfirst.")];
        let aligned = align("---\ntitle: x\n---\n\n> Der Standfirst.\n", &asr, 1.5);
        assert_eq!(aligned.words.len(), 2);
        assert_eq!(aligned.sentences.len(), 1);
        assert_eq!(aligned.sentences[0].text, "Der Standfirst.");
    }

    #[test]
    fn matches_known_words_to_recognized_timings() {
        let asr = vec![
            seg(0.0, 0.4, "Hallo"),
            seg(0.4, 0.6, "Welt"),
            seg(0.6, 0.9, "wie"),
            seg(0.9, 1.2, "geht"),
            seg(1.2, 1.5, "es"),
        ];
        let aligned = align("Hallo Welt, wie geht es?", &asr, 2.0);

        assert_eq!(aligned.words.len(), 5);
        assert_eq!(aligned.words[0].text, "Hallo");
        assert!((aligned.words[0].start - 0.0).abs() < 1e-3);
        assert_eq!(aligned.words[1].text, "Welt,");
        assert!((aligned.words[1].start - 0.4).abs() < 1e-3);
        assert_eq!(aligned.sentences.len(), 1);
        assert!((aligned.sentences[0].start - 0.0).abs() < 1e-3);
    }

    #[test]
    fn interpolates_words_the_recognizer_dropped() {
        let asr = vec![seg(0.0, 0.5, "eins"), seg(2.0, 2.5, "vier")];
        let aligned = align("eins zwei drei vier", &asr, 3.0);

        assert_eq!(aligned.words.len(), 4);
        assert!(aligned.words[1].start >= 0.5);
        assert!(aligned.words[2].start >= aligned.words[1].start);
        assert!(aligned.words[2].end <= 2.0 + 1e-3);
        assert!((aligned.words[3].start - 2.0).abs() < 1e-3);
    }

    #[test]
    fn sentences_group_back_to_their_own_words() {
        let asr = vec![
            seg(0.0, 0.3, "Hallo"),
            seg(0.3, 0.5, "Welt."),
            seg(0.5, 0.8, "Wie"),
            seg(0.8, 1.0, "geht"),
            seg(1.0, 1.2, "es"),
            seg(1.2, 1.5, "dir?"),
        ];
        let aligned = align("Hallo Welt. Wie geht es dir?", &asr, 2.0);

        // mirrors the player's `group()` in main.rs
        let mut next = 0;
        for phrase in &aligned.sentences {
            let start = next;
            while next < aligned.words.len() && aligned.words[next].start < phrase.end - 0.08 {
                next += 1;
            }
            assert!(next > start, "sentence {:?} got no words", phrase.text);
        }
        assert_eq!(next, aligned.words.len());
    }
}

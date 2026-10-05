Task: build a bilingual German->English vocabulary file and a word-index pairing file for one article.

Inputs (read them from disk):
- <ARTICLE_DIR>/transcript.json  (German, {"segments":[{"start":..,"end":..,"text":..}]})
- <ARTICLE_DIR>/translation.json ({"target":"en","sentences":[...]}) aligned 1:1 with the transcript segments.

Write exactly these two files:
- <OUT_DIR>/vocabulary.json
- <OUT_DIR>/pairs.json

Vocabulary selection rules:
- Pick words/short phrases worth learning for an advanced English reader: false friends, and German words/compounds whose English meaning cannot be guessed from the German form.
- Never pick: names of people/places/countries, brands, internationalisms/loanwords that look almost the same in English, or a German word whose English translation is nearly identical.
- Both sides must be quoted EXACTLY as they appear: the German phrase from the segment text, the English phrase from that same segment's translation. Do not inflect or paraphrase.

vocabulary.json schema:
{"target":"en","pairs":[{"de":"<exact German>","en":"<exact English>"}, ...]}

pairs.json schema:
{"model":"<model name>","target":"en","pairs":[{"sentence":<0-based index into transcript segments>,"de":"<exact German>","en":"<exact English>","source":[<word indexes of de within segment.text.split()>],"target":[<word indexes of en within translation.sentences[sentence].split()>]}, ...]}
- Only include a pair in pairs.json if BOTH phrases occur verbatim in that sentence / its translation; otherwise drop it.
- source/target are arrays of indexes into whitespace-split words of the sentence text and the aligned translation sentence.

Do not modify any other files. When done, print a short summary with the number of vocabulary pairs and placed pairs.

# Domain 3 — Thai Word Segmentation (NLP / Character-Level Sequence Labeling)

Competition: `super-ai-engineer-ss-6-word-segmentation`

Source note: Kaggle page is login-walled. Format below follows the standard Super AI Engineer / BEST-2010 / PyThaiNLP convention. On exam day, read the actual Data + Evaluation tabs first and confirm: column names, label `0/1` vs `B/I`, char-level vs word-level F1.

## A. TASK SUMMARY
- Task: Thai has no spaces. Given raw Thai text, predict word boundaries. Framed as `per-character sequence labeling`, not whole-word classification.
- Two equivalent output framings (check Data tab):
  1. Per-character binary boundary: for each char index `i`, `1` if a new word starts at `i`, else `0`. First char always `1` (the PyThaiNLP "starting character" convention).
  2. BIES/BI tag per char: `B`(begin)/`I`(inside) — isomorphic to binary (`B==1`, `I==0`).
- Typical submission CSV: `id,label` — one row PER CHARACTER, `id` = global running char index, `label` = `0/1` (or `B/I`). Output is NOT a list of words.
- Metric: character-level boundary F1 (BEST-2010 / PyThaiNLP standard) on the "is this char a word-start" label.
  - `P=TP/(TP+FP)`, `R=TP/(TP+FN)`, `F1=2PR/(P+R)`. = `sklearn f1_score(..., average='binary', pos_label=1)`.
  - Some variants use word-level F1 (a word correct only if BOTH boundaries match) — stricter. Ref: DeepCut on BEST-2010 = char F1 98.1, word F1 92.6.
- Dataset: BEST-2010 / InterBEST (NECTEC), ~5M words, 4 domains, `|`-annotated boundaries.

## B. DOMAIN CATEGORIZATION
- Top: Sequence Labeling (token/char classification) — same family as NER, POS tagging, chunking.
- Specific: Tokenization / Word Segmentation for scriptio-continua languages (Thai, Chinese, Japanese, Lao, Khmer).
- Granularity: character-level tagging (one label per char / per TCC).
- Adjacent (90% shared pipeline): sentence segmentation, NER, POS tagging — swap label set, reuse everything.

## C. PREDICTED EXAM VARIATIONS
1. Thai NER. Unit = word/subword; labels = BIO over entity types. `AutoModelForTokenClassification`. Metric entity-level F1 via `seqeval`. Watch subword label alignment (`word_ids()`, `-100`).
2. Thai POS tagging. Labels = POS tags (ORCHID/UD). PyThaiNLP `pos_tag(...)` instant baseline. Metric per-token acc/macro-F1.
3. Thai sentence segmentation. Per-space/per-token binary boundary. LST20 corpus. Same code as WS, different positive class.
4. Thai spelling correction. NOT pure labeling — seq2seq/candidate-ranking. PyThaiNLP `correct()`/`spell()` baseline. Needs generation.
5. Thai text classification (sentiment/topic). Whole-sequence label. TF-IDF+LinearSVC baseline OR WangchanBERTa sequence-classification. Different head (pooled CLS).
6. Syllable segmentation / G2P. Finer unit (`ssg` engine) or char seq2seq. Same BI skeleton.

Decision rule: one label per position → sequence labeling (reuse pipeline). One label per text → sequence-classification head. Produce new text → seq2seq.

## D. CORE CONVERSION (the spine — B=1 start, I=0 inside)
```python
def words_to_char_labels(words):     # ['สวัสดี','ครับ'] -> ('สวัสดีครับ',[1,0,0,0,0,0,1,0,0,0])
    text, labels = [], []
    for w in words:
        for i, ch in enumerate(w):
            text.append(ch); labels.append(1 if i == 0 else 0)
    return "".join(text), labels
def char_labels_to_words(text, labels):
    words, cur = [], ""
    for ch, lab in zip(text, labels):
        if lab == 1 and cur: words.append(cur); cur = ""
        cur += ch
    if cur: words.append(cur)
    return words
```

## E. SOTA TABLE (Thai word segmentation, BEST-2010)
| Engine / Model | Approach | Char F1 | Word F1 | When to use |
|---|---|---|---|---|
| `newmm` (PyThaiNLP) | dict + maximal matching + TCC | good in-domain | 0.67 | Fastest baseline, no GPU |
| `attacut` | char+syllable dilated CNN | 97.2 | 0.91 | Best speed/quality NN |
| `deepcut` | char-level CNN (13 conv) | 98.1 | 0.93 | Strong off-the-shelf, near-SOTA |
| `nercut` | newmm + NER merge | ~newmm+ | — | Keep entities whole |
| char CRF (pycrfsuite) | CRF on char window | ~96-98 | ~0.90 | Trainable, CPU, in-domain wins |
| `sefr_cut` | DeepCut + CRF stacked | 98.1 | 0.925 | SOTA / domain adaptation |
| WangchanBERTa token-cls | RoBERTa-Thai tagging | ~98+ | highest | Max accuracy, needs GPU |

Practical ranking under time pressure: `deepcut` (zero-train near-SOTA) → char CRF (CPU, trainable) → WangchanBERTa (GPU + time) → `sefr_cut`.

## F. THAI-SPECIFIC GOTCHAS
- Thai Character Clusters (TCC): combining vowels/tone marks (`ิ ี ึ ื ุ ู ่ ้ ๊ ๋ ์ ั ํ`) and pre-vowels (`เ แ โ ใ ไ`) can NEVER start a word — a boundary must not fall inside a TCC. Restrict predicted boundaries to legal TCC cut positions (`pythainlp.tcc`) to raise precision instantly.
- Count by codepoint; do NOT apply NFC that merges/reorders (desyncs char indices).
- Char-level F1 (lenient ~98) vs word-level F1 (strict ~93) differ a lot — confirm which the LB uses.
- Whitespace: in BEST, spaces are usually their own token. `keep_whitespace=True`.
- Submission alignment is the #1 killer: `len(labels) == len(text)`, same order. Assert it every row. First char always `1`.

## G. QUICK-WIN BASELINE (<30 min)
`pip install pythainlp deepcut`. Read Data+Evaluation tabs. Run `newmm` → words → per-char labels → match sample columns/ids → submit. Re-run with `deepcut` (≈ char F1 98), keep better. Highest-ROI next step: restrict boundaries to legal TCC positions.

## H. KEYWORDS TO STUDY
```
Thai word segmentation, scriptio continua, character-level sequence labeling, BIES BIO tagging,
word boundary detection, boundary F1, character-level F1, word-level F1, CTW correctly tokenized words,
BEST-2010 InterBEST NECTEC, LST20, ORCHID, Wisesight sentiment,
PyThaiNLP word_tokenize, newmm, maximal matching, deepcut, attacut, nercut, sefr_cut SEFR-CUT,
Thai Character Cluster TCC, pythainlp.tcc, combining marks, leading vowels,
CRF python-crfsuite pycrfsuite conditional random field feature template, BiLSTM-CRF, character CNN,
WangchanBERTa airesearch wangchanberta-base-att-spm-uncased, AutoModelForTokenClassification,
token classification, word_ids label alignment, subword sentencepiece, -100 masking, seqeval,
sklearn f1_score average binary, transformers Trainer, Thai NER POS sentence segmentation,
mT5 seq2seq segmentation, domain adaptation, out-of-domain robustness
```

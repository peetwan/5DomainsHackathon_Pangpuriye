# Domain 1 — Thai Language Image Captioning (Multimodal / Vision→Thai Text)

Competition: `super-ai-engineer-ss-6-thai-language-image-captioning`

Source note: the SS6 Kaggle page is login-gated. Task/metric/format below come from the public predecessor (`image-processing-thai-language-image-captioning`, SS5) + the dataset cards (COCO-Thai + NECTEC IPU24) + PyThaiNLP docs. Verify the exact `sample_submission.csv` columns and JSON field names the moment you get dataset access.

## A. TASK SUMMARY
- Task: given one image, generate one Thai sentence describing it. Single image in, single string out.
- Dataset: COCO (2014/2017) captions machine-translated EN→TH (VISTEC-depa NMT) + native-Thai IPU24. Layout: `train/ val/ test/` image folders + `capgen_v1.0_train.json`, `capgen_v1.0_val.json`, `sample_submission.csv`. SS5 scale ~34k files / ~1.95 GB.
- JSON shape: COCO-caption style — list/dict keyed by image filename/id → one or more Thai caption strings.
- Metric: `BLEU`. Critical — Thai has no spaces, so BLEU is computed on `PyThaiNLP newmm`-tokenized text, cumulative BLEU-4 (weights 0.25 each) with smoothing. Public models on this dataset report BLEU via `newmm`. Expect corpus-level cumulative BLEU-4 in [0,1]. Public/Private = 50/50.
  - `BLEU = BP * exp(sum_{n=1..4} 0.25*log p_n)`, `BP = 1 if c>r else exp(1-r/c)`. Tokens = `newmm` words.
- Submission: CSV, 2 cols `id,caption`. One row per test image; `id` = filename/numeric id matching the sample; `caption` = Thai string.

## B. DOMAIN CATEGORIZATION
- Field: Multimodal ML / Vision-Language (VL).
- Task type: Image-to-Text generation (conditional sequence generation) = image captioning.
- Sub-area: low-resource / non-English (Thai) generative captioning. CV image encoder (ViT/CLIP/SigLIP/ConvNeXt) + NLG autoregressive decoder.
- Adjacent: VQA, OCR/scene-text, dense/region captioning, video captioning, image-text retrieval, multimodal translation.
- Eval family: n-gram overlap (BLEU/ROUGE/METEOR/CIDEr/SPICE) — this comp uses BLEU + Thai segmentation.

## C. PREDICTED EXAM VARIATIONS
1. Thai VQA (image + Thai question → Thai answer). Switch to instruction VLM (Qwen2.5-VL, PaliGemma2-mix); question goes into the prompt. Metric likely exact-match/accuracy.
2. OCR / scene-text captioning (read Thai text in image). Use OCR-strong backbone (PaliGemma2, Qwen2.5-VL) or OCR engine (EasyOCR/Typhoon-OCR) + captioner. Metric maybe CER/WER.
3. Multilingual / EN+TH dual captioning. Use mBLIP / PaliGemma2 / Qwen2.5-VL with language-tag prompting.
4. Video captioning. Sample N frames, use Qwen2.5-VL / LLaVA-OneVision; temporal pooling.
5. Dense / grounded captioning (caption + boxes). Use Florence-2 / Qwen2.5-VL / Kosmos-2; structured output.
6. Domain-specific Thai captioning (food/tourism/medical). Same pipeline + heavy LoRA fine-tune + domain `custom_dict` for newmm.

Common thread: same encoder-decoder VLM. What changes = input modality/prompt, target string, metric + its tokenizer.

## D. THAI BLEU EVALUATOR (this IS the leaderboard — validate locally first)
```python
from pythainlp.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
def th_tok(s):
    return [w for w in word_tokenize(str(s), engine="newmm", keep_whitespace=False) if w.strip()]
def thai_bleu(list_of_refs, hyps):   # refs: list[list[str]], hyps: list[str]
    refs = [[th_tok(r) for r in rs] for rs in list_of_refs]
    hyp  = [th_tok(h) for h in hyps]
    return corpus_bleu(refs, hyp, weights=(0.25,0.25,0.25,0.25),
                       smoothing_function=SmoothingFunction().method4)
```

## E. SOTA TABLE (ranked for THIS task)
| Rank | Model | HF id | Params | When to use |
|---|---|---|---|---|
| 1 | PaliGemma2-3B | `google/paligemma2-3b-pt-224` (or `-448`) | 3B | Primary submission. Multilingual + OCR-pretrained. Best public score on this Thai-COCO data. LoRA 4-bit fits 16GB |
| 2 | Qwen2.5-VL 3B/7B | `Qwen/Qwen2.5-VL-3B-Instruct` | 4B/8B | Strong zero/few-shot Thai, great for VQA/OCR variants |
| 3 | Pretrained Thai PaliGemma2 | `MagiBoss/PaliGemma2-3b-224-COCO` | 3B | Warm start / fast strong baseline |
| 4 | Thai ConvNeXt/CLIP captioner | `Natthaphon/thaicapgen-convnext-phayathai` | 0.3-0.6B | Sub-1h baseline, no training, trained on THIS corpus |
| 5 | BLIP-2 + Typhoon | `MagiBoss/Blip2-Typhoon1.5-COCO` | ~3B | Alt baseline |
| 6 | BLIP base | `Salesforce/blip-image-captioning-base` | 0.25B | Fastest from-scratch fine-tune, fits anywhere |
| 7 | GIT | `microsoft/git-base` | 0.2B | Lightweight alt |
| 8 | mBLIP | `Gregor/mblip-mt0-xl` | ~4B | Multilingual fallback |
| 9 | Florence-2 | `microsoft/Florence-2-large` | 0.77B | OCR/grounded variant (English-centric) |

Reality check: best public BLEU-4 on Thai-COCO is low absolute (~4.8 BLEU-4 / ~12 cumulative). The leaderboard is relative — beat the median with a 3B VLM + correct tokenization + length calibration.

## F. THAI-SPECIFIC GOTCHAS
- No spaces between Thai words — never whitespace-split. BLEU MUST use `newmm` tokens.
- Match the host tokenizer (`newmm`). Other engines shift BLEU.
- Don't emit stray spaces inside Thai runs (extra spurious tokens).
- Brevity penalty bites — captions ~10-25 newmm tokens; calibrate `max_new_tokens` to median ref length.
- Decode greedy/beam (`num_beams=3-5`, `do_sample=False`). Sampling lowers BLEU.
- Multi-reference: pass ALL captions per image to `corpus_bleu`.
- Always UTF-8; write submission `utf-8-sig`. Preserve Thai combining marks (tone/vowel).

## G. QUICK-WIN BASELINE (<1 hour, zero training)
Use a model already trained on this exact corpus: `Natthaphon/thaicapgen-convnext-phayathai` (`trust_remote_code=True`) → generate captions for `test/` → map to `sample_submission.csv` → write `utf-8-sig`. Fallback: `MagiBoss/PaliGemma2-3b-224-COCO` with prompt `"caption th"`.

## H. KEYWORDS TO STUDY
```
Thai image captioning, image-to-text, vision-language model VLM, PaliGemma2 caption-th prefix,
SigLIP encoder, Qwen2.5-VL, BLIP, BLIP-2, GIT, Florence-2, mBLIP, LLaVA-OneVision,
encoder-decoder captioning, ViT, CLIP, ConvNeXt-V2, PhayathaiBERT, WangchanBERTa,
PEFT, LoRA, QLoRA, bitsandbytes 4-bit nf4, gradient accumulation, SFTTrainer TRL,
BLEU cumulative BLEU-4, brevity penalty, n-gram precision, smoothing function,
PyThaiNLP word_tokenize newmm, Thai Character Cluster TCC, nltk corpus_bleu, CIDEr METEOR SPICE,
multi-reference BLEU, beam search vs sampling, max_new_tokens, low-resource NLP,
COCO captions, IPU24, VISTEC-depa NMT, Typhoon LLM, EasyOCR Thai, utf-8-sig, VQA, OCR captioning
```

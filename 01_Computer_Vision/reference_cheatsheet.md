# Domain 2 — House Recognition (Computer Vision / Image Classification)

> 🔄 SOTA ล่าสุด (กลางปี 2026) + library เจ๋ง ๆ ดู `../ADVANCED_SOTA.md` และ `../LIBRARIES.md` — ไฟล์นี้คือความรู้พื้นฐาน/กับดักที่ยังใช้ได้

Competition: `super-ai-engineer-season-6-individual-hackathon-house-recognition`

Ground truth confirmed from a participant's public solution repo (`github.com/kullawattana/individual-hackathon-house-recognition-SuperAI-SS6`) since the Kaggle page is login-gated.

## A. TASK SUMMARY
- What it is: `binary image classification` of street-view residential photos into `class 0` vs `class 1`. NOT SVHN digit OCR, NOT price regression, NOT detection.
- Classes: 2 (`0`, `1`).
- Data: Train 2,953 images (cls0=1,520 / cls1=1,433 — nearly balanced). Test 1,550 images.
- Layout: `train/train/` images, `test/test/` images (`{id}.jpg`), `train.csv` (cols `image_name,class`), `sample_submission.csv` (cols `id,answer`).
- Submission: CSV header `id,answer`, one row per test image, `answer ∈ {0,1}` (integer label, not probability), 1,550 rows.
- Metric: `Accuracy` (reference public 0.97777 with balanced classes + integer answers is consistent with plain accuracy). If it's macro-F1, strategy barely changes (balanced classes) — just tune threshold.
- Reference to beat: EfficientNet-B3 @ 300px → val 95.25%, public 0.97777.

## B. DOMAIN CATEGORIZATION
- CV taxonomy: supervised image-level classification → binary, single-label, natural-scene, small dataset (~3k), transfer-learning regime.
- Simplest CV head: one image → one label. No localization, no per-pixel, no sequence.
- Dominant lever on ~3k images = transfer learning + augmentation + TTA, NOT architecture novelty. Gains beyond ~0.97 come from ensembling/TTA/folds.

## C. PREDICTED EXAM VARIATIONS
1. Multi-class single-label (N styles/types). Set `num_classes=N`, keep CrossEntropy, metric macro-F1/top-1; handle imbalance with class weights / WeightedRandomSampler.
2. Fine-grained classification. Higher res (384/448), stronger backbone (convnext/eva02/swin), lower LR, mild geometric aug; TTA helps most.
3. Multi-label (several attributes). Head = `num_labels` logits, loss `BCEWithLogitsLoss`, sigmoid, per-label threshold, metric micro/macro-F1 or mAP.
4. Object detection (boxes). Leave timm, use `YOLOv8/v11` (ultralytics) or `RT-DETR`. Metric mAP@0.5.
5. Segmentation. `segmentation_models_pytorch` (U-Net/DeepLabV3+) or SAM. Metric IoU/Dice, RLE submission.
6. House-number OCR (SVHN multi-digit). CRNN+CTC or `TrOCR`/`Donut`. Metric exact-match/edit distance.

Default assumption: stays image classification (binary or multi-class). The pipeline handles both via one constant `NUM_CLASSES`.

## E. SOTA TABLE (small-data fine-tune, ranked by acc-per-cost)
| Rank | Model | timm id | Params | IN-1k | When to use |
|---|---|---|---|---|---|
| 1 | ConvNeXt-V2 Tiny | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 29M | ~83.9% | Best accuracy/cost for small data; default |
| 2 | EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | 22M | ~84.9% | Fast, light, excellent transfer |
| 3 | EfficientNet-B3 | `efficientnet_b3` | 12M | ~81.6% | Proven on THIS comp (0.978) |
| 4 | Swin-V2 Tiny | `swinv2_tiny_window16_256.ms_in1k` | 28M | ~82.8% | Fine-grained variant |
| 5 | MaxViT Tiny | `maxvit_tiny_tf_224.in1k` | 31M | ~83.6% | Robust, strong TTA gains |
| 6 | ConvNeXt Small | `convnext_small.fb_in22k_ft_in1k` | 50M | ~84.6% | If Tiny underfits |
| 7 | EVA-02 Small | `eva02_small_patch14_336.mim_in22k_ft_in1k` | 22M | ~85.7% | Top acc at 336px, more VRAM |
| 8 | ResNet-50 | `resnet50.a1_in1k` | 25M | ~80.4% | Sanity baseline / fastest |

Strategy: train models 1+2+3 (diverse architectures), average softmax probs. On ~3k images, Tiny/Small beats Base (Base overfits).

## F. PRACTICAL GOTCHAS
- Overfitting ~3k images: prefer Tiny/Small, `drop_rate=0.3`, label smoothing 0.1, MixUp/CutMix, RandomErasing, wd 1e-4, early-stop on val acc, 15-25 epochs.
- Validation must be `StratifiedKFold` on `class`.
- Augmentation: avoid VerticalFlip for houses; keep rotation/perspective mild.
- Imbalance NOT an issue here — don't add class weights.
- Leakage: ImageNet mean/std (don't fit on test). Use GroupKFold if same property appears multiple times.
- TTA: average softmax probs (not votes); step-0 = clean transform; test loader `shuffle=False`.
- Submission traps: header exactly `id,answer`; integer 0/1; 1,550 rows; ids from `sample_submission.csv` (not directory listing); add `.jpg` in code (ids are bare hashes).

## G. QUICK-WIN BASELINE (<1 hour)
EfficientNet-B3 @ 256px, 12 epochs, 1 fold, 3x TTA → ~0.95-0.97 in ~20-30 min on T4/P100. Then upgrade to EfficientNetV2-S @ 300px, 20 epochs, ensemble folds 0/1/2.

Mental checklist: pretrained backbone → `num_classes=2` → ImageNet norm → AdamW 3e-4 + cosine → label smoothing → val-acc early stop → 5x TTA softmax-average → write `id,answer` CSV.

## H. KEYWORDS TO STUDY
```
binary image classification, transfer learning, fine-tuning pretrained CNN, timm create_model,
EfficientNet-B3, EfficientNetV2-S, ConvNeXt-V2 FCMAE, Swin Transformer V2, MaxViT, EVA-02, ViT,
ImageNet normalization mean std, AdamW, weight decay, CosineAnnealingLR, label smoothing,
MixUp, CutMix, RandAugment, RandomErasing cutout, ColorJitter, test-time augmentation TTA,
mixed precision AMP GradScaler autocast, StratifiedKFold, GroupKFold leakage, out-of-fold OOF,
model ensembling probability averaging, WeightedRandomSampler, dropout drop_rate,
decision threshold tuning F1, accuracy vs macro-F1, BCEWithLogitsLoss multi-label, softmax vs sigmoid,
PyTorch Dataset DataLoader, AutoGluon MultiModalPredictor, Kaggle submission.csv,
YOLOv8 YOLOv11 RT-DETR detection, segmentation_models_pytorch U-Net DeepLabV3+, TrOCR Donut OCR
```

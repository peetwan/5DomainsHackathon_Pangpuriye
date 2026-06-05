# SOTA อัปเดต + วิธีดันคะแนน (รีเฟรชกลางปี 2026)

ไฟล์นี้รวม "ของแรง/ท่ายาก" ล่าสุด (verify id กับ HuggingFace/PyPI กลางปี 2026 แล้ว) ถ้าทำ `วิธีที่ 1 (ง่ายสุด)` ในโน้ตบุ๊กได้ submission แล้ว อยากดันคะแนนต่อ มาดูตรงนี้
ดู library เจ๋ง ๆ ที่ทำให้ชีวิตง่ายขึ้นต่อหัวข้อ -> `LIBRARIES.md`

> id ทั้งหมดตรวจแล้วว่ามีจริงกลางปี 2026 แต่ของหนัก (VLM/foundation model) ต้องรันบน Colab/Kaggle ที่มี GPU เสมอ

---

## Computer Vision — SOTA refresh (mid-2026)

แกนเดิม (timm backbone, zero-shot encoder) ยังใช้ได้ แต่ detection/segmentation ขยับไปอีกเจน

### Image classification (timm) — ยังใช่ + เพิ่มของใหม่
- `EVA-02-Large @448` ยังเป็น backbone fine-tune ที่ดีสุดใน timm: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (top-1 ~90%) -> ใส่ใน `MODEL_NAME` (image_classification วิธีที่ 2)
- `DINOv3` (ของใหม่ ดีสุดสาย self-supervised/frozen feature): `timm/vit_large_patch16_dinov3.lvd1689m` หรือ `facebook/dinov3-vitl16-pretrain-lvd1689m` ; มีตัว ConvNeXt distilled `timm/convnext_large.dinov3_lvd1689m` (frozen + linear head ดีตอน data น้อย/โดเมนแปลก)
- `ConvNeXt-V2` ยังเป็น CNN ที่เทรนง่าย/เสถียร
- zero-shot (ไม่ต้องเทรน): `google/siglip2-so400m-patch16-512` (multilingual, ผ่าน pipeline `zero-shot-image-classification`) ; ของใหม่แรงกว่า `facebook/PE-Core-G14-448` (Meta Perception Encoder, ชนะ SigLIP2 บน 0-shot)
- เพดาน ImageNet-1k กลางปี 2026 ยังราว ~91%

### Object detection — เปลี่ยน! (YOLO11/12 -> YOLO26 / DETR family)
- `YOLO26` (Ultralytics) = default ใหม่: NMS-free, เร็วบน CPU/edge -> `pip install ultralytics` แล้ว `YOLO("yolo26n.pt")` (n/s/m/l/x) -- ใช้แทน yolo11/12 ใน object_detection.ipynb
- `RF-DETR` (Roboflow, NAS family Nano..2x-Large) = real-time detector ตัวแรกที่ >60 mAP บน COCO + transfer ข้ามโดเมนเก่งสุด -> `pip install rfdetr` ; มี `RF-DETR-Seg` ด้วย
- `D-FINE` / `DEIMv2` = DETR family ใหม่ (DEIMv2 เสียบ DINOv3 เป็น backbone) -> `Intellindust/DEIMv2_DINOv3_{S,M,L,X}_COCO` ; D-FINE มีใน transformers (`DFineForObjectDetection`)
- `RT-DETRv2` ตอนนี้ตามหลังแล้ว เก็บไว้เป็น baseline ใน transformers

### Segmentation — เปลี่ยน! (SAM2.1 -> SAM3 / SAM3.1)
- `SAM3` (Meta, พ.ย. 2025) = default ใหม่: prompt ด้วย "ข้อความ" ได้ (open-vocabulary เช่น "รถโรงเรียนสีเหลืองทุกคัน") + detect+segment+track ในตัวเดียว -> `facebook/sam3` (ใน transformers)
- `SAM3.1` (มี.ค. 2026) = ติดตามหลายวัตถุเร็วขึ้น -> `facebook/sam3.1`
- `YOLO26-seg` / `RF-DETR-Seg` = real-time trainable masks ; `Mask2Former` = panoptic/semantic reference

### ถ้าอยากได้คะแนนเพิ่ม (CV) ทำตามลำดับ
1. เปลี่ยนรุ่นฟรี ๆ: detection `yolo11/12` -> `yolo26` ; classification `MODEL_NAME` -> `eva02_large_patch14_448...` หรือ frozen `DINOv3`
2. ขยาย scale + เพิ่ม imgsz/IMG_SIZE เท่าที่ GPU ไหว
3. เปิด augment/TTA (cls: MixUp/CutMix/EMA ; det/seg: copy_paste) + ใช้ `sahi` ถ้าวัตถุเล็ก
4. อัปตัวสุด: det -> `RF-DETR`/`DEIMv2` ; seg -> `detector -> SAM3` ; zero-shot cls -> `PE-Core`/`SigLIP2`
5. ensemble หลายโมเดล (`supervision` ช่วยรวม/วาดผล, `WBF` รวมกล่อง)

---

## NLP — SOTA refresh (mid-2026)

### Encoder (ตัวฐาน fine-tune)
- ใหม่: `jhu-clsp/mmBERT-base` (307M, 8K ctx, MIT, เร็วกว่า ชนะ XLM-R บน XTREME, รองรับไทย) = default multilingual ใหม่ -> ใส่แทน `xlm-roberta-large`/`mdeberta-v3-base` ใน text_classification วิธีที่ 2 / word_seg วิธีที่ 3
- ไทยโดยเฉพาะ: `clicknext/phayathaibert` ยังแรงสุดสาย monolingual (เก็บเป็น co-favorite, ensemble กับ mmBERT)

### ตัดคำ / NER / POS
- `PyThaiNLP 5.3.4` (import เบาลง 62x, มี offline mode `PYTHAINLP_OFFLINE=1`, มี `pythainlp.benchmarks` วัด BLEU/ROUGE/WER/CER)
- ตัดคำ: engine `nlpo3`/`deepcut`/`newmm`/`attacut` ยังใช้ได้
- NER: `pythainlp.tag.NER(engine="thainer-v2")` (default ใหม่) หรือ `engine="thai-nner"` (nested NER 104 ชนิด) ; zero-shot label เอง -> `gliner`

### Text classification
- quick-win ของใหม่ (สำคัญ!): `SetFit` (few-shot ~8 ตัวอย่าง/คลาส สู้ fine-tune เต็มได้ ไม่ต้อง prompt) -> `pip install setfit` ใช้ body `BAAI/bge-m3` หรือ `google/embeddinggemma-300m` (เร็ว) -- เพิ่มเป็นวิธีที่ 3 ในโน้ตบุ๊กให้แล้ว
- LLM zero/few-shot ceiling (id ใหม่): `typhoon-ai/typhoon2.5-qwen3-4b` (edge) / `typhoon2.5-qwen3-30b-a3b` (MoE) ; `aisingapore/Qwen-SEA-LION-v4-32B-IT` / `Gemma-SEA-LION-v4.5-27B-IT`
- embeddings: `BAAI/bge-m3` (ยังดี), `Qwen/Qwen3-Embedding-0.6B` (8B นำ MTEB), `google/embeddinggemma-300m` (เล็กบนเครื่อง)

### Text generation / แปล / สรุป
- แปล TH<->EN ของใหม่: `typhoon-ai/typhoon-translate-4b` (เก่งไทยกว่า NLLB ทั่วไป, ต้อง GPU) ; ไม่มี GPU ใช้ `facebook/nllb-200-distilled-600M`
- สรุปไทย: `thanathorn/mt5-cpe-kmutt-thai-sentence-sum` (ยังดี)
- LLM ceiling: Typhoon 2.5 / SEA-LION v4.5 (id ด้านบน)

### Outdated ที่ปรับ
- `typhoon2.1-gemma3-12b` -> `typhoon2.5-qwen3-*` | `Gemma-SEA-LION-v3-9B-IT` -> `v4/v4.5` | เพิ่ม `mmBERT` | เพิ่ม `SetFit`, `typhoon-translate-4b`

### ถ้าอยากได้คะแนนเพิ่ม (NLP)
1. backbone -> `clicknext/phayathaibert` (ไทยล้วน) หรือ `jhu-clsp/mmBERT-base` (ปนอังกฤษ)
2. text-cls data น้อย -> `SetFit` (วิธีที่ 3) | แปล -> `typhoon-translate-4b`
3. จูน epoch/lr/max_length | GPU ใหญ่ -> LLM few-shot (Typhoon 2.5 / SEA-LION v4) ผ่าน `vllm` เร็ว ๆ
4. fine-tune LLM เร็ว/VRAM น้อย -> `unsloth`
5. ensemble หลายโมเดล

---

## Multimodal (Vision-Language) — SOTA refresh (mid-2026)

กฎเหล็ก: BLEU ไทยต้องตัดคำ `newmm` ก่อนเสมอ (cross-check `pythainlp.benchmarks.bleu_score`)

### VLM (captioning/VQA) — ต้อง transformers>=4.57
- `Qwen3-VL` = default ใหม่: `Qwen/Qwen3-VL-4B-Instruct` (16GB GPU) / `Qwen/Qwen3-VL-2B-Instruct` (free-tier) -> `Qwen3VLForConditionalGeneration` ; OCR 32 ภาษารวมไทย -- ใช้แทน Qwen2.5-VL ใน visual_qa/captioning
- `OpenGVLab/InternVL3_5-8B(-HF)` = generalist VQA/doc reasoning แรงสุดคลาส 8B (แทน InternVL3)
- `google/gemma-3-4b-it` / `gemma-3-12b-it` = multilingual VLM ดี
- scale-up: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (รองรับไทย แต่ใหญ่ MoE 109B)
- เก่า (demote เป็น baseline): Florence-2, Molmo, Llama-3.2-Vision

### OCR ไทย / เอกสาร — เปลี่ยนเยอะ
- `Typhoon OCR V1.5` (2B, ฐาน Qwen3-VL) = Thai-doc SOTA (เคลมชนะ GPT-5/Gemini บนเอกสารไทย) -> `pip install typhoon-ocr` ; id โมเดลย้ายเป็น `typhoon-ai/typhoon-ocr-7b` (ของเดิม scb10x/ redirect)
- OCR generalist ใหม่: `deepseek-ai/DeepSeek-OCR` (3.3B, MIT), `rednote-hilab/dots.ocr` (layout/table/formula), `PaddlePaddle/PaddleOCR-VL` (0.96B เล็ก), `nanonets/Nanonets-OCR2-3B`, `ibm-granite/granite-docling-258M` (CPU/edge)
- CPU baseline: `EasyOCR(['th','en'])`, `PaddleOCR`

### ถ้าอยากได้คะแนนเพิ่ม (Multimodal)
1. วัด `thai_bleu()` (newmm) ให้ถูกก่อน
2. captioning/VQA อัปง่าย -> `Qwen/Qwen3-VL-4B-Instruct` zero-shot (ใช้ `qwen-vl-utils` เตรียม input)
3. OCR เอกสารไทย -> `Typhoon OCR V1.5` (จุดคะแนนกระโดดสุด)
4. fine-tune VLM เร็ว/VRAM น้อย -> `unsloth` (FastVisionModel) + QLoRA 4-bit
5. inference เยอะ ๆ เร็ว ๆ -> `vllm` / `lmdeploy` ; เอกสาร PDF -> `docling`/`marker-pdf`/`surya-ocr`

---

## Tabular — SOTA refresh (mid-2026)

### AutoGluon = AutoML ตัวหลัก (bump version)
- `pip install autogluon` -> `1.5.x` ; preset `extreme` ยังแรงสุด (รื้อใหม่ใน 1.5: portfolio = `TabPFNv2 + TabICL + Mitra + TabDPT + TabM + RealMLP + CatBoost/LightGBM/XGBoost + TabPrep-LightGBM`, win-rate 70% เหนือ 1.4)
- `extreme` ต้อง GPU ~20GB+ และ `pip install autogluon.tabular[tabarena]` ; ไม่มี GPU ใช้ preset ใหม่ `best_v150`/`high_v150` (เร็วกว่า, CPU ได้)

### Tabular foundation models
- `TabICLv2` (ICML 2026, ของใหม่ที่ควรเพิ่ม): tuning-free, Apache/permissive, sklearn API, สเกลถึงล้านแถว, ชนะ XGB/CatBoost/LGBM ~80% ของชุด -> `pip install tabicl` (`from tabicl import TabICLClassifier`)
- `TabPFN` -> `pip install tabpfn` (8.x, default = `TabPFN-3`) ; limit แก้ใหม่: v2 = 10K แถว (permissive license), 2.5/2.6 = ~50-100K, TabPFN-3 = สูงสุด ~1M (rows/feature trade-off) -- หมายเหตุ: 2.5/2.6/3  license ห้ามใช้เชิงพาณิชย์, v2 ใช้ได้
- `Mitra` (autogluon/mitra-classifier, Apache-2.0): เก่งสุดบน data เล็ก (<5K)

### Cool: cleanlab (จุดคะแนนกระโดดที่มือใหม่ลืม)
- `pip install cleanlab` -> หา "label ผิด/แถวเสีย" ด้วย `find_label_issues` แล้วลบ/แก้ก่อนเทรน มักดันคะแนนเยอะสุดในงานตาราง

### ถ้าอยากได้คะแนนเพิ่ม (Tabular)
1. `presets="best_quality"` -> `presets="extreme"` (มี GPU) ; ไม่มี GPU -> `best_v150`
2. รัน `cleanlab` ลบ label ผิดก่อนเทรน
3. เพิ่ม `TabICLv2` + `TabPFN` (v2 หรือ 2.5) แล้ว blend OOF
4. เพิ่ม `CatBoost`/`XGBoost` เพื่อ diversity ; จูนด้วย `optuna`
5. stacking/hill-climbing บน OOF

---

## Time-Series / Signal — SOTA refresh (mid-2026)

### Forecasting foundation models (top tier ใหม่)
- `amazon/chronos-2` = default zero-shot (univariate+multivariate+covariates, นำ win-rate บน fev-bench/GIFT-Eval) -> ผ่าน AutoGluon-TS `presets="chronos2"` หรือ pip `chronos-forecasting`
- `google/timesfm-2.5-200m-pytorch` (และ `-transformers`) -> `pip install timesfm`
- `NX-AI/TiRex` (xLSTM ~35M, เล็กแต่แรง, #2 fev-bench) -> `pip install tirex` (license ระวังเชิงพาณิชย์)
- `Datadog/Toto-2.0-*` family (แทน Toto-Open-Base-1.0; `-313m` default, `-2.5B` แม่นสุด) -> `pip install toto-ts`
- `Salesforce/moirai-2.0-R-small`, `thuml/sundial-base-128m`, `tabpfn-time-series` (ซีรีส์สั้น)
- AutoGluon-TS `1.5` : `presets="best_quality"` (auto-ensemble) หรือ `"chronos2"` (zero-shot)

### Classification / biosignal
- ฟีเจอร์เร็ว: `MiniRocket`/`MultiRocket`/`HYDRA` ตอนนี้อยู่ใน `aeon` (มาตรฐานใหม่ของ TSC) -> `from aeon.classification.convolution_based import MiniRocketClassifier`
- TS foundation: `paris-noah/MantisV2`/`MantisPlus` (แทน Mantis-8M), `AutonLab/MOMENT-1-large`
- EEG FM ใหม่ (ผ่าน `braindecode[hub]` API เดียว): `CBraMod` (`braindecode/cbramod-pretrained`), `REVE` (pretrain 25,000 คน) -- แทน LaBraM/EEGPT/BIOT ที่เริ่มเก่า
- ECG FM: `PKUDigitalHealth/ECGFounder` (วินิจฉัย 150 label out-of-the-box)
- sleep: `yasa` 0.7.0 (zero-shot staging) ; ECG/PPG: `neurokit2`

### ถ้าอยากได้คะแนนเพิ่ม (Time-Series)
1. classification: `MiniRocket` (aeon) แทนฟีเจอร์ spectral -> ต่อ AutoGluon/LightGBM
2. forecasting: `Chronos-2` ผ่าน AutoGluon-TS `presets="chronos2"` (zero-shot)
3. เติม `tsfresh`/`pycatch22` ฟีเจอร์ ; sleep ลอง `yasa` ก่อน
4. EEG มี GPU -> `CBraMod`/`REVE` (braindecode) ; ECG -> `ECGFounder`
5. ensemble FM + LightGBM

---

## Audio / Speech — SOTA refresh (mid-2026)

### จำแนกเสียง — วิธีง่ายแต่แรง (ใส่เป็น วิธีที่ 1.5 ในโน้ตบุ๊กแล้ว)
- zero-shot ไม่ต้องเทรน: `laion/clap-htsat-fused` (CLAP) -> `pipeline("zero-shot-audio-classification")` ใส่ชื่อคลาสเป็นข้อความ
- embeddings + classifier ง่าย ๆ (แรงกว่า MFCC เยอะ): `mispeech/ced-base` (AudioSet tagger), `mispeech/dasheng-base` (general), `laion/larger_clap_general` -> ดึง embedding แล้วต่อ LightGBM/AutoGluon
- baseline เดิม: `MIT/ast-finetuned-audioset-...` (AST), PANNs (`panns-inference`)

### Thai ASR (ถอดเสียงไทย)
- `typhoon-ai/typhoon-whisper-turbo` = สมดุลสุด (เร็ว+แม่น) ; `typhoon-ai/typhoon-whisper-large-v3` = แม่นสุด
- `typhoon-ai/typhoon-asr-realtime` = streaming/CPU (115M, ถูกกว่า ~45x)
- `nectec/Pathumma-whisper-th-large-v3`, `biodatlab/whisper-th-large-v3-combined` = Thai baseline ดี
- เร็วขึ้น: `faster-whisper` ; วัด CER/WER ด้วย `jiwer`/pythainlp
- หมายเหตุ: NVIDIA `parakeet`/`canary` แรงแต่ไม่รองรับไทย (ใช้กับอังกฤษ/EU)

### ถ้าอยากได้คะแนนเพิ่ม (Audio)
1. แทน MFCC -> `CLAP zero-shot` หรือ `CED/Dasheng embeddings` + LightGBM/AutoGluon (วิธีที่ 1.5)
2. augment ด้วย `audiomentations` (กัน overfit)
3. มี GPU -> fine-tune หัว AST/CED ; ASR ไทย -> Typhoon Whisper
4. ensemble หลาย embedder

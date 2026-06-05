# 🧰 Library เจ๋ง ๆ ที่ทำให้ชีวิตง่ายขึ้น (ต่อหัวข้อ)

รวมไลบรารีที่ "ช่วยให้ทำเร็วขึ้น/แม่นขึ้น/debug ง่ายขึ้น" สำหรับ hackathon (verify บน PyPI/HF กลางปี 2026)
แต่ละตัว: `pip id` — ช่วยอะไร — เสียบตรงไหน

## ทั่วไป / ทุกหัวข้อ (ลงไว้เลยคุ้มสุด)
- `ydata-profiling` — `ProfileReport(df)` ทำ EDA ครบ (distribution/correlation/missing/warning) บรรทัดเดียว — เซลล์แรกของงานตาราง
- `sweetviz` — `compare(train, test)` ดู drift train vs test เร็ว — ก่อนเทรน
- `cleanlab` — `find_label_issues(...)` หา label ผิด/แถวเสีย — รันก่อนเทรน มักดันคะแนนเยอะสุด
- `optuna` — จูน hyperparameter อัตโนมัติ — ครอบ LightGBM/XGB/CatBoost
- `joblib` — เซฟ/โหลดโมเดล + parallel — กัน train ใหม่ทุกครั้ง

## Computer Vision
- `ultralytics` — train/predict/export YOLO26 (detect/seg/cls/pose) บรรทัดเดียว + export ONNX/TensorRT — `object_detection`/`segmentation`/`image_classification`
- `rfdetr` — SOTA transformer detector + RF-DETR-Seg, transfer ข้ามโดเมนเก่ง — detection/seg
- `timm` — 1000+ backbone (EVA-02, ConvNeXt-V2, DINOv3) `create_model(...)` — image_classification (core) + เป็น backbone
- `transformers` — `pipeline("zero-shot-image-classification")` (SigLIP2/PE-Core), `pipeline("mask-generation")` (SAM3) — zero-shot/SAM
- `supervision` — กาว: รวมผล detect/seg, วาดกล่อง/mask, นับวัตถุใน zone, tracking — postprocess (ทุกตัว)
- `albumentations` — augmentation เร็ว รองรับ bbox/mask/keypoint — data pipeline ทุกงาน
- `fiftyone` — เปิดเบราว์เซอร์ดู dataset+prediction หา label ผิด/false positive — EDA/error analysis
- `sahi` — sliced inference สำหรับวัตถุเล็ก/รูปใหญ่ (โดรน/ดาวเทียม) — detection (เพิ่ม mAP)
- `autodistill` — auto-label โฟลเดอร์รูปด้วย foundation model แล้วเทรนตัวเล็ก — bootstrap dataset
- `segmentation-models-pytorch` — U-Net/DeepLab + timm encoder บรรทัดเดียว — semantic segmentation
- `fastai` — `vision_learner(...).fine_tune(3)` classifier แรงใน 5 บรรทัด — image_classification (baseline เร็วสุด)

## NLP (เน้นไทย)
- `setfit` — few-shot text classification (~8 ตัวอย่าง/คลาส สู้ fine-tune เต็มได้) ไม่ต้อง prompt — text_classification (วิธีที่ 3) ของแรงสุดตอน data น้อย
- `sentence-transformers` — embedding (bge-m3/embeddinggemma) สำหรับ SetFit + semantic similarity — text_classification, คู่ประโยค
- `pythainlp` — ตัดคำ/NER/POS ไทย + `pythainlp.benchmarks` (BLEU/WER/CER) — ทุกงาน NLP ไทย
- `spacy` + `spacy-pythainlp` — pipeline `nlp(text)` + displaCy — word_segmentation
- `gliner` — zero-shot NER ตั้งชื่อ entity เอง ไม่ต้องเทรน — NER ที่ label set แปลก
- `unsloth` — fine-tune LLM เร็ว 2x VRAM น้อย 70% — text_generation/LLM branch
- `vllm` — inference LLM เร็วมาก (batched) — ตอน prompt LLM เยอะ ๆ
- `simpletransformers` — `ClassificationModel`/`Seq2SeqModel` 3 บรรทัด — baseline เร็ว

## Multimodal (Vision-Language)
- `transformers>=4.57.0` — ต้องมีถึงจะโหลด Qwen3-VL (`Qwen3VLForConditionalGeneration`) + InternVL3.5 — ทั้งสองโน้ตบุ๊ก
- `qwen-vl-utils==0.0.14` — `process_vision_info(messages)` เตรียม input ให้ Qwen-VL ถูกต้อง — visual_qa/captioning
- `unsloth` — fine-tune VLM เร็ว/VRAM น้อย (FastVisionModel) ลง T4 ฟรีได้ — captioning วิธีที่ 3
- `peft` + `bitsandbytes` — LoRA/QLoRA 4-bit ให้โมเดล 8B ลง 16GB — fine-tune
- `vllm` / `lmdeploy` — inference VLM เร็ว batched — สร้าง caption/คำตอบทั้ง test set
- `typhoon-ocr` — OCR เอกสารไทย (Typhoon OCR V1.5) ออก Markdown รักษาเลย์เอาต์ — visual_qa (จุดคะแนนกระโดด)
- `docling` / `marker-pdf` / `surya-ocr` — PDF/สแกน -> Markdown/JSON — preprocess เอกสาร
- `easyocr` / `paddleocr` — OCR baseline CPU เร็ว — visual_qa

## Tabular
- `autogluon` — AutoML ตัวหลัก `fit(presets="extreme")` รวม ensemble ให้หมด — classification/regression/AUTO_SOLVER
- `tabicl` — TabICLv2 tuning-free SOTA (permissive, sklearn API, สเกลล้านแถว) — baseline แรง + ensemble
- `tabpfn` (+ `tabpfn-extensions`) — TabPFN ตารางเล็กไม่ต้องจูน — baseline + ensemble (ระวัง license 2.5+)
- `pytabkit` — RealMLP/TabM standalone — diversity ใน ensemble
- `flaml` / `mljar-supervised` / `pycaret` — AutoML เร็ว/รายงานสวย — second opinion / prototype
- `cleanlab` — หา label ผิด — ก่อนเทรน
- `category_encoders` / `feature-engine` — encode categorical + feature engineering — preprocess

## Time-Series / Signal
- `aeon` — มาตรฐานใหม่ TSC, มี MiniRocket/Rocket/HYDRA/InceptionTime — signal_classification (backbone หลัก)
- `sktime` — pipeline + cv splitter + forecasting — ทั้งสองโน้ตบุ๊ก
- `tsfresh` / `pycatch22` — สกัดฟีเจอร์ TS อัตโนมัติ — signal_classification
- `nixtla` (`statsforecast`/`mlforecast`/`neuralforecast`) — forecasting เร็วสุดสายโปรดักชัน (AutoARIMA baseline + deep models) — forecasting
- `darts` — forecasting + anomaly + backtest API เดียว — forecasting
- `autogluon.timeseries` — AutoML forecasting (Chronos-2 ในตัว) — forecasting (one-call)
- `mne` + `braindecode[hub]` — EEG I/O/filtering + EEG FM (CBraMod/REVE) API เดียว — signal_classification (EEG)
- `neurokit2` — ECG/PPG (R-peak, HRV) — signal_classification (ECG)
- `yasa` — sleep staging สำเร็จรูป — signal_classification (sleep)
- `tsai` — deep learning TS (InceptionTime ฯลฯ) — signal_classification (GPU)

## Audio / Speech
- `transformers` — `pipeline("zero-shot-audio-classification")` (CLAP), `pipeline("automatic-speech-recognition")` (Thai Whisper) — audio_classification (วิธีที่ 1.5 + ASR)
- `librosa` — โหลด/สกัดฟีเจอร์เสียง (MFCC/mel) — audio_classification (core)
- `torchaudio` — I/O + mel-spectrogram บน GPU — feed โมเดล + วิธีที่ 3
- `faster-whisper` — Whisper เร็ว 4x VRAM น้อย — ASR
- `speechbrain` — recipe ASR/speaker/emotion + BEATs features — งานเสียงขั้นสูง
- `audiomentations` — augmentation เสียง (pitch/time/noise) — กัน overfit
- `pyannote.audio` — diarization (ใครพูดเมื่อไหร่) — เสียงหลายคนก่อน ASR
- `panns-inference` / `openl3` — audio embeddings สำเร็จรูป — วิธีที่ 1.5 ทางเลือก

---

## 3 ตัวที่ลงทุก hackathon (ROI สูงสุด)
1. `cleanlab` — ลบ label ผิด (งานตาราง/รูป/ข้อความ)
2. `ydata-profiling` — EDA บรรทัดเดียว
3. `optuna` — จูน GBDT อัตโนมัติ

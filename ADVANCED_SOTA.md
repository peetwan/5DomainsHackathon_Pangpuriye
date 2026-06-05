# SOTA อัปเดต + วิธีดันคะแนน (2025–2026)

ไฟล์นี้รวม "ท่ายาก / ของใหม่" สำหรับแต่ละหัวข้อใหญ่ ถ้าทำ `วิธีที่ 1 (ง่ายสุด)` ในโน้ตบุ๊กได้ submission แล้ว และอยากดันคะแนนต่อ ให้มาดูตรงนี้
ทุกหัวข้อบอกชัดว่า: ใช้โมเดล/เทคนิคอะไร, id ไหน, ช่วยยังไง, เหนื่อยแค่ไหน, และ `แก้ไฟล์ไหน ตัวแปรไหน`

วิธีใช้: หาหัวข้อใหญ่ของคุณ -> ดู "ถ้าอยากได้คะแนนเพิ่ม ทำตามลำดับนี้" ท้ายแต่ละหัวข้อ -> ทำจากบนลงล่าง (ผลตอบแทนต่อแรงมากสุดอยู่บน)

> หมายเหตุ: id ทั้งหมดตรวจกับ HuggingFace/เอกสารทางการแล้ว แต่ของหนัก (VLM/foundation model) ต้องรันบน Colab/Kaggle ที่มี GPU เสมอ

---

## Computer Vision — Advanced / SOTA (อัปเดต)

หมายเหตุ: ทุกอย่างด้านล่างคือ "ของจริงที่ใช้ได้ปี 2025–2026" และเลือกเฉพาะที่ plug เข้ากับ notebook เดิมได้เร็ว ตัวเลข top-1 อ้างอิงจาก timm/HF model cards และ COCO benchmark ของผู้พัฒนา

### (a) Image Classification — `image_classification.ipynb`

จุดที่แก้ง่ายสุดคือ `วิธีที่ 2` (timm) แค่เปลี่ยน `MODEL_NAME` (ปัจจุบัน `"tf_efficientnetv2_s.in21k_ft_in1k"`) + ปรับ `IMG_SIZE` ให้ตรงกับโมเดล timm จะ auto-download น้ำหนักให้เอง

| เทคนิค/โมเดล | id (ใส่ใน `MODEL_NAME`) | ทำไมช่วย | effort | แก้ตรงไหน |
|---|---|---|---|---|
| `ConvNeXt-V2-Large` | `convnextv2_large.fcmae_ft_in22k_in1k` (ใช้ `_384` ถ้า GPU ไหว → top-1 88.2%) | อัปจาก EfficientNetV2-S ทันที, top-1 ~87–88% สาย CNN เทรนง่าย เสถียร | ต่ำ | `MODEL_NAME` ใน `วิธีที่ 2`; ถ้าใช้ `_384` ตั้ง `IMG_SIZE = 384` |
| `EVA-02-Large @448` (แชมป์ accuracy) | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (top-1 90.05%) | จุดสูงสุด accuracy ที่ใช้ได้จริงบน timm | กลาง (กิน VRAM, `IMG_SIZE=448`, batch เล็ก) | `MODEL_NAME` + `IMG_SIZE = 448` |
| `DINOv3` backbone (ใหม่สุด ส.ค. 2025) | `vit_large_patch16_dinov3.lvd1689m` (ต้อง `timm>=1.0.20`) | ฟีเจอร์ self-supervised เทพ; เก่งมากตอน data น้อย/โดเมนแปลก (frozen + linear head ก็ได้) | ต่ำ–กลาง | `MODEL_NAME` ใน `วิธีที่ 2` |
| `SigLIP2` (zero-shot ไม่ต้องเทรน) | `google/siglip2-so400m-patch14-384` (zero-shot ImageNet 84.1%) | ได้ baseline แบบ 0 epoch ด้วย text labels; ดีตอนเวลาน้อย/label น้อย | ต่ำ | เพิ่ม cell ใช้ `pipeline("zero-shot-image-classification", model=...)` |

เทคนิคเทรน (เปิดใน `วิธีที่ 2` ได้): `MixUp + CutMix` (timm.data.Mixup) กัน overfit +0.5–1.5%, `EMA` (timm.utils.ModelEmaV3) +0.3–0.8%, `TTA` ตอน inference +0.3–1%, `label smoothing 0.1` + cosine LR

### (b) Object Detection — `object_detection.ipynb`

ของเดิม `YOLO("yolov8n.pt")`. ลำดับอัป: เปลี่ยนรุ่น YOLO (effort เกือบ 0) → ขยายไซซ์ → RF-DETR ถ้าต้องการ accuracy สูงสุด

| เทคนิค/โมเดล | id | ทำไมช่วย | effort | แก้ตรงไหน |
|---|---|---|---|---|
| `YOLO11` | `model = YOLO("yolo11n.pt")` (n/s/m/l/x) | ใหม่กว่า v8, mAP สูงขึ้นที่พารามิเตอร์ใกล้กัน, API เหมือนเป๊ะ | แทบ 0 | เปลี่ยน `YOLO("yolov8n.pt")` → `YOLO("yolo11n.pt")` |
| `YOLO12` (attention-centric) | `model = YOLO("yolo12n.pt")` | เพิ่ม attention ดัน accuracy ต่อ | แทบ 0 | เปลี่ยนชื่อรุ่นใน `YOLO(...)` |
| ขยาย scale + `imgsz` | `YOLO("yolo11m.pt")` + `train(imgsz=1024)` | ขยับ `n→s→m` + เพิ่ม `imgsz` คือวิธีดันคะแนนที่ชัวร์สุด | ต่ำ | args ใน `model.train(...)` |
| `RF-DETR` (SOTA บน COCO, real-time) | `pip install rfdetr` → `from rfdetr import RFDETRNano` | transformer detector ชนะ YOLO ด้าน accuracy; รับ dataset YOLO/COCO; Apache-2.0 | กลาง | สร้าง cell ใหม่แทน block ultralytics |
| `RT-DETRv2` | `from ultralytics import RTDETR; RTDETR("rtdetr-l.pt")` | NMS-free, แม่นกว่าที่ฉากซับซ้อน, API เดิม | ต่ำ–กลาง | ใช้ `RTDETR` แทน `YOLO` |

ทริค: เทรนยาว + `imgsz` ใหญ่, เปิด `augment=True` ตอน predict (TTA), ensemble หลาย scale รวมกล่องด้วย `WBF` (`pip install ensemble-boxes`)

### (c) Segmentation — `segmentation.ipynb`

| เทคนิค/โมเดล | id | ทำไมช่วย | effort | แก้ตรงไหน |
|---|---|---|---|---|
| `YOLO11-seg` / `YOLO12-seg` | `YOLO("yolo11n-seg.pt")` | drop-in แทน v8-seg, mask แม่นขึ้น | แทบ 0 | เปลี่ยน `YOLO("yolov8n-seg.pt")` → `YOLO("yolo11n-seg.pt")` |
| ขยาย scale + imgsz | `YOLO("yolo11m-seg.pt")` + `imgsz=1024` | ดัน mask mAP ตรง ๆ | ต่ำ | args `model.train(...)` |
| `SAM2` (promptable, zero-shot) | `from ultralytics import SAM; SAM("sam2.1_b.pt")` | mask คุณภาพสูงสุดไม่ต้องเทรน; ใช้ box จาก detector เป็น prompt | ต่ำ–กลาง | เพิ่ม cell: box จาก YOLO → prompt ให้ SAM2 |

ทริค: TTA flip/multi-scale, `copy_paste=0.3` ใน ultralytics, pipeline `detector → SAM2 refine mask`

### ถ้าอยากได้คะแนนเพิ่ม ทำตามลำดับนี้ (CV)

1. เปลี่ยนรุ่นฟรี ๆ ก่อน: detection/seg `yolov8*` → `yolo11*`/`yolo12*`; classification `MODEL_NAME` → `convnextv2_large.fcmae_ft_in22k_in1k`
2. ขยาย scale + เพิ่ม `IMG_SIZE`/`imgsz` เท่าที่ GPU ไหว
3. เปิดทริคเทรน: `MixUp/CutMix + EMA + label smoothing` (cls); `augment/copy_paste` (det/seg)
4. `TTA` ตอน inference ทุก task
5. อัปตัวสุด: cls → `eva02_large_patch14_448...` (90%) หรือ frozen `DINOv3`; det → `RF-DETR`; seg → `detector → SAM2`
6. `Ensemble` ปิดท้าย: cls เฉลี่ย logits หลายโมเดล; det รวมกล่องด้วย `WBF`

---

## NLP — Advanced / SOTA (อัปเดต)

หมายเหตุ: ทุก id ตรวจบน HuggingFace แล้ว (2025–2026) วางทับของเดิมได้ แก้แค่ตัวแปรตามที่ระบุ

### (a) ตัดคำ / NER / POS — `thai_word_segmentation.ipynb`

ตัดคำ:
- `nlpo3` engine (`pip install nlpo3`) — newmm เขียนใหม่ด้วย Rust เร็วกว่า ~2.5x ผลตัดเหมือน newmm — แก้ที่ `to_labels(text, engine="deepcut")` วิธีที่ 1 เปลี่ยนเป็น `engine="nlpo3"` (ถ้าต้องการแม่นสุดคง `deepcut`)
- ensemble ง่าย: รัน `deepcut` + `attacut` + CRF แล้ว vote ขอบคำต่อตัวอักษร

NER/POS (token classification, วิธีที่ 3):
- `clicknext/phayathaibert` — WangchanBERTa ที่ขยาย vocab + เทรนต่อ ชนะเกือบทุกงาน โดยเฉพาะข้อความปนอังกฤษ (drop-in แทนได้ สถาปัตยกรรม camembert เหมือนกัน) — แก้ที่ค่า `MODEL` วิธีที่ 3
- NER สำเร็จรูป: `Pavarissy/phayathaibert-thainer`; POS สำเร็จรูป: `lunarlist/pos_thai_phayathai` (zero-train predict ได้เลย)
- multilingual: `microsoft/mdeberta-v3-base` (Thai XNLI 76.4 > xlm-roberta-base 74.6) หรือ `FacebookAI/xlm-roberta-large`

### (b) จำแนกข้อความ — `text_classification.ipynb`

วิธีที่ 2 ของเดิม `MODEL="airesearch/wangchanberta-base-att-spm-uncased"` → level-up:
- `clicknext/phayathaibert` — monolingual ไทยที่แรงสุดสำหรับ fine-tune ตอนนี้ — `เปลี่ยน MODEL = "clicknext/phayathaibert"`
- `microsoft/mdeberta-v3-base` — ถ้าข้อความปนอังกฤษ/เป็นข้อความแปลเยอะ
- `FacebookAI/xlm-roberta-large` — พารามิเตอร์เยอะ มักดันคะแนนสุด (ลด batch=8 กัน OOM)
- LLM zero/few-shot (เพิ่มวิธีที่ 3): prompt `typhoon-ai/typhoon2.1-gemma3-12b` / `Qwen/Qwen2.5-7B-Instruct` / `aisingapore/Gemma-SEA-LION-v3-9B-IT` ให้แปะ label — ดีตอน train เล็ก
- ฟรีคะแนน (วิธีที่ 1): เปลี่ยน `engine="newmm"` → `"nlpo3"` ใน `th_tokenize`, ลอง `LinearSVC` แทน LogReg
- งานจับคู่ประโยค/semantic: `kornwtp/SCT-model-phayathaibert` หรือ `BAAI/bge-m3` (multilingual, ไทยดีมาก)

### (c) สร้าง/แปลงข้อความ Seq2Seq — `text_generation.ipynb`

วิธีที่ 1 (pipeline) เปลี่ยน `MODEL` ตามงาน:
- แปลภาษา: `facebook/nllb-200-distilled-600M` (เบา) / `facebook/nllb-200-3.3B` (แม่นสุด) — SOTA แปล low-resource รวมไทย ต้องตั้ง `src_lang="tha_Thai"`, `tgt_lang="eng_Latn"`
- สรุปไทย: `thanathorn/mt5-cpe-kmutt-thai-sentence-sum` (fine-tune ไทยแล้ว) หรือ `csebuetnlp/mT5_multilingual_XLSum` (เดิม)
- prompt LLM (มักได้สูงสุดถ้าวัด generative): `typhoon-ai/typhoon2.1-gemma3-12b` / `aisingapore/Gemma-SEA-LION-v3-9B-IT` / `Qwen/Qwen2.5-7B-Instruct`

วิธีที่ 2 (fine-tune) `MODEL="google/mt5-small"` → `google/mt5-base` (แม่นกว่า); งานแปลใช้ `facebook/nllb-200-distilled-600M` (ใส่ `forced_bos_token_id`); ภาษาไม่มีเว้นวรรค/สะกดผิด ลอง `google/byt5-small`

### ถ้าอยากได้คะแนนเพิ่ม ทำตามลำดับนี้ (NLP)

1. เปลี่ยน backbone เป็น `clicknext/phayathaibert` ทุกที่ที่ใช้ WangchanBERTa (text_classification วิธีที่ 2, word_segmentation วิธีที่ 3)
2. งานแปล: pipeline `facebook/nllb-200-distilled-600M` + ตั้ง `src_lang/tgt_lang`
3. งานสรุป: `thanathorn/mt5-cpe-kmutt-thai-sentence-sum`
4. จูน: `num_train_epochs` 3→4–5, `learning_rate` 1e-5/2e-5/3e-5, `max_length` 256→384
5. GPU ใหญ่: ดันเป็น `mt5-base` / `xlm-roberta-large` (ลด batch)
6. LLM zero/few-shot (Typhoon 2.1 / SEA-LION v3 / Qwen2.5) เป็นตัวเทียบเพดาน + ใช้จริงงาน generative
7. Ensemble: เฉลี่ย/โหวต 2–3 โมเดล (phayathaibert + mdeberta-v3; deepcut + nlpo3 + CRF)

---

## Multimodal (Vision-Language) — Advanced / SOTA (อัปเดต)

กฎเหล็กการวัด: BLEU ไทยต้องตัดคำ `newmm` ก่อนเสมอ (`thai_bleu()` ในโน้ตบุ๊กทำถูกแล้ว) cross-check ด้วย `pythainlp.benchmarks.bleu_score(refs, hyps, tokenize="newmm")` วัดในเครื่องก่อนส่งทุกครั้ง

### A. Image Captioning ไทย — `thai_image_captioning.ipynb`

ลำดับความแรง: วิธีที่ 1 (pretrained) < BLIP < PaliGemma2+LoRA < `Qwen2.5-VL / Qwen3-VL` fine-tune

1. `Qwen/Qwen2.5-VL-7B-Instruct` (เบา `-3B-`) — VLM แรง รองรับไทย OCR ดี สั่งเป็นไทยได้ "อธิบายภาพนี้เป็นไทยสั้น ๆ 1 ประโยค" zero-shot แข็งกว่าวิธีที่ 1 — แก้ที่ วิธีที่ 1 เปลี่ยนเป็น chat template แบบ Qwen ใน `visual_qa.ipynb` แล้วเก็บผลใส่ `pred[...]` → `write_submission(pred)`
2. `Qwen/Qwen3-VL-4B-Instruct` (มี `-2B-`/`-8B-`) — ใหม่สุดปลายปี 2025 ดีขึ้นทุกด้าน `2B` รัน free-tier ได้ — โหลดด้วย `Qwen3VLForConditionalGeneration`
3. PaliGemma2 (วิธีที่ 3): ใช้ `google/paligemma2-3b-mix-448` (instruction-tuned แล้ว zero-shot ได้) หรือ `-pt-448` (คมขึ้น) prompt `"caption th\n"`
4. QLoRA 4-bit fine-tune (`peft` + `bitsandbytes`, `r=8-16, alpha=16-32`) — ปรับ VLM 3B-8B บน GPU 16GB, เทรน <1% params ได้ BLEU ขยับ
5. เสริม: `florence-community/Florence-2-large` (ไม่ต้อง trust_remote_code, caption อังกฤษ แปลทีหลัง), vision encoder `google/siglip2-so400m-patch14-384`

### B. Visual QA / OCR ไทย — `visual_qa.ipynb`

1. เลื่อน `Qwen2.5-VL` เป็นวิธีหลัก: ปลดคอมเมนต์บล็อก Qwen (ทางเลือก) เปลี่ยนเป็น `-7B-` ถ้าไหว — BLIP-VQA ตอบไทยไม่ได้/OCR อ่อน, Qwen ตอบไทย + อ่านตัวอักษรได้
2. `Qwen/Qwen3-VL-4B-Instruct` (`-2B-`) — DocVQA/ChartVQA ไทยดีขึ้น
3. OCR ไทยเฉพาะทาง:
   - `scb10x/typhoon-ocr-7b` (`pip install typhoon-ocr`) — document parsing ไทย-อังกฤษ ออก Markdown/HTML รักษาเลย์เอาต์ ตัวท็อปสำหรับเอกสารไทย
   - `stepfun-ai/GOT-OCR-2.0-hf` — OCR generalist เล็ก เร็ว (716M) ไม่ต้อง trust_remote_code
   - `scb10x/typhoon2-qwen2vl-7b-vision-instruct` — VQA ไทยที่ต้องเข้าใจบริบท
   - `EasyOCR` (`['th','en']`) / `PaddleOCR` — baseline CPU เร็ว
4. VLM open อื่น: `OpenGVLab/InternVL3-8B`, `meta-llama/Llama-3.2-11B-Vision-Instruct` (รองรับ th), `allenai/Molmo-7B-D-0924`

### ถ้าอยากได้คะแนนเพิ่ม ทำตามลำดับนี้ (Multimodal)

1. วัดให้ถูก: ยืนยัน `thai_bleu()` (newmm) cross-check `pythainlp.benchmarks.bleu_score(..., tokenize="newmm")`
2. รัน วิธีที่ 1 เดิมให้ได้ baseline 1 ไฟล์
3. Captioning อัปง่าย: `Qwen/Qwen2.5-VL-3B-Instruct` zero-shot สั่งเป็นไทย วัด BLEU เทียบ
4. VQA/OCR อัปง่าย: ปลดคอมเมนต์ Qwen ใน `visual_qa.ipynb` ใช้แทน BLIP-VQA
5. OCR เอกสารไทย: เสียบ `scb10x/typhoon-ocr-7b` (จุดที่คะแนนกระโดดเยอะสุด)
6. ดันสุด (GPU 16GB + train JSON): QLoRA fine-tune `Qwen2.5-VL-7B` / `PaliGemma2-3b-pt-448`
7. ของใหม่สุด: `Qwen/Qwen3-VL-4B-Instruct` (`2B` บนเครื่องเล็ก)
8. รีดท้าย: ensemble (Typhoon OCR + Qwen3-VL) หรือ few-shot แล้วเลือกไฟล์ `thai_bleu()` สูงสุด

---

## Tabular — Advanced / SOTA (อัปเดต)

ภาพรวม 2025-2026 (TabArena, NeurIPS 2025): tabular foundation models (in-context learning) มาแรงสุดสำหรับข้อมูลเล็ก-กลาง — `TabPFNv2`, `TabPFN-2.5` (~50k แถว), `TabICL` (classification, ~500k), `Mitra`; GBDT (`CatBoost` นำ) ยังเป็นกระดูกสันหลัง; และ `weighted ensemble` ของหลายตระกูลชนะโมเดลเดี่ยวเสมอ

### Level-up 1 — AutoGluon preset `extreme` (ฟรีเกือบทุกอย่างในคำสั่งเดียว)
AutoGluon 1.4+ preset `extreme` รวม `TabPFNv2 + TabICL + Mitra + TabM + RealMLP + CatBoost/LightGBM/XGBoost` แล้ว ensemble ให้ (1.5 เพิ่ม `TabPFN-2.5` + `TabDPT`) — `pip install -U "autogluon.tabular[all]"` — แก้ที่ cell-9 ทั้ง classification/regression เปลี่ยน `presets="best_quality"` → `presets="extreme"` (ถ้าเวอร์ชันเก่าไม่รู้จัก ให้อัปก่อน หรือ fallback `"best_quality"`)

### Level-up 2 — เพิ่ม `TabPFN v2` แล้ว blend (จุดทองสำหรับ heart disease n<10k)
`pip install tabpfn` (รีดเพิ่ม `pip install "tabpfn-extensions[post_hoc_ensembles]"`) — SOTA ตารางเล็ก (≤10k แถว, ≤500 ฟีเจอร์) มักเก่งคนละทางกับ GBDT blend แล้ว +0.5-2 จุด — เพิ่มเป็นเซลล์ใหม่ต่อท้ายวิธีที่ 2 ใน `classification.ipynb`:

```python
# === เซลล์ใหม่: TabPFN v2 (OOF) + blend ===  (classification.ipynb วิธีที่ 2)
# !pip -q install tabpfn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
Xn  = X.apply(lambda c: c.cat.codes if str(c.dtype)=="category" else c).astype("float32")
Xten= Xte.apply(lambda c: c.cat.codes if str(c.dtype)=="category" else c).astype("float32")
oof_tp=np.zeros(len(y)); pred_tp=np.zeros(len(Xten))
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(Xn,y):
    clf=TabPFNClassifier(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    clf.fit(Xn.iloc[tr],y[tr])
    oof_tp[va]=clf.predict_proba(Xn.iloc[va])[:,1]; pred_tp+=clf.predict_proba(Xten)[:,1]/5
print("TabPFN OOF AUC =", round(roc_auc_score(y,oof_tp),4))
best_w,best=0.5,0
for w in np.linspace(0,1,21):
    s=roc_auc_score(y, w*oof_tp+(1-w)*oof)   # oof = LightGBM OOF (หรือ predictor.predict_proba_oof() ของ AutoGluon)
    if s>best: best,best_w=s,w
print(f"best blend w(TabPFN)={best_w:.2f} OOF AUC={best:.4f}")
out=sample.copy(); out[TARGET]=best_w*pred_tp+(1-best_w)*pred   # AUC ส่ง prob
out.to_csv("submission_blend.csv",index=False)
```
blend กับ AutoGluon ตรง ๆ: ดึง OOF ด้วย `predictor.predict_proba_oof()` (ต้อง fit ด้วย preset ที่ทำ bagging) แล้วใช้แทน `oof`

### Level-up 3 — `TabICL` (classification ตารางใหญ่ n หลายหมื่น-แสน)
`pip install tabicl` — foundation model classification สเกล ~500k แถว เร็วกว่า TabPFNv2 ~10x และชนะบนชุด >10k — drop-in แทน `TabPFNClassifier` (`from tabicl import TabICLClassifier`) (regression ใช้ไม่ได้)

### Level-up 4 — `RealMLP` / `TabM` (deep tabular เพิ่ม diversity)
`pip install pytabkit` (`RealMLP_TD_Classifier`/`Regressor`) — ต่างทางจาก GBDT/TabPFN, blend แล้วช่วย, ใช้ได้ทั้ง cls/reg — หรือได้มาฟรีใน AutoGluon `extreme`

### Level-up 5 — เพิ่ม `CatBoost` + `XGBoost` แล้ว blend
`pip install catboost xgboost` — CatBoost เก่งสุดกลุ่ม default จัดการ categorical native; 3 ตระกูล GBDT error คนละทาง blend แล้วได้เพิ่ม — ก็อปลูป KFold เดิม เก็บ `oof_cat/oof_xgb` ไป blend

### Level-up 6 — จูน GBDT ด้วย `Optuna` (50+ trials)
`pip install optuna` — ครอบ `LGBMClassifier` ด้วย study หา `num_leaves, learning_rate, min_child_samples, reg_lambda, feature_fraction` โดยใช้ OOF เป็น objective

### Level-up 7 — `Hill climbing` / `Stacking` (เทคนิคชนะ Kaggle 2025)
รวบ OOF ทุกตัว (`oof, oof_tp, oof_cat, oof_ag`) → (ก) hill climbing เติมโมเดลทีละตัวพร้อม weight เก็บเฉพาะที่ทำ OOF ดีขึ้น, หรือ (ข) `Ridge().fit(oof_matrix, y)` เป็น meta-model

ข้อควรระวัง: blend/stacking ต้องทำบน OOF เดียวกัน (fold/seed เดียว) ไม่งั้น leak; fit Imputer/Scaler/SMOTE ในแต่ละ fold; เลือก submission จาก CV OOF ไม่ใช่ public LB; TabPFNv2 จำกัด ≤10k แถว (เกินใช้ TabPFN-2.5 / TabICL)

### ถ้าอยากได้คะแนนเพิ่ม ทำตามลำดับนี้ (Tabular)

1. `presets="best_quality"` → `presets="extreme"` (cell-9 ทั้งสองไฟล์) — ผลตอบแทนต่อแรงสูงสุด
2. เพิ่ม `TabPFN v2` blend OOF (heart disease n<10k คือจุดแข็ง)
3. เพิ่ม `CatBoost` (+ `XGBoost`) เพื่อ diversity
4. เปลี่ยนเฉลี่ยตรง ๆ เป็น `hill climbing` / `Ridge stacking` บน OOF
5. จูน GBDT ด้วย `Optuna` 50+ trials
6. ข้อมูลใหญ่ (>10k): TabPFN → `TabICL`; เพิ่ม `RealMLP`
7. regression/imbalanced: ตรวจ metric, multi-seed averaging ก่อน stacking หลายชั้น

---

## Time-Series / Signal — Advanced / SOTA (อัปเดต)

### A) Signal / Time-Series Classification

`MiniRocket` — easy big win
- `pip install sktime` — `from sktime.classification.kernel_based import RocketClassifier` (หรือ `MiniRocket` transformer)
- แปลง raw window เป็นฟีเจอร์ด้วย convolution สุ่มตายตัว เร็วมาก (CPU ไหว) แล้วต่อ `RidgeClassifierCV` แม่นระดับ near-SOTA บน UCR ดีกว่าฟีเจอร์ spectral เดิมเกือบทุกครั้ง
- แก้ที่ `signal_classification.ipynb` cell `[9]`: ป้อน raw window เข้า `MiniRocket().fit_transform(X)` แล้วเสียบเป็น `Xtr` ของ cell `[11]` (AutoGluon) หรือ `[13]` (LightGBM + GroupKFold) — โครง CV เดิมไม่ต้องแก้
- ตัวพี่: `MultiRocket`, `HYDRA`

`Catch22 / tsfresh` (feature-based เสริม): `pip install pycatch22` (22 ฟีเจอร์เด็ด เร็ว) หรือ `tsfresh` (+`select_features`) — เติมต่อท้ายฟีเจอร์ spectral ใน cell `[9]`

`Mantis-8M` (TS classification foundation model): `pip install mantis-tsfm` / HF `paris-noah/Mantis-8M` — ViT-based 8M เบา ดึง embedding zero-shot หรือ fine-tune — เพิ่ม "วิธีที่ 4" ใต้ cell `[13]`

`MOMENT` (general TS foundation model): `pip install momentfm` / HF `AutonLab/MOMENT-1-large` — embedding zero-shot/linear-probe (univariate, กิน VRAM)

EEG เฉพาะทาง (ถ้าโจทย์ EEG/sleep): `LaBraM` (HF `braindecode/labram-pretrained`), `EEGPT`, `BIOT`, และทางลัด `YASA` (`pip install yasa`, sleep staging สำเร็จรูป ~87% ไม่ต้องเทรน) — ใช้แทน 1D-CNN (วิธีที่ 3)

baseline academic: `HIVECOTEV2` (sktime) แม่นสุด UCR แต่ช้า; backbone สมัยใหม่ `InceptionTime`, `PatchTST`

### B) Forecasting

`Chronos-Bolt` / `Chronos-2` — zero-shot upgrade (แนะนำสุด)
- ผ่าน AutoGluon: `pip install -U "autogluon.timeseries>=1.5"` / HF `amazon/chronos-bolt-base`, `amazon/chronos-2`
- Chronos-Bolt zero-shot แม่น เร็วกว่า Chronos ~250x (CPU ได้); Chronos-2 (Dec 2025) รองรับ covariates นำ leaderboard
- แก้ที่ `forecasting.ipynb` cell `[11]`: `TimeSeriesPredictor(...).fit(tsdf, presets="chronos2")` (หรือ `"bolt_base"`) + อัป AG `>=1.5` ที่ cell `[3]`

`TimesFM 2.5` (Google): `pip install timesfm` / HF `google/timesfm-2.5-200m-transformers` — decoder-only forecasting zero-shot แข็ง long-horizon — เพิ่ม "วิธีที่ 3" ใต้ cell `[11]`, ensemble กับ LightGBM

`TabPFN-TS` (ซีรีส์สั้น/น้อย): `pip install tabpfn-time-series` — แปลง forecasting เป็น tabular regression + TabPFN-v2 zero-shot

`AutoGluon-TimeSeries 1.5` (ตัวรวมร่าง): รวม Chronos-2/Toto/stack ensembling/covariate regressor — เปลี่ยน preset อย่างเดียวได้คะแนนเพิ่ม

### ถ้าอยากได้คะแนนเพิ่ม ทำตามลำดับนี้ (Time-Series)

1. `MiniRocket` แทนฟีเจอร์ spectral ใน `signal_classification.ipynb` cell `[9]` → ต่อ `[11]`/`[13]` (classification คะแนนเด้งสุด)
2. `Chronos-Bolt`/`Chronos-2` ใน `forecasting.ipynb` cell `[11]` ผ่าน AutoGluon `presets="chronos2"` (อัป AG `>=1.5`)
3. เติม `pycatch22` + `tsfresh` เข้าฟีเจอร์ cell `[9]` รวมกับ MiniRocket
4. sleep โดยตรง: ลอง `YASA` zero-shot เทียบก่อน
5. `Mantis-8M` / `MOMENT` เป็นวิธีที่ 4 (embedding + classifier เดิม, มี GPU)
6. Forecasting เสริม: `TimesFM 2.5` / `TabPFN-TS` แล้ว ensemble กับ LightGBM + Chronos
7. ensemble หลายโมเดลปิดท้ายทั้งสอง notebook (มักได้อีก 1-2%)
8. EEG ที่มี GPU/เวลา: fine-tune `LaBraM`/`EEGPT`/`BIOT` แทน 1D-CNN

# สิ่งที่ทดสอบแล้วจริง (Verification)

อัปเดต: 2026-06-05 | ทดสอบบน numpy 2.0.2 / scikit-learn 1.6.1 / lightgbm 4.6.0 / scipy 1.13.1
(เวอร์ชันใกล้เคียง Google Colab / Kaggle ปัจจุบัน)

## 1. Master Router ทำงานจริง (ไทย + อังกฤษ)
- ทดสอบ `34 โจทย์` (ทั้งไทยและอังกฤษ) -> route ถูกหมด `34/34`
- รันโค้ด router ที่ฝังในโน้ตบุ๊ก `00_MASTER_ROUTER.ipynb` จริง -> ชี้ไฟล์ถูกทุกตัวอย่าง
- แก้ปัญหา false match เช่น `ner` ไปโดน `ge-ner-ate` แล้ว (อังกฤษ match แบบมีขอบคำ, ไทย match แบบ substring)
- รันซ้ำได้: `python3 _build/verify_router.py`

## 2. โค้ดส่วนที่รันได้จริง — ทดสอบด้วยข้อมูลจำลอง (ผ่านหมด)
รันจริงด้วย `python3 _build/verify_logic.py` ครอบคลุม:
- helper ตัดคำ (words_to_char_labels / char_labels_to_words / boundary_f1)
- สกัดฟีเจอร์สัญญาณ (Welch PSD, Hjorth, band power) — ใช้ `np.trapezoid` รองรับ numpy 2.0
- LightGBM multiclass + `StratifiedGroupKFold` (แบ่งตาม subject)
- LightGBM classification + `early_stopping` + `StratifiedKFold`
- LightGBM regression + RMSE แบบ version-safe (`np.sqrt(mean_squared_error)` เพราะ sklearn 1.6 เอา `squared=False` ออก)
- forecasting lag/date features
- TF-IDF + Logistic Regression pipeline + cross-validation

## 3. API ไลบรารีหนัก — ตรวจกับเอกสารจริง (Context7)
รันเองในเครื่องนี้ไม่ได้ (ต้อง GPU/ดาวน์โหลดโมเดลใหญ่) แต่ตรวจ API กับเอกสารทางการแล้ว:
- `AutoGluon` TabularPredictor(label, problem_type, eval_metric).fit(..., time_limit) + MultiModalPredictor(label, path).fit + TimeSeriesPredictor — ตรงเอกสาร
- `Ultralytics` YOLO("yolov8n.pt"/"yolov8n-seg.pt").train(data=yaml, epochs, imgsz) + predict -> boxes.xyxy/conf/cls, masks — ตรงเอกสาร
- `transformers` Trainer / Seq2SeqTrainer / pipeline(visual-question-answering, summarization, translation) — ตรง API ปัจจุบัน

## 4. ทุกโน้ตบุ๊ก compile ผ่าน
- 13/13 notebooks เป็น JSON ถูกต้อง + ทุก code cell `compile()` ผ่าน 0 error
- รันซ้ำ: ดู `_build/` แล้วรัน generator ใหม่

## ข้อควรรู้ (ตรงไปตรงมา)
- โน้ตบุ๊กที่ใช้ของหนัก (AutoGluon, timm, YOLO, transformers, VLM) ควรรันบน `Colab/Kaggle` ที่มี GPU
  ในเครื่องนี้ไม่มี GPU/ไม่ได้ลงโมเดลใหญ่ จึงตรวจระดับ API + รันตรรกะหลักด้วยข้อมูลจำลองแทน
- โจทย์ `detection / segmentation / forecasting` รูปแบบ submission หลากหลายมาก
  โค้ดเตรียมโครงให้ + คอมเมนต์จุดที่ต้องปรับ (`# << แก้`, `# TODO`) ตามรูปแบบจริงของ comp
- `transformers` ถ้าเวอร์ชันเก่ากว่า 4.46 ให้เปลี่ยน `eval_strategy` -> `evaluation_strategy` (มีคอมเมนต์เตือนในเซลล์แล้ว)

## รันการตรวจทั้งหมดอีกครั้ง
```bash
cd 5Domains
python3 _build/verify_router.py     # router 34 เคส ไทย+อังกฤษ
python3 _build/verify_logic.py      # ตรรกะหลักด้วยข้อมูลจำลอง
bash   _build/run_checks.sh         # รันทั้งหมด + syntax check
```

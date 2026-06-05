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

## จำลองสอบจริง End-to-End (สำคัญสุด) — `python _build/simulate.py`

สร้างชุดข้อมูลปลอมหน้าตาเหมือน Kaggle (train.csv/test.csv/sample_submission.csv + รูป) แล้ว `exec โค้ดในโน้ตบุ๊กจริง`
ตั้งแต่โหลดข้อมูล -> เทรน -> สร้าง submission แล้วตรวจว่า submission ตรงรูปแบบ sample_submission
(ไลบรารีหนัก autogluon/timm/pythainlp ถูก mock เพราะหนัก/ต้องเน็ต ; lightgbm/sklearn/scipy/torch รันจริง)

ผลลัพธ์: ผ่าน 13/13 (8 หัวข้อ + AUTO SOLVER 5 แบบ) ได้ submission ถูกคอลัมน์ + จำนวนแถวตรง + ไม่มี NaN
- `00_AUTO_SOLVER.ipynb` (ปุ่มเดียวจบ) เดาประเภทงานถูกทั้ง 5: tabular-clf, regression, text, image, multiclass-proba แล้วสร้าง submission ถูกฟอร์แมต
- forecasting ทดสอบกับข้อมูลที่มีฟีเจอร์เพิ่ม (เทสบั๊ก KeyError ที่ audit เจอ) -> ผ่าน

รอบ audit + polish (All-in-one): แก้บั๊กที่ audit agent เจอ
- forecasting recursive เคย KeyError ถ้า train มีฟีเจอร์อื่นนอกจาก date/target -> แก้ให้เอาค่าฟีเจอร์จากแถว test
- AUTO_SOLVER เคย crash ถ้า test.csv ชื่ออื่น -> เพิ่ม assert บอกให้แก้ชื่อ
- เซลล์ดึง metric: kaggle เวอร์ชันใหม่คืน response ไม่ใช่ list -> แก้ใช้ `.competitions` + ชื่อ attribute `evaluation_metric`
เพิ่ม (All-in-one): seed_everything, auto-EDA, ส่ง baseline ก่อน, คะแนน validation ในเครื่อง, blend หลายไฟล์, หัวข้อใหม่ Audio,
`PLAYBOOK.md` (ลำดับทำงาน), `TROUBLESHOOTING.md` (error->แก้)
- Tabular classification (ทั้ง accuracy และ AUC->ส่ง prob), Tabular regression
- Text classification, Word segmentation (1 แถวต่อตัวอักษร 34 แถวถูกต้อง)
- Signal classification (แบ่ง CV ตาม subject), Forecasting (recursive ไม่มี NaN), Image classification

แปลว่า "เจอข้อสอบแล้วเปิดโน้ตบุ๊ก แก้ config แล้วได้ submission ส่งได้จริง" สำหรับทุกแนวที่ submission เป็นตาราง

## ความพร้อม + ข้อจำกัดที่เหลือ (ตรงไปตรงมา)

พร้อมใช้สอบจริง:
- 8 หัวข้อหลักรันจบ end-to-end ได้ submission ถูกฟอร์แมต
- เพิ่ม "ตรวจฟอร์แมตก่อนส่งอัตโนมัติ" ในทุกโน้ตบุ๊ก (เทียบคอลัมน์/จำนวนแถวกับ sample แล้ว assert กันส่งผิดได้ 0)

ยังต้องปรับตามโจทย์จริง (เป็นธรรมชาติของงาน มีคำเตือน+ตัวอย่างให้แล้ว):
- detection / segmentation: ต้องแปลงข้อมูลเป็นรูปแบบ YOLO เอง + submission (box/RLE) ต่างกันทุก comp -> ยากสุด
- captioning / VQA: รันได้แต่ต้องโหลดโมเดลใหญ่ + GPU (เปิด GPU บน Colab/Kaggle)
- AutoGluon ครั้งแรกติดตั้ง ~5-10 นาทีบน Colab -> ปกติ รอได้
- โน้ตบุ๊กเดาชื่อไฟล์/คอลัมน์ให้อัตโนมัติ ถ้า layout แปลกต้องแก้ `# << แก้` เอง (มีตัวอย่างค่าที่ต้องใส่ให้ทุกจุด)

## รอบ Recheck ละเอียด (แก้บั๊ก + เพิ่มตัวอย่างทุกบรรทัด)

ใช้ review agent อ่านทุกโน้ตบุ๊กทีละบรรทัด เจอและแก้แล้ว:
- forecasting: เดิมเติม lag ของ test เป็น NaN -> คะแนนพังเงียบ ๆ ; แก้เป็น recursive forecasting (เติม lag จากประวัติจริง) -- ทดสอบแล้วได้ค่าจริง ไม่มี NaN
- signal_classification: `StratifiedGroupKFold` crash เมื่อ `GROUP_COL=None` (มี 1 group) ; แก้ให้ fallback ไป `StratifiedKFold` อัตโนมัติ
- tabular classification: AUC เดิมต้องแก้มือ + `iloc[:,1]` เปราะ ; แก้ให้เลือกแบบส่งอัตโนมัติตาม metric + ใช้ `predictor.positive_class`
- tabular regression: รองรับชื่อ metric ที่ Kaggle เขียนสั้น (RMSE/MAE/R2) -> แปลงเป็นชื่อ AutoGluon ให้
- NLP word_seg: `labels[0]=1` crash เมื่อข้อความว่าง -> ใส่ guard `if labels:` ; เปลี่ยน id จาก `r.get('id',_)` (เงียบ ๆ ใช้เลขแถว) เป็น `ID_COL` ชัดเจน + ตัวอย่างรูปแบบ id
- NLP transformers: `Trainer/Seq2SeqTrainer(tokenizer=)` ถูกถอดใน v5 -> เปลี่ยนเป็น `processing_class=` ; แยก `TGT_COL` (train) กับ `ANS_COL` (submission)
- CV image_classification: `torch.cuda.amp` (deprecated) -> `torch.amp` ; `.astype(int)` ที่ crash กับป้ายข้อความ -> เอาออก
- CV detection/segmentation: ไล่เลขขั้นตอนใหม่ (เดิม "ขั้นที่ 2" ซ้ำ) ; เพิ่มรุ่น `yolo11` ; เพิ่มคำเตือน `No labels found` ; segmentation เปลี่ยน placeholder เป็น RLE encode จริง + resize mask กลับขนาดรูป
- Multimodal: เพิ่มเซลล์วัด `thai_bleu()` ในเครื่องก่อนส่ง ; `write_submission` เพิ่มการจับคู่ id + เตือนถ้า fallback เยอะ (กันส่งแล้วได้ 0) ; visual_qa เพิ่ม assert ชื่อคอลัมน์ + เตือน BLIP ตอบอังกฤษ + แถบ progress
- ทุกบรรทัด `# << แก้` เพิ่ม "ตัวอย่างค่าที่ต้องใส่" (เช่น `COMP` -> "titanic", `METRIC` -> "f1"/"roc_auc", `FS` -> EEG=100/accelerometer=50)

ทดสอบซ้ำหลังแก้: 13 notebooks syntax ผ่าน, logic test ผ่าน (รวม recursive forecasting + CV fallback), router 34/34, JS router ใน HTML 10/10

## รันการตรวจทั้งหมดอีกครั้ง
```bash
cd 5Domains
python3 _build/verify_router.py     # router 34 เคส ไทย+อังกฤษ
python3 _build/verify_logic.py      # ตรรกะหลักด้วยข้อมูลจำลอง
bash   _build/run_checks.sh         # รันทั้งหมด + syntax check
```

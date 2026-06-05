# Super AI Engineer SS6 — ชุดเตรียมสอบ (จัดตามหัวข้อใหญ่)

ชุดนี้แปลง 5 โจทย์ Kaggle ให้เป็น `เทมเพลตหัวข้อใหญ่` ที่ใช้ซ้ำกับโจทย์อะไรก็ได้ในหมวดนั้น
เน้น `ง่ายที่สุด` สำหรับมือใหม่ ทุกโน้ตบุ๊กรันจากบนลงล่างได้เลย คอมเมนต์ไทยทุกจุด

อ่านก่อน: `START_HERE.md` (คู่มือมือใหม่) แล้วเปิด `00_MASTER_ROUTER.ipynb`

## โครงสร้าง (จัดตามหัวข้อใหญ่)

```
START_HERE.md                       <- อ่านก่อน (มือใหม่)
00_MASTER_ROUTER.ipynb              <- เจอโจทย์แบบไหน เปิดไฟล์ไหน (พิมพ์โจทย์แล้วมันบอก)
01_Computer_Vision/                 <- รูป -> คลาส / กล่อง / mask
    README.md                       (เกร็งโจทย์: ออกได้กี่แบบ)
    image_classification.ipynb
    object_detection.ipynb
    segmentation.ipynb
    reference_cheatsheet.md
02_NLP/                             <- ข้อความ -> ป้าย / ข้อความใหม่
    thai_word_segmentation.ipynb
    text_classification.ipynb
    text_generation.ipynb
03_Multimodal_VisionLanguage/       <- รูป -> ข้อความ
    thai_image_captioning.ipynb
    visual_qa.ipynb
04_Tabular/                         <- ตาราง -> ค่าทำนาย
    classification.ipynb
    regression.ipynb
05_TimeSeries_Signal/               <- สัญญาณ -> คลาส / พยากรณ์
    signal_classification.ipynb
    forecasting.ipynb
```
(รวม 13 โน้ตบุ๊ก ครอบคลุมทุกแนวของ 5 หัวข้อใหญ่)
(ทุกโฟลเดอร์มี `README.md` เกร็งโจทย์ + `reference_cheatsheet.md` ความรู้ลึก)

## 5 หัวข้อใหญ่ + วิธีที่ง่ายที่สุด

| หัวข้อใหญ่ | input -> output | วิธีง่ายสุด (มือใหม่) |
|---|---|---|
| Computer Vision | รูป -> คลาส | AutoGluon MultiModal |
| NLP | ข้อความ -> ป้าย | PyThaiNLP / TF-IDF + LogReg |
| Multimodal | รูป -> ข้อความ | โมเดล Thai-COCO สำเร็จรูป |
| Tabular | ตาราง -> ค่าทำนาย | AutoGluon Tabular |
| Time-Series | สัญญาณ -> คลาส | สกัดฟีเจอร์ + AutoGluon |

แนวคิด: ทุกโน้ตบุ๊กมี `วิธีที่ 1 (ง่ายสุด)` อยู่บนสุด มือใหม่ทำแค่นี้ก็ได้คะแนนแล้ว
แล้วมีวิธีที่ 2/3 (ขั้นสูง) ไว้ดันคะแนนถ้ามีเวลา

## โจทย์ที่คาดว่าจะออก (เกร็งโจทย์ต่อหัวข้อใหญ่)

ดูละเอียดในไฟล์ `README.md` ของแต่ละโฟลเดอร์ สรุปสั้น:
- CV: จำแนกรูป / หาวัตถุ (detection) / segmentation / OCR
- NLP: ตัดคำ / NER / POS / จำแนกข้อความ (sentiment)
- Multimodal: บรรยายรูป / VQA / OCR captioning
- Tabular: จำแนก 2 คลาส / หลายคลาส / regression / imbalanced
- Time-Series: จำแนกสัญญาณ / forecasting / anomaly

## ลำดับทำงานในห้องสอบ

1. เปิด `00_MASTER_ROUTER.ipynb` พิมพ์โจทย์ -> รู้ว่าเปิดไฟล์ไหน
2. อ่านหน้า Kaggle จริง tab `Data` + `Evaluation` จด: คอลัมน์, metric, จำนวนคลาส, รูปแบบ submission
3. แก้ CONFIG ตามคอมเมนต์ `# << แก้`
4. รัน `วิธีที่ 1 (ง่ายสุด)` ให้มี submission ก่อน
5. เหลือเวลาค่อยทำวิธีที่ 2/3 ดันคะแนน

หมายเหตุ: หน้า Kaggle SS6 ส่วนใหญ่ login-gated ตอนเตรียมชุดนี้ จึงยืนยันรูปแบบ submission เป๊ะ ๆ ไม่ได้
แต่โค้ดทุกตัวอ่าน `sample_submission.csv` มาเป็นแม่แบบ จึงปรับตามจริงให้อัตโนมัติ

## เว็บคู่มือ (HTML) + เอกสารเพิ่มเติม

- `index.html` — เว็บคู่มือรวมทุกอย่าง (responsive มือถือ 100%) มี Router แบบพิมพ์โจทย์แล้วบอกโน้ตบุ๊ก
  - รันในเครื่อง: `python server.py` แล้วเปิด `http://localhost:8000`
  - deploy Railway: New Project -> Deploy from GitHub repo -> เลือก repo นี้ (อ่าน `Procfile` ให้เอง) ดู `index.html` ส่วน Deploy
- `ADVANCED_SOTA.md` — SOTA อัปเดต 2025-2026 + วิธีดันคะแนน (อยากได้คะแนนเพิ่มไปใช้อะไร แก้ตรงไหน)
- `VERIFY.md` — สรุปสิ่งที่ทดสอบจริง (router 34 เคส, โค้ดรันจริง, API ตรวจกับเอกสาร)
- สร้างเว็บใหม่หลังแก้เนื้อหา: `python _build/build_site.py`

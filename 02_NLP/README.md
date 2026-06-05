# หัวข้อใหญ่: NLP (งานเกี่ยวกับข้อความ)

input คือ `ข้อความ` (มักเป็นภาษาไทย ซึ่งไม่มีเว้นวรรคระหว่างคำ)

## เกร็งโจทย์ — ออกได้กี่แบบ?

| แบบโจทย์ | output | metric | ใช้ไฟล์ | ง่ายแค่ไหน |
|---|---|---|---|---|
| 1. ตัดคำ (word segmentation) | ป้ายต่อ "ตัวอักษร" (0/1) | char/word F1 | `thai_word_segmentation.ipynb` | ⭐ ง่าย (ใช้ของสำเร็จรูป) |
| 2. NER (ดึงชื่อเฉพาะ) | ป้ายต่อ "คำ" (B-PER, ...) | entity F1 (seqeval) | `thai_word_segmentation.ipynb` (เปลี่ยน label) | ⭐⭐ |
| 3. POS tagging (ชนิดคำ) | ป้ายต่อ "คำ" | accuracy | `thai_word_segmentation.ipynb` (เปลี่ยน label) | ⭐⭐ |
| 4. จำแนกข้อความ (sentiment/topic) | 1 ป้ายต่อ "ทั้งข้อความ" | accuracy, macro-F1 | `text_classification.ipynb` | ⭐ ง่าย |

## วิธีดูว่าเป็นแบบไหน (ดู output)
- ป้าย "ต่อหน่วยในลำดับ" (ทีละคำ/ตัวอักษร) -> sequence labeling -> `thai_word_segmentation.ipynb`
- ป้าย "เดียวต่อทั้งข้อความ" (รีวิวบวก/ลบ, หมวดข่าว) -> `text_classification.ipynb`

## วิธีที่ง่ายที่สุด (มือใหม่)
- ตัดคำ: `PyThaiNLP deepcut` ตัดให้เลย ไม่ต้องเทรน
- จำแนกข้อความ: ตัดคำไทย + `TF-IDF` + `Logistic Regression` (เร็ว ไม่ต้อง GPU)

## กับดักที่ต้องระวัง (ดูเต็มใน reference_cheatsheet.md)
- ไทยไม่มีเว้นวรรค ต้องตัดคำก่อนเสมอ (อย่า split ด้วยช่องว่าง)
- งานตัดคำ: ป้ายต้องมี 1 ป้ายต่อ 1 ตัวอักษร จำนวนต้องเท่ากับความยาวข้อความเป๊ะ (พลาดบ่อยสุด)
- ตัวอักษรไทยที่เป็นสระ/วรรณยุกต์ ห้ามเป็นจุดขึ้นคำ
- รูปแบบ id ใน submission ต้องตรง sample (ตรวจให้ดี)

ไฟล์ในโฟลเดอร์นี้:
- `thai_word_segmentation.ipynb` — ตัดคำ / sequence labeling (NER/POS ก็ใช้ฐานนี้)
- `text_classification.ipynb` — จำแนกข้อความทั้งประโยค
- `text_generation.ipynb` — แปล/สรุป/ตอบคำถาม (seq2seq)
- `reference_cheatsheet.md` — ความรู้ลึก + ตาราง SOTA + keywords

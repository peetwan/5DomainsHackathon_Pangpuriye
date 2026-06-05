# หัวข้อใหญ่: Multimodal / Vision-Language (รูป + ข้อความ)

input มี `รูป` (อาจมีข้อความด้วย) และ output เป็น `ข้อความ`

## เกร็งโจทย์ — ออกได้กี่แบบ?

| แบบโจทย์ | input -> output | metric | ใช้ไฟล์ | ง่ายแค่ไหน |
|---|---|---|---|---|
| 1. บรรยายรูป (image captioning) | รูป -> ประโยค | BLEU (ตัดคำ newmm) | `thai_image_captioning.ipynb` | ⭐⭐ |
| 2. VQA (ตอบคำถามจากรูป) | รูป+คำถาม -> คำตอบ | accuracy/exact-match | `thai_image_captioning.ipynb` (ใส่คำถามใน prompt) | ⭐⭐⭐ |
| 3. OCR captioning (อ่านตัวอักษรในรูป) | รูป -> ข้อความ | CER/WER | ใช้ caption + โมเดล OCR | ⭐⭐⭐ |

## วิธีดูว่าเป็นแบบไหน
- มีแค่รูป -> ตอบประโยคบรรยาย = captioning
- มีรูป + คำถาม -> ตอบตามคำถาม = VQA (เปลี่ยนเป็นโมเดล instruction เช่น Qwen2.5-VL)
- เน้นอ่านตัวอักษรในรูป = OCR (ใช้โมเดลที่เก่ง OCR เช่น PaliGemma2)

## วิธีที่ง่ายที่สุด (มือใหม่)
ใช้โมเดลที่เทรนกับ Thai-COCO มาแล้ว (เช่น `Natthaphon/thaicapgen-convnext-phayathai`) generate ได้เลย ไม่ต้องเทรน

## กับดักที่ต้องระวัง (สำคัญมาก ดูเต็มใน reference_cheatsheet.md)
- metric เป็น BLEU ที่ต้อง `ตัดคำไทยด้วย newmm ก่อน` -> ใช้ `thai_bleu()` วัดในเครื่องก่อนส่งทุกครั้ง
- decode แบบ greedy/beam อย่า sample (คะแนน BLEU จะตก)
- ความยาวประโยคพอดี ๆ (สั้นไป/ยาวไป BLEU ตก)
- เขียนไฟล์ submission ด้วย `utf-8-sig` (กันภาษาไทยเพี้ยน)

ไฟล์ในโฟลเดอร์นี้:
- `thai_image_captioning.ipynb` — บรรยายรูปเป็นไทย (3 วิธี ง่าย -> ขั้นสูง)
- `visual_qa.ipynb` — ตอบคำถามจากรูป / OCR (BLIP-VQA / Qwen2.5-VL)
- `reference_cheatsheet.md` — ความรู้ลึก + ตาราง SOTA (PaliGemma2/Qwen2.5-VL/BLIP) + keywords

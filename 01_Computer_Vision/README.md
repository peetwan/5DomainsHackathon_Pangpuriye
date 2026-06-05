# หัวข้อใหญ่: Computer Vision (งานเกี่ยวกับรูปภาพ)

input คือ `รูปภาพ` เกร็งโจทย์ว่าออกได้แบบไหนบ้าง แล้วใช้โน้ตบุ๊กไหน

## เกร็งโจทย์ — ออกได้กี่แบบ?

| แบบโจทย์ | output | metric ที่เจอบ่อย | ใช้ไฟล์ | ง่ายแค่ไหน |
|---|---|---|---|---|
| 1. จำแนกรูป (classification) | 1 คลาส/รูป | Accuracy, F1 | `image_classification.ipynb` | ⭐ ง่ายสุด |
| 2. หาวัตถุ (detection) | กล่อง + คลาส | mAP | `object_detection.ipynb` | ⭐⭐⭐ ยากกว่า |
| 3. แบ่งส่วนภาพ (segmentation) | mask ระบายสีต่อพิกเซล | IoU, Dice | ใช้ detection เป็นฐาน + U-Net | ⭐⭐⭐ |
| 4. OCR (อ่านตัวอักษรในรูป) | ข้อความ | CER, accuracy | ไปหัวข้อ 03 Multimodal | ⭐⭐ |

แบบที่เจอบ่อยสุดในสอบ = `จำแนกรูป (classification)` เริ่มจากอันนี้ก่อนเสมอ

## วิธีดูว่าเป็นแบบไหน (ดู output ของโจทย์)
- ตอบว่า "รูปนี้คือคลาสอะไร" -> classification
- ตอบว่า "วัตถุอยู่ตรงไหน (พิกัดกล่อง)" -> detection
- ตอบว่า "พิกเซลไหนเป็นอะไร" -> segmentation
- ตอบว่า "ในรูปเขียนว่าอะไร" -> OCR

## วิธีที่ง่ายที่สุด (มือใหม่)
- classification: `AutoGluon MultiModalPredictor` -> บอก path รูป + label เดี๋ยวมันทำให้หมด
- detection: `YOLOv8` (ultralytics) -> สั่ง `model.train()` 2-3 บรรทัด

## กับดักที่ต้องระวัง (สรุปสั้น ดูเต็มใน reference_cheatsheet.md)
- ข้อมูลน้อย (~ไม่กี่พันรูป) ใช้โมเดลเล็ก-กลาง + augmentation อย่า overfit
- แบ่ง train/val ให้สมดุลคลาส (StratifiedKFold)
- รูปแบบ submission: ส่งเลขคลาส (เช่น 0/1) ไม่ใช่ความน่าจะเป็น ตรวจ header ให้ตรง sample

ไฟล์ในโฟลเดอร์นี้:
- `image_classification.ipynb` — จำแนกรูป (เริ่มที่นี่ ง่ายสุด)
- `object_detection.ipynb` — หาวัตถุด้วย YOLOv8
- `segmentation.ipynb` — แบ่งส่วนภาพด้วย YOLOv8-seg
- `reference_cheatsheet.md` — ความรู้ลึก + ตาราง SOTA + keywords

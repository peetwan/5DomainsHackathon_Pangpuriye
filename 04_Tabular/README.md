# หัวข้อใหญ่: Tabular (ข้อมูลตาราง / สเปรดชีต)

input คือ `ตาราง` (แต่ละแถว = 1 ตัวอย่าง, คอลัมน์ = ฟีเจอร์) เป็นหัวข้อที่ `ง่ายที่สุด` สำหรับมือใหม่

## เกร็งโจทย์ — ออกได้กี่แบบ?

| แบบโจทย์ | output | metric | ใช้ไฟล์ | ง่ายแค่ไหน |
|---|---|---|---|---|
| 1. จำแนก 2 คลาส (binary) | 0/1 | Accuracy, AUC, F1 | `classification.ipynb` | ⭐ ง่ายสุด |
| 2. จำแนกหลายคลาส (multiclass) | 1 ใน N คลาส | macro-F1, log-loss | `classification.ipynb` (เปลี่ยน metric) | ⭐ |
| 3. ทำนายตัวเลข (regression) | ตัวเลขต่อเนื่อง | RMSE, MAE | `regression.ipynb` | ⭐ |
| 4. ข้อมูลไม่สมดุล (imbalanced) | 0/1 (คลาสน้อยมาก) | AUC-PR, F1 | `classification.ipynb` + class_weight | ⭐⭐ |

## วิธีดูว่าเป็นแบบไหน (ดูคอลัมน์เป้าหมาย)
- เป้าหมายเป็นหมวด/ป้าย -> classification
- เป้าหมายเป็นตัวเลขต่อเนื่อง (ราคา/คะแนน) -> regression
- เป้าหมายมี 2 ค่า แต่ค่านึงน้อยมาก (เช่น 2%) -> imbalanced (ใช้ AUC/F1 + ถ่วงน้ำหนัก)

## วิธีที่ง่ายที่สุด (มือใหม่ ทำแค่นี้พอ)
`AutoGluon TabularPredictor` -> บอก label + metric + time_limit เดี๋ยวมันลองหลายโมเดล + รวมให้เอง
จัดการ missing/categorical ให้หมด ไม่ต้องเตรียมข้อมูลเอง

## กับดักที่ต้องระวัง (ดูเต็มใน reference_cheatsheet.md)
- อย่าลืม drop คอลัมน์ `id` ออกจากฟีเจอร์
- เลือก metric ให้ตรงโจทย์: AUC -> ส่งความน่าจะเป็น, Accuracy/F1 -> ส่งป้าย 0/1
- เชื่อคะแนน cross-validation ของเราเอง มากกว่า public leaderboard (ข้อมูลน้อย LB หลอกได้)
- ข้อมูลการแพทย์มักไม่สมดุล -> ดู F1/recall ไม่ใช่ accuracy อย่างเดียว

ไฟล์ในโฟลเดอร์นี้:
- `classification.ipynb` — ทำนายคลาส (เริ่มที่นี่)
- `regression.ipynb` — ทำนายตัวเลข
- `reference_cheatsheet.md` — ความรู้ลึก + LightGBM/CatBoost/TabPFN + keywords

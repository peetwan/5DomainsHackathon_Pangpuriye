# หัวข้อใหญ่: Time-Series / Signal (สัญญาณ/ข้อมูลตามเวลา)

input คือ `ค่าตามเวลา` หรือ `สัญญาณ` (EEG, ECG, เซนเซอร์, ยอดขายรายวัน ฯลฯ)

## เกร็งโจทย์ — ออกได้กี่แบบ?

| แบบโจทย์ | output | metric | ใช้ไฟล์ | ง่ายแค่ไหน |
|---|---|---|---|---|
| 1. จำแนกหน้าต่างสัญญาณ (classification) | 1 คลาส/หน้าต่าง | macro-F1, kappa, accuracy | `signal_classification.ipynb` | ⭐⭐ |
| 2. ทำนายค่าในอนาคต (forecasting) | ตัวเลขในอนาคต | RMSE, MAE, SMAPE | `forecasting.ipynb` | ⭐⭐ |
| 3. ตรวจจับความผิดปกติ (anomaly) | ปกติ/ผิดปกติ | F1, AUPRC | `signal_classification.ipynb` (binary) | ⭐⭐ |

ตัวอย่างหน้าต่างสัญญาณ -> คลาส: EEG -> ระยะการนอน, ECG -> ชนิดการเต้น, เซนเซอร์มือถือ -> ท่าทาง (HAR)

## วิธีดูว่าเป็นแบบไหน
- ตอบ "หน้าต่างนี้คือคลาสอะไร" -> classification -> `signal_classification.ipynb`
- ตอบ "ค่าถัดไป/อนาคตเป็นเท่าไหร่" -> forecasting -> `forecasting.ipynb`

## วิธีที่ง่ายที่สุด (มือใหม่)
- classification: `สกัดฟีเจอร์` (พลังงานในแต่ละความถี่ + สถิติ) แล้วโยนเข้า `AutoGluon` เหมือนตารางธรรมดา
- forecasting: ทำ `ฟีเจอร์วันที่ + ค่าย้อนหลัง (lag)` แล้วใช้ `LightGBM` ทำนายเหมือน regression

## กับดักที่ต้องระวัง (สำคัญมาก ดูเต็มใน reference_cheatsheet.md)
- แบ่ง train/val ตาม `subject/recording` ไม่ใช่สุ่ม (ไม่งั้นคะแนนหลอกสูงมาก) -- พลาดบ่อยสุด
- คลาสมักไม่สมดุล -> วัด macro-F1 / kappa ไม่ใช่ accuracy
- normalize สัญญาณต่อ recording (อย่า normalize รวมทั้งชุด)
- forecasting: ค่าย้อนหลัง (lag) ของ test ต้องต่อจาก history จริง ระวังข้อมูลรั่ว

ไฟล์ในโฟลเดอร์นี้:
- `signal_classification.ipynb` — จำแนกหน้าต่างสัญญาณ (เริ่มที่นี่)
- `forecasting.ipynb` — ทำนายค่าในอนาคต
- `reference_cheatsheet.md` — ความรู้ลึก + 1D-CNN/TinySleepNet + keywords

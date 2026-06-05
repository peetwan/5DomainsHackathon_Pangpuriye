# เริ่มอ่านที่นี่ (สำหรับมือใหม่) 👋

ชุดนี้ช่วยให้สอบ hackathon ได้ แม้ยังไม่ค่อยมีความรู้ จัดเป็น `หัวข้อใหญ่` ไว้ให้แล้ว
แต่ละหัวข้อมีโน้ตบุ๊กที่ `รันจากบนลงล่างได้เลย` คอมเมนต์ไทยทุกจุด

## ใช้ยังไง (3 ขั้น)

ขั้น 1: เปิด `00_MASTER_ROUTER.ipynb` -> พิมพ์โจทย์ที่เจอ -> มันบอกว่าใช้ไฟล์ไหน
ขั้น 2: เปิดไฟล์ที่มันบอก -> แก้จุดที่มีคอมเมนต์ `# << แก้` (ส่วนใหญ่แค่ชื่อ competition กับชื่อคอลัมน์)
ขั้น 3: กดรันจากบนลงล่าง -> ทำแค่ `วิธีที่ 1 (ง่ายสุด)` ก็ได้ไฟล์ submission แล้ว

## 5 หัวข้อใหญ่ (จับให้ถูกหมวด)

- `01_Computer_Vision/` — input เป็นรูป (จำแนกรูป / หาวัตถุ)
- `02_NLP/` — input เป็นข้อความ (ตัดคำ / จำแนกข้อความ)
- `03_Multimodal_VisionLanguage/` — รูป -> ข้อความ (บรรยายรูป)
- `04_Tabular/` — ข้อมูลตาราง (ทำนายคลาส / ทำนายตัวเลข)
- `05_TimeSeries_Signal/` — สัญญาณตามเวลา (จำแนกสัญญาณ / พยากรณ์)

วิธีจับหมวดเร็ว ๆ: ถามตัวเอง 2 ข้อ -> "input เป็นอะไร" และ "output เป็นอะไร"

## รันบน Google Colab (อ่าน `COLAB_GUIDE.md`)

เรื่อง GPU / เอาข้อมูลเข้า 3 วิธี (Kaggle/อัปโหลดเอง/Drive) / submission หลายรูปแบบ / สิ่งที่ควรกังวล
อ่านละเอียดใน `COLAB_GUIDE.md` สรุปสั้น:
- เปิด GPU: เมนู `Runtime > Change runtime type > T4 GPU` (โค้ดเลือก GPU/CPU ให้เองแล้ว) — เปิดเฉพาะงาน deep learning
- อัปโหลดข้อมูลเอง: ลาก zip ไปวางในแถบ Files ของ Colab แล้วตั้ง `DATA_DIR = "/content"` (เซลล์แตก zip ให้เอง)
- submission: เริ่มจาก sample เสมอ + มี assert ตรวจฟอร์แมตให้ก่อนส่ง

## ต้องเตรียมอะไรก่อน

1. มีบัญชี Kaggle และเข้าร่วม competition แล้ว
2. มีไฟล์ `kaggle.json` (เข้า kaggle.com -> รูปโปรไฟล์ -> Settings -> Account -> Create New Token)
   - ถ้ารันบน `Kaggle Notebook` หรือ `Google Colab` จะง่ายสุด (มี GPU ฟรี)
3. ในแต่ละโน้ตบุ๊ก เซลล์ตั้งค่า Kaggle จะบอกว่าต้องใส่ token ตรงไหน

## เคล็ดลับมือใหม่

- ทำ `วิธีที่ 1 (ง่ายสุด)` ให้มีคะแนนขึ้นกระดานก่อนเสมอ แล้วค่อยลองวิธีอื่น
- `AutoGluon` คือเพื่อนซี้ของมือใหม่ -> ใช้ได้กับ ตาราง / รูป / สัญญาณ แทบกดปุ่มเดียวจบ
- รันบน `Google Colab` หรือ `Kaggle Notebook` (มี GPU ฟรี + ไลบรารีพร้อม) จะง่ายและเร็วสุด
- ทุกโฟลเดอร์มี `README.md` (เกร็งโจทย์ว่าออกแบบไหนได้) และ `reference_cheatsheet.md` (ความรู้ลึก)

## มีกี่โน้ตบุ๊ก (ครอบคลุมทุกแนว)

รวม 13 โน้ตบุ๊ก แต่ละหัวข้อใหญ่มีหลายแบบ:
- CV: image_classification, object_detection, segmentation
- NLP: thai_word_segmentation, text_classification, text_generation
- Multimodal: thai_image_captioning, visual_qa
- Tabular: classification, regression
- Time-Series: signal_classification, forecasting

อยากรู้ว่าตรวจอะไรไปบ้าง (router/โค้ดทำงานจริงไหม) ดู `VERIFY.md`

## ถ้างง ให้ดูตามนี้

โจทย์ให้รูป -> ตอบคลาส?  ->  `01_Computer_Vision/image_classification.ipynb`
โจทย์ให้ตาราง -> ทำนายค่า?  ->  `04_Tabular/classification.ipynb` หรือ `regression.ipynb`
โจทย์ให้ข้อความไทย -> ตัดคำ?  ->  `02_NLP/thai_word_segmentation.ipynb`
โจทย์ให้รูป -> บรรยายเป็นไทย?  ->  `03_Multimodal_VisionLanguage/thai_image_captioning.ipynb`
โจทย์ให้สัญญาณ/EEG -> ตอบคลาส?  ->  `05_TimeSeries_Signal/signal_classification.ipynb`

โชคดีนะครับ 💪

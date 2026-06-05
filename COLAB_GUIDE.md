# คู่มือใช้บน Google Colab (อ่านก่อนสอบจริง)

รวมทุกเรื่องที่ต้องรู้ตอนรันบน Colab + เรื่องที่ควรกังวล (ตอบคำถาม Peet แบบ proactive)

---

## 1) GPU — เมื่อไหร่ต้องใช้ + เปิดยังไง + ต้องเขียนโค้ดเพิ่มไหม

ต้องเขียนโค้ดเพิ่มไหม: `ไม่ต้อง` โน้ตบุ๊กเช็ค + เลือก GPU/CPU ให้อัตโนมัติแล้ว (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`)
สิ่งเดียวที่ต้องทำเองคือ "เปิด GPU runtime" จากเมนู (ไม่ใช่โค้ด):

วิธีเปิด GPU: เมนู `Runtime` -> `Change runtime type` -> เลือก `T4 GPU` -> `Save` (ฟรี)
เซลล์ตั้งค่าข้อมูลจะ print ให้เห็นว่ามี GPU ไหม เช่น `GPU: Tesla T4`

งานไหนต้องใช้ GPU (เปิดก่อนรัน):
- ต้องใช้ GPU: captioning, VQA, timm (วิธี 2 ของ image), 1D-CNN, WangchanBERTa/mT5 (วิธีขั้นสูงของ NLP), YOLO (detection/segmentation)
- ไม่ต้องใช้ GPU ก็ได้: Tabular (AutoGluon/LightGBM), Word segmentation (PyThaiNLP/CRF), Text classification (TF-IDF), Signal features+LightGBM, Forecasting

เคล็ดลับ: เปิด GPU เฉพาะตอนจะใช้ deep learning เพราะโควต้า GPU ฟรีมีจำกัด ถ้าทำงานตารางไม่ต้องเปิด (ประหยัดโควต้า)

ถ้าขึ้น `CUDA out of memory`: ลด `BATCH` (เช่น 32 -> 16 -> 8) หรือลด `IMG_SIZE`/`max_length`

---

## 2) เอาข้อมูลเข้า Colab — 3 วิธี (เลือก 1)

เซลล์ "ขั้นที่ 2" ในทุกโน้ตบุ๊กรองรับครบ 3 วิธีแล้ว แก้แค่ตัวแปรบนสุด

### วิธี A — ดาวน์โหลดจาก Kaggle อัตโนมัติ (แนะนำ ถ้าโจทย์อยู่บน Kaggle)
1. เอา token: kaggle.com -> รูปโปรไฟล์ -> Settings -> Account -> Create New Token (ได้ไฟล์ `kaggle.json`)
2. เปิด `kaggle.json` คัดลอก username กับ key มาใส่:
```python
COMP = "ชื่อ-competition"
KAGGLE_USERNAME = "ของคุณ"
KAGGLE_KEY      = "คีย์ยาว ๆ"
DATA_DIR = ""     # ปล่อยว่าง
```
หมายเหตุ: ต้องกด `Join Competition` / ยอมรับ rules บนหน้า Kaggle ก่อน ไม่งั้นดาวน์โหลดไม่ได้ (403)

### วิธี B — โหลดข้อมูลมาเครื่องตัวเองแล้วอัปขึ้น Colab เอง (ตอบคำถามข้อนี้โดยตรง)
ต้องเปลี่ยนโค้ดอะไรไหม: `แค่ตั้ง DATA_DIR` บรรทัดเดียว ไม่ต้องแตะอย่างอื่น
1. ในเครื่องตัวเอง ดาวน์โหลดไฟล์จาก Kaggle (ปุ่ม Download Data) จะได้ `xxx.zip`
2. บน Colab: เปิดแถบ `Files` (ไอคอนโฟลเดอร์ ซ้ายมือ) แล้ว `ลากไฟล์ zip ไปวาง` (รอจนอัปโหลดเสร็จ จะอยู่ที่ `/content/xxx.zip`)
3. ตั้งในเซลล์:
```python
DATA_DIR = "/content"     # << โฟลเดอร์ที่วาง zip ไว้ -- เซลล์จะแตก zip ให้เอง
# COMP / KAGGLE_* ไม่ต้องใส่
```
4. รันเซลล์ -> มันจะแตก zip ในโฟลเดอร์นั้น แล้ว print รายชื่อไฟล์ออกมา (เอาไปแก้ path ในเซลล์ถัดไปให้ตรง)

ข้อควรระวังวิธี B: ไฟล์ใหญ่ (รูปเยอะ) อัปโหลดผ่านเบราว์เซอร์ช้ามาก -> ใช้วิธี A หรือ C ดีกว่าถ้าข้อมูลใหญ่

### วิธี C — ต่อ Google Drive (เหมาะกับข้อมูลใหญ่/ใช้ซ้ำหลายรอบ)
1. เอา zip ไปวางใน Google Drive ก่อน (เช่น MyDrive/data/)
2. บน Colab รัน:
```python
from google.colab import drive
drive.mount('/content/drive')     # กดยืนยันสิทธิ์
```
3. ตั้ง:
```python
DATA_DIR = "/content/drive/MyDrive/data"   # << โฟลเดอร์ใน Drive ที่มีไฟล์/zip
```
ข้อดี: ไม่ต้องอัปโหลดซ้ำทุกครั้งที่ Colab หลุด

---

## 3) Submission มีหลายรูปแบบ เปลี่ยนไป ต้องทำยังไง

หลักการทอง: `เริ่มจาก sample_submission.csv เสมอ` โน้ตบุ๊กทำให้แล้ว (`out = sample.copy()` แล้วเติมคำตอบ)
และมี "ตรวจฟอร์แมตก่อนส่งอัตโนมัติ" (เทียบคอลัมน์/จำนวนแถว + assert) -> ถ้าผิดจะเตือนทันที ไม่ส่งมั่ว

วิธีดูว่าโจทย์ต้องการรูปแบบไหน: เปิด `sample_submission.csv` (โน้ตบุ๊ก print หัวตารางให้) + อ่าน tab `Evaluation` บน Kaggle

รูปแบบที่เจอบ่อย + วิธีรับมือ (แก้ตรงบรรทัดสร้าง submission):

| รูปแบบ sample | หมายความว่า | ทำยังไง |
|---|---|---|
| `id,label` ค่าเป็น 0/1 | ส่งป้าย | `out[col] = preds` (ป้าย) — ค่าเริ่มต้นของโน้ตบุ๊ก |
| `id,target` ค่าเป็นทศนิยม 0-1 | ส่งความน่าจะเป็น (AUC) | `out[col] = proba` (โน้ตบุ๊ก tabular สลับให้อัตโนมัติตาม METRIC) |
| หลายคอลัมน์ (เช่น `id,class_0,class_1,..`) | ส่ง prob ทุกคลาส | `proba_df = predictor.predict_proba(...); out[คอลัมน์คลาส] = proba_df.values` |
| 1 แถวต่อ "ตัวอักษร/พิกเซล" | per-unit (ตัดคำ/segmentation) | สร้าง row ต่อหน่วย (โน้ตบุ๊กตัดคำทำให้แล้ว) เช็ค id เช่น `5_0,5_1` |
| คอลัมน์ `rle`/`EncodedPixels` | segmentation แบบ RLE | ใช้ `rle_encode()` ในโน้ตบุ๊ก segmentation |
| คอลัมน์กล่อง (`xmin,ymin,..`) หรือ string | detection | เขียนตามที่ Evaluation บอก (1 แถวต่อกล่อง หรือรวมเป็น string) |

ถ้าฟอร์แมตแปลกที่โน้ตบุ๊กยังไม่รองรับ: ดู `sample.head()` แล้วสร้าง DataFrame ให้คอลัมน์ตรงเป๊ะ ตัว assert ก่อนส่งจะช่วยจับว่ายังไม่ตรง

ถ้า assert ฟ้อง "คอลัมน์ไม่ตรง": ปรับชื่อ/จำนวนคอลัมน์ของ `out` ให้เท่ากับ sample เช่น `out = out.rename(columns={...})` หรือเพิ่ม/ลบคอลัมน์

---

## 4) เรื่องอื่น ๆ ที่ควรกังวล (proactive — สิ่งที่มือใหม่มักลืม)

ก่อนเริ่ม:
- กด `Join Competition` + ยอมรับ rules บน Kaggle ก่อน ไม่งั้นโหลดข้อมูล/ส่งไม่ได้
- บาง competition จำกัดจำนวนส่งต่อวัน (เช่น 5 ครั้ง/วัน) -> อย่าส่งมั่ว เก็บโควต้าไว้
- เปิด GPU ก่อน ถ้างานเป็น deep learning (ดูข้อ 1)

ระหว่างทำ:
- Colab หลุดเอง: idle ~90 นาที / สูงสุด ~12 ชม. -> อย่าทิ้งไว้นาน, งานยาวให้เซฟผลลง Drive
- ดิสก์/แรมหาย: พอ Colab reconnect ไฟล์ใน `/content` หายหมด ต้องโหลด/อัปใหม่ (ใช้ Drive กันหาย)
- GPU โควต้าหมด: ขึ้น "cannot connect to GPU" -> รอ/เปลี่ยนบัญชี/ใช้ CPU สำหรับงานตาราง
- Out of memory: ลด `BATCH`, `IMG_SIZE`, `max_length`, หรือจำนวนข้อมูลต่อรอบ
- เวลาเทรนนาน: ตั้ง `epochs`/`time_limit` น้อย ๆ ก่อนให้รันผ่าน แล้วค่อยเพิ่ม

ความถูกต้อง:
- ตั้ง seed แล้ว (random_state=42) เพื่อให้ผลซ้ำได้
- อย่าให้ข้อมูลรั่ว: tabular ใช้ StratifiedKFold, signal แบ่งตาม subject (โน้ตบุ๊กทำให้แล้ว)
- ภาษาไทยใน CSV: เขียนด้วย `utf-8-sig` (โน้ตบุ๊กงานไทยทำให้แล้ว) กันตัวอักษรเพี้ยน
- path ใน Linux (Colab) ตัวพิมพ์เล็ก/ใหญ่ต่างกัน (`Train` != `train`) -> ดูชื่อจริงจาก output แล้วแก้ให้ตรง

ตอนส่ง:
- เลือกไฟล์ส่งจากคะแนน CV ของเราเอง ไม่ใช่ public LB อย่างเดียว (public น้อย หลอกได้)
- ถ้าส่งได้หลายไฟล์/วัน ส่งทั้งวิธี 1 และ 2 เก็บไว้เทียบ
- อย่า commit/แชร์ `kaggle.json` (เป็นความลับ) — repo นี้มี .gitignore กันให้แล้ว

ติดตั้งไลบรารี:
- AutoGluon ครั้งแรกติดตั้ง ~5-10 นาที (ปกติ รอได้) ; ถ้า error เรื่องเวอร์ชัน ลอง `Runtime > Restart` แล้วรันเซลล์ติดตั้งใหม่
- ถ้า import error หลังติดตั้ง: `Runtime > Restart session` แล้วรันใหม่ (Colab ต้อง restart หลังลงบางแพ็กเกจ)
- transformers เก่ามาก: ถ้าเจอ error `eval_strategy`/`tokenizer` ให้ `!pip install -U transformers`

---

## สรุปสั้น (จำแค่นี้พอ)
1. เปิด GPU จากเมนูถ้างานเป็น deep learning (โค้ดเลือกให้เองแล้ว)
2. เอาข้อมูลเข้า: Kaggle (วิธี A) / อัปโหลด zip แล้วตั้ง `DATA_DIR="/content"` (วิธี B) / Drive (วิธี C)
3. Submission เริ่มจาก sample เสมอ + มี assert ตรวจให้ ถ้าฟ้องให้ปรับคอลัมน์ให้ตรง
4. ระวัง: join comp ก่อน, โควต้าส่ง/วัน, Colab หลุด (เซฟลง Drive), OOM (ลด batch), อย่าแชร์ kaggle.json

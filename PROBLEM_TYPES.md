# แผนที่โจทย์ทั้งหมด — เจอแบบนี้ ทำแบบนี้ (รวมโจทย์ยาก)

ตารางนี้ proactive ครอบคลุมโจทย์ที่ "น่าจะเจอ" ในแต่ละหัวข้อใหญ่ (รวมพวกยาก/แปลก) + บอกว่าใช้อะไร + `AUTO_SOLVER ดักให้ไหม`

คอลัมน์ "ดัก": 🟢 = `00_AUTO_SOLVER` ทำให้อัตโนมัติ | 📓 = มีโน้ตบุ๊กเฉพาะ | 🛠️ = ปรับจากโน้ตบุ๊กใกล้เคียง (มีสูตรด้านล่าง) | ⚠️ = AUTO จะเตือน/บอกให้ไปใช้โน้ตบุ๊กเฉพาะ

---

## Computer Vision (รูป)

| โจทย์ | output | metric ที่เจอ | submission | ใช้อะไร | ดัก |
|---|---|---|---|---|---|
| จำแนกรูป (1 ป้าย) | คลาส | Accuracy/F1/AUC | id,label | AUTO / `image_classification` | 🟢📓 |
| จำแนกรูปหลายป้าย (multi-label) | หลายป้าย/รูป | macro-F1/mAP | หลายคอลัมน์ 0/1 หรือ string คั่นช่องว่าง | 🛠️ ปรับ image (sigmoid+threshold) | 🛠️ |
| fine-grained (คล้ายกันมาก) | คลาส | Accuracy | id,label | `image_classification` (โมเดลใหญ่+res สูง) | 📓 |
| หาวัตถุ (detection) | กล่อง+คลาส | mAP | box/string | `object_detection` (YOLO) | 📓⚠️ |
| แบ่งส่วนภาพ (segmentation) | mask | IoU/Dice | RLE/polygon | `segmentation` (YOLO-seg) | 📓⚠️ |
| อ่านตัวอักษรในรูป (OCR) | ข้อความ | CER/accuracy | id,text | `visual_qa` (Typhoon-OCR) | 📓 |
| keypoint/pose | จุด | OKS/mAP | พิกัดจุด | 🛠️ YOLO-pose (`YOLO("yolo11n-pose.pt")`) | 🛠️ |
| image retrieval/หาภาพคล้าย | embedding | mAP/recall@k | คู่ภาพ | 🛠️ DINOv2/CLIP embedding + cosine | 🛠️ |

## NLP (ข้อความ)

| โจทย์ | output | metric | submission | ใช้อะไร | ดัก |
|---|---|---|---|---|---|
| ตัดคำ/NER/POS | ป้ายต่อ token | F1 (seqeval/boundary) | 1 แถวต่อตัวอักษร/คำ | `thai_word_segmentation` | 📓⚠️ |
| จำแนกข้อความ | 1 ป้าย | Accuracy/F1 | id,label | AUTO / `text_classification` | 🟢📓 |
| จำแนกข้อความหลายป้าย | หลายป้าย | macro-F1 | หลายคอลัมน์/คั่นช่องว่าง | 🛠️ ปรับ text (sigmoid) | 🛠️ |
| ให้คะแนนข้อความ (essay scoring/regression) | ตัวเลข | RMSE/QWK | id,score | 🟢 AUTO (MultiModal เดา regression ให้) | 🟢 |
| คู่ประโยค (NLI/similarity) | ป้าย/คะแนน | Accuracy/Pearson | id,label | 🛠️ รวม 2 คอลัมน์ข้อความ แล้วใช้ text/BERT cross-encoder | 🛠️ |
| แปล/สรุป/ตอบคำถาม (seq2seq) | ข้อความใหม่ | BLEU/ROUGE | id,text | `text_generation` | 📓 |
| QA แบบดึงช่วงคำตอบ (extractive) | ช่วงข้อความ | EM/F1 | id,answer | 🛠️ BERT QA (AutoModelForQuestionAnswering) | 🛠️ |

## Multimodal (รูป+ข้อความ)

| โจทย์ | output | metric | submission | ใช้อะไร | ดัก |
|---|---|---|---|---|---|
| บรรยายรูป (captioning) | ประโยค | BLEU (newmm) | id,caption | `thai_image_captioning` | 📓 |
| ตอบคำถามจากรูป (VQA) | คำตอบ | accuracy/BLEU | id,answer | `visual_qa` | 📓 |
| เข้าใจเอกสาร/ตาราง (DocVQA) | คำตอบ | accuracy/ANLS | id,answer | `visual_qa` (Qwen-VL/Typhoon-OCR) | 📓 |

## Tabular (ตาราง)

| โจทย์ | output | metric | submission | ใช้อะไร | ดัก |
|---|---|---|---|---|---|
| จำแนก 2 คลาส | 0/1 หรือ prob | Accuracy/F1/AUC | id,target | AUTO / `classification` | 🟢📓 |
| จำแนกหลายคลาส | คลาส หรือ prob ต่อคลาส | macro-F1/log-loss | id,label หรือหลายคอลัมน์ prob | 🟢 AUTO (เติม prob ต่อคลาสให้) | 🟢 |
| ทำนายตัวเลข (regression) | ตัวเลข | RMSE/MAE | id,value | AUTO / `regression` | 🟢📓 |
| ข้อมูลไม่สมดุล (imbalanced) | 0/1 | AUC-PR/F1 | id,target | `classification` (class_weight) | 📓 |
| ordinal (คะแนนมีลำดับ) | คลาสมีลำดับ | QWK | id,label | 🛠️ classification + QWK threshold | 🛠️ |
| multi-label tabular | หลายป้าย | macro-F1 | หลายคอลัมน์ | 🛠️ วน LightGBM ต่อป้าย | 🛠️ |
| survival/time-to-event | เวลา+censor | C-index | id,risk | 🛠️ lifelines/xgboost-cox | 🛠️ |

## Time-Series / Signal (สัญญาณ/เวลา)

| โจทย์ | output | metric | submission | ใช้อะไร | ดัก |
|---|---|---|---|---|---|
| จำแนกสัญญาณ (EEG/ECG/HAR) | คลาส | macro-F1/kappa | id,label | `signal_classification` (หรือ AUTO ทำเป็นตาราง) | 📓🟢 |
| พยากรณ์ (forecasting) | ค่าอนาคต | RMSE/MAE/SMAPE | id,value | `forecasting` | 📓 |
| ตรวจจับความผิดปกติ (anomaly) | ปกติ/ผิดปกติ | F1/AUPRC | id,label | `signal_classification` (binary) | 📓🛠️ |
| จำแนกเสียง (audio) | คลาส | Accuracy | id,label | `06_Audio/audio_classification` | 📓 |

## Audio / Speech (เสียง)

| โจทย์ | output | metric | submission | ใช้อะไร | ดัก |
|---|---|---|---|---|---|
| จำแนกเสียง (1 คลาส/ไฟล์) | คลาส | Accuracy/F1 | id,label | `06_Audio/audio_classification` | 📓 |
| ตรวจจับเหตุการณ์เสียง | ป้าย | F1 | หลายคอลัมน์ | 🛠️ ปรับ audio (multi-label) | 🛠️ |
| ถอดเสียงเป็นข้อความ (ASR ไทย) | ข้อความ | CER/WER | id,text | 🛠️ Whisper (โน้ตin 06_Audio) | 🛠️ |

---

## สูตรลัดสำหรับโจทย์ยาก (🛠️)

multi-label (รูป/ข้อความ/ตาราง):
- เปลี่ยน loss เป็น `BCEWithLogitsLoss` (sigmoid) แทน softmax ; ทำนายแต่ละป้ายแยก แล้ว threshold 0.5
- ง่ายสุด: วนเทรน LightGBM/โมเดล ทีละป้าย (one-vs-rest) แล้วเติมหลายคอลัมน์/รวมเป็น string คั่นช่องว่าง
- submission: ถ้าหลายคอลัมน์ -> เติม 0/1 ต่อป้าย ; ถ้า string -> `" ".join(ป้ายที่ผ่าน threshold)`

คู่ประโยค (NLI/similarity):
- ง่าย: รวม 2 คอลัมน์เป็นข้อความเดียว `text = a + " [SEP] " + b` แล้วใช้ `text_classification`
- ดี: ใช้ cross-encoder (`AutoModelForSequenceClassification` ป้อนคู่ประโยค) หรือ `sentence-transformers` (BAAI/bge-m3) วัด cosine

essay scoring / text regression:
- ใช้ AUTO_SOLVER ได้เลย (AutoGluon MultiModal เดาว่าเป็น regression เมื่อ label เป็นตัวเลขต่อเนื่อง) ; metric มัก QWK -> ปัดเป็นจำนวนเต็มก่อนส่ง

ordinal / QWK:
- ทำนายเป็น regression แล้วปัดเป็นคลาส หรือหา threshold ที่ทำให้ QWK สูงสุดบน OOF

audio classification:
- `librosa.feature.melspectrogram` -> บันทึกเป็นรูป -> ใช้ `image_classification` ; หรือสกัดฟีเจอร์เสียง -> ใช้ `signal_classification`

keypoint/pose:
- `YOLO("yolo11n-pose.pt")` (ultralytics) คล้าย detection แต่ได้จุด

extractive QA:
- `AutoModelForQuestionAnswering` (เช่น xlm-roberta) ป้อน (คำถาม, บทความ) -> ได้ช่วงคำตอบ

---

## submission ทุกรูปแบบ + วิธีเติม (สรุป)

| sample เป็นแบบนี้ | แปลว่า | เติมยังไง |
|---|---|---|
| `id,label` ค่า 0/1 หรือชื่อคลาส | ส่งป้าย | `out[col] = preds` |
| `id,target` ค่าทศนิยม 0-1 | ส่งความน่าจะเป็น (AUC/log-loss) | `out[col] = proba_ของคลาสบวก` |
| `id,value` ค่าหลากหลาย | regression | `out[col] = preds (ตัวเลข)` |
| `id,class_0,class_1,...` | prob ต่อคลาส (multiclass) | `out[คอลัมน์คลาส] = predict_proba` (AUTO ทำให้) |
| `id,a,b,c` ค่า 0/1 หลายคอลัมน์ | multi-label | ทำนายแต่ละป้าย เติม 0/1 |
| 1 แถวต่อตัวอักษร/พิกเซล | ตัดคำ/segmentation | สร้าง row ต่อหน่วย (โน้ตบุ๊กเฉพาะ) |
| คอลัมน์ `rle`/`EncodedPixels` | mask แบบ RLE | `rle_encode()` ใน segmentation |

วิธีรู้แน่ ๆ: รันเซลล์ `🔎 โจทย์นี้ต้องส่งอะไร` (มีในทุกโน้ตบุ๊ก + AUTO_SOLVER) -- มันดึง metric จาก Kaggle + บอกชนิดคอลัมน์ + บอกว่าส่งเป็นป้าย/prob/ตัวเลข

---

## สรุปสั้น
- โจทย์ทั่วไป (รูป/ข้อความ/ตาราง/regression/multiclass-prob) -> `00_AUTO_SOLVER` ดักให้หมด 🟢
- โจทย์มีโน้ตบุ๊กเฉพาะ (ตัดคำ/detection/segmentation/captioning/VQA/forecasting/signal) -> AUTO บอกให้ไปเปิด 📓
- โจทย์ยาก/แปลก (multi-label/คู่ประโยค/audio/pose/QA/survival) -> มีสูตรลัดด้านบน 🛠️
- ทุกโน้ตบุ๊กมีเซลล์ `🔎` บอก metric + รูปแบบ submission ให้ก่อนเริ่ม + assert ตรวจฟอร์แมตก่อนส่ง

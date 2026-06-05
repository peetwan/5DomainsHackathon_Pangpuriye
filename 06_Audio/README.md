# หัวข้อใหญ่: Audio / Speech (เสียง)

input คือ `ไฟล์เสียง` (.wav/.mp3/...)

## เกร็งโจทย์ — ออกได้กี่แบบ?

| โจทย์ | output | metric | ใช้ไฟล์ | ง่ายแค่ไหน |
|---|---|---|---|---|
| จำแนกเสียง (1 คลาส/ไฟล์) | คลาส | Accuracy/F1 | `audio_classification.ipynb` | ⭐⭐ |
| ตรวจจับเหตุการณ์เสียง (sound event) | ป้าย/ช่วงเวลา | F1/mAP | ปรับ `audio_classification` (multi-label) | ⭐⭐⭐ |
| ถอดเสียงเป็นข้อความ (ASR) | ข้อความ | CER/WER | Whisper (หมายเหตุในโน้ตบุ๊ก) | ⭐⭐⭐ |

## วิธีที่ง่ายที่สุด
สกัดฟีเจอร์เสียง (`MFCC` + spectral) ด้วย `librosa` -> โยนเข้า `AutoGluon` เหมือนตารางธรรมดา
(เสียงพูดใช้ sample rate 16000, เพลงใช้ 22050)

## กับดักที่ต้องระวัง
- ไฟล์เสียงยาวไม่เท่ากัน -> สกัดเป็น "เฉลี่ย/ส่วนเบี่ยงเบนต่อไฟล์" (โน้ตบุ๊กทำให้แล้ว) ไฟล์สั้นเกินมี pad ให้
- ถ้าโหลด mp3 ไม่ได้ ลง `ffmpeg`: `!apt-get -qq install ffmpeg`
- imbalance: ดู macro-F1 ไม่ใช่ accuracy
- ASR ภาษาไทย: Whisper `small`/`medium` พอใช้ ; วัด CER/WER ในเครื่องก่อนส่ง

ไฟล์ในโฟลเดอร์นี้:
- `audio_classification.ipynb` — จำแนกเสียง (features + AutoGluon/LightGBM) + หมายเหตุ ASR (Whisper)

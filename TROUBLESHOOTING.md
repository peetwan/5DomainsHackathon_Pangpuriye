# 🔧 แก้ปัญหาที่เจอบ่อย (อาการ -> วิธีแก้)

ค้นหาอาการที่เจอ แล้วทำตาม

## Kaggle / ข้อมูล
| อาการ | สาเหตุ + วิธีแก้ |
|---|---|
| `403 Forbidden` ตอนโหลดข้อมูล | ยังไม่ได้กด Join Competition / ยอมรับ rules -> เข้าหน้า comp กด Join ก่อน |
| `401 / OSError: kaggle.json` | token ผิด/ไม่มี -> สร้าง token ใหม่ (Settings > Create New Token) ใส่ username+key ให้ตรง |
| `Could not find kaggle.json` (Colab) | ใส่ `KAGGLE_USERNAME`/`KAGGLE_KEY` ในเซลล์ หรือใช้วิธีอัปโหลด zip (`DATA_DIR="/content"`) |
| ดาวน์โหลดมาแล้วหา train.csv ไม่เจอ | ดูชื่อจริงจาก output (`ไฟล์ที่มี:`) แล้วแก้ `# << แก้` ให้ตรง (อาจชื่อ train_data.csv) |
| `unzip: command not found` | `!apt-get -qq install unzip` (Colab มักมีอยู่แล้ว) |
| โหลด mp3/เสียงไม่ได้ | `!apt-get -qq install ffmpeg` |

## ส่ง submission
| อาการ | วิธีแก้ |
|---|---|
| ส่งแล้วได้ `0 คะแนน` | ฟอร์แมตผิด: คอลัมน์/ลำดับไม่ตรง sample, หรือส่งป้ายแต่ metric เป็น AUC (ต้องส่ง prob) -> รันเซลล์ `🔎 ดูโจทย์` ดูว่าต้องส่งแบบไหน |
| `Submission scoring error` | จำนวนแถวไม่ครบ / มี NaN / ชนิดข้อมูลผิด -> เซลล์ตรวจก่อนส่งจะ assert ให้ ดูข้อความ |
| ส่งไม่ได้ "max submissions reached" | ครบโควต้าวันนี้แล้ว -> รอวันถัดไป (เลยต้องส่ง baseline ก่อน + เลือกไฟล์ดี ๆ ส่ง) |
| คอลัมน์ไม่ตรง (assert ฟ้อง) | `out = out.rename(columns={...})` หรือเพิ่ม/ลบคอลัมน์ให้เท่ากับ sample |

## GPU / หน่วยความจำ / เวลา
| อาการ | วิธีแก้ |
|---|---|
| `CUDA out of memory` | ลด `BATCH` (32->16->8), ลด `IMG_SIZE`/`max_length`, หรือใช้โมเดลเล็กลง |
| `cannot connect to GPU` (Colab) | โควต้า GPU ฟรีหมด -> รอ / ใช้ CPU สำหรับงานตาราง / ลองบัญชีอื่น |
| รันค้างนาน ไม่ขยับ | งานใหญ่/ไม่มี GPU -> ลด epochs/time_limit, เปิด GPU, หรือ sample ข้อมูลมาลองก่อน |
| Colab หลุด/รีเซ็ตเอง | idle 90 นาที หรือครบ 12 ชม. -> ไฟล์ใน /content หาย ต้องโหลดใหม่ (งานยาวเซฟลง Google Drive) |
| `RAM ไม่พอ` ตอนอ่าน CSV ใหญ่ | `pd.read_csv(..., dtype=...)` ลดชนิด, อ่านบางส่วน `nrows=...` มาลองก่อน |

## ไลบรารี
| อาการ | วิธีแก้ |
|---|---|
| ลง AutoGluon แล้ว import error | `Runtime > Restart session` แล้วรันเซลล์ใหม่ (Colab ต้อง restart หลังลงบางแพ็กเกจ) |
| `unexpected keyword eval_strategy` | transformers เก่า -> `!pip install -U transformers` หรือเปลี่ยนเป็น `evaluation_strategy` |
| `Trainer got unexpected keyword tokenizer` | transformers เก่ามาก -> เปลี่ยน `processing_class=` กลับเป็น `tokenizer=` |
| `np.trapz` / `np.int` หาย (numpy 2.0) | โน้ตบุ๊กใช้ `np.trapezoid` fallback แล้ว ถ้าโค้ดอื่นเจอ ให้เปลี่ยนชื่อตาม |
| YOLO `No labels found` | ยังไม่ได้แปลงข้อมูลเป็นรูปแบบ YOLO (ไฟล์ label .txt ต่อรูป) -> ดู object_detection.ipynb |

## คะแนน/โมเดล
| อาการ | วิธีแก้ |
|---|---|
| CV ในเครื่องสูง แต่ public LB ต่ำมาก | ข้อมูลรั่ว (leakage) หรือ overfit -> เช็คว่าไม่มีคอลัมน์รั่ว, แบ่ง CV ให้ถูก (signal แบ่งตาม subject) |
| accuracy สูงแต่ดูแล้วทายคลาสเดียว | ข้อมูลไม่สมดุล -> ดู macro-F1/AUC, ใช้ class_weight |
| ผลรันไม่ซ้ำเดิม | ตั้ง seed: เรียก `seed_everything(42)` (AUTO_SOLVER มีให้) |

ยังไม่หาย? เปิด `COLAB_GUIDE.md` (ตั้งค่า Colab) หรือ `PROBLEM_TYPES.md` (โจทย์แปลก ๆ)

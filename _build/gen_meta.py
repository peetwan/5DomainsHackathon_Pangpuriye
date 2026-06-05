import sys, json; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from router import CATS   # the SAME dict that verify_router.py tested

c = []
c.append(md(r"""# 00 — MASTER ROUTER (เปิดอันนี้ก่อนเสมอ)

ตัวนำทาง: เจอโจทย์แบบไหน เปิดโฟลเดอร์/โน้ตบุ๊กไหน (ครอบคลุมทุกแนวของ 5 หัวข้อใหญ่)

หลักคิดง่าย ๆ: ดูแค่ 2 อย่าง -> `input เป็นอะไร` กับ `output เป็นอะไร`

| ถ้า input/output เป็นแบบนี้ | เปิดไฟล์ |
|---|---|
| รูป -> 1 คลาส/ป้าย | `01_Computer_Vision/image_classification.ipynb` |
| รูป -> กล่องวัตถุ (detection) | `01_Computer_Vision/object_detection.ipynb` |
| รูป -> mask ต่อพิกเซล (segmentation) | `01_Computer_Vision/segmentation.ipynb` |
| ข้อความ -> ป้ายต่อคำ/ตัวอักษร (ตัดคำ/NER/POS) | `02_NLP/thai_word_segmentation.ipynb` |
| ข้อความ -> 1 ป้ายต่อทั้งข้อความ (sentiment/หัวข้อ) | `02_NLP/text_classification.ipynb` |
| ข้อความ -> ข้อความใหม่ (แปล/สรุป/ตอบคำถาม) | `02_NLP/text_generation.ipynb` |
| รูป -> ข้อความบรรยาย (captioning) | `03_Multimodal_VisionLanguage/thai_image_captioning.ipynb` |
| รูป+คำถาม -> คำตอบ / OCR | `03_Multimodal_VisionLanguage/visual_qa.ipynb` |
| ตาราง -> คลาส/ป้าย | `04_Tabular/classification.ipynb` |
| ตาราง -> ตัวเลขต่อเนื่อง | `04_Tabular/regression.ipynb` |
| สัญญาณตามเวลา -> 1 คลาส | `05_TimeSeries_Signal/signal_classification.ipynb` |
| ค่าตามเวลา -> ทำนายอนาคต (forecast) | `05_TimeSeries_Signal/forecasting.ipynb` |

ทุกโน้ตบุ๊กมี `วิธีที่ 1 (ง่ายสุด)` อยู่บนสุด มือใหม่ทำแค่วิธีที่ 1 ก็ได้คะแนนแล้ว
"""))

c.append(md(r"""## ตัวช่วยอัตโนมัติ — พิมพ์โจทย์ที่เจอ (ไทย/อังกฤษ) แล้วให้มันบอกว่าเปิดไฟล์ไหน

ตัวจับคำนี้ผ่านการทดสอบกับโจทย์จริง 34 แบบ (ไทย+อังกฤษ) แล้ว แก้ `PROBLEM` แล้วรัน"""))

# ---- build the router code cell from the tested CATS + route() (single source of truth) ----
router_src = (
    "import re\n"
    "PROBLEM = \"ทำนายระยะการนอนจากสัญญาณ EEG ทุก 30 วินาที\"   # << แก้: ใส่คำอธิบายโจทย์ที่เจอ (ไทยหรืออังกฤษก็ได้)\n\n"
    "CATS = " + json.dumps(CATS, ensure_ascii=False, indent=1) + "\n\n"
    "def _hit(kw, p):\n"
    "    # คำอังกฤษ -> match แบบมีขอบคำ (กัน 'ner' ไปโดน 'generate') ; คำไทย -> substring\n"
    "    kw = kw.lower()\n"
    "    if kw.isascii():\n"
    "        return re.search(r\"(?<![a-z])\" + re.escape(kw) + r\"(?![a-z])\", p) is not None\n"
    "    return kw in p\n\n"
    "def route(problem):\n"
    "    p = (problem or \"\").lower()\n"
    "    items = [(sum(1 for k in v[\"kw\"] if _hit(k, p)), name) for name, v in CATS.items()]\n"
    "    return sorted(items, key=lambda x: -x[0])\n\n"
    "scored = route(PROBLEM)\n"
    "best_score, best = scored[0]\n"
    "print(\"โจทย์:\", PROBLEM, \"\\n\")\n"
    "if best_score == 0:\n"
    "    print(\"เดาไม่ออกจากคำ ลองดูตารางด้านบน (input เป็นอะไร -> output เป็นอะไร)\")\n"
    "else:\n"
    "    print(\">> เปิด:\", CATS[best][\"path\"])\n"
    "    print(\">> หมวด:\", CATS[best][\"label\"])\n"
    "    print(\">> เริ่มจาก:\", CATS[best][\"first\"], \"\\n\")\n"
    "print(\"คะแนนแต่ละหมวด (มากสุด = ตรงสุด):\")\n"
    "for s, name in scored:\n"
    "    if s > 0: print(f\"  {s}  {CATS[name]['path']}\")\n"
)
c.append(code(router_src))

c.append(md(r"""## ลองหลายโจทย์พร้อมกัน (ตัวอย่างไทย+อังกฤษ)

แก้ลิสต์ `EXAMPLES` เพิ่มโจทย์ที่อยากลองได้"""))
c.append(code(r"""EXAMPLES = [
    "จำแนกรูปบ้านเป็น 2 ประเภท",
    "Detect and count cars in the image",
    "ตัดคำภาษาไทย",
    "classify movie reviews sentiment",
    "แปลอังกฤษเป็นไทย",
    "บรรยายรูปเป็นภาษาไทย วัด BLEU",
    "อ่านตัวอักษรในรูป OCR",
    "predict house price from tabular data",
    "ทำนายระยะการนอนจาก EEG",
    "forecast next week sales",
]
for q in EXAMPLES:
    sc = route(q); best = sc[0][1]
    print(f"{q[:42]:44s} -> {CATS[best]['path']}")"""))

c.append(md(r"""## สิ่งที่ต้องแก้เหมือนกันทุกโน้ตบุ๊ก (หาคำว่า `# << แก้`)

1. `COMP` = slug ของ competition (ส่วนท้าย URL หลัง `/competitions/`)
2. `KAGGLE_USERNAME`, `KAGGLE_KEY` = ใส่เฉพาะตอนรันบน Colab/เครื่องตัวเอง (บน Kaggle ไม่ต้อง)
3. ชื่อไฟล์ / โฟลเดอร์รูป / ชื่อคอลัมน์ = ดูจาก output ตอนดาวน์โหลดแล้วแก้ให้ตรง
4. `METRIC` = ดูจาก tab `Evaluation` บนหน้า Kaggle จริง
5. `time_limit` / `EPOCHS` = ปรับตามเวลาสอบและ GPU

กฎเหล็ก: ไฟล์ `submission.csv` ต้องมีคอลัมน์ + จำนวนแถว ตรงกับ `sample_submission.csv` เป๊ะ
(โค้ดทุกตัว copy โครงจาก sample มาให้แล้ว)

แนะนำ runtime: รันบน `Google Colab` หรือ `Kaggle Notebook` (มี GPU ฟรี + ไลบรารีพร้อม) จะง่ายสุด"""))

c.append(md(r"""## ลำดับทำงานในห้องสอบ

1. อ่านโจทย์ -> ใช้ตัวช่วยด้านบน หรือดูตาราง -> เปิดไฟล์ให้ถูก
2. เปิดหน้า Kaggle อ่าน tab `Data` + `Evaluation` จด: คอลัมน์, metric, จำนวนคลาส, รูปแบบ submission
3. แก้ CONFIG ตาม `# << แก้`
4. รัน `วิธีที่ 1 (ง่ายสุด)` ให้มี submission ขึ้นกระดานก่อน
5. เหลือเวลาค่อยทำวิธีที่ 2/3 เพื่อดันคะแนน"""))

write_nb("00_MASTER_ROUTER.ipynb", c)

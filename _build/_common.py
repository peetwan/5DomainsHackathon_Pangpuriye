# Shared beginner-friendly cells reused across all topic notebooks.
import sys; sys.path.insert(0, "_build")
from _nbutil import md, code

def kaggle_md():
    return md(r"""## ขั้นที่ 2 — เอาข้อมูลเข้า (เลือก 1 ใน 3 วิธี) + เช็ค GPU

เซลล์ล่างรองรับ 3 วิธี แก้แค่ตัวแปรบนสุดให้ตรงกับวิธีที่ใช้:

วิธี A (แนะนำ) ดาวน์โหลดจาก Kaggle อัตโนมัติ: ต้องมี `kaggle.json` (kaggle.com -> รูปโปรไฟล์ -> Settings -> Account -> Create New Token)
- บน `Kaggle Notebook`: ข้อมูลต่อไว้ที่ `/kaggle/input/...` แล้ว ไม่ต้องใส่ token
- บน `Colab/เครื่อง`: ใส่ `KAGGLE_USERNAME` + `KAGGLE_KEY`

วิธี B โหลดข้อมูลมาเครื่องตัวเองแล้วอัปขึ้น Colab เอง: ลากไฟล์ (เช่น `data.zip`) ไปวางในแถบ Files ซ้ายมือของ Colab
แล้วตั้ง `DATA_DIR = "/content"` (หรือโฟลเดอร์ที่วาง) -> เซลล์จะแตก zip ให้เอง ไม่ต้องใช้ token

วิธี C ต่อ Google Drive: รัน `from google.colab import drive; drive.mount('/content/drive')` ก่อน แล้วตั้ง `DATA_DIR = "/content/drive/MyDrive/โฟลเดอร์ของคุณ"`

เซลล์นี้เช็ค GPU ให้ด้วย ถ้างานเป็น deep learning (รูป/ข้อความ/สัญญาณดิบ) ควรเปิด GPU: เมนู `Runtime > Change runtime type > T4 GPU`""")

def kaggle_cell(comp):
    src = r'''import os, glob, subprocess

# ----- วิธี A: ดาวน์โหลดจาก Kaggle -----
COMP = "__COMP__"      # << แก้: slug ท้าย URL เช่น kaggle.com/competitions/titanic -> "titanic"  (ใช้เฉพาะวิธี A)
KAGGLE_USERNAME = ""   # << แก้ (วิธี A, เฉพาะ Colab/เครื่อง) เช่น "peetwan1"  (บน Kaggle Notebook เว้นว่างได้)
KAGGLE_KEY      = ""   # << แก้ (วิธี A, เฉพาะ Colab/เครื่อง) เช่น "0a1b2c3d4e5f..." (คีย์ยาว ~32 ตัวจาก kaggle.json)
# ----- วิธี B/C: อัปโหลดเอง / ต่อ Google Drive -----
DATA_DIR = ""          # << แก้: ถ้าอัปโหลดข้อมูลเอง ใส่ path โฟลเดอร์ เช่น "/content" (วิธี B) หรือ "/content/drive/MyDrive/data" (วิธี C) ; ใช้ Kaggle (วิธี A) ปล่อยว่าง

# เช็ค GPU (ถ้าไม่มี + งานเป็น deep learning -> เปิดที่เมนู Runtime > Change runtime type > T4 GPU)
try:
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ไม่มี (งาน deep learning จะช้า; งานตาราง/ตัดคำ ไม่ต้องใช้ GPU ก็ได้)")
except Exception:
    pass

def get_data(comp):
    # วิธี B/C: ผู้ใช้ตั้ง DATA_DIR เอง (อัปโหลด/ต่อ Drive) -- แตก zip ในโฟลเดอร์นั้นให้ด้วยถ้ามี
    if DATA_DIR and os.path.isdir(DATA_DIR):
        for z in glob.glob(os.path.join(DATA_DIR, "*.zip")):
            subprocess.run(["unzip", "-o", "-q", z, "-d", DATA_DIR])
        print("ใช้ข้อมูลจากโฟลเดอร์ที่ตั้งเอง:", DATA_DIR); return DATA_DIR
    # บน Kaggle Notebook ข้อมูลต่อไว้แล้ว
    kpath = f"/kaggle/input/{comp}"
    if os.path.isdir(kpath):
        print("อยู่บน Kaggle ข้อมูลอยู่ที่", kpath); return kpath
    # วิธี A: ดาวน์โหลดด้วย Kaggle API
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        open(os.path.expanduser("~/.kaggle/kaggle.json"), "w").write(
            '{"username":"%s","key":"%s"}' % (KAGGLE_USERNAME, KAGGLE_KEY))
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    os.makedirs("data", exist_ok=True)
    subprocess.run(["kaggle","competitions","download","-c",comp,"-p","data"], check=True)
    for z in glob.glob("data/*.zip"):
        subprocess.run(["unzip","-o","-q",z,"-d","data"], check=True)
    return "data"

DATA_DIR = get_data(COMP)
print("\nไฟล์ที่มี (ดูชื่อไฟล์/โฟลเดอร์ แล้วแก้ในเซลล์ถัดไปให้ตรง):")
for p in sorted(glob.glob(os.path.join(DATA_DIR, "**"), recursive=True))[:40]:
    print("  ", p)'''
    return code(src.replace("__COMP__", comp))

def inspect_md():
    return md(r"""## 🔎 โจทย์นี้ต้องส่งอะไร + วัดด้วยอะไร (รันก่อนทำ submission)

เซลล์นี้ตอบ 3 คำถามที่มือใหม่มักไม่รู้:
- ต้องเติม "คอลัมน์อะไร" ลง submission และเป็น "ชนิดไหน" (ดูจาก sample_submission)
- โจทย์วัดด้วย "metric อะไร" (ดึงจาก Kaggle ให้อัตโนมัติ)
- ต้องส่งเป็น "ป้าย / ความน่าจะเป็น / ตัวเลข" แบบไหน""")

def inspect_cell():
    src = (
        '# บอกว่าต้องเติมอะไรลง submission + ตัวอย่างค่าที่ sample ให้มา\n'
        'print("ไฟล์ส่งต้องมีคอลัมน์:", list(sample.columns), "| จำนวนแถว:", len(sample))\n'
        'for _c in list(sample.columns)[1:]:\n'
        '    print(f"  - เติม \'{_c}\': ชนิด {sample[_c].dtype}, ตัวอย่างค่าใน sample = {list(sample[_c].dropna().unique()[:3])}")\n'
        '# ดึง metric จาก Kaggle อัตโนมัติ (ถ้าต่อ API ได้)\n'
        '_metric = None\n'
        'try:\n'
        '    from kaggle.api.kaggle_api_extended import KaggleApi\n'
        '    _api = KaggleApi(); _api.authenticate()\n'
        '    _resp = _api.competitions_list(search=COMP)\n'
        '    for _co in (getattr(_resp, "competitions", _resp) or []):   # kaggle ใหม่คืน response (ใช้ .competitions), เก่าคืน list\n'
        '        if str(getattr(_co, "ref", "")).rstrip("/").split("/")[-1] == COMP:\n'
        '            _metric = getattr(_co, "evaluation_metric", None) or getattr(_co, "evaluationMetric", None); break\n'
        'except Exception:\n'
        '    pass\n'
        'def _how_to_send(m):\n'
        '    m = (m or "").lower()\n'
        '    if any(k in m for k in ["auc","roc","log loss","logloss","brier","probab"]): return "ความน่าจะเป็น (เลข 0-1)"\n'
        '    if any(k in m for k in ["rmse","mae","mse","r2","rmsle","error","mape","smape"]): return "ตัวเลขต่อเนื่อง"\n'
        '    return "ป้าย/คลาส (เช่น 0/1 หรือชื่อคลาส)"\n'
        'print("\\nMetric (ดึงจาก Kaggle):", _metric or "ดึงไม่ได้ -> เปิด tab Evaluation บนหน้า Kaggle อ่านเอง")\n'
        'print("=> ปกติต้องส่งเป็น:", _how_to_send(_metric), "(เช็คกับ tab Evaluation อีกที)")'
    )
    return code(src)

def submit_md():
    return md(r"""## ขั้นสุดท้าย — ส่ง submission ขึ้น Kaggle

เช็คก่อนส่งทุกครั้ง: ไฟล์ submission ต้องมีชื่อคอลัมน์ + จำนวนแถว ตรงกับ `sample_submission.csv` เป๊ะ ๆ
- บน Kaggle Notebook: กดปุ่ม `Submit` ที่แถบขวา (ง่ายสุด) หรือใช้คำสั่งข้างล่าง
- บน Colab/เครื่อง: รันเซลล์ข้างล่าง (ต้องตั้ง token แล้ว)""")

def submit_cell(msg):
    src = (
        'import pandas as pd, glob, os\n'
        'FILE = "submission.csv"   # << แก้เป็นไฟล์ที่จะส่ง เช่น "submission_lgbm.csv" หรือ "submission_timm.csv"\n'
        '# ตรวจรูปแบบก่อนส่งอัตโนมัติ (กันแก้ config ผิดแล้วส่งฟอร์แมตเพี้ยน -> ได้ 0 คะแนน)\n'
        '_sub = pd.read_csv(FILE)\n'
        '_sam = glob.glob(os.path.join(DATA_DIR, "*ample*ubmission*.csv"))\n'
        'if _sam:\n'
        '    _s = pd.read_csv(_sam[0])\n'
        '    print("คอลัมน์ตรง sample:", list(_sub.columns)==list(_s.columns), "| จำนวนแถวตรง:", len(_sub)==len(_s))\n'
        '    assert list(_sub.columns)==list(_s.columns), f"คอลัมน์ไม่ตรง sample! {list(_sub.columns)} != {list(_s.columns)} -> แก้ก่อนส่ง"\n'
        'print("พร้อมส่ง:", FILE, _sub.shape)\n'
        '!kaggle competitions submit -c {COMP} -f {FILE} -m "__MSG__"\n'
        '!kaggle competitions submissions -c {COMP} | head'
    )
    return code(src.replace("__MSG__", msg))

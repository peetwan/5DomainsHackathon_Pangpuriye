# Shared beginner-friendly cells reused across all topic notebooks.
import sys; sys.path.insert(0, "_build")
from _nbutil import md, code

def kaggle_md():
    return md(r"""## ขั้นที่ 2 — ตั้งค่า Kaggle แล้วดาวน์โหลดข้อมูล

ต้องมีไฟล์ `kaggle.json` ก่อน วิธีได้มา: เข้า kaggle.com -> กดรูปโปรไฟล์ -> Settings -> Account -> Create New Token
จะได้ไฟล์ `kaggle.json` หน้าตา `{"username":"...","key":"..."}`

- ถ้ารันบน `Kaggle Notebook`: ข้อมูลอยู่ที่ `/kaggle/input/...` แล้ว ไม่ต้องทำอะไร รันผ่านได้เลย
- ถ้ารันบน `Google Colab / เครื่องตัวเอง`: เอา username กับ key ใส่ในเซลล์ข้างล่าง (จุด `# << แก้`)""")

def kaggle_cell(comp):
    src = r'''import os, glob, subprocess

COMP = "__COMP__"      # << แก้ตรงนี้: slug ของ competition (คือส่วนท้าย URL หลังคำว่า /competitions/)
KAGGLE_USERNAME = ""   # << แก้ (เฉพาะ Colab/เครื่องตัวเอง) เช่น "peetwan1"
KAGGLE_KEY      = ""   # << แก้ (เฉพาะ Colab/เครื่องตัวเอง) คีย์ยาว ๆ จาก kaggle.json

def get_data(comp):
    # ถ้าอยู่บน Kaggle ข้อมูลถูกต่อไว้ให้แล้ว
    kpath = f"/kaggle/input/{comp}"
    if os.path.isdir(kpath):
        print("อยู่บน Kaggle แล้ว ข้อมูลอยู่ที่", kpath); return kpath
    # ถ้าไม่ใช่ Kaggle: เขียน token แล้วโหลดด้วย kaggle CLI
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
print("ไฟล์ที่ดาวน์โหลดมา (ดูชื่อไฟล์/โฟลเดอร์ แล้วเอาไปแก้ในเซลล์ถัดไป):")
for p in sorted(glob.glob(os.path.join(DATA_DIR, "**"), recursive=True))[:40]:
    print("  ", p)'''
    return code(src.replace("__COMP__", comp))

def submit_md():
    return md(r"""## ขั้นสุดท้าย — ส่ง submission ขึ้น Kaggle

เช็คก่อนส่งทุกครั้ง: ไฟล์ submission ต้องมีชื่อคอลัมน์ + จำนวนแถว ตรงกับ `sample_submission.csv` เป๊ะ ๆ
- บน Kaggle Notebook: กดปุ่ม `Submit` ที่แถบขวา (ง่ายสุด) หรือใช้คำสั่งข้างล่าง
- บน Colab/เครื่อง: รันเซลล์ข้างล่าง (ต้องตั้ง token แล้ว)""")

def submit_cell(msg):
    src = ('FILE = "submission.csv"   # << แก้เป็นชื่อไฟล์ที่คะแนนดีสุด\n'
           '!kaggle competitions submit -c {COMP} -f {FILE} -m "__MSG__"\n'
           '!kaggle competitions submissions -c {COMP} | head')
    return code(src.replace("__MSG__", msg))

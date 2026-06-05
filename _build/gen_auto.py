import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell

c = []
c.append(md(r"""# 🪄 AUTO SOLVER — ปุ่มเดียวจบ (สำหรับคนที่อยากง่ายที่สุด)

โน้ตบุ๊กนี้ "เดาเอง" ว่าโจทย์เป็นงานประเภทไหน แล้วเลือกวิธี + เทรน + สร้าง submission ให้อัตโนมัติ
คุณแก้แค่ `ที่เดียว` (ชื่อ competition หรืออัปโหลดไฟล์) แล้วกดรันทีละเซลล์จากบนลงล่าง

ทำอัตโนมัติให้ได้: จำแนกรูป (image), จำแนกข้อความ (text), ตาราง-จำแนก/ทำนายตัวเลข (tabular/regression), สัญญาณ/พยากรณ์ (ทำเป็นตาราง)
ทำให้ไม่ได้ (จะบอกให้ไปเปิดโน้ตบุ๊กเฉพาะ): ตัดคำ, segmentation, บรรยายรูป/VQA, detection -- พวกนี้ฟอร์แมตพิเศษ

ขั้นตอน: (1) ติดตั้ง (2) ใส่ข้อมูล/ชื่อ comp (3) กดรันเซลล์ AUTO (4) ส่ง -- จบ
"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง (กดรันรอ ~5-10 นาทีครั้งแรก)"""))
c.append(code(r"""!pip -q install -U "autogluon.tabular[all]" "autogluon.multimodal" pandas numpy pillow"""))

c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — กดรัน AUTO (เดา+เทรน+สร้าง submission ให้อัตโนมัติ)

ปกติไม่ต้องแก้อะไรในเซลล์นี้ มันจะ print ว่าเดาว่าเป็นงานอะไร + คอลัมน์ไหน ให้เช็คได้
ถ้าเดาผิด ค่อยตั้ง `FORCE_TASK` / `METRIC` ที่บรรทัดบนสุดของเซลล์"""))
c.append(code(r"""import os, glob, pandas as pd, numpy as np
IMG_EXT = (".jpg",".jpeg",".png",".bmp")

# ===== ปรับเฉพาะถ้า auto เดาผิด (ปกติปล่อยไว้) =====
FORCE_TASK = None    # << ใส่ได้: "image" / "text" / "tabular" / "regression"
METRIC     = None    # << ใส่ "roc_auc" ถ้าโจทย์วัด AUC (จะได้ส่งความน่าจะเป็น)

def _find(name):
    h = glob.glob(os.path.join(DATA_DIR, "**", name), recursive=True); return h[0] if h else None
def _img_dir(keyword):
    cand = [d for d,_,fs in os.walk(DATA_DIR) if keyword in d.lower()
            and any(f.lower().endswith(IMG_EXT) for f in fs)]
    return max(cand, key=lambda d: len(os.listdir(d))) if cand else None

sample = pd.read_csv(_find("sample_submission.csv"))
ID_COL, ANS_COL = sample.columns[0], sample.columns[1]
train = pd.read_csv(_find("train.csv")) if _find("train.csv") else None
test  = pd.read_csv(_find("test.csv"))  if _find("test.csv")  else None
train_img, test_img = _img_dir("train"), _img_dir("test")
print("ไฟล์ส่งต้องมีคอลัมน์:", list(sample.columns), "| id =", ID_COL, "| คำตอบ =", ANS_COL)

assert train is not None, "หา train.csv ไม่เจอ -> เช็ค path/ใช้โน้ตบุ๊กเฉพาะ"

# ----- กันงานฟอร์แมตพิเศษ (submission ไม่ใช่ 1 แถวต่อ 1 ตัวอย่าง test) -----
if test is not None and len(sample) != len(test) and not (train_img and test_img):
    raise SystemExit("งานนี้ submission ไม่ใช่ 1 แถวต่อ 1 แถวของ test (น่าจะเป็นตัดคำ/segmentation) "
                     "-> เปิด 00_MASTER_ROUTER.ipynb พิมพ์โจทย์ แล้วใช้โน้ตบุ๊กเฉพาะ")

# ----- เดาคอลัมน์เป้าหมาย (คอลัมน์ที่อยู่ใน train แต่ไม่อยู่ใน test) -----
def detect_target():
    if test is not None:
        cand = [c for c in train.columns if c not in test.columns and c != ID_COL]
        if len(cand) == 1: return cand[0]
    if ANS_COL in train.columns: return ANS_COL
    return [c for c in train.columns if c != ID_COL][-1]

# ----- เดาประเภทงาน -----
def detect_task():
    if FORCE_TASK: return FORCE_TASK
    if train_img and test_img:
        imgcols = [c for c in train.columns if train[c].astype(str).str.lower().str.endswith(IMG_EXT).mean() > 0.5]
        if imgcols: return "image"
    textcols = [c for c in train.columns if train[c].dtype == object and c != ID_COL
                and train[c].astype(str).str.len().mean() >= 10]
    if textcols: return "text"
    tgt = detect_target()
    if pd.api.types.is_numeric_dtype(train[tgt]) and train[tgt].nunique() > 20: return "regression"
    return "tabular"

task = detect_task()
print("ตรวจพบว่าเป็นงาน:", {"image":"จำแนกรูป","text":"จำแนกข้อความ","tabular":"ตาราง-จำแนกคลาส",
                          "regression":"ตาราง-ทำนายตัวเลข"}.get(task, task))

out = sample.copy()
if task == "image":
    from autogluon.multimodal import MultiModalPredictor
    IMG_COL = [c for c in train.columns if train[c].astype(str).str.lower().str.endswith(IMG_EXT).mean() > 0.5][0]
    LABEL_COL = [c for c in train.columns if c not in (IMG_COL, ID_COL)][0]
    exts = [os.path.splitext(f)[1] for f in os.listdir(test_img) if f.lower().endswith(IMG_EXT)]
    ext = exts[0] if exts else ".jpg"
    print("  รูป:", IMG_COL, "| ป้าย:", LABEL_COL, "| นามสกุลไฟล์ test:", ext)
    tr = train.copy(); tr["image"] = tr[IMG_COL].apply(lambda n: os.path.join(train_img, str(n)))
    p = MultiModalPredictor(label=LABEL_COL, path="ag_auto").fit(tr[["image", LABEL_COL]], time_limit=900)
    td = sample.copy(); td["image"] = td[ID_COL].apply(lambda i: os.path.join(test_img, str(i)+ext))
    out[ANS_COL] = p.predict(td[["image"]]).values

elif task == "text":
    from autogluon.multimodal import MultiModalPredictor
    TEXT_COL = [c for c in train.columns if train[c].dtype == object and c != ID_COL
                and train[c].astype(str).str.len().mean() >= 10][0]
    LABEL_COL = detect_target()
    print("  ข้อความ:", TEXT_COL, "| ป้าย:", LABEL_COL)
    p = MultiModalPredictor(label=LABEL_COL, path="ag_auto").fit(train[[TEXT_COL, LABEL_COL]], time_limit=600)
    out[ANS_COL] = p.predict(test[[TEXT_COL]]).values

else:  # tabular / regression
    from autogluon.tabular import TabularPredictor
    LABEL_COL = detect_target()
    is_reg = (task == "regression")
    print("  เป้าหมาย:", LABEL_COL, "| ชนิด:", "ทำนายตัวเลข" if is_reg else "จำแนกคลาส")
    tr = train.drop(columns=[c for c in [ID_COL] if c in train.columns])
    te = test.drop(columns=[c for c in [ID_COL] if c in test.columns])
    p = TabularPredictor(label=LABEL_COL, problem_type=("regression" if is_reg else None),
                         path="ag_auto").fit(tr, presets="best_quality", time_limit=600)
    if (not is_reg) and METRIC in ("roc_auc","auc") and getattr(p, "positive_class", None) is not None:
        out[ANS_COL] = p.predict_proba(te)[p.positive_class].values   # ส่งความน่าจะเป็น (AUC)
    else:
        out[ANS_COL] = p.predict(te).values
        # งานจำแนก 2 คลาส: เซฟไฟล์ความน่าจะเป็นไว้ด้วย เผื่อ metric เป็น AUC
        if (not is_reg) and getattr(p, "positive_class", None) is not None and train[LABEL_COL].nunique() == 2:
            o2 = sample.copy(); o2[ANS_COL] = p.predict_proba(te)[p.positive_class].values
            o2.to_csv("submission_proba.csv", index=False)
            print("  (เซฟ submission_proba.csv ไว้ด้วย เผื่อโจทย์วัดด้วย AUC)")

out.to_csv("submission.csv", index=False)
print("\nเสร็จ! บันทึก submission.csv", out.shape);
try: display(out.head())
except Exception: print(out.head())"""))

c.append(submit_md()); c.append(submit_cell("auto solver"))
write_nb("00_AUTO_SOLVER.ipynb", c)

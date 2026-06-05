import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell

c = []
c.append(md(r"""# 🪄 AUTO SOLVER — ปุ่มเดียวจบ (All-in-one สำหรับคนที่อยากง่ายที่สุด)

โน้ตบุ๊กนี้ "เดาเอง" ว่าโจทย์เป็นงานอะไร แล้วเลือกวิธี + เทรน + สร้าง submission ให้อัตโนมัติ
คุณแก้แค่ `ที่เดียว` (ชื่อ competition หรืออัปโหลดไฟล์) แล้วกดรันทีละเซลล์จากบนลงล่าง

ในนี้มีให้ครบ: ดูโจทย์ (metric/ฟอร์แมต) -> ดูข้อมูล (EDA) -> ส่ง baseline ก่อน -> เทรนจริง + ดูคะแนนในเครื่อง -> รวมไฟล์ (blend) -> ส่ง
ทำอัตโนมัติได้: จำแนกรูป / จำแนกข้อความ / ตาราง-จำแนก/ทำนายตัวเลข (รวม multiclass แบบส่ง prob)
ทำให้ไม่ได้ (จะบอกให้ไปเปิดโน้ตบุ๊กเฉพาะ): ตัดคำ, segmentation, บรรยายรูป/VQA, detection, เสียง(audio)
"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง (กดรันรอ ~5-10 นาทีครั้งแรก)"""))
c.append(code(r"""!pip -q install -U "autogluon.tabular[all]" "autogluon.multimodal" pandas numpy pillow"""))

c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — ดูโจทย์ (ต้องส่งอะไร/metric) + ดูข้อมูล (EDA) + เดาประเภทงาน

รันแล้วมันบอกเอง: ต้องเติมคอลัมน์อะไร, metric อะไร (ดึงจาก Kaggle), ข้อมูลหน้าตาเป็นไง, และเดาว่าเป็นงานอะไร
ถ้าเดาผิด ค่อยตั้ง `FORCE_TASK` / `METRIC` ที่บรรทัดบนสุด"""))
c.append(code(r"""import os, glob, random, pandas as pd, numpy as np
IMG_EXT = (".jpg",".jpeg",".png",".bmp")
AUDIO_EXT = (".wav",".mp3",".flac",".ogg",".m4a")

# ===== ปรับเฉพาะถ้า auto เดาผิด (ปกติปล่อยไว้) =====
FORCE_TASK = None    # << ใส่ได้: "image" / "text" / "tabular" / "regression"
METRIC     = None    # << ใส่ "roc_auc" ถ้าโจทย์วัด AUC (จะได้ส่งความน่าจะเป็น)

def seed_everything(s=42):   # ตั้ง seed ให้ผลรันซ้ำได้
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception: pass
seed_everything(42)

def _find(name):
    h = glob.glob(os.path.join(DATA_DIR, "**", name), recursive=True); return h[0] if h else None
def _dir_with(exts, keyword=None):
    cand = [d for d,_,fs in os.walk(DATA_DIR) if any(f.lower().endswith(exts) for f in fs)
            and (keyword is None or keyword in d.lower())]
    return max(cand, key=lambda d: len(os.listdir(d))) if cand else None

sample = pd.read_csv(_find("sample_submission.csv"))
ID_COL, ANS_COL = sample.columns[0], sample.columns[1]
train = pd.read_csv(_find("train.csv")) if _find("train.csv") else None
test  = pd.read_csv(_find("test.csv"))  if _find("test.csv")  else None
train_img, test_img = _dir_with(IMG_EXT, "train"), _dir_with(IMG_EXT, "test")
has_audio = _dir_with(AUDIO_EXT) is not None

# 🔎 ต้องส่งอะไร + metric อะไร
print("ไฟล์ส่งต้องมีคอลัมน์:", list(sample.columns), "| จำนวนแถว:", len(sample))
for _c in list(sample.columns)[1:]:
    print(f"  - เติม '{_c}': ชนิด {sample[_c].dtype}, ตัวอย่าง = {list(sample[_c].dropna().unique()[:3])}")
_metric = None
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    _api = KaggleApi(); _api.authenticate()
    _resp = _api.competitions_list(search=COMP)
    for _co in (getattr(_resp, "competitions", _resp) or []):
        if str(getattr(_co, "ref", "")).rstrip("/").split("/")[-1] == COMP:
            _metric = getattr(_co, "evaluation_metric", None) or getattr(_co, "evaluationMetric", None); break
except Exception:
    pass
_ml = (_metric or "").lower()
WANT_PROB = (METRIC in ("roc_auc","auc")) or any(k in _ml for k in ["auc","roc","log loss","logloss","brier","probab"])
MULTICOL  = len(sample.columns) > 2
print("Metric (จาก Kaggle):", _metric or "ดึงไม่ได้ (เปิด tab Evaluation อ่านเอง)",
      "| auto จะส่งเป็น:", "ความน่าจะเป็น" if (WANT_PROB or MULTICOL) else "ป้าย/ตัวเลข")

# 👀 EDA สั้น ๆ
if train is not None:
    print("\nEDA: train", train.shape, "| แถวซ้ำ", int(train.duplicated().sum()))
    _miss = (train.isnull().mean()*100).round(1); _miss = _miss[_miss > 0]
    if len(_miss): print("  คอลัมน์มีค่าว่าง(%):", _miss.to_dict())
    if test is not None:
        print("  คอลัมน์มีใน train ไม่มีใน test:", (set(train.columns)-set(test.columns)) or "(ไม่มี)")

# 🚧 ดักงานที่ AUTO ทำไม่ได้ -> บอกให้ไปโน้ตบุ๊กเฉพาะ
if has_audio and not (train_img and test_img):
    raise SystemExit("งานนี้เป็นเสียง (audio) -> เปิด 06_Audio/audio_classification.ipynb")
assert train is not None, "หา train.csv ไม่เจอ -> เช็ค path / ใช้โน้ตบุ๊กเฉพาะ"
if test is not None and len(sample) != len(test) and not (train_img and test_img):
    raise SystemExit("submission ไม่ใช่ 1 แถวต่อ 1 แถวของ test (ตัดคำ/segmentation) -> เปิด 00_MASTER_ROUTER เลือกโน้ตบุ๊กเฉพาะ")
assert (test is not None) or (train_img and test_img), \
    "หา test.csv ไม่เจอ (อาจชื่ออื่น) -> เปลี่ยนชื่อเป็น test.csv หรือใช้โน้ตบุ๊กเฉพาะ"

def detect_target():
    if test is not None:
        cand = [c for c in train.columns if c not in test.columns and c != ID_COL]
        if len(cand) == 1: return cand[0]
    if ANS_COL in train.columns: return ANS_COL
    return [c for c in train.columns if c != ID_COL][-1]
def detect_task():
    if FORCE_TASK: return FORCE_TASK
    if train_img and test_img:
        ic = [c for c in train.columns if train[c].astype(str).str.lower().str.endswith(IMG_EXT).mean() > 0.5]
        if ic: return "image"
    tc = [c for c in train.columns if train[c].dtype == object and c != ID_COL and train[c].astype(str).str.len().mean() >= 10]
    if tc: return "text"
    tgt = detect_target()
    if pd.api.types.is_numeric_dtype(train[tgt]) and train[tgt].nunique() > 20: return "regression"
    return "tabular"
task = detect_task()
print("\nตรวจพบว่าเป็นงาน:", {"image":"จำแนกรูป","text":"จำแนกข้อความ","tabular":"ตาราง-จำแนกคลาส",
                          "regression":"ตาราง-ทำนายตัวเลข"}.get(task, task))"""))

c.append(md(r"""## ขั้นที่ 4 — ส่ง baseline ก่อน (สำคัญมาก!)

เซลล์นี้สร้าง submission แบบโง่ ๆ (ทายคลาสที่เจอบ่อยสุด / ค่าเฉลี่ย) ใน ~0 วินาที
ส่งอันนี้ "ก่อน" เทรนจริง เพื่อ (1) ล็อกว่าฟอร์แมตถูก ส่งได้ (2) ได้คะแนนพื้นไว้ก่อน กันพลาดเสียสิทธิ์ส่ง
วิธีส่ง: ไปขั้นสุดท้าย เปลี่ยน `FILE = "submission_baseline.csv"` แล้วรัน"""))
c.append(code(r"""base = sample.copy()
if (not MULTICOL) and (ANS_COL in train.columns):
    _t = train[ANS_COL]
    base[ANS_COL] = _t.mean() if (pd.api.types.is_numeric_dtype(_t) and _t.nunique() > 20) else _t.mode()[0]
base.to_csv("submission_baseline.csv", index=False)
assert list(base.columns) == list(sample.columns), "ฟอร์แมต baseline ไม่ตรง sample"
print("บันทึก submission_baseline.csv -> แนะนำส่งอันนี้ก่อนเลย")
print("ค่าที่เติม:", base[ANS_COL].iloc[0] if ANS_COL in base else "(หลายคอลัมน์ = 0)")"""))

c.append(md(r"""## ขั้นที่ 5 — AUTO เทรนจริง + สร้าง submission + ดูคะแนนในเครื่อง

มันจะ print "คะแนน validation ในเครื่อง" ให้ด้วย จะได้รู้ว่าดีพอจะส่งไหม โดยไม่ต้องเปลือง quota"""))
c.append(code(r"""out = sample.copy()
if task == "image":
    from autogluon.multimodal import MultiModalPredictor
    IMG_COL = [c for c in train.columns if train[c].astype(str).str.lower().str.endswith(IMG_EXT).mean() > 0.5][0]
    LABEL_COL = [c for c in train.columns if c not in (IMG_COL, ID_COL)][0]
    exts = [os.path.splitext(f)[1] for f in os.listdir(test_img) if f.lower().endswith(IMG_EXT)]
    ext = exts[0] if exts else ".jpg"
    print("  รูป:", IMG_COL, "| ป้าย:", LABEL_COL, "| นามสกุล test:", ext)
    tr = train.copy(); tr["image"] = tr[IMG_COL].apply(lambda n: os.path.join(train_img, str(n)))
    p = MultiModalPredictor(label=LABEL_COL, path="ag_auto").fit(tr[["image", LABEL_COL]], time_limit=900)
    td = sample.copy(); td["image"] = td[ID_COL].apply(lambda i: os.path.join(test_img, str(i)+ext))
    out[ANS_COL] = p.predict(td[["image"]]).values
elif task == "text":
    from autogluon.multimodal import MultiModalPredictor
    TEXT_COL = [c for c in train.columns if train[c].dtype == object and c != ID_COL and train[c].astype(str).str.len().mean() >= 10][0]
    LABEL_COL = detect_target(); print("  ข้อความ:", TEXT_COL, "| ป้าย:", LABEL_COL)
    p = MultiModalPredictor(label=LABEL_COL, path="ag_auto").fit(train[[TEXT_COL, LABEL_COL]], time_limit=600)
    out[ANS_COL] = p.predict(test[[TEXT_COL]]).values
else:
    from autogluon.tabular import TabularPredictor
    LABEL_COL = detect_target(); is_reg = (task == "regression")
    print("  เป้าหมาย:", LABEL_COL, "| ชนิด:", "ทำนายตัวเลข" if is_reg else "จำแนกคลาส")
    tr = train.drop(columns=[c for c in [ID_COL] if c in train.columns])
    te = test.drop(columns=[c for c in [ID_COL] if c in test.columns])
    p = TabularPredictor(label=LABEL_COL, problem_type=("regression" if is_reg else None),
                         path="ag_auto").fit(tr, presets="best_quality", time_limit=600)
    if is_reg:
        out[ANS_COL] = p.predict(te).values
    elif MULTICOL:
        proba = p.predict_proba(te); name_map = {str(cc): cc for cc in proba.columns}
        for col in list(sample.columns[1:]):
            key = name_map.get(str(col)) or name_map.get(str(col).split("_")[-1])
            if key is None: print("  เตือน: จับคู่คอลัมน์", col, "กับคลาสไม่ได้ -> เติม 0 (ตรวจชื่อคอลัมน์)")
            out[col] = proba[key].values if key is not None else 0.0
        print("  (เติมความน่าจะเป็นต่อคลาส", len(sample.columns)-1, "คอลัมน์)")
    elif WANT_PROB and getattr(p, "positive_class", None) is not None:
        out[ANS_COL] = p.predict_proba(te)[p.positive_class].values
    else:
        out[ANS_COL] = p.predict(te).values
        if getattr(p, "positive_class", None) is not None and train[LABEL_COL].nunique() == 2:
            o2 = sample.copy(); o2[ANS_COL] = p.predict_proba(te)[p.positive_class].values
            o2.to_csv("submission_proba.csv", index=False); print("  (เซฟ submission_proba.csv เผื่อ metric เป็น AUC)")

# คะแนน validation ในเครื่อง (ไม่ต้องเดา/ไม่เปลือง quota ส่ง)
try:
    _lb = p.leaderboard()
    print("  คะแนน validation (ในเครื่อง) ดีสุด ~", round(float(_lb["score_val"].max()), 4))
except Exception:
    pass
out.to_csv("submission.csv", index=False)
print("\nเสร็จ! บันทึก submission.csv", out.shape)
try: display(out.head())
except Exception: print(out.head())"""))

c.append(md(r"""## ขั้นที่ 6 — รวมหลายไฟล์ (blend) ดันคะแนนฟรี (ไม่บังคับ)

ถ้าทำหลายวิธี/หลายโมเดล (มีหลายไฟล์ submission) การเฉลี่ยรวมกันมักได้คะแนนเพิ่มแบบไม่ต้องเทรนใหม่
เซลล์นี้รวมไฟล์ `submission*.csv` ด้วย rank-average (ดีสุดสำหรับความน่าจะเป็น)"""))
c.append(code(r"""import glob
FILES = [f for f in glob.glob("submission*.csv") if "baseline" not in f and "blend" not in f]   # << เลือกไฟล์ที่จะรวม
print("ไฟล์ที่จะรวม:", FILES)
if len(FILES) >= 2:
    dfs = [pd.read_csv(f) for f in FILES]; bl = dfs[0].copy()
    for col in bl.columns[1:]:
        if pd.api.types.is_numeric_dtype(bl[col]):
            bl[col] = sum(d[col].rank(pct=True) for d in dfs) / len(dfs)   # rank-avg
    bl.to_csv("submission_blend.csv", index=False)
    print("บันทึก submission_blend.csv -> ลองส่งเทียบกับ submission.csv")
else:
    print("มีไฟล์เดียว ยังไม่ต้อง blend (ทำหลายวิธีก่อนแล้วค่อยกลับมารวม)")"""))

c.append(submit_md()); c.append(submit_cell("auto solver"))
write_nb("00_AUTO_SOLVER.ipynb", c)

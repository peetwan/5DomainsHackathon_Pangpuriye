# จำลองสถานการณ์สอบจริง: สร้างชุดข้อมูลปลอมหน้าตาเหมือน Kaggle แล้ว exec โค้ดในโน้ตบุ๊กจริง
# ตั้งแต่โหลดข้อมูล -> เทรน -> สร้าง submission แล้วตรวจว่า submission ตรงรูปแบบ sample_submission
# ไลบรารีหนัก/ต้องโหลดเน็ต (autogluon, timm, pythainlp) ถูก mock ; lightgbm/sklearn/scipy/torch รันจริง
import os, sys, json, types, tempfile, traceback, warnings, importlib.util
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
rng = np.random.default_rng(0)
ROOT = os.path.abspath(".")

# ---------------- fake heavy modules ----------------
def install_fakes():
    # pythainlp.tokenize.word_tokenize (ถ้าไม่มี) -> ตัดคำง่าย ๆ ให้ join กลับเท่าเดิม
    if "pythainlp" not in sys.modules and importlib.util.find_spec("pythainlp") is None:
        pt = types.ModuleType("pythainlp"); tok = types.ModuleType("pythainlp.tokenize")
        def word_tokenize(text, engine="newmm", keep_whitespace=True):
            text = str(text); out = []; i = 0
            while i < len(text):
                if text[i] == " ": out.append(" "); i += 1
                else: out.append(text[i:i+3]); i += 3
            return out
        tok.word_tokenize = word_tokenize; pt.tokenize = tok
        sys.modules["pythainlp"] = pt; sys.modules["pythainlp.tokenize"] = tok
    # autogluon.tabular / autogluon.multimodal
    if importlib.util.find_spec("autogluon") is None:
        ag = types.ModuleType("autogluon"); tab = types.ModuleType("autogluon.tabular"); mm = types.ModuleType("autogluon.multimodal")
        class TabularPredictor:
            def __init__(self, label, problem_type=None, eval_metric=None, path=None):
                self.label = label; self.problem_type = problem_type; self._cls = None; self.positive_class = None
            def fit(self, train, presets=None, time_limit=None, **k):
                y = train[self.label]
                if self.problem_type != "regression" and y.nunique() <= 20:
                    self._cls = sorted(y.unique())
                    if len(self._cls) == 2: self.positive_class = self._cls[-1]
                return self
            def predict(self, X):
                if self._cls is not None: return pd.Series(rng.choice(self._cls, len(X)))
                return pd.Series(rng.normal(size=len(X)))
            def predict_proba(self, X):
                p = rng.random((len(X), len(self._cls))); p = p / p.sum(1, keepdims=True)
                return pd.DataFrame(p, columns=self._cls)
            def leaderboard(self, *a, **k): return pd.DataFrame({"model": ["fake_ensemble"], "score_val": [0.9]})
        class MultiModalPredictor:
            def __init__(self, label, eval_metric=None, path=None): self.label = label; self._cls = None
            def fit(self, train, time_limit=None, **k): self._cls = sorted(train[self.label].unique()); return self
            def predict(self, X): return pd.Series(rng.choice(self._cls, len(X)))
        tab.TabularPredictor = TabularPredictor; mm.MultiModalPredictor = MultiModalPredictor
        ag.tabular = tab; ag.multimodal = mm
        sys.modules["autogluon"] = ag; sys.modules["autogluon.tabular"] = tab; sys.modules["autogluon.multimodal"] = mm
    # timm.create_model -> โมเดล torch จิ๋วจริง (รับ (B,3,H,W) -> (B,nc))
    if importlib.util.find_spec("timm") is None:
        import torch.nn as nn
        timm = types.ModuleType("timm")
        class _Net(nn.Module):
            def __init__(self, nc): super().__init__(); self.head = nn.Linear(3, nc)
            def forward(self, x): return self.head(x.mean(dim=(2, 3)))
        def create_model(name, pretrained=False, num_classes=2, drop_rate=0.0, **k): return _Net(num_classes)
        timm.create_model = create_model
        sys.modules["timm"] = timm

# ---------------- mock competitions ----------------
def mk_dir():
    d = tempfile.mkdtemp(prefix="mockcomp_"); return d

def comp_tabular_clf():
    d = mk_dir(); n = 200
    df = pd.DataFrame({"age": rng.integers(30, 80, n), "sex": rng.choice(["M", "F"], n),
                       "cp": rng.integers(0, 4, n), "chol": rng.normal(240, 40, n)})
    df["target"] = ((df["age"] > 55) ^ (df["sex"] == "M")).astype(int)
    tr = df.iloc[:150].copy(); tr.insert(0, "id", range(150))
    te = df.iloc[150:].drop(columns=["target"]).reset_index(drop=True); te.insert(0, "id", range(50))
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    pd.DataFrame({"id": range(50), "target": 0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d

def comp_tabular_reg():
    d = mk_dir(); n = 200
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    df["SalePrice"] = (df["x1"] * 3 + df["x2"] + rng.normal(size=n)) * 1000 + 50000
    tr = df.iloc[:150].copy(); tr.insert(0, "id", range(150))
    te = df.iloc[150:].drop(columns=["SalePrice"]).reset_index(drop=True); te.insert(0, "id", range(50))
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    pd.DataFrame({"id": range(50), "SalePrice": 0.0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d

def comp_text_clf():
    d = mk_dir()
    pos = ["ดีมากชอบเลย", "ประทับใจสุดยอด", "คุ้มค่าน่าซื้อ"]; neg = ["แย่มากไม่ชอบ", "เสียดายเงิน", "ห่วยสุด"]
    rows = [{"text": rng.choice(pos), "label": 1} for _ in range(60)] + [{"text": rng.choice(neg), "label": 0} for _ in range(60)]
    tr = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    te = pd.DataFrame({"id": range(20), "text": [rng.choice(pos + neg) for _ in range(20)]})
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    pd.DataFrame({"id": range(20), "label": 0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d

def comp_word_seg():
    d = mk_dir()
    sents = ["ผม|รัก|แมว", "เรา|ไป|โรงเรียน", "วันนี้|อากาศ|ดี"]
    tr = pd.DataFrame({"text": [rng.choice(sents) for _ in range(50)]})
    raw = ["ผมรักแมว", "เราไปโรงเรียน", "วันนี้อากาศดี"]
    te = pd.DataFrame({"id": range(len(raw)), "text": raw})
    n_chars = sum(len(t) for t in raw)
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    pd.DataFrame({"id": [f"{i}_{j}" for i, t in enumerate(raw) for j in range(len(t))], "label": 0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d, n_chars

def comp_signal():
    d = mk_dir(); n = 180; L = 300
    X = rng.normal(size=(n, L)); y = rng.integers(0, 3, n); subj = rng.integers(0, 8, n)
    cols = [f"s{i}" for i in range(L)]
    tr = pd.DataFrame(X[:140], columns=cols); tr["label"] = y[:140]; tr["subject"] = subj[:140]; tr.insert(0, "id", range(140))
    te = pd.DataFrame(X[140:], columns=cols); te.insert(0, "id", range(40))
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    pd.DataFrame({"id": range(40), "label": 0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d

def comp_forecasting():
    d = mk_dir()
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    sales = (np.sin(np.arange(200) / 7) * 50 + 200 + rng.normal(0, 5, 200)).round(1)
    tr = pd.DataFrame({"id": range(200), "date": dates, "promo": rng.integers(0, 2, 200), "sales": sales})
    fut = pd.date_range("2023-07-20", periods=30, freq="D")
    te = pd.DataFrame({"id": range(30), "date": fut, "promo": rng.integers(0, 2, 30)})  # ฟีเจอร์เพิ่ม (เทสบั๊ก KeyError)
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    pd.DataFrame({"id": range(30), "sales": 0.0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d

def comp_image():
    d = mk_dir(); os.makedirs(f"{d}/train", exist_ok=True); os.makedirs(f"{d}/test", exist_ok=True)
    rows = []
    for i in range(24):
        c = i % 2; arr = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
        if c == 1: arr[:, :, 0] = 200
        Image.fromarray(arr).save(f"{d}/train/img{i}.jpg"); rows.append({"image_name": f"img{i}.jpg", "class": c})
    for i in range(10):
        Image.fromarray((rng.random((16, 16, 3)) * 255).astype(np.uint8)).save(f"{d}/test/{i}.jpg")
    pd.DataFrame(rows).to_csv(f"{d}/train.csv", index=False)
    pd.DataFrame({"id": range(10), "answer": 0}).to_csv(f"{d}/sample_submission.csv", index=False)
    return d

def comp_multiclass():
    d = mk_dir(); n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    X["target"] = rng.integers(0, 3, n)
    tr = X.iloc[:150].copy(); tr.insert(0, "id", range(150))
    te = X.iloc[150:].drop(columns=["target"]).reset_index(drop=True); te.insert(0, "id", range(50))
    tr.to_csv(f"{d}/train.csv", index=False); te.to_csv(f"{d}/test.csv", index=False)
    s = pd.DataFrame({"id": range(50)})
    for cc in [0, 1, 2]: s[f"class_{cc}"] = 0.0
    s.to_csv(f"{d}/sample_submission.csv", index=False)
    return d

# ---------------- notebook runner ----------------
SKIP = ["get_data(", "kaggle competitions submit", "kaggle competitions submissions",
        "AutoModelForSequenceClassification", "Seq2SeqTrainer", "AutoModelForSeq2SeqLM",
        "pipeline(", "PaliGemma", "BlipForConditional", "AutoModel.from_pretrained", "AutoModel,",
        "TabPFN", "from ultralytics", "YOLO(", "pycrfsuite", "Qwen", "load_refs(",
        "from setfit", "SetFitModel"]
def strip_magics(src):
    return "\n".join("pass  # magic" if l.lstrip().startswith(("!", "%")) else l for l in src.split("\n"))

def run_notebook(nb_path, data_dir, overrides=None):
    d = json.load(open(nb_path, encoding="utf-8"))
    sandbox = tempfile.mkdtemp(prefix="run_"); old = os.getcwd(); os.chdir(sandbox)
    ns = {"DATA_DIR": data_dir, "display": lambda *a, **k: None, "__name__": "__main__"}
    try:
        for c in d["cells"]:
            if c["cell_type"] != "code": continue
            src = "".join(c["source"])
            if any(k in src for k in SKIP): continue
            exec(strip_magics(src).replace("num_workers=2","num_workers=0"), ns)
            if overrides:
                for k, v in overrides.items(): ns[k] = v
                overrides = None  # apply once after first non-skipped cell (config)
        sub = pd.read_csv(os.path.join(sandbox, "submission.csv"))
        return sub, ns
    finally:
        os.chdir(old)

# ---------------- run all sims ----------------
def check(name, sub, sample_path, expect_rows=None, label_set=None, no_nan=True):
    s = pd.read_csv(sample_path)
    issues = []
    if list(sub.columns) != list(s.columns): issues.append(f"คอลัมน์ไม่ตรง {list(sub.columns)} != {list(s.columns)}")
    if expect_rows is not None and len(sub) != expect_rows: issues.append(f"จำนวนแถว {len(sub)} != {expect_rows}")
    if no_nan and sub.iloc[:, 1].isna().any(): issues.append("มี NaN ในคำตอบ")
    if label_set is not None and not set(sub.iloc[:, 1].unique()).issubset(label_set): issues.append(f"label แปลก {set(sub.iloc[:,1].unique())}")
    ok = not issues
    print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} rows={len(sub)} cols={list(sub.columns)}" + ("" if ok else " | " + "; ".join(issues)))
    return ok

install_fakes()
results = []
def trial(name, fn):
    try: results.append(fn())
    except Exception:
        print(f"[ERROR] {name}"); traceback.print_exc(); results.append(False)

print("==== จำลองสอบจริง: รันโค้ดในโน้ตบุ๊กกับข้อมูลปลอม ====\n")

def sim_tab_clf():
    d = comp_tabular_clf(); sub, _ = run_notebook(f"{ROOT}/04_Tabular/classification.ipynb", d)
    return check("Tabular classification (acc)", sub, f"{d}/sample_submission.csv", 50, {0, 1})
trial("tab_clf", sim_tab_clf)

def sim_tab_clf_auc():
    d = comp_tabular_clf(); sub, _ = run_notebook(f"{ROOT}/04_Tabular/classification.ipynb", d, overrides={"METRIC": "roc_auc"})
    s = pd.read_csv(f"{d}/sample_submission.csv")
    ok = list(sub.columns) == list(s.columns) and len(sub) == 50 and ((sub.iloc[:, 1] >= 0).all() and (sub.iloc[:, 1] <= 1).all())
    print(f"[{'PASS' if ok else 'FAIL'}] {'Tabular classification (AUC->prob)':32s} rows={len(sub)} (prob in 0..1: {ok})")
    return ok
trial("tab_clf_auc", sim_tab_clf_auc)

def sim_tab_reg():
    d = comp_tabular_reg(); sub, _ = run_notebook(f"{ROOT}/04_Tabular/regression.ipynb", d)
    return check("Tabular regression", sub, f"{d}/sample_submission.csv", 50)
trial("tab_reg", sim_tab_reg)

def sim_text():
    d = comp_text_clf(); sub, _ = run_notebook(f"{ROOT}/02_NLP/text_classification.ipynb", d)
    return check("Text classification", sub, f"{d}/sample_submission.csv", 20, {0, 1})
trial("text_clf", sim_text)

def sim_wordseg():
    d, nchars = comp_word_seg(); sub, _ = run_notebook(f"{ROOT}/02_NLP/thai_word_segmentation.ipynb", d)
    return check("Word segmentation (per-char)", sub, f"{d}/sample_submission.csv", nchars, {0, 1})
trial("word_seg", sim_wordseg)

def sim_signal():
    d = comp_signal(); sub, _ = run_notebook(f"{ROOT}/05_TimeSeries_Signal/signal_classification.ipynb", d)
    return check("Signal classification", sub, f"{d}/sample_submission.csv", 40, {0, 1, 2})
trial("signal", sim_signal)

def sim_forecast():
    d = comp_forecasting(); sub, _ = run_notebook(f"{ROOT}/05_TimeSeries_Signal/forecasting.ipynb", d)
    return check("Forecasting (recursive)", sub, f"{d}/sample_submission.csv", 30)
trial("forecast", sim_forecast)

def sim_image():
    d = comp_image(); sub, _ = run_notebook(f"{ROOT}/01_Computer_Vision/image_classification.ipynb", d)
    return check("Image classification", sub, f"{d}/sample_submission.csv", 10, {0, 1})
trial("image", sim_image)

print("\n---- 🪄 AUTO SOLVER (ปุ่มเดียวจบ: เดาประเภทงานเอง) ----")
AUTO = f"{ROOT}/00_AUTO_SOLVER.ipynb"
def sim_auto(name, comp_fn, expect_task, rows, cols):
    d = comp_fn(); d = d[0] if isinstance(d, tuple) else d
    sub, ns = run_notebook(AUTO, d)
    ok = ns.get("task") == expect_task and len(sub) == rows and list(sub.columns) == cols
    print(f"[{'PASS' if ok else 'FAIL'}] {('AUTO -> ' + name):28s} เดาได้ task={ns.get('task')} (คาด {expect_task}) rows={len(sub)}")
    return ok
trial("auto_tabclf", lambda: sim_auto("tabular clf", comp_tabular_clf, "tabular", 50, ["id", "target"]))
trial("auto_tabreg", lambda: sim_auto("tabular reg", comp_tabular_reg, "regression", 50, ["id", "SalePrice"]))
trial("auto_text",   lambda: sim_auto("text clf", comp_text_clf, "text", 20, ["id", "label"]))
trial("auto_image",  lambda: sim_auto("image clf", comp_image, "image", 10, ["id", "answer"]))
trial("auto_multicol", lambda: sim_auto("multiclass-proba", comp_multiclass, "tabular", 50, ["id", "class_0", "class_1", "class_2"]))

print(f"\n==== สรุป: {sum(1 for r in results if r)}/{len(results)} ผ่าน ====")
sys.exit(0 if all(results) else 1)

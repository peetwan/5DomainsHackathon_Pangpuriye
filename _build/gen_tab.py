import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell

# ============================================================
# โน้ตบุ๊ก 1: Tabular Classification (ทำนายคลาส)
# ============================================================
c = []
c.append(md(r"""# Tabular — โจทย์แบบ "ทำนายคลาสจากตาราง" (Classification)

ข้อมูลเป็นตาราง (แต่ละแถว = 1 ตัวอย่าง, คอลัมน์ = ฟีเจอร์) -> ทำนายป้าย (เช่น ป่วย/ไม่ป่วย, ซื้อ/ไม่ซื้อ)

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด มือใหม่ทำแค่นี้) = `AutoGluon` กดรันแล้วมันลองหลายโมเดล + รวมให้เอง
- วิธีที่ 2 (ไม่บังคับ) = `LightGBM` + cross-validation คุมเอง
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ: ข้อมูลเป็นตาราง และทำนาย "ป้าย/หมวด" (มีกี่หมวดก็ได้)
ถ้าทำนาย "ตัวเลขต่อเนื่อง" (ราคา/คะแนน) -> ไปใช้ `regression.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, ชื่อไฟล์ csv, `TARGET` (คอลัมน์ที่ทำนาย), `METRIC`"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install -U "autogluon.tabular[all]"      # วิธีที่ 1
!pip -q install lightgbm scikit-learn pandas numpy   # วิธีที่ 2"""))
c.append(kaggle_md())
c.append(kaggle_cell("super-ai-engineer-ss-6-heart-disease-prediction"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, pandas as pd, numpy as np
TRAIN_CSV  = os.path.join(DATA_DIR,"train.csv")   # << แก้ชื่อไฟล์
TEST_CSV   = os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR,"sample_submission.csv")
train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
ID_COL = sample.columns[0]    # เดาอัตโนมัติ: คอลัมน์แรกของ sample_submission คือ id
TARGET = sample.columns[1]    # << แก้ถ้าผิด: คอลัมน์ที่ต้องทำนาย เช่น "target", "Survived", "num" (ปกติคือคอลัมน์ที่ 2 ของ sample)
# << แก้ METRIC ให้ตรง tab Evaluation บน Kaggle: ถ้าเขียน Accuracy ใส่ "accuracy", ROC AUC ใส่ "roc_auc", F1 ใส่ "f1"
METRIC = "accuracy"
print("คอลัมน์ train:", list(train.columns)); display(train.head()); display(sample.head())
print("เป้าหมาย =", TARGET, "| metric =", METRIC)"""))

c.append(md(r"""## วิธีที่ 1 — AutoGluon (ง่ายสุด มือใหม่ทำแค่นี้พอ)

AutoGluon จัดการ missing/categorical ให้เอง + ลองหลายโมเดล + รวมเป็น ensemble แค่บอก label กับ metric"""))
c.append(code(r"""from autogluon.tabular import TabularPredictor
# แปลงชื่อ metric -> ชื่อของ AutoGluon (ถ้าไม่รู้จัก ส่งตรง ๆ ให้ AutoGluon ตัดสิน)
AG_METRIC = {"accuracy":"accuracy","acc":"accuracy","roc_auc":"roc_auc","auc":"roc_auc","f1":"f1"}.get(METRIC, METRIC)
train_ag = train.drop(columns=[c for c in [ID_COL] if c in train.columns])
test_ag  = test.drop(columns=[c for c in [ID_COL] if c in test.columns])
predictor = TabularPredictor(label=TARGET, eval_metric=AG_METRIC, path="ag_tab").fit(
    train_ag, presets="best_quality", time_limit=600)   # << แก้ time_limit: วินาที (600=10นาที) ลอง 120 ก่อน, ส่งจริงเพิ่มเป็น 1800
out = sample.copy()
if AG_METRIC == "roc_auc":
    # AUC ส่ง "ความน่าจะเป็นของคลาสบวก" -- ใช้ positive_class ของ AutoGluon ปลอดภัยกว่า iloc[:,1]
    out[TARGET] = predictor.predict_proba(test_ag)[predictor.positive_class].values
else:
    out[TARGET] = predictor.predict(test_ag).values                  # acc/f1 ส่งป้าย (เลข/ข้อความ)
out.to_csv("submission.csv", index=False)
print("บันทึก submission.csv"); display(out.head())
print(predictor.leaderboard().head())   # AutoGluon รุ่นใหม่เอา silent= ออกแล้ว"""))

c.append(md(r"""## วิธีที่ 2 — LightGBM + cross-validation (ไม่บังคับ)

คุมเองได้ละเอียด ใช้ StratifiedKFold กัน overfit"""))
c.append(code(r"""import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
y = train[TARGET].values
X = train.drop(columns=[c for c in [ID_COL,TARGET] if c in train.columns])
Xte = test.drop(columns=[c for c in [ID_COL] if c in test.columns])
for df in (X,Xte):
    for col in df.columns:
        if df[col].dtype=="object": df[col]=df[col].astype("category")
oof=np.zeros(len(y)); pred=np.zeros(len(Xte))
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
    m=lgb.LGBMClassifier(n_estimators=1500,learning_rate=0.02,num_leaves=31,random_state=42,verbose=-1)
    m.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],callbacks=[lgb.early_stopping(80,verbose=False)])
    oof[va]=m.predict_proba(X.iloc[va])[:,1]; pred+=m.predict_proba(Xte)[:,1]/5
print("OOF AUC =", round(roc_auc_score(y,oof),4), "(ยิ่งใกล้ 1 ยิ่งดี, 0.5 = เดามั่ว)")
# pred = ความน่าจะเป็นว่าเป็นคลาส 1 ; เลือกแบบส่งให้ตรง metric อัตโนมัติ
out=sample.copy()
if METRIC in ("roc_auc","auc"):
    out[TARGET]=pred                      # AUC -> ส่งความน่าจะเป็น (ทศนิยม 0..1)
else:
    out[TARGET]=(pred>=0.5).astype(int)   # accuracy/f1 -> ส่งป้าย 0/1 (ตัดที่ 0.5)
out.to_csv("submission_lgbm.csv",index=False); print("บันทึก submission_lgbm.csv")"""))

c.append(submit_md()); c.append(submit_cell("autogluon tabular classification"))
write_nb("04_Tabular/classification.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 2: Tabular Regression (ทำนายตัวเลข)
# ============================================================
c = []
c.append(md(r"""# Tabular — โจทย์แบบ "ทำนายตัวเลขจากตาราง" (Regression)

ข้อมูลตาราง -> ทำนายตัวเลขต่อเนื่อง (เช่น ราคาบ้าน, คะแนนความเสี่ยง, ยอดขาย)

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด) = `AutoGluon` แค่บอก problem_type='regression'
- วิธีที่ 2 (ไม่บังคับ) = `LightGBM` regressor + cross-validation
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อคำตอบเป็น "ตัวเลขต่อเนื่อง" ไม่ใช่หมวด metric มักเป็น `RMSE` หรือ `MAE`
ถ้าคำตอบเป็น "หมวด/ป้าย" -> ไปใช้ `classification.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, ไฟล์ csv, `TARGET`, `METRIC`"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install -U "autogluon.tabular[all]"
!pip -q install lightgbm scikit-learn pandas numpy"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-regression-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, pandas as pd, numpy as np
TRAIN_CSV=os.path.join(DATA_DIR,"train.csv"); TEST_CSV=os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB=os.path.join(DATA_DIR,"sample_submission.csv")
train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
ID_COL=sample.columns[0]   # เดาอัตโนมัติ: คอลัมน์แรกของ sample คือ id
TARGET=sample.columns[1]   # << แก้ถ้าผิด: คอลัมน์ตัวเลขที่ทำนาย เช่น "SalePrice", "price", "target"
# << แก้ METRIC ตาม tab Evaluation: Kaggle เขียน RMSE ใส่ "root_mean_squared_error", MAE ใส่ "mean_absolute_error", R2 ใส่ "r2"
METRIC="root_mean_squared_error"
print("คอลัมน์:",list(train.columns)); display(train.head()); display(sample.head())"""))

c.append(md(r"""## วิธีที่ 1 — AutoGluon Regression (ง่ายสุด)"""))
c.append(code(r"""from autogluon.tabular import TabularPredictor
# Kaggle มักเขียนสั้น (RMSE/MAE/R2) แต่ AutoGluon ใช้ชื่อยาว -> แปลงให้ (ถ้าไม่รู้จัก ส่งตรง ๆ)
AG_METRIC = {"rmse":"root_mean_squared_error","mae":"mean_absolute_error","r2":"r2"}.get(METRIC.lower(), METRIC)
train_ag=train.drop(columns=[c for c in [ID_COL] if c in train.columns])
test_ag =test.drop(columns=[c for c in [ID_COL] if c in test.columns])
predictor=TabularPredictor(label=TARGET, problem_type="regression", eval_metric=AG_METRIC, path="ag_reg").fit(
    train_ag, presets="best_quality", time_limit=600)   # << แก้ time_limit: วินาที (600=10นาที) ลอง 120 ก่อน
out=sample.copy(); out[TARGET]=predictor.predict(test_ag).values
out.to_csv("submission.csv",index=False); print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — LightGBM Regressor + CV (ไม่บังคับ)"""))
c.append(code(r"""import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
y=train[TARGET].values
X=train.drop(columns=[c for c in [ID_COL,TARGET] if c in train.columns])
Xte=test.drop(columns=[c for c in [ID_COL] if c in test.columns])
for df in (X,Xte):
    for col in df.columns:
        if df[col].dtype=="object": df[col]=df[col].astype("category")
oof=np.zeros(len(y)); pred=np.zeros(len(Xte))
for tr,va in KFold(5,shuffle=True,random_state=42).split(X):
    m=lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.02,num_leaves=31,random_state=42,verbose=-1)
    m.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],callbacks=[lgb.early_stopping(80,verbose=False)])
    oof[va]=m.predict(X.iloc[va]); pred+=m.predict(Xte)/5
print("OOF RMSE =", round(np.sqrt(mean_squared_error(y,oof)),4))   # sklearn 1.6 เอา squared=False ออกแล้ว ใช้ np.sqrt แทน
out=sample.copy(); out[TARGET]=pred; out.to_csv("submission_lgbm.csv",index=False)
print("บันทึก submission_lgbm.csv")"""))

c.append(submit_md()); c.append(submit_cell("autogluon regression"))
write_nb("04_Tabular/regression.ipynb", c)

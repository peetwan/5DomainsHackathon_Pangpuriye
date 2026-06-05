import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell

# ============================================================
# โน้ตบุ๊ก 1: Signal / Time-Series Classification
# ============================================================
c = []
c.append(md(r"""# Time-Series / Signal — โจทย์แบบ "สัญญาณ -> คลาส" (Classification)

หน้าต่างสัญญาณ 1 ช่วง -> 1 คลาส (เช่น EEG -> ระยะการนอน, ECG -> ชนิดการเต้นหัวใจ, เซนเซอร์ -> ท่าทาง)

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด มือใหม่ทำแค่นี้) = สกัดฟีเจอร์ (พลังงานในแต่ละความถี่/สถิติ) + `AutoGluon`
- วิธีที่ 2 (ไม่บังคับ) = ฟีเจอร์เดิม + `LightGBM` แบ่ง CV ตาม subject (กันข้อมูลรั่ว)
- วิธีที่ 3 (ขั้นสูง ต้อง GPU) = `1D-CNN` บนสัญญาณดิบ (ดู reference_cheatsheet.md)
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ input เป็นสัญญาณตามเวลา (มี sampling rate) แบ่งเป็นหน้าต่าง แล้วทำนาย 1 คลาสต่อหน้าต่าง
ถ้าต้อง "ทำนายค่าตัวเลขในอนาคต" -> ไปใช้ `forecasting.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, `FS` (sampling rate), `LABEL_COL`, `GROUP_COL` (คอลัมน์ subject)"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install scipy lightgbm scikit-learn pandas numpy
!pip -q install -U "autogluon.tabular[all]"   # วิธีที่ 1"""))
c.append(kaggle_md())
c.append(kaggle_cell("super-ai-engineer-ss-6-sleep-stage-classification"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG

รองรับ CSV ที่แต่ละแถว = 1 หน้าต่าง คอลัมน์เป็นค่าสัญญาณ (เช่น s0..s2999) + คอลัมน์ label"""))
c.append(code(r"""import os, pandas as pd, numpy as np
FS        = 100      # << แก้: sampling rate (Hz) เช่น EEG=100
LABEL_COL = "label"  # << แก้: คอลัมน์คลาส
GROUP_COL = None     # << แก้: คอลัมน์ subject เช่น "subject" ถ้าไม่มีใส่ None
TRAIN_CSV = os.path.join(DATA_DIR,"train.csv"); TEST_CSV=os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB= os.path.join(DATA_DIR,"sample_submission.csv")
tr=pd.read_csv(TRAIN_CSV); te=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
ID_COL=sample.columns[0]; ANS_COL=sample.columns[1]
print("คอลัมน์ train:",list(tr.columns)[:8],"..."); display(tr.head())
drop=[c for c in [LABEL_COL,ID_COL,GROUP_COL] if c and c in tr.columns]
sig_cols=[c for c in tr.columns if c not in drop and pd.api.types.is_numeric_dtype(tr[c])]
Xtr_raw=tr[sig_cols].values; y_raw=tr[LABEL_COL].values
groups=tr[GROUP_COL].values if (GROUP_COL and GROUP_COL in tr.columns) else np.zeros(len(tr))
Xte_raw=te[[c for c in sig_cols if c in te.columns]].values
test_ids=te[ID_COL].values if ID_COL in te.columns else sample[ID_COL].values
classes=sorted(pd.unique(y_raw)); c2i={c:i for i,c in enumerate(classes)}; i2c={i:c for c,i in c2i.items()}
y=np.array([c2i[v] for v in y_raw]); N_CLASSES=len(classes)
print("จำนวนคลาส =",N_CLASSES,"| สัญญาณยาว",len(sig_cols),"ค่า/หน้าต่าง")"""))

c.append(md(r"""## สกัดฟีเจอร์ต่อหน้าต่าง (ใช้ทั้งวิธีที่ 1 และ 2)"""))
c.append(code(r"""from scipy.signal import welch
from scipy.stats import skew, kurtosis
BANDS={"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"sigma":(11,16),"beta":(16,30),"gamma":(30,49)}
_trap=getattr(np,"trapezoid",np.trapz)   # numpy 2.0 เปลี่ยนชื่อ np.trapz -> np.trapezoid (รองรับทั้งสองเวอร์ชัน)
def feat(sig):
    sig=np.asarray(sig,dtype=np.float64)
    f={"mean":sig.mean(),"std":sig.std(),"min":sig.min(),"max":sig.max(),"ptp":np.ptp(sig),
       "skew":skew(sig),"kurt":kurtosis(sig),"rms":np.sqrt(np.mean(sig**2)),
       "zcr":np.mean(np.abs(np.diff(np.sign(sig))))/2}
    d1=np.diff(sig); d2=np.diff(d1); v0=sig.var()+1e-12; v1=d1.var()+1e-12; v2=d2.var()+1e-12
    f["hj_mob"]=np.sqrt(v1/v0); f["hj_comp"]=np.sqrt(v2/v1)/(np.sqrt(v1/v0)+1e-12)
    fr,psd=welch(sig,fs=FS,nperseg=min(len(sig),FS*2)); tot=_trap(psd,fr)+1e-12
    for nm,(lo,hi) in BANDS.items():
        idx=(fr>=lo)&(fr<hi); bp=_trap(psd[idx],fr[idx]) if idx.any() else 0.0
        f["bp_"+nm]=bp; f["rbp_"+nm]=bp/tot
    f["spec_ent"]=-np.sum((psd/tot)*np.log(psd/tot+1e-12)); return f
Xtr=pd.DataFrame([feat(w) for w in Xtr_raw]); Xte=pd.DataFrame([feat(w) for w in Xte_raw])
print("ฟีเจอร์:",Xtr.shape[1],"ตัว")"""))

c.append(md(r"""## วิธีที่ 1 — ฟีเจอร์ + AutoGluon (ง่ายสุด)"""))
c.append(code(r"""from autogluon.tabular import TabularPredictor
ag=Xtr.copy(); ag[LABEL_COL]=y
pred=TabularPredictor(label=LABEL_COL,eval_metric="f1_macro",path="ag_sig").fit(
    ag, presets="best_quality", time_limit=600)   # << แก้ time_limit
yhat=pred.predict(Xte).values
out=sample.copy(); out[ANS_COL]=[i2c[int(v)] for v in yhat]
out.to_csv("submission.csv",index=False); print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — ฟีเจอร์ + LightGBM แบ่ง CV ตาม subject (ไม่บังคับ แต่สำคัญถ้ามี subject)

แบ่ง CV ตาม subject เพื่อไม่ให้หน้าต่างจากคนเดียวกันอยู่ทั้ง train และ val (กันคะแนนหลอก)"""))
c.append(code(r"""import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
cls,cnt=np.unique(y,return_counts=True); cw={c:len(y)/(len(cls)*n) for c,n in zip(cls,cnt)}
sw=np.array([cw[v] for v in y])
params=dict(objective="multiclass",num_class=N_CLASSES,learning_rate=0.03,num_leaves=64,
            min_child_samples=40,n_jobs=-1,verbosity=-1)
oof=np.zeros((len(y),N_CLASSES)); tp=np.zeros((len(Xte),N_CLASSES))
for tr_i,va_i in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(Xtr,y,groups):
    d=lgb.Dataset(Xtr.iloc[tr_i],y[tr_i],weight=sw[tr_i]); dv=lgb.Dataset(Xtr.iloc[va_i],y[va_i])
    m=lgb.train(params,d,2000,valid_sets=[dv],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    oof[va_i]=m.predict(Xtr.iloc[va_i]); tp+=m.predict(Xte)/5
print("OOF macro-F1 =", round(f1_score(y,oof.argmax(1),average="macro"),4))
out=sample.copy(); out[ANS_COL]=[i2c[int(v)] for v in tp.argmax(1)]
out.to_csv("submission_lgbm.csv",index=False); print("บันทึก submission_lgbm.csv")"""))

c.append(submit_md()); c.append(submit_cell("features autogluon signal"))
write_nb("05_TimeSeries_Signal/signal_classification.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 2: Forecasting (ทำนายค่าในอนาคต)
# ============================================================
c = []
c.append(md(r"""# Time-Series — โจทย์แบบ "ทำนายค่าในอนาคต" (Forecasting)

มีค่าตามเวลาในอดีต -> ทำนายค่าในอนาคต (เช่น ยอดขายพรุ่งนี้, ราคาสัปดาห์หน้า)

วิธีในโน้ตบุ๊กนี้ (มือใหม่):
- วิธีที่ 1 (ง่ายสุด) = สร้างฟีเจอร์จากวันที่ + ค่าย้อนหลัง (lag) แล้วใช้ `LightGBM` ทำนายเหมือน regression
- วิธีที่ 2 (ไม่บังคับ) = `AutoGluon TimeSeries` (เฉพาะทาง forecasting)
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อต้องทำนาย "ค่าตัวเลขในอนาคต" จากประวัติตามเวลา (มีคอลัมน์วันที่/เวลา)
ถ้าทำนาย "คลาสจากหน้าต่างสัญญาณ" -> ไปใช้ `signal_classification.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, `TIME_COL` (คอลัมน์วันที่), `TARGET`, `SERIES_COL` (ถ้ามีหลายซีรีส์)"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install lightgbm scikit-learn pandas numpy
!pip -q install -U "autogluon.timeseries"   # วิธีที่ 2 เท่านั้น"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-forecasting-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, pandas as pd, numpy as np
TRAIN_CSV=os.path.join(DATA_DIR,"train.csv"); TEST_CSV=os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB=os.path.join(DATA_DIR,"sample_submission.csv")
train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
TIME_COL  = "date"      # << แก้: คอลัมน์วันที่/เวลา
TARGET    = sample.columns[1]   # << แก้ถ้าผิด: คอลัมน์ค่าที่ทำนาย
SERIES_COL= None        # << แก้: ถ้ามีหลายซีรีส์ (เช่น "store_id") ใส่ชื่อ, ถ้าซีรีส์เดียวใส่ None
ID_COL=sample.columns[0]
print("คอลัมน์:",list(train.columns)); display(train.head()); display(sample.head())"""))

c.append(md(r"""## วิธีที่ 1 — ฟีเจอร์วันที่ + lag + LightGBM (ง่ายสุด ใช้ได้จริง)

หลักคิด: เปลี่ยนปัญหา forecasting ให้เป็น regression ธรรมดา โดยทำฟีเจอร์จากวันที่ (ปี/เดือน/วัน/วันในสัปดาห์)
+ ค่าย้อนหลัง (lag) แล้วให้ LightGBM เรียนรู้"""))
c.append(code(r"""import lightgbm as lgb
def add_time_features(df):
    df=df.copy(); df[TIME_COL]=pd.to_datetime(df[TIME_COL])
    df["year"]=df[TIME_COL].dt.year; df["month"]=df[TIME_COL].dt.month
    df["day"]=df[TIME_COL].dt.day; df["dow"]=df[TIME_COL].dt.dayofweek
    df["doy"]=df[TIME_COL].dt.dayofyear
    return df
def add_lags(df, lags=(1,7,14,28)):   # << แก้ค่า lag ตามความถี่ข้อมูล (รายวัน/รายชั่วโมง)
    df=df.sort_values(TIME_COL).copy()
    g=df.groupby(SERIES_COL)[TARGET] if SERIES_COL else df[TARGET]
    for L in lags:
        df[f"lag_{L}"]= (g.shift(L) if SERIES_COL else df[TARGET].shift(L))
    return df
tr=add_lags(add_time_features(train)).dropna()
feat_cols=[c for c in tr.columns if c not in [TARGET,TIME_COL,ID_COL] and pd.api.types.is_numeric_dtype(tr[c])]
m=lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.02,num_leaves=63,random_state=42,verbose=-1)
m.fit(tr[feat_cols], tr[TARGET])
# เตรียม test: ต้องมีฟีเจอร์เดียวกัน (ถ้า test ไม่มี lag ต้องต่อ history จาก train ก่อน) -- ปรับตามรูปแบบจริง
te=add_time_features(test)
for col in feat_cols:
    if col not in te.columns: te[col]=np.nan   # << แก้: เติม lag ของ test จาก history จริงถ้าจำเป็น
out=sample.copy(); out[TARGET]=m.predict(te[feat_cols])
out.to_csv("submission.csv",index=False); print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — AutoGluon TimeSeries (ไม่บังคับ เฉพาะทาง forecasting)

เหมาะเมื่อข้อมูลเป็นรูปแบบ long (item_id, timestamp, target) ชัดเจน"""))
c.append(code(r"""# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
# tsdf = TimeSeriesDataFrame.from_data_frame(train, id_column=SERIES_COL, timestamp_column=TIME_COL)
# predictor = TimeSeriesPredictor(target=TARGET, prediction_length=28).fit(tsdf, time_limit=600)
# pred = predictor.predict(tsdf)   # แล้วแมปกลับเข้า sample_submission ตามรูปแบบจริง
print("ปลดคอมเมนต์เซลล์นี้ถ้าข้อมูลเป็นรูปแบบ long ชัดเจน")"""))

c.append(submit_md()); c.append(submit_cell("lightgbm forecasting"))
write_nb("05_TimeSeries_Signal/forecasting.ipynb", c)

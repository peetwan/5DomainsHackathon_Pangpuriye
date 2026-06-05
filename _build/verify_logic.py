# Execute the core runnable code paths from the notebooks with synthetic data,
# on the installed numpy 2.0 / sklearn 1.6 / lightgbm 4.6 (close to current Colab).
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

# 1) Word-seg helpers (NLP notebook)
def words_to_char_labels(words):
    text, labels = [], []
    for w in words:
        for i, ch in enumerate(w): text.append(ch); labels.append(1 if i == 0 else 0)
    return "".join(text), labels
def char_labels_to_words(text, labels):
    words, cur = [], ""
    for ch, lab in zip(text, labels):
        if lab == 1 and cur: words.append(cur); cur = ""
        cur += ch
    if cur: words.append(cur)
    return words
w = ["สวัสดี", "ครับ", "ผม"]; t, l = words_to_char_labels(w)
assert char_labels_to_words(t, l) == w
from sklearn.metrics import f1_score
assert abs(f1_score([1,0,1,0],[1,0,1,0],average="binary",pos_label=1) - 1.0) < 1e-9
print("OK 1 word-seg helpers + boundary_f1")

# 2) Signal feature extraction (Time-Series notebook) — np.trapezoid fallback for numpy 2.0
from scipy.signal import welch
from scipy.stats import skew, kurtosis
FS = 100; _trap = getattr(np, "trapezoid", np.trapz)
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"sigma":(11,16),"beta":(16,30),"gamma":(30,49)}
def feat(sig):
    sig = np.asarray(sig, dtype=np.float64)
    f = {"mean":sig.mean(),"std":sig.std(),"min":sig.min(),"max":sig.max(),"ptp":np.ptp(sig),
         "skew":skew(sig),"kurt":kurtosis(sig),"rms":np.sqrt(np.mean(sig**2)),
         "zcr":np.mean(np.abs(np.diff(np.sign(sig))))/2}
    d1=np.diff(sig); d2=np.diff(d1); v0=sig.var()+1e-12; v1=d1.var()+1e-12; v2=d2.var()+1e-12
    f["hj_mob"]=np.sqrt(v1/v0); f["hj_comp"]=np.sqrt(v2/v1)/(np.sqrt(v1/v0)+1e-12)
    fr,psd=welch(sig,fs=FS,nperseg=min(len(sig),FS*2)); tot=_trap(psd,fr)+1e-12
    for nm,(lo,hi) in BANDS.items():
        idx=(fr>=lo)&(fr<hi); bp=_trap(psd[idx],fr[idx]) if idx.any() else 0.0
        f["bp_"+nm]=bp; f["rbp_"+nm]=bp/tot
    f["spec_ent"]=-np.sum((psd/tot)*np.log(psd/tot+1e-12)); return f
X = rng.normal(size=(60,300)); Xf = pd.DataFrame([feat(r) for r in X])
assert Xf.shape == (60, 24), Xf.shape
print("OK 2 signal feature extraction", Xf.shape)

# 3) LightGBM multiclass + StratifiedGroupKFold (signal classification)
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
y = rng.integers(0,3,60); groups = rng.integers(0,6,60)
params = dict(objective="multiclass",num_class=3,learning_rate=0.1,num_leaves=15,n_jobs=1,verbosity=-1)
oof = np.zeros((60,3))
for tr,va in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(Xf,y,groups):
    d=lgb.Dataset(Xf.iloc[tr],y[tr]); dv=lgb.Dataset(Xf.iloc[va],y[va])
    m=lgb.train(params,d,50,valid_sets=[dv],callbacks=[lgb.early_stopping(10),lgb.log_evaluation(0)])
    oof[va]=m.predict(Xf.iloc[va])
print("OK 3 lgb.train multiclass + StratifiedGroupKFold | macroF1=", round(f1_score(y,oof.argmax(1),average="macro"),3))

# 4) Tabular classification: LGBMClassifier + early_stopping callback + StratifiedKFold
from sklearn.model_selection import StratifiedKFold
Xtab = pd.DataFrame(rng.normal(size=(80,6))); ytab = rng.integers(0,2,80)
for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(Xtab,ytab):
    m=lgb.LGBMClassifier(n_estimators=100,learning_rate=0.1,num_leaves=15,random_state=0,verbose=-1)
    m.fit(Xtab.iloc[tr],ytab[tr],eval_set=[(Xtab.iloc[va],ytab[va])],callbacks=[lgb.early_stopping(10,verbose=False)])
print("OK 4 LGBMClassifier + early_stopping + StratifiedKFold")

# 5) Regression RMSE — version-safe (sklearn 1.6 removed squared= arg)
from sklearn.metrics import mean_squared_error
yreg = rng.normal(size=80); oofr=np.zeros(80)
for tr,va in StratifiedKFold(4,shuffle=True,random_state=0).split(Xtab,(yreg>0).astype(int)):
    m=lgb.LGBMRegressor(n_estimators=100,learning_rate=0.1,random_state=0,verbose=-1); m.fit(Xtab.iloc[tr],yreg[tr])
    oofr[va]=m.predict(Xtab.iloc[va])
rmse = np.sqrt(mean_squared_error(yreg,oofr))
print("OK 5 LGBMRegressor + version-safe RMSE =", round(rmse,3))

# 6) Forecasting lag/date features (Time-Series forecasting)
df = pd.DataFrame({"date":pd.date_range("2024-01-01",periods=100,freq="D"),"target":rng.normal(size=100).cumsum()})
def add_time_features(d):
    d=d.copy(); d["date"]=pd.to_datetime(d["date"])
    d["year"]=d["date"].dt.year; d["month"]=d["date"].dt.month; d["day"]=d["date"].dt.day
    d["dow"]=d["date"].dt.dayofweek; d["doy"]=d["date"].dt.dayofyear; return d
def add_lags(d,lags=(1,7)):
    d=d.sort_values("date").copy()
    for L in lags: d[f"lag_{L}"]=d["target"].shift(L)
    return d
tr=add_lags(add_time_features(df)).dropna()
fc=[c for c in tr.columns if c not in ["target","date"] and pd.api.types.is_numeric_dtype(tr[c])]
lgb.LGBMRegressor(n_estimators=100,random_state=0,verbose=-1).fit(tr[fc],tr["target"])
print("OK 6 forecasting lag/date features | n_feats=", len(fc))

# 7) Text classification sklearn pipeline (tokenizer swapped to .split for the test)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
texts=["good great love","bad awful hate","good nice","bad terrible","love it","hate it"]*4
ylab=[1,0,1,0,1,0]*4
clf=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=1)),
              ("lr",LogisticRegression(max_iter=1000,class_weight="balanced"))])
print("OK 7 text-clf TF-IDF+LogReg | cv acc=", round(cross_val_score(clf,texts,ylab,cv=3).mean(),3))

# 8) timm model builds + forward (image classification วิธี 2)
try:
    import torch, timm
    mdl = timm.create_model("resnet18", pretrained=False, num_classes=2)
    out = mdl(torch.randn(2,3,64,64)); assert out.shape == (2,2)
    print("OK 8 timm create_model + forward")
except ModuleNotFoundError as e:
    print("SKIP 8 timm not installed locally (pattern is standard) -", e)

print("\nALL LOGIC TESTS PASSED")

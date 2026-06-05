import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell, inspect_md, inspect_cell

c = []
c.append(md(r"""# หัวข้อใหญ่ 6 — Audio / Speech (เสียง)

โจทย์ตัวอย่าง: จำแนกเสียง (ชนิดเสียง/คำสั่งเสียง/อารมณ์), จำแนกประเภทดนตรี, ตรวจจับเหตุการณ์เสียง
(ถ้าเป็น "ถอดเสียงเป็นข้อความ" = ASR ดูหมายเหตุท้ายโน้ตบุ๊ก -> ใช้ Whisper)

วิธีในโน้ตบุ๊กนี้ (จำแนกเสียง):
- วิธีที่ 1 (ง่ายสุด) = สกัดฟีเจอร์เสียง (MFCC/spectral) ด้วย `librosa` แล้วโยนเข้า `AutoGluon` เหมือนตาราง
- วิธีที่ 2 (ไม่บังคับ) = ฟีเจอร์เดิม + `LightGBM`
- วิธีที่ 3 (ขั้นสูง) = แปลงเป็น `mel-spectrogram` (รูป) แล้วใช้โน้ตบุ๊ก 01 image_classification
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ input เป็นไฟล์เสียง (.wav/.mp3/...) และตอบ "1 คลาสต่อ 1 ไฟล์"
ถ้าต้อง "ถอดเสียงเป็นข้อความ" (ASR) -> ใช้ Whisper (หมายเหตุท้ายไฟล์)

ต้องแก้ (`# << แก้`): ชื่อ competition, โฟลเดอร์เสียง, `FILE_COL` (คอลัมน์ชื่อไฟล์), `LABEL_COL`, `SR`"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install librosa soundfile pandas numpy scikit-learn lightgbm
!pip -q install -U "autogluon.tabular[all]"   # วิธีที่ 1"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-audio-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, glob, pandas as pd, numpy as np
AUDIO_EXT = (".wav",".mp3",".flac",".ogg",".m4a")
def find(name):
    h=glob.glob(os.path.join(DATA_DIR,"**",name),recursive=True); return h[0] if h else None
def audio_dir(keyword):
    cand=[d for d,_,fs in os.walk(DATA_DIR) if keyword in d.lower() and any(f.lower().endswith(AUDIO_EXT) for f in fs)]
    return max(cand,key=lambda d:len(os.listdir(d))) if cand else None

TRAIN_CSV = find("train.csv"); SAMPLE_SUB = find("sample_submission.csv")
TRAIN_AUD = audio_dir("train")   # << แก้ถ้าผิด เช่น os.path.join(DATA_DIR,"audio/train")
TEST_AUD  = audio_dir("test")
SR        = 16000      # << แก้: sample rate ที่จะอ่าน (16000 พอสำหรับเสียงพูด, เพลงใช้ 22050)
df = pd.read_csv(TRAIN_CSV); sample = pd.read_csv(SAMPLE_SUB)
ID_COL, ANS_COL = sample.columns[0], sample.columns[1]
print("train คอลัมน์:", list(df.columns)); print("เสียง train:", TRAIN_AUD, "| test:", TEST_AUD)
display(df.head()); display(sample.head())
FILE_COL  = "filename"   # << แก้: คอลัมน์ชื่อไฟล์เสียงใน train.csv เช่น "fname", "audio", "id"
LABEL_COL = "label"      # << แก้: คอลัมน์คลาส เช่น "label", "category", "target"
TEST_EXT  = ".wav"       # << แก้: นามสกุลไฟล์เสียง test (ถ้าใน sample id มีนามสกุลแล้ว ใส่ "")"""))

c.append(inspect_md()); c.append(inspect_cell())

c.append(md(r"""## สกัดฟีเจอร์เสียงต่อไฟล์ (ใช้ทั้งวิธีที่ 1 และ 2)

ฟีเจอร์มาตรฐาน: MFCC (รูปร่างของเสียง) + spectral + zero-crossing + พลังงาน -> เฉลี่ย/ส่วนเบี่ยงเบนต่อไฟล์"""))
c.append(code(r"""import librosa
def audio_features(path, sr=SR):
    y,_ = librosa.load(path, sr=sr, mono=True)
    if len(y) < sr: y = np.pad(y, (0, sr-len(y)))   # กันไฟล์สั้นเกิน
    f = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20): f[f"mfcc{i}_m"]=mfcc[i].mean(); f[f"mfcc{i}_s"]=mfcc[i].std()
    f["cent"]=librosa.feature.spectral_centroid(y=y,sr=sr).mean()
    f["bw"]=librosa.feature.spectral_bandwidth(y=y,sr=sr).mean()
    f["roll"]=librosa.feature.spectral_rolloff(y=y,sr=sr).mean()
    f["zcr"]=librosa.feature.zero_crossing_rate(y).mean()
    f["rms"]=librosa.feature.rms(y=y).mean()
    chroma=librosa.feature.chroma_stft(y=y,sr=sr);
    for i in range(12): f[f"chroma{i}"]=chroma[i].mean()
    return f
def build(file_list, audio_dir):
    rows=[]
    for fn in file_list:
        p=os.path.join(audio_dir, str(fn))
        try: rows.append(audio_features(p))
        except Exception: rows.append({})
    return pd.DataFrame(rows).fillna(0)
Xtr = build(df[FILE_COL].tolist(), TRAIN_AUD)
Xte = build([str(i)+TEST_EXT for i in sample[ID_COL].tolist()], TEST_AUD)
print("ฟีเจอร์ต่อไฟล์:", Xtr.shape[1], "ตัว")"""))

c.append(md(r"""## วิธีที่ 1 — ฟีเจอร์ + AutoGluon (ง่ายสุด)"""))
c.append(code(r"""from autogluon.tabular import TabularPredictor
ag = Xtr.copy(); ag[LABEL_COL] = df[LABEL_COL].values
pred = TabularPredictor(label=LABEL_COL, path="ag_audio").fit(ag, presets="best_quality", time_limit=600)   # << แก้ time_limit
out = sample.copy(); out[ANS_COL] = pred.predict(Xte).values
out.to_csv("submission.csv", index=False); print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — ฟีเจอร์ + LightGBM + CV (ไม่บังคับ)"""))
c.append(code(r"""import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
le = LabelEncoder(); y = le.fit_transform(df[LABEL_COL])
oof = np.zeros(len(y)); pred = np.zeros((len(Xte), len(le.classes_)))
for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(Xtr,y):
    m=lgb.LGBMClassifier(n_estimators=800,learning_rate=0.03,random_state=42,verbose=-1)
    m.fit(Xtr.iloc[tr],y[tr],eval_set=[(Xtr.iloc[va],y[va])],callbacks=[lgb.early_stopping(50,verbose=False)])
    oof[va]=m.predict(Xtr.iloc[va]); pred+=m.predict_proba(Xte)/5
print("OOF macro-F1 =", round(f1_score(y,oof,average="macro"),4))
out=sample.copy(); out[ANS_COL]=le.inverse_transform(pred.argmax(1))
out.to_csv("submission_lgbm.csv",index=False); print("บันทึก submission_lgbm.csv")"""))

c.append(md(r"""## หมายเหตุ — ถ้าเป็น ASR (ถอดเสียงเป็นข้อความ)

ใช้ Whisper สำเร็จรูป (zero-shot, รองรับไทย):
```python
!pip install -U openai-whisper
import whisper
m = whisper.load_model("small")   # tiny/base/small/medium/large
text = m.transcribe("ไฟล์.wav", language="th")["text"]
```
แล้วเขียน text ลง submission ตามฟอร์แมต (วัดด้วย CER/WER -> ใช้ pythainlp.util.word_error_rate วัดในเครื่อง)"""))

c.append(submit_md()); c.append(submit_cell("audio features autogluon"))
write_nb("06_Audio/audio_classification.ipynb", c)

import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell

# ============================================================
# โน้ตบุ๊ก 1: Thai Word Segmentation (ป้ายต่อ "ตัวอักษร")
# ============================================================
c = []
c.append(md(r"""# NLP — โจทย์แบบ "ตัดคำ / ป้ายต่อตัวอักษร" (Sequence Labeling)

ภาษาไทยไม่มีเว้นวรรคระหว่างคำ งานคือบอกว่า "ตรงไหนคือจุดขึ้นคำใหม่"
มองเป็น: ตัวอักษรแต่ละตัว -> ป้าย 1 (ขึ้นคำใหม่) หรือ 0 (อยู่ในคำเดิม)

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด มือใหม่ทำแค่นี้) = `PyThaiNLP deepcut` ตัดคำสำเร็จรูป ไม่ต้องเทรน
- วิธีที่ 2 (ไม่บังคับ) = `CRF` เทรนเองบน CPU ได้
- วิธีที่ 3 (ขั้นสูง ต้อง GPU) = `WangchanBERTa`
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ output เป็น "ป้ายต่อหน่วยในลำดับ" เช่น ตัดคำ, NER (ดึงชื่อเฉพาะ), POS (ชนิดคำ)
ถ้า output เป็น "1 ป้ายต่อทั้งประโยค" (เช่น บวก/ลบ) -> ไปใช้ `text_classification.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, ชื่อไฟล์/คอลัมน์ข้อความ, ตัวคั่นคำในไฟล์เฉลย, รูปแบบ id ใน sample"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install pythainlp deepcut tensorflow python-crfsuite scikit-learn pandas numpy
!pip -q install transformers datasets torch   # วิธีที่ 3 เท่านั้น"""))
c.append(kaggle_md())
c.append(kaggle_cell("super-ai-engineer-ss-6-word-segmentation"))

c.append(md(r"""## ฟังก์ชันหลัก — แปลงไป-กลับระหว่าง "คำ" กับ "ป้ายต่อตัวอักษร" """))
c.append(code(r"""def words_to_char_labels(words):     # ['สวัสดี','ครับ'] -> ('สวัสดีครับ',[1,0,0,0,0,0,1,0,0,0])
    text,labels=[],[]
    for w in words:
        for i,ch in enumerate(w): text.append(ch); labels.append(1 if i==0 else 0)
    return "".join(text), labels
def char_labels_to_words(text,labels):
    words,cur=[],""
    for ch,lab in zip(text,labels):
        if lab==1 and cur: words.append(cur); cur=""
        cur+=ch
    if cur: words.append(cur)
    return words
from sklearn.metrics import f1_score
def boundary_f1(yt,yp): return f1_score(yt,yp,average="binary",pos_label=1)   # ใช้วัดในเครื่อง"""))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, pandas as pd, numpy as np
TRAIN_CSV  = os.path.join(DATA_DIR,"train.csv")   # << แก้ชื่อไฟล์
TEST_CSV   = os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR,"sample_submission.csv")
TEXT_COL   = "text"     # << แก้: คอลัมน์ข้อความ
SEP        = "|"        # << แก้: ตัวคั่นคำในไฟล์เฉลย (อาจเป็น "|" หรือเว้นวรรค " ")
train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
print("train คอลัมน์:",list(train.columns)); print("sample คอลัมน์:",list(sample.columns))
display(train.head()); display(sample.head())
def load_segmented(df):
    out=[]
    for s in df[TEXT_COL].astype(str):
        out.append(words_to_char_labels([w for w in s.split(SEP) if w!=""]))
    return out
train_pairs=load_segmented(train)
print("ตัวอย่าง:", train_pairs[0][0][:30], "->", train_pairs[0][1][:30])"""))

c.append(md(r"""## วิธีที่ 1 — PyThaiNLP สำเร็จรูป (ง่ายสุด ไม่ต้องเทรน)

ลอง engine `deepcut` (แม่น ใกล้ SOTA) หรือ `newmm` (เร็วมาก) เก็บอันที่คะแนนดีกว่า"""))
c.append(code(r"""from pythainlp.tokenize import word_tokenize
def to_labels(text, engine="deepcut"):   # << แก้ engine ได้: deepcut / newmm / attacut
    words=word_tokenize(str(text), engine=engine, keep_whitespace=True)
    _,labels=words_to_char_labels(words)
    return labels[:len(text)] + [0]*max(0,len(text)-len(labels))
rows=[]
for _,r in test.iterrows():
    labels=to_labels(str(r[TEXT_COL])); labels[0]=1   # ตัวแรกเป็นขอบคำเสมอ
    for ci,lab in enumerate(labels):
        rows.append({sample.columns[0]: f"{r.get('id',_)}_{ci}", sample.columns[1]: lab})  # << แก้รูปแบบ id ให้ตรง sample
sub=pd.DataFrame(rows); sub.to_csv("submission.csv",index=False)
print("บันทึก submission.csv", sub.shape); display(sub.head())"""))

c.append(md(r"""## วิธีที่ 2 — CRF เทรนเอง (ไม่บังคับ, CPU ได้, คะแนนดีในโดเมน)"""))
c.append(code(r"""import pycrfsuite
def feats(chars,i):
    n=len(chars); f={"bias":1.0,"c0":chars[i]}
    for d in (1,2,3):
        f[f"c-{d}"]=chars[i-d] if i-d>=0 else "BOS"; f[f"c+{d}"]=chars[i+d] if i+d<n else "EOS"
    f["is_digit"]=chars[i].isdigit(); f["is_space"]=chars[i]==" "
    return f
tr=pycrfsuite.Trainer(verbose=False)
for text,labels in train_pairs:
    ch=list(text); tr.append([feats(ch,i) for i in range(len(ch))],[str(l) for l in labels])
tr.set_params({"c1":0.1,"c2":0.1,"max_iterations":100,"feature.possible_transitions":True})
tr.train("ws.crfsuite")
tag=pycrfsuite.Tagger(); tag.open("ws.crfsuite")
rows=[]
for _,r in test.iterrows():
    ch=list(str(r[TEXT_COL])); labels=[int(t) for t in tag.tag([feats(ch,i) for i in range(len(ch))])]
    labels[0]=1
    for ci,lab in enumerate(labels):
        rows.append({sample.columns[0]: f"{r.get('id',_)}_{ci}", sample.columns[1]: lab})  # << แก้รูปแบบ id
pd.DataFrame(rows).to_csv("submission_crf.csv",index=False); print("บันทึก submission_crf.csv")"""))

c.append(md(r"""## วิธีที่ 3 — WangchanBERTa (ขั้นสูง ต้อง GPU, ไม่บังคับ)

ดูโค้ดเต็มใน reference_cheatsheet.md หัวข้อ D.3 (token classification ด้วยโมเดลภาษาไทย)
มือใหม่ข้ามได้ ใช้วิธีที่ 1 ก็พอแล้วสำหรับการทำคะแนน"""))

c.append(submit_md()); c.append(submit_cell("deepcut word segmentation"))
write_nb("02_NLP/thai_word_segmentation.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 2: Text Classification (1 ป้ายต่อทั้งข้อความ)
# ============================================================
c = []
c.append(md(r"""# NLP — โจทย์แบบ "จำแนกข้อความ" (Text Classification)

ข้อความ 1 ชิ้น -> 1 ป้าย (เช่น รีวิว บวก/ลบ/กลาง, หัวข้อข่าว, สแปม/ไม่สแปม)

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด มือใหม่ทำแค่นี้) = ตัดคำไทย + `TF-IDF` + `Logistic Regression` (เร็ว ไม่ต้อง GPU)
- วิธีที่ 2 (ไม่บังคับ ต้อง GPU) = `WangchanBERTa` คะแนนสูงกว่า
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ input เป็นข้อความ และตอบ "1 ป้ายต่อทั้งข้อความ"
ถ้าต้องป้ายทีละคำ/ตัวอักษร (ตัดคำ/NER) -> ใช้ `thai_word_segmentation.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, คอลัมน์ข้อความ `TEXT_COL`, คอลัมน์ป้าย `LABEL_COL`"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install pythainlp scikit-learn pandas numpy
!pip -q install transformers datasets torch   # วิธีที่ 2 เท่านั้น"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-text-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, pandas as pd, numpy as np
TRAIN_CSV  = os.path.join(DATA_DIR,"train.csv")   # << แก้ชื่อไฟล์
TEST_CSV   = os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR,"sample_submission.csv")
TEXT_COL   = "text"      # << แก้: คอลัมน์ข้อความ
LABEL_COL  = "label"     # << แก้: คอลัมน์ป้าย
train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
ID_COL=sample.columns[0]; ANS_COL=sample.columns[1]
print("train คอลัมน์:",list(train.columns)); display(train.head()); display(sample.head())"""))

c.append(md(r"""## วิธีที่ 1 — ตัดคำ + TF-IDF + Logistic Regression (ง่ายสุด เร็ว ไม่ต้อง GPU)

หลักคิด: เปลี่ยนข้อความเป็นตัวเลข (นับคำสำคัญด้วย TF-IDF) แล้วให้โมเดลเส้นตรงเรียนรู้
ตัดคำไทยก่อนด้วย PyThaiNLP เพราะไทยไม่มีเว้นวรรค"""))
c.append(code(r"""from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def th_tokenize(s): return word_tokenize(str(s), engine="newmm")   # ตัดคำไทย
clf = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=th_tokenize, ngram_range=(1,2), min_df=2)),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
# เช็คคะแนนในเครื่องก่อน (cross-validation)
print("CV accuracy:", cross_val_score(clf, train[TEXT_COL].astype(str), train[LABEL_COL], cv=5).mean().round(4))
clf.fit(train[TEXT_COL].astype(str), train[LABEL_COL])
out = sample.copy(); out[ANS_COL] = clf.predict(test[TEXT_COL].astype(str))
out.to_csv("submission.csv", index=False)
print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — WangchanBERTa (ไม่บังคับ ต้อง GPU คะแนนสูงกว่า)

ใช้โมเดลภาษาไทยแบบ fine-tune ทำ sequence classification"""))
c.append(code(r"""import numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding)
MODEL="airesearch/wangchanberta-base-att-spm-uncased"   # << แก้โมเดลได้
labels=sorted(train[LABEL_COL].unique()); l2i={l:i for i,l in enumerate(labels)}; i2l={i:l for l,i in l2i.items()}
tok=AutoTokenizer.from_pretrained(MODEL)
ds=Dataset.from_dict({"text":train[TEXT_COL].astype(str).tolist(),
                      "label":[l2i[v] for v in train[LABEL_COL]]}).train_test_split(0.1,seed=42)
def enc(b): return tok(b["text"], truncation=True, max_length=256)
ds=ds.map(enc, batched=True)
model=AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))
# หมายเหตุ: ถ้า transformers เก่า (ก่อน 4.46) เปลี่ยน eval_strategy -> evaluation_strategy
args=TrainingArguments(output_dir="out", learning_rate=2e-5, per_device_train_batch_size=16,
    num_train_epochs=3, eval_strategy="epoch", save_strategy="no",
    fp16=torch.cuda.is_available(), report_to="none")
tr=Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
           tokenizer=tok, data_collator=DataCollatorWithPadding(tok))
tr.train()
import torch
preds=[]
model.eval()
for i in range(0,len(test),64):
    batch=tok(test[TEXT_COL].astype(str).tolist()[i:i+64], truncation=True, max_length=256,
              padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad(): preds+=model(**batch).logits.argmax(1).cpu().tolist()
out=sample.copy(); out[ANS_COL]=[i2l[p] for p in preds]
out.to_csv("submission_bert.csv",index=False); print("บันทึก submission_bert.csv")"""))

c.append(submit_md()); c.append(submit_cell("tfidf logreg text classification"))
write_nb("02_NLP/text_classification.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 3: Text Generation / Seq2Seq (สร้าง/แปลงข้อความ)
# ============================================================
c = []
c.append(md(r"""# NLP — โจทย์แบบ "สร้าง/แปลงข้อความ" (Text Generation / Seq2Seq)

ข้อความเข้า -> ข้อความออก เช่น แปลภาษา, สรุปความ, ตอบคำถามจากบทความ, แก้คำผิด, ถอดความ

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด) = `pipeline` ของ HuggingFace ใช้โมเดลสำเร็จรูป generate เลย ไม่ต้องเทรน
- วิธีที่ 2 (ไม่บังคับ ต้อง GPU) = fine-tune `mT5` ด้วย Seq2SeqTrainer
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ output เป็น "ข้อความใหม่" (ไม่ใช่ป้าย/คลาส) วัดด้วย BLEU/ROUGE/accuracy แล้วแต่งาน
ถ้า output เป็นป้าย -> ใช้ `text_classification.ipynb` หรือ `thai_word_segmentation.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, `SRC_COL` (ข้อความเข้า), `TGT_COL` (ข้อความเป้าหมาย), `TASK`/โมเดล"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install -U transformers sentencepiece sacremoses pandas torch
!pip -q install datasets evaluate sacrebleu rouge_score   # วิธีที่ 2 (เทรน/วัดผล)"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-textgen-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — โหลดข้อมูล + CONFIG"""))
c.append(code(r"""import os, pandas as pd
TRAIN_CSV=os.path.join(DATA_DIR,"train.csv"); TEST_CSV=os.path.join(DATA_DIR,"test.csv")
SAMPLE_SUB=os.path.join(DATA_DIR,"sample_submission.csv")
train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sample=pd.read_csv(SAMPLE_SUB)
SRC_COL="source"   # << แก้: คอลัมน์ข้อความเข้า (เช่น ประโยคต้นทาง/บทความ)
TGT_COL=sample.columns[1]   # << แก้ถ้าผิด: คอลัมน์ข้อความเป้าหมาย
print("train คอลัมน์:",list(train.columns)); display(train.head()); display(sample.head())"""))

c.append(md(r"""## วิธีที่ 1 — pipeline สำเร็จรูป (ง่ายสุด ไม่ต้องเทรน)

เลือก TASK + โมเดลให้ตรงงาน:
- สรุปความไทย: model="csebuetnlp/mT5_multilingual_XLSum", task="summarization"
- แปล ไทย<->อังกฤษ: model="Helsinki-NLP/opus-mt-th-en" (หรือ -en-th), task="translation"
- ทั่วไป seq2seq: task="text2text-generation" """))
c.append(code(r"""from transformers import pipeline
import torch
TASK  = "summarization"                         # << แก้: summarization / translation / text2text-generation
MODEL = "csebuetnlp/mT5_multilingual_XLSum"     # << แก้โมเดลให้ตรงงาน/ภาษา
gen = pipeline(TASK, model=MODEL, device=0 if torch.cuda.is_available() else -1)
outs=[]
for txt in test[SRC_COL].astype(str).tolist():
    r = gen(txt, max_length=128, truncation=True)[0]   # << แก้ max_length ตามความยาวเป้าหมาย
    outs.append(r.get("summary_text") or r.get("translation_text") or r.get("generated_text"))
out=sample.copy(); out[TGT_COL]=outs
out.to_csv("submission.csv",index=False,encoding="utf-8-sig")
print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — fine-tune mT5 (ไม่บังคับ ต้อง GPU)

เทรน seq2seq บนคู่ (source -> target) ของเราเอง"""))
c.append(code(r"""import numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
MODEL="google/mt5-small"   # << แก้: mt5-small เร็ว / mt5-base แม่นกว่า
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL)
ds=Dataset.from_dict({"src":train[SRC_COL].astype(str).tolist(),
                      "tgt":train[TGT_COL].astype(str).tolist()}).train_test_split(test_size=0.1,seed=42)
def prep(b):
    enc=tok(b["src"], max_length=256, truncation=True)
    lab=tok(text_target=b["tgt"], max_length=128, truncation=True)
    enc["labels"]=lab["input_ids"]; return enc
ds=ds.map(prep, batched=True, remove_columns=ds["train"].column_names)
# หมายเหตุ: ถ้า transformers เก่า (ก่อน 4.46) เปลี่ยน eval_strategy -> evaluation_strategy
args=Seq2SeqTrainingArguments(output_dir="out", learning_rate=3e-4, per_device_train_batch_size=8,
    num_train_epochs=3, eval_strategy="epoch", save_strategy="no",
    predict_with_generate=True, fp16=torch.cuda.is_available(), report_to="none")
trainer=Seq2SeqTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
    tokenizer=tok, data_collator=DataCollatorForSeq2Seq(tok, model=model))
trainer.train()
model.eval(); outs=[]
for txt in test[SRC_COL].astype(str).tolist():
    ids=tok(txt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad(): g=model.generate(**ids, max_new_tokens=128, num_beams=4)
    outs.append(tok.decode(g[0], skip_special_tokens=True))
out=sample.copy(); out[TGT_COL]=outs; out.to_csv("submission_mt5.csv",index=False,encoding="utf-8-sig")
print("บันทึก submission_mt5.csv")"""))

c.append(submit_md()); c.append(submit_cell("pipeline text generation"))
write_nb("02_NLP/text_generation.ipynb", c)

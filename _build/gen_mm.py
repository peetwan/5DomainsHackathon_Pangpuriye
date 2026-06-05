import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell

c = []
c.append(md(r"""# Multimodal (Vision-Language) — โจทย์แบบ "รูป -> ข้อความ"

เช่น บรรยายรูปเป็นภาษาไทย (image captioning), ตอบคำถามจากรูป (VQA), อ่านตัวอักษรในรูป (OCR)

วิธีในโน้ตบุ๊กนี้ (สำหรับ image captioning ไทย):
- วิธีที่ 1 (ง่ายสุด มือใหม่ทำแค่นี้) = ใช้โมเดลที่เทรนกับ Thai-COCO มาแล้ว generate ได้เลย ไม่ต้องเทรน
- วิธีที่ 2 (ไม่บังคับ) = `BLIP` fine-tune (เบา)
- วิธีที่ 3 (ขั้นสูง ต้อง GPU 16GB) = `PaliGemma2` + LoRA คะแนนดีสุด
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ input มีรูป และ output เป็น "ข้อความ" (ประโยค/คำตอบ)
ถ้า output เป็น "1 คลาส" -> ไปหัวข้อ 01 Computer Vision

สำคัญมาก: metric คือ `BLEU` ที่ต้องตัดคำไทยด้วย `newmm` ก่อน
ใช้ฟังก์ชัน `thai_bleu()` ในโน้ตบุ๊กวัดคะแนนในเครื่องก่อนส่งทุกครั้ง
ต้องแก้ (`# << แก้`): ชื่อ competition, path โฟลเดอร์รูป/ไฟล์ JSON"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install -U transformers accelerate datasets pillow pandas torch sentencepiece
!pip -q install pythainlp nltk          # ใช้วัด Thai BLEU
# !pip -q install peft bitsandbytes     # ปลดคอมเมนต์เฉพาะถ้าจะทำวิธีที่ 3 (PaliGemma2 LoRA) -- มือใหม่ไม่ต้องลง"""))
c.append(kaggle_md())
c.append(kaggle_cell("super-ai-engineer-ss-6-thai-language-image-captioning"))

c.append(md(r"""## ขั้นที่ 3 — path + ตัววัด Thai BLEU (นี่คือ metric จริง)"""))
c.append(code(r"""import os, glob, json, pandas as pd, torch
from PIL import Image
TRAIN_IMG  = os.path.join(DATA_DIR,"train")     # << แก้: ให้ตรงโฟลเดอร์รูป train เช่นถ้าเห็น "images/train" ใช้ os.path.join(DATA_DIR,"images","train")
TEST_IMG   = os.path.join(DATA_DIR,"test")      # << แก้: ให้ตรงโฟลเดอร์รูป test เช่น os.path.join(DATA_DIR,"images","test")
SAMPLE_SUB = os.path.join(DATA_DIR,"sample_submission.csv")  # << แก้: ถ้าไฟล์ไม่ได้ชื่อนี้ ดูชื่อจริงจากผลเซลล์ดาวน์โหลด
js = glob.glob(os.path.join(DATA_DIR,"*train*.json")); TRAIN_JSON = js[0] if js else None
sample = pd.read_csv(SAMPLE_SUB); ID_COL, CAP_COL = sample.columns[0], sample.columns[1]
print("รูป test:", TEST_IMG, "| sample คอลัมน์:", list(sample.columns)); display(sample.head())

# BLEU = วัดว่าประโยคที่โมเดลเขียน เหมือนเฉลยแค่ไหน (0-1 ยิ่งสูงยิ่งดี) ; ไทยต้องตัดคำด้วย newmm ก่อน ไม่งั้นคะแนนเพี้ยน
from pythainlp.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction   # ไม่ต้อง nltk.download อะไร
def th_tok(s): return [w for w in word_tokenize(str(s),engine="newmm",keep_whitespace=False) if w.strip()]
def thai_bleu(list_of_refs, hyps):
    refs=[[th_tok(r) for r in rs] for rs in list_of_refs]; hyp=[th_tok(h) for h in hyps]
    return corpus_bleu(refs,hyp,weights=(0.25,0.25,0.25,0.25),smoothing_function=SmoothingFunction().method4)
def write_submission(pred_dict, out="submission.csv"):
    def look(idv):
        s=str(idv)
        # ลองหลายแบบ: ตรง ๆ / เติม .jpg / เอาเฉพาะชื่อไฟล์ / ตัดนามสกุลออก
        for k in (s, s+".jpg", os.path.basename(s), os.path.splitext(os.path.basename(s))[0]):
            if k in pred_dict: return pred_dict[k]
        return "ภาพ"   # fallback ไทยสั้น ๆ กัน BLEU=0
    o=sample.copy(); o[CAP_COL]=o[ID_COL].map(look)
    n_fb=(o[CAP_COL]=="ภาพ").sum()
    if n_fb: print("เตือน: เติม fallback", n_fb, "แถว (ถ้าเยอะแปลว่า id ไม่ตรง -> เช็ค ID_COL/นามสกุลไฟล์)")
    o.to_csv(out,index=False,encoding="utf-8-sig"); print("บันทึก",out,o.shape); return o"""))

c.append(md(r"""## วิธีที่ 1 — โมเดล Thai-COCO สำเร็จรูป (ง่ายสุด ไม่ต้องเทรน)

ใช้โมเดลที่เคยเทรนกับคลังข้อมูลเดียวกัน generate คำบรรยายได้เลยใน <1 ชม."""))
c.append(code(r"""from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
M="Natthaphon/thaicapgen-convnext-phayathai"   # << แก้โมเดลได้
dev="cuda" if torch.cuda.is_available() else "cpu"
fe=AutoImageProcessor.from_pretrained(M); tk=AutoTokenizer.from_pretrained(M)
model=AutoModel.from_pretrained(M,trust_remote_code=True).to(dev).eval()
pred={}
for p in sorted(glob.glob(os.path.join(TEST_IMG,"*"))):
    px=fe(images=[Image.open(p).convert("RGB")],return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad(): ids=model.generate(px,max_length=120,num_beams=4)   # ถ้าไม่มี GPU จะช้ามาก แนะนำเปิด GPU
    pred[os.path.basename(p)]=tk.batch_decode(ids,skip_special_tokens=True)[0].strip()
print("ตัวอย่างคำบรรยายที่ได้:", list(pred.items())[:2])   # ควรเห็นข้อความไทย เช่น 'ผู้ชายกำลังเล่นเซิร์ฟ'
write_submission(pred,"submission.csv")"""))

c.append(md(r"""## ตรวจคะแนน BLEU ในเครื่องก่อนส่ง (สำคัญ! ทำทุกครั้ง)

เอารูปฝั่ง train (ที่รู้คำเฉลย) มาลองทำนายแล้วเรียก `thai_bleu()` ดูคะแนน ถ้าต่ำมาก (ใกล้ 0) แปลว่ามีอะไรผิด (path/id) อย่าเพิ่งส่ง"""))
c.append(code(r"""def load_refs(json_path, img_dir, limit=50):
    if not json_path: return []
    data=json.load(open(json_path,encoding="utf-8")); pairs={}
    def add(fn,cap):
        if not fn or not cap: return
        p=os.path.join(img_dir, os.path.basename(str(fn)))
        if os.path.exists(p): pairs.setdefault(p,[]).append(str(cap).strip())
    if isinstance(data,dict) and "annotations" in data:
        id2n={im["id"]:im.get("file_name") or im.get("filename") for im in data.get("images",[])}
        for a in data["annotations"]: add(id2n.get(a.get("image_id")), a.get("caption") or a.get("th_caption"))
    elif isinstance(data,dict):
        for k,v in data.items():
            for c in (v if isinstance(v,list) else [v]): add(k, c if isinstance(c,str) else (c.get("caption") if isinstance(c,dict) else None))
    elif isinstance(data,list):
        for r in data: add(r.get("image") or r.get("file_name"), r.get("caption") or r.get("th_caption"))
    return list(pairs.items())[:limit]
val=load_refs(TRAIN_JSON, TRAIN_IMG, 50)
if val:
    hyps=[]; refs=[]
    for p,caps in val:
        px=fe(images=[Image.open(p).convert("RGB")],return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad(): ids=model.generate(px,max_length=120,num_beams=4)
        hyps.append(tk.batch_decode(ids,skip_special_tokens=True)[0].strip()); refs.append(caps)
    print("Thai BLEU บน train (ลอง 50 รูป) =", round(thai_bleu(refs,hyps),4), "(ยิ่งสูงยิ่งดี; ใกล้ 0 = มีอะไรผิด อย่าเพิ่งส่ง)")
else:
    print("โหลด reference จาก TRAIN_JSON ไม่ได้ -> ข้ามการวัดในเครื่อง (เช็ค path/รูปแบบ JSON)")"""))

c.append(md(r"""## วิธีที่ 2 — BLIP fine-tune (ไม่บังคับ)

โค้ดเต็มอยู่ใน reference_cheatsheet.md (โหลดคู่รูป-คำบรรยายจาก JSON มาเทรนต่อ)
มือใหม่ข้ามได้ ใช้วิธีที่ 1 ก็ได้คะแนนแล้ว"""))
c.append(md(r"""## วิธีที่ 3 — PaliGemma2 + LoRA (ขั้นสูง ต้อง GPU 16GB คะแนนดีสุด)

โค้ดเต็มอยู่ใน reference_cheatsheet.md หัวข้อ D (prompt `caption th` + LoRA 4-bit)"""))

c.append(submit_md()); c.append(submit_cell("thai coco pretrained captioning"))
write_nb("03_Multimodal_VisionLanguage/thai_image_captioning.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 2: Visual Question Answering / OCR (รูป + คำถาม -> คำตอบ)
# ============================================================
c = []
c.append(md(r"""# Multimodal — โจทย์แบบ "ตอบคำถามจากรูป / อ่านข้อความในรูป" (VQA / OCR)

รูป (+ คำถาม) -> คำตอบสั้น ๆ เช่น "ในรูปมีกี่คน", "ป้ายเขียนว่าอะไร"

วิธีในโน้ตบุ๊กนี้:
- วิธีที่ 1 (ง่ายสุด) = VLM สำเร็จรูปแบบ zero-shot (BLIP-VQA / Qwen2.5-VL) ไม่ต้องเทรน
- วิธีที่ 2 (ขั้นสูง) = fine-tune VLM (ดู reference_cheatsheet.md)
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อมี "รูป + คำถาม" -> ตอบ หรือ "อ่านตัวอักษรในรูป (OCR)"
ถ้าแค่บรรยายรูป (ไม่มีคำถาม) -> `thai_image_captioning.ipynb`

ต้องแก้ (`# << แก้`): ชื่อ competition, path โฟลเดอร์รูป, คอลัมน์คำถาม `Q_COL`, คอลัมน์ id รูป"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install -U transformers pillow pandas torch
!pip -q install pythainlp   # เผื่อวัด/ตัดคำไทย"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-vqa-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — path + CONFIG"""))
c.append(code(r"""import os, glob, pandas as pd, torch
from PIL import Image
TEST_IMG = os.path.join(DATA_DIR,"test")            # << แก้: โฟลเดอร์รูป test เช่น os.path.join(DATA_DIR,"images","test")
SAMPLE_SUB = os.path.join(DATA_DIR,"sample_submission.csv")
test = pd.read_csv(os.path.join(DATA_DIR,"test.csv"))   # << แก้: ถ้าไฟล์ไม่ชื่อ test.csv เช่น "test_questions.csv"
sample = pd.read_csv(SAMPLE_SUB)
IMG_ID_COL = "image"     # << แก้: ชื่อคอลัมน์ไฟล์รูปใน test (ดูจาก "test คอลัมน์:") เช่น "image_id", "filename"
Q_COL      = "question"  # << แก้: ชื่อคอลัมน์คำถาม เช่น "question_th" (ถ้าเป็น OCR ล้วน ไม่มีคำถาม ตั้งเป็น None)
ANS_COL    = sample.columns[1]
print("test คอลัมน์:",list(test.columns)); display(test.head()); display(sample.head())
# ตรวจชื่อคอลัมน์ก่อน กันพังกลางทาง
assert IMG_ID_COL in test.columns, f'ไม่มีคอลัมน์ {IMG_ID_COL} -- เลือกจาก {list(test.columns)}'
assert (Q_COL is None) or (Q_COL in test.columns), f'ไม่มีคอลัมน์ {Q_COL} -- เลือกจาก {list(test.columns)}'"""))

c.append(md(r"""## วิธีที่ 1 — BLIP-VQA สำเร็จรูป (ง่ายสุด ไม่ต้องเทรน)

ตอบคำถามจากรูปแบบ zero-shot สำหรับภาษาไทยที่ซับซ้อนแนะนำเปลี่ยนเป็น Qwen2.5-VL (ดูคอมเมนต์)"""))
c.append(code(r"""from transformers import pipeline
from tqdm.auto import tqdm
dev = 0 if torch.cuda.is_available() else -1
print("เตือน: BLIP ตอบเป็นภาษาอังกฤษ + อ่านตัวอักษร(OCR)ไม่เก่ง -- ถ้าโจทย์ต้องการคำตอบไทย/OCR ให้ใช้ Qwen2.5-VL เซลล์ถัดไปแทน")
vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base", device=dev)
ans=[]
for _,r in tqdm(test.iterrows(), total=len(test)):   # มีแถบ progress ให้เห็นว่าไม่ค้าง
    img = Image.open(os.path.join(TEST_IMG, str(r[IMG_ID_COL]))).convert("RGB")
    q = str(r[Q_COL]) if Q_COL else "What is in the image?"
    ans.append(vqa(image=img, question=q, top_k=1)[0]["answer"])
out=sample.copy(); out[ANS_COL]=ans
out.to_csv("submission.csv",index=False,encoding="utf-8-sig")
print("บันทึก submission.csv"); display(out.head())"""))

c.append(md(r"""## ทางเลือก — Qwen2.5-VL (รองรับไทย + OCR ดีกว่า, ต้อง GPU)

ปลดคอมเมนต์ถ้าต้องการคำตอบภาษาไทยหรืออ่านตัวอักษรในรูป"""))
c.append(code(r"""# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
#         torch_dtype="auto", device_map="auto")
# proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# def ask(img_path, question):
#     msgs=[{"role":"user","content":[{"type":"image","image":img_path},{"type":"text","text":question}]}]
#     text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
#     inputs=proc(text=[text], images=[Image.open(img_path).convert("RGB")], return_tensors="pt").to(mdl.device)
#     out=mdl.generate(**inputs, max_new_tokens=64)
#     return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
print("ปลดคอมเมนต์เซลล์นี้ถ้าจะใช้ Qwen2.5-VL (ภาษาไทย/OCR)")"""))

c.append(submit_md()); c.append(submit_cell("blip vqa zero-shot"))
write_nb("03_Multimodal_VisionLanguage/visual_qa.ipynb", c)

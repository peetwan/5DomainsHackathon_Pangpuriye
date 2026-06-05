import sys; sys.path.insert(0, "_build")
from _nbutil import md, code, write_nb
from _common import kaggle_md, kaggle_cell, submit_md, submit_cell, inspect_md, inspect_cell

# ============================================================
# โน้ตบุ๊ก 1: Image Classification
# ============================================================
c = []
c.append(md(r"""# Computer Vision — โจทย์แบบ "จำแนกรูปภาพ" (Image Classification)

รูป 1 รูป -> 1 ป้าย (เช่น บ้าน/ไม่ใช่บ้าน, สุนัข/แมว, ปกติ/ผิดปกติ, ชนิดสินค้า)

วิธีในโน้ตบุ๊กนี้ (ทำจากบนลงล่าง):
- วิธีที่ 1 (ง่ายสุด แนะนำมือใหม่ทำแค่นี้) = `AutoGluon` กดรันแล้วมันเลือกโมเดล+เทรนให้เอง
- วิธีที่ 2 (ไม่บังคับ ทำถ้าอยากได้คะแนนสูงขึ้น) = `timm` เลือกโมเดลเอง + TTA
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ: input เป็นรูป และต้องตอบว่า "รูปนี้คือคลาสอะไร" (ตอบเป็นป้าย/ตัวเลข ไม่ใช่ข้อความยาว)
ถ้าโจทย์ให้ "หากรอบวัตถุ (กล่อง)" -> ไปใช้ `object_detection.ipynb`
ถ้าโจทย์ให้ "บรรยายรูปเป็นข้อความ" -> ไปหัวข้อ 03 Multimodal

ต้องแก้อะไรบ้าง (หาคำว่า `# << แก้`): ชื่อ competition, path โฟลเดอร์รูป, ชื่อคอลัมน์, จำนวนคลาส"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install -U "autogluon.multimodal"        # วิธีที่ 1
!pip -q install timm torch torchvision pillow scikit-learn pandas numpy tqdm   # วิธีที่ 2"""))

c.append(kaggle_md())
c.append(kaggle_cell("super-ai-engineer-season-6-individual-hackathon-house-recognition"))

c.append(md(r"""## ขั้นที่ 3 — ตั้งค่า path + CONFIG (ดูจาก output ด้านบนแล้วแก้ให้ตรง)"""))
c.append(code(r"""import os, glob, pandas as pd, numpy as np

def find(name):   # หาไฟล์ตามชื่อ
    h = glob.glob(os.path.join(DATA_DIR, "**", name), recursive=True); return h[0] if h else None
def img_dir(keyword):   # หาโฟลเดอร์รูปที่ชื่อมี keyword (train/test)
    cand = [d for d,_,fs in os.walk(DATA_DIR)
            if keyword in d.lower() and any(f.lower().endswith((".jpg",".png",".jpeg")) for f in fs)]
    return max(cand, key=lambda d: len(os.listdir(d))) if cand else None

TRAIN_CSV  = find("train.csv")               # << แก้ถ้าหาไม่เจอ: ใส่ path เอง เช่น "data/train_labels.csv"
SAMPLE_SUB = find("sample_submission.csv")
TRAIN_IMG  = img_dir("train")                # << แก้ถ้าผิด: ใส่ path เอง เช่น "data/train_images"
TEST_IMG   = img_dir("test")                 # << แก้ถ้าผิด: เช่น "data/test_images"

df = pd.read_csv(TRAIN_CSV); sample = pd.read_csv(SAMPLE_SUB)
print("train.csv คอลัมน์:", list(df.columns)); print("sample คอลัมน์:", list(sample.columns))
print("รูป train:", TRAIN_IMG, "| รูป test:", TEST_IMG)
display(df.head()); display(sample.head())

IMG_COL    = "image_name"        # << แก้: ชื่อคอลัมน์ไฟล์รูปใน train.csv (ดูจาก "train.csv คอลัมน์:") เช่น "filename", "id"
LABEL_COL  = "class"             # << แก้: ชื่อคอลัมน์คำตอบใน train.csv เช่น "label", "target", "category"
ID_COL     = sample.columns[0]   # เดาอัตโนมัติ: คอลัมน์แรกของ sample คือ id
ANSWER_COL = sample.columns[1]   # เดาอัตโนมัติ: คอลัมน์ที่สองคือคำตอบ
TEST_EXT   = ".jpg"              # << แก้ถ้านามสกุลไฟล์ test ไม่ใช่ .jpg เช่น ".png" (ถ้าใน sample id มีนามสกุลอยู่แล้ว ใส่ "")
NUM_CLASSES = int(df[LABEL_COL].nunique())
print("จำนวนคลาส =", NUM_CLASSES)"""))

c.append(inspect_md()); c.append(inspect_cell())
c.append(md(r"""## วิธีที่ 1 — AutoGluon (ง่ายสุด มือใหม่ทำแค่นี้ก็ได้คะแนนดี)

แค่ทำตารางที่มี path รูป + ป้าย แล้วสั่ง fit เดี๋ยว AutoGluon เลือกโมเดล + เทรน + รวมโมเดลให้เอง
ปรับ `time_limit` (วินาที) ตามเวลาที่มี"""))
c.append(code(r"""from autogluon.multimodal import MultiModalPredictor

train_df = df.copy()
train_df["image"] = train_df[IMG_COL].apply(lambda n: os.path.join(TRAIN_IMG, str(n)))
predictor = MultiModalPredictor(label=LABEL_COL, eval_metric="accuracy", path="ag_imgcls")
predictor.fit(train_df[["image", LABEL_COL]], time_limit=900)   # << แก้ time_limit: วินาที (900=15นาที) ลอง 300 ก่อน, ส่งจริงเพิ่มเป็น 1800-3600

test_df = sample.copy()
test_df["image"] = test_df[ID_COL].apply(lambda i: os.path.join(TEST_IMG, str(i)+TEST_EXT))
out = sample.copy(); out[ANSWER_COL] = predictor.predict(test_df[["image"]]).values   # ไม่ astype(int) เพื่อรองรับป้ายที่เป็นข้อความด้วย
out.to_csv("submission.csv", index=False)
print("บันทึก submission.csv เรียบร้อย"); display(out.head())"""))

c.append(md(r"""## วิธีที่ 2 — timm fine-tune + TTA (ไม่บังคับ ทำถ้าอยากได้คะแนนสูงขึ้น ต้องมี GPU)

โหลดโมเดล pretrained จาก `timm` มาเทรนต่อ + augmentation + TTA
เลือกโมเดลอื่นได้ที่ `MODEL_NAME` (ดูตัวเลือกใน reference_cheatsheet.md)"""))
c.append(code(r"""import torch, torch.nn as nn, torch.nn.functional as F, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm.auto import tqdm

MODEL_NAME="tf_efficientnetv2_s.in21k_ft_in1k"  # << แก้โมเดลได้ เช่น "resnet50" (เบา/เร็ว), "convnext_small.fb_in22k_ft_in1k" (แม่น)
IMG_SIZE=300; EPOCHS=12; BATCH=32; LR=3e-4      # << แก้ตามเวลา/GPU
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; AMP=(DEVICE=="cuda")
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]
tf_tr=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3,.3,.3,.1),transforms.RandomRotation(15),transforms.ToTensor(),
    transforms.Normalize(MEAN,STD),transforms.RandomErasing(p=0.2)])
tf_ev=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
tf_tta=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
class DS(Dataset):
    def __init__(s,frame,d,tf,test=False,idc=None): s.f=frame.reset_index(drop=True);s.d=d;s.tf=tf;s.t=test;s.idc=idc
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        r=s.f.iloc[i]
        if s.t: return s.tf(Image.open(os.path.join(s.d,str(r[s.idc])+TEST_EXT)).convert("RGB")),0
        return s.tf(Image.open(os.path.join(s.d,str(r[IMG_COL]))).convert("RGB")),int(r[LABEL_COL])
skf=StratifiedKFold(5,shuffle=True,random_state=42)
tr,va=next(iter(skf.split(df,df[LABEL_COL])))
model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=NUM_CLASSES,drop_rate=0.3).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
crit=nn.CrossEntropyLoss(label_smoothing=0.1); scaler=torch.amp.GradScaler("cuda", enabled=AMP)
dl=DataLoader(DS(df.iloc[tr],TRAIN_IMG,tf_tr),BATCH,shuffle=True,num_workers=2,drop_last=True)
vl=DataLoader(DS(df.iloc[va],TRAIN_IMG,tf_ev),BATCH,num_workers=2)
for ep in range(EPOCHS):
    model.train()
    for x,y in tqdm(dl,desc=f"ep{ep+1}/{EPOCHS}",leave=False):
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=AMP): loss=crit(model(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    sch.step()
    model.eval(); cc=tt=0
    with torch.no_grad():
        for x,y in vl:
            with torch.amp.autocast(device_type=DEVICE, enabled=AMP): pr=model(x.to(DEVICE)).argmax(1).cpu()
            cc+=(pr==y).sum().item(); tt+=len(y)
    print(f"ep{ep+1} val_acc={cc/tt:.4f}")
probs=np.zeros((len(sample),NUM_CLASSES))
for s_ in range(5):
    tf=tf_ev if s_==0 else tf_tta
    dl=DataLoader(DS(sample,TEST_IMG,tf,test=True,idc=ID_COL),BATCH,num_workers=2)
    o=[]
    with torch.no_grad():
        for x,_ in tqdm(dl,desc=f"TTA{s_+1}",leave=False):
            with torch.amp.autocast(device_type=DEVICE, enabled=AMP): o.append(F.softmax(model(x.to(DEVICE)),1).float().cpu().numpy())
    probs+=np.vstack(o)
out=sample.copy(); out[ANSWER_COL]=probs.argmax(1).astype(int)
out.to_csv("submission_timm.csv",index=False); print("บันทึก submission_timm.csv"); display(out.head())"""))

c.append(submit_md()); c.append(submit_cell("image classification"))
write_nb("01_Computer_Vision/image_classification.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 2: Object Detection
# ============================================================
c = []
c.append(md(r"""# Computer Vision — โจทย์แบบ "หาวัตถุในรูป (กล่อง)" Object Detection

รูป 1 รูป -> หากล่อง (bounding box) + ป้ายของวัตถุ (เช่น หาบ้าน/รถ/ป้ายทะเบียนในรูป)

วิธีในโน้ตบุ๊กนี้: `YOLOv8` (ไลบรารี ultralytics) เป็นวิธีที่ง่ายและแรงสุดสำหรับมือใหม่
สั่งเทรน 2-3 บรรทัดจบ
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ: โจทย์ให้ทำนาย "ตำแหน่งกล่อง + ชนิดวัตถุ" metric มักเป็น `mAP`
ถ้าโจทย์แค่ "รูปนี้คือคลาสอะไร" (ไม่มีกล่อง) -> กลับไป `image_classification.ipynb` (ง่ายกว่ามาก)

หมายเหตุมือใหม่: detection ยากกว่า classification เพราะข้อมูลต้องมี label เป็นกล่อง (รูปแบบ YOLO)
ถ้าเวลาน้อยและโจทย์เป็น classification ได้ ให้เลือก classification ก่อน"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install ultralytics pandas pyyaml"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-detection-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — เตรียมข้อมูลแบบ YOLO

YOLO ต้องการ: โฟลเดอร์รูป + ไฟล์ label `.txt` (คลาส x_center y_center w h เป็นค่า 0-1) ต่อรูป 1 ไฟล์
และไฟล์ `data.yaml` ที่บอก path รูป train/val + ชื่อคลาส

สำคัญสำหรับมือใหม่: ถ้า competition ให้ข้อมูลเป็นรูปแบบอื่น (COCO json / csv ของกล่อง) ต้องแปลงเป็นรูปแบบ YOLO ก่อน
ถ้าตอนเทรนขึ้น error `No labels found in ...` แปลว่ายังไม่ได้แปลงข้อมูลเป็นรูปแบบ YOLO (ต้องสร้างไฟล์ .txt ต่อรูปก่อน)"""))
c.append(code(r"""import yaml, os
# << แก้ path เหล่านี้ให้ตรงกับข้อมูลจริง (โฟลเดอร์ที่มีรูป + label .txt)
data_cfg = {
    "path": os.path.abspath(DATA_DIR),   # โฟลเดอร์หลัก (เขียน path สั้น ๆ ต่อจากนี้ในบรรทัดล่าง)
    "train": "images/train",             # << แก้: โฟลเดอร์รูป train เช่น "train/images" หรือ "dataset/images/train"
    "val":   "images/val",               # << แก้: โฟลเดอร์รูป val เช่น "valid/images" (ไม่มี val ให้แบ่งจาก train เองก่อน)
    "names": {0: "object"},              # << แก้: ชื่อคลาส เช่น {0:"house",1:"car"} (เลขต้องตรงกับเลขคลาสในไฟล์ label .txt)
}
with open("data.yaml","w") as f: yaml.safe_dump(data_cfg, f, allow_unicode=True)
print(open("data.yaml").read())"""))

c.append(md(r"""## ขั้นที่ 4 — เทรน YOLO (ง่ายสุด)

`yolo11n` = เล็ก/เร็ว, `yolo11s/m` = ใหญ่/แม่นขึ้น (รุ่นใหม่กว่า yolov8) ปรับ `epochs`/`imgsz` ตามเวลา/GPU
เคล็ดลับ: ลอง `epochs=10` ก่อนให้รันผ่าน แล้วค่อยเพิ่ม (epochs สูงใช้เวลานานมากถ้าไม่มี GPU)"""))
c.append(code(r"""from ultralytics import YOLO
model = YOLO("yolo11n.pt")      # << แก้: ใหม่สุดปี 2026 "yolo26n.pt" (n/s/m/l/x) เร็ว+แม่นกว่า ; accuracy สูงสุด -> rfdetr (ดู ADVANCED_SOTA.md)
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16)   # << แก้: ลอง epochs=10 ก่อน; GPU เล็ก ลด batch=8 หรือ imgsz=512
metrics = model.val(); print(metrics)   # ถ้าไม่มีชุด val แยก .val() อาจ error -> ก๊อปรูป train ส่วนหนึ่งไป images/val ก่อน"""))

c.append(md(r"""## ขั้นที่ 5 — ทำนาย test + สร้าง submission

รูปแบบ submission ของ detection แตกต่างกันมากในแต่ละ comp (อ่าน sample_submission ให้ดี)
ตัวอย่างล่างสร้างตาราง: id, class, confidence, x_min, y_min, x_max, y_max — แก้ให้ตรงรูปแบบจริง
ถ้าไฟล์ที่ได้ว่าง (0 แถว) แปลว่าโมเดลยังไม่เจอวัตถุ -> เทรนเพิ่มหรือลด conf"""))
c.append(code(r"""import glob, pandas as pd
TEST_IMG_DIR = os.path.join(DATA_DIR, "images/test")   # << แก้: path รูป test เช่น os.path.join(DATA_DIR,"test/images")
rows = []
for p in sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*"))):
    r = model.predict(p, verbose=False)[0]
    img_id = os.path.splitext(os.path.basename(p))[0]
    for b in r.boxes:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        rows.append({"id": img_id, "class": int(b.cls[0]), "confidence": float(b.conf[0]),
                     "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2})
sub = pd.DataFrame(rows)            # << แก้คอลัมน์ให้ตรง sample เช่น sub = sub.rename(columns={"class":"label"})
sub.to_csv("submission.csv", index=False)
print("บันทึก submission.csv", sub.shape); display(sub.head())"""))

c.append(submit_md()); c.append(submit_cell("yolov8 detection"))
write_nb("01_Computer_Vision/object_detection.ipynb", c)


# ============================================================
# โน้ตบุ๊ก 3: Image Segmentation
# ============================================================
c = []
c.append(md(r"""# Computer Vision — โจทย์แบบ "แบ่งส่วนภาพ" (Segmentation)

ระบุว่าพิกเซลไหนเป็นวัตถุอะไร (เช่น แยกพื้นที่บ้าน/ถนน/ต้นไม้ในภาพ)

วิธีในโน้ตบุ๊กนี้: `YOLOv8-seg` (ultralytics) ง่ายสุดสำหรับมือใหม่ ใช้คล้าย detection แต่ได้ mask
"""))
c.append(md(r"""## เมื่อไหร่ใช้โน้ตบุ๊กนี้

ใช้เมื่อ output เป็น "mask ต่อพิกเซล" (ระบายว่าพิกเซลไหนเป็นคลาสอะไร) metric มักเป็น `IoU`/`Dice`
ถ้าแค่ "รูปนี้คือคลาสอะไร" -> `image_classification.ipynb` (ง่ายกว่ามาก)
ถ้าแค่ "กล่องรอบวัตถุ" -> `object_detection.ipynb`

หมายเหตุมือใหม่: segmentation ต้องมี label เป็น mask/polygon (รูปแบบ YOLO-seg) ถ้าเวลาน้อยและทำ classification ได้ ให้เลือก classification"""))

c.append(md(r"""## ขั้นที่ 1 — ติดตั้ง"""))
c.append(code(r"""!pip -q install ultralytics pyyaml numpy"""))
c.append(kaggle_md())
c.append(kaggle_cell("ใส่-slug-ของ-competition-segmentation-ตรงนี้"))

c.append(md(r"""## ขั้นที่ 3 — เตรียม data.yaml (รูปแบบ YOLO-seg)

YOLO-seg ต้องการ: โฟลเดอร์รูป + ไฟล์ label `.txt` (คลาส + พิกัด polygon เป็นค่า 0-1) ต่อรูป
ถ้าข้อมูลเป็นรูปแบบอื่น (mask PNG / COCO json) ต้องแปลงเป็น YOLO-seg ก่อน
ถ้าเทรนขึ้น error `No labels found` แปลว่ายังไม่ได้แปลงข้อมูลเป็นรูปแบบ YOLO-seg"""))
c.append(code(r"""import yaml, os
data_cfg = {
    "path": os.path.abspath(DATA_DIR),
    "train": "images/train",   # << แก้: โฟลเดอร์รูป train เช่น "train/images"
    "val":   "images/val",     # << แก้: โฟลเดอร์รูป val เช่น "valid/images" (ไม่มีให้แบ่งจาก train)
    "names": {0: "object"},    # << แก้: ชื่อคลาส เช่น {0:"building",1:"road"}
}
with open("data_seg.yaml","w") as f: yaml.safe_dump(data_cfg, f, allow_unicode=True)
print(open("data_seg.yaml").read())"""))

c.append(md(r"""## ขั้นที่ 4 — เทรน YOLO-seg (ง่ายสุด)

`RLE` = วิธีบีบอัด mask ให้เป็นตัวเลขสั้น ๆ ต่อ 1 วัตถุ (comp ส่วนใหญ่ใช้แบบนี้)
`IoU`/`Dice` = วัดว่าพื้นที่ที่ทายทับกับเฉลยมากแค่ไหน ยิ่งสูงยิ่งดี"""))
c.append(code(r"""from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")    # << แก้: ใหม่สุดปี 2026 "yolo26n-seg.pt" ; mask คุณภาพสูงไม่ต้องเทรน -> SAM3 (facebook/sam3)
model.train(data="data_seg.yaml", epochs=50, imgsz=640, batch=16)   # << แก้: ลอง epochs=10 ก่อน; GPU เล็ก batch=8
print(model.val())"""))

c.append(md(r"""## ขั้นที่ 5 — ทำนาย test + สร้าง submission (RLE)

comp ส่วนใหญ่ใช้ RLE ด้านล่างมีฟังก์ชัน `rle_encode` มาตรฐานให้ + resize mask กลับเป็นขนาดรูปจริงก่อน encode
(ของ ultralytics คืน mask ขนาดตาม imgsz ไม่ใช่ขนาดรูปจริง ต้อง resize ก่อน ไม่งั้น RLE จะผิดตำแหน่ง)"""))
c.append(code(r"""import glob, numpy as np, pandas as pd, cv2
def rle_encode(mask):   # mask 0/1 -> สตริง RLE แบบ Kaggle (นับตามคอลัมน์)
    pix = mask.flatten(order="F"); pix = np.concatenate([[0], pix, [0]])
    runs = np.where(pix[1:] != pix[:-1])[0] + 1; runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
TEST_IMG_DIR = os.path.join(DATA_DIR, "images/test")   # << แก้: path รูป test เช่น os.path.join(DATA_DIR,"test/images")
rows = []
for p in sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*"))):
    r = model.predict(p, verbose=False)[0]
    img_id = os.path.splitext(os.path.basename(p))[0]
    H, W = r.orig_shape   # ขนาดรูปจริง
    if r.masks is None: continue
    for ci in range(len(r.masks.data)):
        m = r.masks.data[ci].cpu().numpy().astype(np.uint8)         # ขนาดตาม imgsz ของโมเดล
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)  # resize กลับขนาดรูปจริง
        cls = int(r.boxes.cls[ci]) if r.boxes is not None else 0
        rows.append({"id": img_id, "class": cls, "rle": rle_encode(m)})  # << แก้คอลัมน์ให้ตรง sample
sub = pd.DataFrame(rows); sub.to_csv("submission.csv", index=False)
print("บันทึก submission.csv", sub.shape); display(sub.head())"""))

c.append(submit_md()); c.append(submit_cell("yolov8 segmentation"))
write_nb("01_Computer_Vision/segmentation.ipynb", c)

# Single source of truth for the master router.
# Tested by verify_router.py AND embedded into 00_MASTER_ROUTER.ipynb by gen_meta.py.
import re

CATS = {
 "cv_classification": {
   "label": "Computer Vision — จำแนกรูป (Image Classification)",
   "path": "01_Computer_Vision/image_classification.ipynb",
   "first": "วิธีที่ 1 = AutoGluon MultiModal (ตั้ง NUM_CLASSES ให้ตรง)",
   "kw": ["classify image","image classification","image recognition","จำแนกรูป","จำแนกภาพ","รูปบ้าน",
          "ชนิดรูป","ป้ายรูป","cat dog","สุนัขแมว","label the image","what class is this image",
          "photo classification","ภาพนี้คือ","categorize image","ประเภทของรูป","photo","photos",
          "categories","category","image into","ประเภทไหน","หมวดของรูป","classify each"]},
 "cv_detection": {
   "label": "Computer Vision — หาวัตถุ (Object Detection)",
   "path": "01_Computer_Vision/object_detection.ipynb",
   "first": "วิธีที่ 1 = YOLOv8 (เตรียมข้อมูลรูปแบบ YOLO + data.yaml)",
   "kw": ["object detection","detection","detect","detects","ตรวจจับวัตถุ","ตรวจจับ","bounding box",
          "bounding","กล่อง","หากล่อง","locate","locate object","หาตำแหน่งวัตถุ","map","นับวัตถุ",
          "count objects","count","นับ","object","objects","ป้ายทะเบียน","หาวัตถุในรูป","draw box","yolo"]},
 "cv_segmentation": {
   "label": "Computer Vision — แบ่งส่วนภาพ (Segmentation)",
   "path": "01_Computer_Vision/segmentation.ipynb",
   "first": "วิธีที่ 1 = YOLOv8-seg (ง่ายสุด) หรือ U-Net",
   "kw": ["segmentation","แบ่งส่วนภาพ","แบ่งพื้นที่","mask","ระบายพิกเซล","semantic segmentation",
          "instance segmentation","pixel level","ต่อพิกเซล","iou","dice","แยกพื้นที่ในรูป","ระบายสีตามพื้นที่"]},
 "nlp_sequence_labeling": {
   "label": "NLP — ตัดคำ / ป้ายต่อ token (Sequence Labeling)",
   "path": "02_NLP/thai_word_segmentation.ipynb",
   "first": "วิธีที่ 1 = PyThaiNLP deepcut",
   "kw": ["word segmentation","ตัดคำ","tokenize","tokenization","ner","named entity","entity",
          "ดึงชื่อเฉพาะ","pos tagging","part of speech","ชนิดคำ","sequence labeling","ป้ายต่อคำ",
          "ป้ายต่อตัวอักษร","ขอบเขตคำ","แท็กต่อคำ","tag each","each token","token","tokens","tagging","แท็ก"]},
 "nlp_text_classification": {
   "label": "NLP — จำแนกข้อความ (Text Classification)",
   "path": "02_NLP/text_classification.ipynb",
   "first": "วิธีที่ 1 = ตัดคำ + TF-IDF + Logistic Regression",
   "kw": ["text classification","จำแนกข้อความ","sentiment","ความรู้สึก","อารมณ์","รีวิว","review",
          "บวกลบ","positive negative","topic classification","หมวดข่าว","หมวดหมู่ข้อความ","spam","สแปม",
          "intent","เจตนา","จัดประเภทข้อความ","topic","topics","headline","headlines","news","ข่าว",
          "label text","classify text","classify the text","classify review","emotion",
          "spam detection","detect spam","email","อีเมล"]},
 "nlp_text_generation": {
   "label": "NLP — สร้าง/แปลง ข้อความ (Text Generation / Seq2Seq)",
   "path": "02_NLP/text_generation.ipynb",
   "first": "วิธีที่ 1 = HuggingFace pipeline หรือ mT5 fine-tune",
   "kw": ["translation","translate","แปลภาษา","แปลข้อความ","แปล","summarization","summarize","สรุปความ",
          "สรุปข้อความ","สรุป","generate text","สร้างข้อความ","seq2seq","question answering",
          "ตอบคำถามจากบทความ","paraphrase","ถอดความ","spelling correction","แก้คำผิด","grammar",
          "เขียนใหม่","machine translation","article","บทความ","rewrite","สรุปบทความ"]},
 "mm_captioning": {
   "label": "Multimodal — บรรยายรูป (Image Captioning)",
   "path": "03_Multimodal_VisionLanguage/thai_image_captioning.ipynb",
   "first": "วิธีที่ 1 = โมเดล Thai-COCO สำเร็จรูป + วัด thai_bleu()",
   "kw": ["image captioning","caption","captioning","image caption","บรรยายรูป","บรรยายภาพ","describe image",
          "อธิบายรูป","คำบรรยาย","image to text","รูปเป็นข้อความ","เขียนคำบรรยายรูป","bleu","generate caption"]},
 "mm_vqa": {
   "label": "Multimodal — ตอบคำถามจากรูป / OCR (Visual QA)",
   "path": "03_Multimodal_VisionLanguage/visual_qa.ipynb",
   "first": "วิธีที่ 1 = VLM สำเร็จรูป (Qwen2.5-VL / BLIP-VQA) แบบ zero-shot",
   "kw": ["vqa","visual question answering","ตอบคำถามจากรูป","ถามรูป","ถามจากภาพ","question about image",
          "ocr","อ่านข้อความในรูป","read text in image","อ่านตัวอักษรในภาพ","text in image","extract text from image"]},
 "tab_classification": {
   "label": "Tabular — ทำนายคลาส (Classification)",
   "path": "04_Tabular/classification.ipynb",
   "first": "วิธีที่ 1 = AutoGluon Tabular (ตั้ง METRIC ให้ตรง)",
   "kw": ["tabular classification","จำแนกจากตาราง","ทำนายโรค","predict disease","ป่วย","churn","ลูกค้าเลิกใช้",
          "fraud","ทุจริต","predict class","จำแนกข้อมูลตาราง","binary classification","หลายคลาส","multiclass",
          "default","อนุมัติสินเชื่อ","credit"]},
 "tab_regression": {
   "label": "Tabular — ทำนายตัวเลข (Regression)",
   "path": "04_Tabular/regression.ipynb",
   "first": "วิธีที่ 1 = AutoGluon problem_type=regression",
   "kw": ["regression","ทำนายราคา","predict price","house price","ราคาบ้าน","คะแนน","score","ยอดขาย",
          "sales amount","predict value","ตัวเลขต่อเนื่อง","continuous value","rmse","mae","ประมาณค่า","estimate value"]},
 "ts_signal_classification": {
   "label": "Time-Series / Signal — จำแนกสัญญาณ (Classification)",
   "path": "05_TimeSeries_Signal/signal_classification.ipynb",
   "first": "วิธีที่ 1 = สกัดฟีเจอร์ + AutoGluon (แบ่ง CV ตาม subject)",
   "kw": ["eeg","ecg","emg","sleep stage","ระยะการนอน","การนอน","biosignal","สัญญาณชีวภาพ","signal classification",
          "จำแนกสัญญาณ","accelerometer","เซนเซอร์","sensor","human activity","har","ท่าทาง","arrhythmia",
          "seizure","sampling rate","หน้าต่างเวลา","epoch 30","classify window"]},
 "ts_forecasting": {
   "label": "Time-Series — พยากรณ์ค่าอนาคต (Forecasting)",
   "path": "05_TimeSeries_Signal/forecasting.ipynb",
   "first": "วิธีที่ 1 = ฟีเจอร์วันที่ + lag + LightGBM",
   "kw": ["forecast","forecasting","พยากรณ์","พยากรณ์ยอดขาย","ทำนายอนาคต","predict future","future value",
          "demand forecast","demand","ยอดขายพรุ่งนี้","ราคาล่วงหน้า","next day","next week",
          "time series prediction","ทำนายยอดขาย","predict next","แนวโน้ม","trend prediction",
          "ย้อนหลัง","ข้อมูลย้อนหลัง","historical"]},
}

def _hit(kw, p):
    # English keyword -> match on letter boundaries (so "ner" doesn't match "generate").
    # Thai keyword -> plain substring (Thai has no spaces).
    kw = kw.lower()
    if kw.isascii():
        return re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", p) is not None
    return kw in p

def route(problem):
    p = (problem or "").lower()
    # score in CATS insertion order, then stable sort by score desc (ties keep CATS order)
    items = [(sum(1 for k in v["kw"] if _hit(k, p)), name) for name, v in CATS.items()]
    scored = sorted(items, key=lambda x: -x[0])
    best_score, best_name = scored[0]
    return best_name, best_score, scored

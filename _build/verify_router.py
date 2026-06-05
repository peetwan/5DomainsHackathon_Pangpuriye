import sys; sys.path.insert(0, "_build")
from router import route, CATS

# (problem description, expected category key) — mix of Thai and English, realistic exam phrasings
TESTS = [
 # Computer Vision
 ("จำแนกรูปบ้านว่าเป็นประเภทไหน", "cv_classification"),
 ("Classify each photo into one of 5 categories", "cv_classification"),
 ("ตรวจจับวัตถุในภาพและวาดกล่องล้อมรอบ", "cv_detection"),
 ("Detect and locate all cars with bounding boxes, evaluated by mAP", "cv_detection"),
 ("ทำ semantic segmentation แยกพื้นที่ในรูปต่อพิกเซล", "cv_segmentation"),
 ("Predict a pixel-level mask for each image (IoU metric)", "cv_segmentation"),
 # NLP
 ("ตัดคำภาษาไทยจากข้อความที่ไม่มีเว้นวรรค", "nlp_sequence_labeling"),
 ("Named entity recognition: tag each token", "nlp_sequence_labeling"),
 ("จำแนกรีวิวว่าเป็นบวกหรือลบ (sentiment)", "nlp_text_classification"),
 ("Classify news headlines into topics", "nlp_text_classification"),
 ("แปลข้อความอังกฤษเป็นไทย", "nlp_text_generation"),
 ("Summarize the article into a short paragraph", "nlp_text_generation"),
 # Multimodal
 ("บรรยายรูปภาพเป็นประโยคภาษาไทย วัดด้วย BLEU", "mm_captioning"),
 ("Generate an image caption", "mm_captioning"),
 ("ตอบคำถามจากรูปภาพ (visual question answering)", "mm_vqa"),
 ("Read the text written in the image (OCR)", "mm_vqa"),
 # Tabular
 ("ทำนายว่าผู้ป่วยเป็นโรคหัวใจหรือไม่ จากตารางข้อมูล", "tab_classification"),
 ("Predict customer churn (yes/no) from tabular features", "tab_classification"),
 ("ทำนายราคาบ้านจากคุณสมบัติต่าง ๆ", "tab_regression"),
 ("Predict a continuous sales amount, RMSE metric", "tab_regression"),
 # Time-Series
 ("จำแนกระยะการนอนจากสัญญาณ EEG ทุก 30 วินาที", "ts_signal_classification"),
 ("Classify ECG signal windows into arrhythmia types", "ts_signal_classification"),
 ("พยากรณ์ยอดขายของพรุ่งนี้จากข้อมูลย้อนหลัง", "ts_forecasting"),
 ("Forecast next week demand from the time series", "ts_forecasting"),
 # extra varied phrasings
 ("Image classification of flowers into species", "cv_classification"),
 ("instance segmentation of cells", "cv_segmentation"),
 ("POS tagging ภาษาไทย", "nlp_sequence_labeling"),
 ("spam detection ของอีเมล", "nlp_text_classification"),
 ("machine translation อังกฤษเป็นไทย", "nlp_text_generation"),
 ("OCR อ่านตัวอักษรจากใบเสร็จ", "mm_vqa"),
 ("predict house price regression", "tab_regression"),
 ("human activity recognition จาก accelerometer", "ts_signal_classification"),
 ("detect objects and count them", "cv_detection"),
 ("จัดประเภทข้อความว่าเป็นหมวดไหน", "nlp_text_classification"),
 ("จำแนกเสียงสัตว์จากไฟล์ wav", "audio_classification"),
 ("audio sound classification of music genre", "audio_classification"),
 ("speech command keyword spotting", "audio_classification"),
]

passed = 0
fails = []
for desc, expected in TESTS:
    name, score, scored = route(desc)
    ok = (name == expected)
    passed += ok
    if not ok:
        fails.append((desc, expected, name, scored[:3]))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] -> {name:26s} (expected {expected:26s}) score={score} | {desc[:50]}")

print(f"\n{passed}/{len(TESTS)} routed correctly")
if fails:
    print("\n--- FAILURES (need keyword tuning) ---")
    for desc, exp, got, top in fails:
        print(f"  '{desc}'\n    expected {exp}, got {got}; top3={[ (s,n) for n,s in [] ] or top}")
    sys.exit(1)
# sanity: every category has a notebook path and is reachable
print("\nทุกหมวดมี path:", all('path' in v and v['path'].endswith('.ipynb') for v in CATS.values()))

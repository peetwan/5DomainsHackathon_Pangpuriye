# Builds a single responsive index.html containing ALL project info.
# Reads the .md files + .ipynb notebooks + the tested router (router.py).
import sys, os, re, json, html
sys.path.insert(0, "_build")
from router import CATS

ROOT = "."

def esc(s):
    return html.escape(s, quote=False)

def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", lambda m: "<code>" + m.group(1) + "</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: "<strong>" + m.group(1) + "</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: '<a href="' + m.group(2) + '" target="_blank" rel="noopener">' + m.group(1) + "</a>", s)
    return s

def md2html(text):
    lines = text.split("\n"); out = []; i = 0; n = len(lines); stack = []
    def closel():
        while stack: out.append("</%s>" % stack.pop())
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("```"):
            i += 1; buf = []
            while i < n and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1; closel()
            out.append('<div class="codewrap"><button class="copy" onclick="cp(this)">copy</button><pre><code>'
                       + esc("\n".join(buf)) + "</code></pre></div>")
            continue
        if "|" in line and i + 1 < n and "-" in lines[i + 1] and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            closel()
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2; rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            t = ['<div class="tablewrap"><table><thead><tr>']
            for h in header: t.append("<th>" + inline(h) + "</th>")
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>")
                for c in r: t.append("<td>" + inline(c) + "</td>")
                t.append("</tr>")
            t.append("</tbody></table></div>"); out.append("".join(t)); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            closel(); lvl = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (lvl, inline(m.group(2).strip()), lvl)); i += 1; continue
        if re.match(r"^\s*---+\s*$", line):
            closel(); out.append("<hr>"); i += 1; continue
        if line.lstrip().startswith(">"):
            closel(); out.append("<blockquote>" + inline(line.lstrip()[1:].strip()) + "</blockquote>"); i += 1; continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            if not stack or stack[-1] != "ul":
                closel(); out.append("<ul>"); stack.append("ul")
            out.append("<li>" + inline(m.group(1)) + "</li>"); i += 1; continue
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            if not stack or stack[-1] != "ol":
                closel(); out.append("<ol>"); stack.append("ol")
            out.append("<li>" + inline(m.group(1)) + "</li>"); i += 1; continue
        if not line.strip():
            closel(); i += 1; continue
        closel(); out.append("<p>" + inline(line) + "</p>"); i += 1
    closel()
    return "\n".join(out)

def read(path):
    p = os.path.join(ROOT, path)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

def nb2html(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p): return ""
    d = json.load(open(p, encoding="utf-8"))
    parts = []
    for c in d["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            parts.append('<div class="nbmd">' + md2html(src) + "</div>")
        else:
            parts.append('<div class="codewrap"><button class="copy" onclick="cp(this)">copy</button>'
                         '<pre><code>' + esc(src) + "</code></pre></div>")
    return "\n".join(parts)

TOPICS = [
    ("cv", "🖼️", "Computer Vision", "01_Computer_Vision",
     ["image_classification", "object_detection", "segmentation"]),
    ("nlp", "📝", "NLP", "02_NLP",
     ["thai_word_segmentation", "text_classification", "text_generation"]),
    ("mm", "🎨", "Multimodal", "03_Multimodal_VisionLanguage",
     ["thai_image_captioning", "visual_qa"]),
    ("tab", "📊", "Tabular", "04_Tabular",
     ["classification", "regression"]),
    ("ts", "📈", "Time-Series", "05_TimeSeries_Signal",
     ["signal_classification", "forecasting"]),
    ("audio", "🔊", "Audio", "06_Audio", ["audio_classification"]),
]

def topic_section(folder, notebooks):
    h = [md2html(read(folder + "/README.md"))]
    for nb in notebooks:
        h.append('<details class="nb"><summary>📓 ' + nb + ".ipynb</summary>"
                 + nb2html(folder + "/" + nb + ".ipynb") + "</details>")
    if os.path.exists(os.path.join(ROOT, folder, "reference_cheatsheet.md")):
        h.append('<details class="nb"><summary>📚 reference_cheatsheet.md (ความรู้ลึก)</summary>'
                 + md2html(read(folder + "/reference_cheatsheet.md")) + "</details>")
    return "\n".join(h)

# ---- router widget ----
router_widget = (
    '<div class="router">'
    '<h2>🧭 ถามตัวช่วย: เจอโจทย์แบบไหน เปิดไฟล์ไหน</h2>'
    '<p>พิมพ์คำอธิบายโจทย์ (ไทยหรืออังกฤษ) แล้วกดปุ่ม</p>'
    '<div class="rrow"><input id="q" placeholder="เช่น ทำนายระยะการนอนจาก EEG / classify movie reviews" '
    'onkeydown="if(event.key===\'Enter\')doRoute()"><button onclick="doRoute()">หา</button></div>'
    '<div class="chips" id="chips"></div>'
    '<div id="rout"></div>'
    "</div>"
)

deploy_md = """# 🚂 Deploy บน Railway + GitHub

## GitHub
โค้ดทั้งหมดอยู่ที่ `https://github.com/peetwan/5DomainsHackathon_Pangpuriye`
clone แล้วใช้ได้เลย: `git clone https://github.com/peetwan/5DomainsHackathon_Pangpuriye`

## รันเว็บนี้บนเครื่อง
```bash
python server.py          # แล้วเปิด http://localhost:8000
```

## Deploy ขึ้น Railway (ให้เพื่อนในทีมเปิดดูได้)
1. ไปที่ railway.app -> New Project -> Deploy from GitHub repo -> เลือก repo นี้
2. Railway จะอ่าน `Procfile` (`web: python server.py`) แล้วรันให้เอง
3. กด Generate Domain จะได้ลิงก์เว็บสาธารณะ ส่งให้เพื่อนได้เลย

ไฟล์ที่เกี่ยวกับ deploy: `server.py`, `Procfile`, `requirements.txt`

## รันเว็บใหม่หลังแก้เนื้อหา
```bash
python _build/build_site.py    # สร้าง index.html ใหม่จากไฟล์ .md และ .ipynb
```
"""

SECTIONS = []
SECTIONS.append(("auto", "🪄", "AUTO SOLVER (ปุ่มเดียว)",
                 md2html("## 🪄 ขี้เกียจสุด? ใช้อันนี้\n\nแก้แค่ชื่อ competition (หรืออัปโหลด zip) แล้วกดรัน "
                         "มันเดาประเภทงานเอง + เทรน + สร้าง submission ให้อัตโนมัติ "
                         "ครอบคลุม: จำแนกรูป / จำแนกข้อความ / ตาราง (จำแนก/ทำนายตัวเลข) / สัญญาณ-พยากรณ์ (ทำเป็นตาราง)\n\n"
                         "เปิดไฟล์ `00_AUTO_SOLVER.ipynb` โค้ดทุกเซลล์อยู่ข้างล่างนี้ (กดปุ่ม copy ได้):")
                 + nb2html("00_AUTO_SOLVER.ipynb")))
SECTIONS.append(("start", "🏁", "เริ่มที่นี่", md2html(read("START_HERE.md"))))
SECTIONS.append(("colab", "🖥️", "คู่มือ Colab", md2html(read("COLAB_GUIDE.md"))))
SECTIONS.append(("playbook", "📋", "Playbook (ทำตามนี้)", md2html(read("PLAYBOOK.md"))))
SECTIONS.append(("ptypes", "🗺️", "แผนที่โจทย์ทั้งหมด", md2html(read("PROBLEM_TYPES.md"))))
SECTIONS.append(("trouble", "🔧", "แก้ปัญหา (errors)", md2html(read("TROUBLESHOOTING.md"))))
SECTIONS.append(("router", "🧭", "ตัวนำทาง (Router)",
                 router_widget + md2html(read("README.md"))))
for tid, icon, title, folder, nbs in TOPICS:
    SECTIONS.append((tid, icon, title, topic_section(folder, nbs)))
SECTIONS.append(("sota", "🚀", "SOTA อัปเดต", md2html(read("ADVANCED_SOTA.md"))))
SECTIONS.append(("verify", "✅", "Verify (ตรวจแล้ว)", md2html(read("VERIFY.md"))))
SECTIONS.append(("deploy", "🚂", "Deploy / GitHub", md2html(deploy_md)))

nav = []
secs = []
for sid, icon, title, body in SECTIONS:
    nav.append('<a class="navlink" data-t="%s" href="#%s">%s <span>%s</span></a>' % (sid, sid, icon, esc(title)))
    secs.append('<section id="%s" class="section">%s</section>' % (sid, body))

CSS = """
:root{--bg:#0f1419;--bg2:#161b22;--card:#1c2330;--bd:#2a3340;--tx:#e6edf3;--mut:#9aa7b4;--ac:#4cc2ff;--ac2:#7ee787;--code:#0b1020}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans Thai","Sarabun",sans-serif;background:var(--bg);color:var(--tx);line-height:1.65;font-size:16px}
header{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:12px;padding:12px 16px;background:rgba(15,20,25,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--bd)}
header h1{font-size:16px;margin:0;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
header .sub{color:var(--mut);font-size:12px}
#burger{display:none;background:var(--card);border:1px solid var(--bd);color:var(--tx);font-size:20px;border-radius:8px;width:40px;height:40px;cursor:pointer}
.wrap{display:flex;min-height:calc(100vh - 58px)}
#sidebar{width:270px;flex:0 0 270px;background:var(--bg2);border-right:1px solid var(--bd);padding:14px;overflow-y:auto;position:sticky;top:58px;height:calc(100vh - 58px)}
.navlink{display:flex;align-items:center;gap:10px;padding:11px 12px;border-radius:10px;color:var(--tx);text-decoration:none;font-size:15px;margin-bottom:4px;border:1px solid transparent}
.navlink span{font-size:14px}
.navlink:hover{background:var(--card)}
.navlink.active{background:var(--card);border-color:var(--ac);color:var(--ac)}
main{flex:1;min-width:0;padding:22px 20px 80px;max-width:920px;margin:0 auto;width:100%}
.section{display:none}
.section.active{display:block;animation:fade .25s}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
h1,h2,h3,h4{line-height:1.3;margin:1.3em 0 .55em}
h1{font-size:1.7rem;color:#fff}h2{font-size:1.32rem;color:var(--ac)}h3{font-size:1.12rem;color:var(--ac2)}h4{font-size:1rem}
p{margin:.6em 0}a{color:var(--ac)}
ul,ol{padding-left:1.4em;margin:.5em 0}li{margin:.3em 0}
hr{border:0;border-top:1px solid var(--bd);margin:1.6em 0}
blockquote{border-left:3px solid var(--ac);background:var(--card);margin:.8em 0;padding:.5em .9em;border-radius:0 8px 8px 0;color:var(--mut)}
code{background:#232b39;color:#ffd9a0;padding:.12em .4em;border-radius:5px;font-size:.86em;font-family:"SF Mono",Menlo,Consolas,monospace;word-break:break-word}
.codewrap{position:relative;margin:.8em 0}
.codewrap pre{background:var(--code);border:1px solid var(--bd);border-radius:10px;padding:14px;overflow-x:auto;font-size:13px}
.codewrap pre code{background:none;color:#d7e2ee;padding:0;font-size:13px;white-space:pre}
.copy{position:absolute;top:8px;right:8px;background:var(--card);border:1px solid var(--bd);color:var(--mut);font-size:11px;padding:4px 9px;border-radius:6px;cursor:pointer;z-index:2}
.copy:hover{color:var(--ac);border-color:var(--ac)}
.tablewrap{overflow-x:auto;margin:.9em 0;border:1px solid var(--bd);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{border-bottom:1px solid var(--bd);padding:9px 11px;text-align:left;vertical-align:top}
th{background:var(--card);color:var(--ac);font-weight:600;white-space:nowrap}
tr:last-child td{border-bottom:0}
details.nb{border:1px solid var(--bd);border-radius:10px;margin:.7em 0;background:var(--bg2);overflow:hidden}
details.nb>summary{cursor:pointer;padding:12px 14px;font-weight:600;list-style:none;user-select:none}
details.nb>summary::-webkit-details-marker{display:none}
details.nb[open]>summary{border-bottom:1px solid var(--bd);color:var(--ac)}
details.nb>*:not(summary){padding:0 14px}
.nbmd{padding-top:6px}
.router{background:var(--card);border:1px solid var(--ac);border-radius:14px;padding:16px;margin-bottom:20px}
.router h2{margin-top:0}
.rrow{display:flex;gap:8px;margin:10px 0}
.rrow input{flex:1;min-width:0;background:var(--code);border:1px solid var(--bd);color:var(--tx);padding:12px;border-radius:10px;font-size:15px}
.rrow button{background:var(--ac);color:#001018;border:0;font-weight:700;padding:0 20px;border-radius:10px;cursor:pointer;font-size:15px}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin:6px 0}
.chip{background:var(--bg2);border:1px solid var(--bd);color:var(--mut);font-size:12px;padding:6px 10px;border-radius:20px;cursor:pointer}
.chip:hover{color:var(--ac);border-color:var(--ac)}
#rout{margin-top:12px}
.hit{background:var(--bg2);border:1px solid var(--ac2);border-radius:10px;padding:12px;margin-top:8px}
.hit b{color:var(--ac2)}
.scorebar{font-size:12.5px;color:var(--mut);margin-top:8px}
.foot{color:var(--mut);font-size:12px;text-align:center;padding:24px 0}
@media(max-width:860px){
 #burger{display:block}
 #sidebar{position:fixed;left:0;top:58px;transform:translateX(-105%);transition:.25s;z-index:40;box-shadow:0 0 40px rgba(0,0,0,.6);height:calc(100vh - 58px)}
 #sidebar.open{transform:none}
 .scrim{display:none;position:fixed;inset:58px 0 0;background:rgba(0,0,0,.5);z-index:35}
 .scrim.open{display:block}
 main{padding:16px 14px 70px}
 header h1{font-size:14px}
 body{font-size:15px}
}
"""

JS = """
const CATS=__CATS__;
function hit(kw,p){kw=kw.toLowerCase();const a=/^[\\x00-\\x7F]*$/.test(kw);if(a){const e=kw.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&');return new RegExp("(^|[^a-z])"+e+"($|[^a-z])").test(p);}return p.indexOf(kw)>=0;}
function route(p){p=(p||'').toLowerCase();let sc=[];for(const k in CATS){let s=0;for(const w of CATS[k].kw){if(hit(w,p))s++;}sc.push([s,k]);}sc.sort((a,b)=>b[0]-a[0]);return sc;}
function gotoSec(id){document.querySelector('.navlink[data-t="'+id+'"]')?.click();}
function doRoute(){const p=document.getElementById('q').value;const sc=route(p);const box=document.getElementById('rout');if(!p.trim()){box.innerHTML='';return;}if(sc[0][0]===0){box.innerHTML='<div class="hit">เดาไม่ออกจากคำนี้ ลองดูตารางด้านล่าง (input เป็นอะไร -> output เป็นอะไร)</div>';return;}const best=CATS[sc[0][1]];let topic=sc[0][1].split('_')[0];const map={cv:'cv',nlp:'nlp',mm:'mm',tab:'tab',ts:'ts'};let h='<div class="hit">📂 เปิด: <b>'+best.path+'</b><br>หมวด: '+best.label+'<br>เริ่มจาก: '+best.first+'<br><button class="chip" style="margin-top:8px" onclick="gotoSec(\\''+(map[topic]||'router')+'\\')">ไปที่หัวข้อนี้ →</button></div>';h+='<div class="scorebar">คะแนนแมตช์: ';h+=sc.filter(x=>x[0]>0).map(x=>CATS[x[1]].path.split('/').pop()+' ('+x[0]+')').join(' · ');h+='</div>';box.innerHTML=h;}
const EX=["ทำนายระยะการนอนจาก EEG","classify movie reviews sentiment","ตรวจจับวัตถุในรูป","บรรยายรูปเป็นภาษาไทย","predict house price","ตัดคำภาษาไทย","forecast next week sales","อ่านตัวอักษรในรูป OCR"];
window.addEventListener('DOMContentLoaded',function(){
 const ch=document.getElementById('chips');if(ch){EX.forEach(function(e){const b=document.createElement('span');b.className='chip';b.textContent=e;b.onclick=function(){document.getElementById('q').value=e;doRoute();};ch.appendChild(b);});}
 const links=document.querySelectorAll('.navlink');const secs=document.querySelectorAll('.section');
 const sb=document.getElementById('sidebar');const scr=document.getElementById('scrim');
 function show(id){secs.forEach(s=>s.classList.toggle('active',s.id===id));links.forEach(l=>l.classList.toggle('active',l.dataset.t===id));sb.classList.remove('open');if(scr)scr.classList.remove('open');window.scrollTo(0,0);}
 links.forEach(l=>l.addEventListener('click',e=>{show(l.dataset.t);}));
 const init=(location.hash||'#start').slice(1);show(document.getElementById(init)?init:'start');
 document.getElementById('burger').onclick=()=>{sb.classList.toggle('open');if(scr)scr.classList.toggle('open');};
 if(scr)scr.onclick=()=>{sb.classList.remove('open');scr.classList.remove('open');};
});
function cp(b){const c=b.parentElement.querySelector('code');navigator.clipboard.writeText(c.innerText).then(()=>{const o=b.textContent;b.textContent='copied!';setTimeout(()=>b.textContent=o,1200);});}
"""

JS = JS.replace("__CATS__", json.dumps(CATS, ensure_ascii=False))

HTML = (
    "<!DOCTYPE html><html lang=\"th\"><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<meta name=\"theme-color\" content=\"#0f1419\">"
    "<title>5 Domains Hackathon — Pangpuriye | คู่มือสอบ</title>"
    "<style>" + CSS + "</style></head><body>"
    "<header><button id=\"burger\" aria-label=\"menu\">☰</button>"
    "<div><h1>5 Domains Hackathon · Pangpuriye</h1>"
    "<div class=\"sub\">คู่มือสอบ AI 5 หัวข้อใหญ่ · เปิดที่ไหนก็ได้ ทั้งคอมและมือถือ</div></div></header>"
    "<div class=\"wrap\">"
    "<nav id=\"sidebar\">" + "".join(nav) + "</nav><div class=\"scrim\" id=\"scrim\"></div>"
    "<main>" + "".join(secs) +
    "<div class=\"foot\">สร้างโดยทีม Pangpuriye · ข้อมูล SOTA อัปเดต 2025–2026 · เปิด Router แล้วพิมพ์โจทย์เพื่อหาโน้ตบุ๊ก</div>"
    "</main></div>"
    "<script>" + JS + "</script></body></html>"
)

open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(HTML)
print("wrote index.html", len(HTML), "bytes |", len(SECTIONS), "sections")

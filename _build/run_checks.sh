#!/usr/bin/env bash
# รันการตรวจทั้งหมด: regenerate notebooks -> router test -> logic test -> syntax check
set -e
cd "$(dirname "$0")/.."
echo "==== regenerate 13 notebooks ===="
for g in gen_cv gen_nlp gen_mm gen_tab gen_ts gen_meta; do python3 "_build/$g.py"; done
echo ""
echo "==== router test (Thai+English) ===="
python3 _build/verify_router.py | tail -2
echo ""
echo "==== runnable-logic test ===="
python3 _build/verify_logic.py | tail -2
echo ""
echo "==== syntax check all code cells ===="
python3 - <<'PY'
import json, glob
def strip(s): return "\n".join("pass  # magic" if l.lstrip().startswith(("!","%")) else l for l in s.split("\n"))
nbs=sorted(glob.glob("[0-9]*/*.ipynb"))+sorted(glob.glob("00_*.ipynb")); bad=0
for nb in nbs:
    d=json.load(open(nb,encoding="utf-8"))
    for i,c in enumerate(d["cells"]):
        if c["cell_type"]!="code": continue
        try: compile(strip("".join(c["source"])),f"{nb}#c{i}","exec")
        except SyntaxError as e: bad+=1; print("SYNTAX",nb,"c",i,e.msg)
print(f"notebooks={len(nbs)} syntax_problems={bad}")
PY
echo ""
echo "==== end-to-end exam simulation ===="
python3 _build/simulate.py 2>&1 | grep -E "^\[(PASS|FAIL|ERROR)|สรุป"
echo ""
echo "ALL CHECKS DONE"

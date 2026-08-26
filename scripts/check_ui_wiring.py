"""G5: the page really exposes the review queue and the correction control."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gate_common import isolated_app, seed
d, database, web, c = isolated_app()
sid = seed(database, files=3)
html = c.get("/library").get_data(as_text=True)
need = [
    ("review panel", "Needs review"),
    ("find button", "Find likely mistakes"),
    ("review loader", "async function loadReview"),
    ("correction handler", "async function fixLabel"),
    ("category dropdown builder", "function categoryOptions"),
    ("review endpoint call", "/review"),
    ("relabel endpoint call", "/relabel"),
    ("categories exposed to js", "LIBRARY_CATEGORIES"),
]
missing = [name for name, tok in need if tok not in html]
if missing:
    print("missing from rendered page:", missing); sys.exit(1)
# the control must be a real select bound to fixLabel, not inert markup
if not re.search(r"onchange=\"fixLabel\(", html):
    print("correction dropdown is not wired to fixLabel"); sys.exit(1)
print(f"all {len(need)} review/correction elements present and wired")
print("UI_WIRING_OK")

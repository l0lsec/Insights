"""G3: a correction is written to every file of the event and reads back."""
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gate_common import isolated_app, seed
d, database, web, c = isolated_app()
sid = seed(database, files=6, category="MMA", conf=0.5)
r = c.post(f"/library/scan/{sid}/relabel",
           json={"event_key": "2020/01/01#0", "category": "Beach Trips"})
if r.status_code != 200:
    print("relabel failed:", r.status_code, r.data[:120]); sys.exit(1)
n = r.get_json().get("files_updated", 0)
if n != 6:
    print(f"expected 6 files relabelled, got {n}"); sys.exit(1)
with sqlite3.connect(database.DB_PATH) as conn:
    cats = [x[0] for x in conn.execute(
        "SELECT DISTINCT category FROM library_files WHERE scan_id=?", (sid,))]
    pinned = conn.execute(
        "SELECT COUNT(*) FROM library_files WHERE scan_id=? AND pinned=1", (sid,)).fetchone()[0]
if cats != ["Beach Trips"]:
    print("category not persisted, got", cats); sys.exit(1)
if pinned != 6:
    print(f"expected 6 pinned, got {pinned}"); sys.exit(1)
# readable back through the browse API
files = c.get(f"/library/scan/{sid}/files?limit=10").get_json()["files"]
if any(f["category"] != "Beach Trips" for f in files):
    print("browse API still returns the old label"); sys.exit(1)
print(f"relabelled=6 pinned=6 readback=ok")
print("RELABEL_PERSIST_OK")

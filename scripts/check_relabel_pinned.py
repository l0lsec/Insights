"""G4: a reclassification pass cannot overwrite a hand correction."""
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gate_common import isolated_app, seed
d, database, web, c = isolated_app()
import content_library as cl
sid = seed(database, files=5, category="MMA", conf=0.5)
c.post(f"/library/scan/{sid}/relabel", json={"event_key": "2020/01/01#0", "category": "Documents"})

# 1) the engine must not offer a pinned event as pending, even on a retry pass
rows, _ = database.query_library_files(sid, limit=1000, db_path=database.DB_PATH)
files = [web._scanned_from_row(r) for r in rows]
ev = {}
for f in files:
    ev.setdefault(f.event_key, []).append(f)
pend = cl.pending_events(ev, retry_unresolved=True)
if pend:
    print("pinned event was offered for reclassification:", list(pend)); sys.exit(1)

# 2) even a direct label write must not clobber it
database.apply_event_labels(sid, "2020/01/01#0", "Screenshots", 0.9, "vision", "x",
                            db_path=database.DB_PATH)
with sqlite3.connect(database.DB_PATH) as conn:
    cats = [x[0] for x in conn.execute(
        "SELECT DISTINCT category FROM library_files WHERE scan_id=?", (sid,))]
if cats != ["Documents"]:
    print("pinned label was overwritten, got", cats); sys.exit(1)

# positive control: an UNPINNED event must still be overwritable, or the guard
# would be indistinguishable from a broken writer
sid2 = seed(database, files=3, category="MMA", conf=0.5)
database.apply_event_labels(sid2, "2020/01/01#0", "Screenshots", 0.9, "vision", "x",
                            db_path=database.DB_PATH)
with sqlite3.connect(database.DB_PATH) as conn:
    cats2 = [x[0] for x in conn.execute(
        "SELECT DISTINCT category FROM library_files WHERE scan_id=?", (sid2,))]
if cats2 != ["Screenshots"]:
    print("control failed: unpinned label did not update, got", cats2); sys.exit(1)
print("pinned survived retry+direct write; unpinned control still updates")
print("RELABEL_PINNED_OK")

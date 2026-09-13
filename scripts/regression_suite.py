"""Durable regression suite for the Content Library.

Lives in the repo rather than a scratch directory, because the previous copies
were wiped twice by scratchpad cleanup and took the only proof of these
behaviours with them.

Covers the behaviours earlier sessions established and that later work could
plausibly break: the scan -> classify -> plan -> approve -> apply -> undo
lifecycle in both copy and link mode, year scoping, resume, stats merging, and
the guarantee that undo never touches an original.
"""
from __future__ import annotations
import json, os, random, shutil, sqlite3, sys, tempfile, time

# The checkout this suite belongs to, found from the file rather than written
# down, so it runs wherever the repository is cloned.
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)

def build(tmp):
    from PIL import Image
    random.seed(3)
    for cat in ("MMA", "Cybersecurity"):
        os.makedirs(f"{tmp}/tax/{cat}", exist_ok=True)
        Image.new("RGB", (8, 8)).save(f"{tmp}/tax/{cat}/ex.jpg")
    n = 0
    for yr in (2019, 2023):
        for cat in ("MMA", "Cybersecurity"):
            d = f"{tmp}/src/{yr}/06/{cat}"
            os.makedirs(d, exist_ok=True)
            for i in range(3):
                im = Image.new("RGB", (48, 48), (random.randrange(256),) * 3)
                im.putpixel((i, i), (255, 0, 0))
                im.save(f"{d}/IMG_{n:04d}.jpg")
                n += 1
    return n

def wait(database, web, sid, states, limit=90):
    """Block until the scan's background job has actually finished.

    Waiting on the status alone is not enough, and was the reason this suite
    failed about half its runs. Apply, classify and thumbnails all run in
    threads, and a scan keeps the previous job's status until its next job
    overwrites it, so a wait for "applied" placed straight after a second apply
    returned at once on the *first* apply's status. The suite then walked a
    destination that was still being written, and read "nothing placed" or
    "undo left N behind" depending on how far the copy had got.

    The job flag is the real signal: it is claimed in the request, so it is
    already true by the time the POST returns. A terminal status with the flag
    down is the one state that means this job, not a previous one, is done.
    """
    deadline = time.time() + limit * 0.5
    while time.time() < deadline:
        s = database.get_library_scan(sid, db_path=database.DB_PATH)
        if s["status"] in states and not web._library_job_running(sid):
            return s
        time.sleep(0.1)
    s = database.get_library_scan(sid, db_path=database.DB_PATH)
    raise AssertionError(
        f"job did not reach {sorted(states)} within {limit * 0.5:.0f}s "
        f"(status={s['status']}, running={web._library_job_running(sid)})")

def main() -> int:
    tmp = tempfile.mkdtemp(prefix="lib_regress_")
    os.chdir(tmp)
    import database
    database.DB_PATH = os.path.join(tmp, "insights.db")
    database.init_db(database.DB_PATH)
    import insights_web as web
    from werkzeug.security import generate_password_hash
    uid = database.create_user("r", generate_password_hash("p"), db_path=database.DB_PATH)
    c = web.app.test_client()
    with c.session_transaction() as s:
        s["user_id"] = uid
    total = build(tmp)

    c.post("/library/roots", data={"path": f"{tmp}/tax", "role": "taxonomy", "label": "tax"})
    c.post("/library/roots", data={"path": f"{tmp}/src", "role": "source", "label": "src"})
    roots = {r["label"]: r["id"] for r in database.list_library_roots(db_path=database.DB_PATH)}
    c.post("/library/taxonomy/learn", data={"root_id": roots["tax"]})
    c.post("/library/scan", data={"root_id": roots["src"]})
    sid = database.latest_library_scan(db_path=database.DB_PATH)["id"]
    s = wait(database, web, sid, ("scanned", "failed"))
    assert s["status"] == "scanned", f"scan failed: {s['error_message']}"
    assert s["files_total"] == total, f"scanned {s['files_total']} of {total}"

    # year scoping narrows the estimate
    a = c.get(f"/library/scan/{sid}/estimate?calibrate=0").get_json()
    b = c.get(f"/library/scan/{sid}/estimate?calibrate=0&year_from=2019&year_to=2019").get_json()
    assert b["events_total"] < a["events_total"], "year filter did not narrow the estimate"

    # classify only 2019
    c.post(f"/library/scan/{sid}/classify", json={"budget_gb": 1, "year_from": 2019, "year_to": 2019})
    wait(database, web, sid, ("classified", "failed"))
    by = {}
    with sqlite3.connect(database.DB_PATH) as conn:
        for y, cat in conn.execute("SELECT year, category FROM library_files WHERE scan_id=?", (sid,)):
            by.setdefault(y, set()).add(cat or "Unsorted")
    assert by.get(2023) == {"Unsorted"}, f"2023 should be untouched, got {by.get(2023)}"

    # resume is a no-op once nothing is pending
    r = c.post(f"/library/scan/{sid}/classify", json={"budget_gb": 1, "resume": "1"})
    assert r.status_code == 200, f"resume refused: {r.get_json()}"
    s = wait(database, web, sid, ("classified", "scanned", "failed"))
    assert s["status"] in ("classified", "scanned"), \
        f"resume left the scan in a bad state: {s['status']}"

    # stats from different jobs must coexist
    r = c.post(f"/library/scan/{sid}/thumbnails", json={"fetch_gb": 0})
    assert r.status_code == 200, f"thumbnails refused: {r.get_json()}"
    s = wait(database, web, sid, ("classified", "scanned", "applied", "failed"))
    st = json.loads(s["stats"] or "{}")
    assert "classify" in st and "thumbnails" in st, f"stats clobbered: {sorted(st)}"

    results = {}
    for mode in ("copy", "link"):
        dest = os.path.join(tmp, f"out_{mode}")
        c.post(f"/library/scan/{sid}/plan", json={"min_confidence": 0.4})
        c.post(f"/library/scan/{sid}/plan/approve", json={"state": "approved"})
        r = c.post(f"/library/scan/{sid}/apply", json={"dest_root": dest, "mode": mode}).get_json()
        assert r.get("ok"), f"{mode} apply refused: {r}"
        s = wait(database, web, sid, ("applied", "failed"))
        assert s["status"] == "applied", f"{mode} apply failed: {s['error_message']}"
        placed = [os.path.join(dp, f) for dp, _, fs in os.walk(dest)
                  for f in fs if not f.startswith(".")]
        assert placed, f"{mode}: nothing placed"
        if mode == "link":
            assert all(os.path.islink(p) for p in placed), "link mode produced real files"
        else:
            assert not any(os.path.islink(p) for p in placed), "copy mode produced links"
        # re-apply must be idempotent
        c.post(f"/library/scan/{sid}/plan/approve", json={"state": "approved"})
        again_r = c.post(f"/library/scan/{sid}/apply",
                         json={"dest_root": dest, "mode": mode}).get_json()
        assert again_r.get("ok"), f"{mode} re-apply refused: {again_r}"
        s = wait(database, web, sid, ("applied", "failed"))
        assert s["status"] == "applied", f"{mode} re-apply failed: {s['error_message']}"
        again = [os.path.join(dp, f) for dp, _, fs in os.walk(dest)
                 for f in fs if not f.startswith(".")]
        assert len(again) == len(placed), f"{mode} apply not idempotent: {len(placed)} -> {len(again)}"
        # undo removes only what it placed
        before_src = sum(1 for _, _, fs in os.walk(f"{tmp}/src") for _ in fs)
        u = c.post(f"/library/scan/{sid}/undo",
                   json={"manifest": os.path.join(dest, ".insights-library-manifest.jsonl")}).get_json()
        after_src = sum(1 for _, _, fs in os.walk(f"{tmp}/src") for _ in fs)
        assert after_src == before_src, f"{mode} undo deleted originals: {before_src} -> {after_src}"
        left = [f for _, _, fs in os.walk(dest) for f in fs if not f.startswith(".")]
        assert not left, f"{mode} undo left {len(left)} behind"
        results[mode] = (len(placed), u["removed"])

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"lifecycle ok | copy placed/removed {results['copy']} | link placed/removed {results['link']}")
    print("year scoping, resume no-op, stats merge, idempotent apply, safe undo: all ok")
    print("REGRESSIONS_OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

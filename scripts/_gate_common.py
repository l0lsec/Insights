import os, sys, sqlite3, tempfile, shutil
W = "/Users/sedriclouissaint/tools/Insights/.claude/worktrees/instagram-preview-aspect-ratios-9a6cd8"
sys.path.insert(0, W)

def isolated_app():
    """A test client on a throwaway DB seeded with one classified event."""
    d = tempfile.mkdtemp(prefix="gate_")
    os.chdir(d)
    import database
    database.DB_PATH = os.path.join(d, "insights.db")
    database.init_db(database.DB_PATH)
    import insights_web as web
    web.database.DB_PATH = database.DB_PATH
    from werkzeug.security import generate_password_hash
    uid = database.create_user("g", generate_password_hash("p"), db_path=database.DB_PATH)
    c = web.app.test_client()
    with c.session_transaction() as s:
        s["user_id"] = uid
    return d, database, web, c

def seed(database, files=4, category="MMA", caption="two fighters in a cage", conf=0.55):
    rid = database.add_library_root("/tmp/gate_src", "src", "source", db_path=database.DB_PATH)
    sid = database.create_library_scan(rid, db_path=database.DB_PATH)
    rows = [{"path": f"/tmp/gate_src/2020/01/01/f{i}.jpg", "rel_path": f"2020/01/01/f{i}.jpg",
             "name": f"f{i}.jpg", "ext": "jpg", "kind": "image", "size": 1000, "mtime": 0,
             "materialized": 1, "captured_at": None, "date_source": "path", "year": 2020,
             "month": 1, "event_key": "2020/01/01#0", "dup_group": 1,
             "category": category, "confidence": conf, "classified_by": "vision"}
            for i in range(files)]
    database.bulk_insert_library_files(sid, rows, db_path=database.DB_PATH)
    database.upsert_library_event(sid, {
        "event_key": "2020/01/01#0", "directory": "2020/01/01", "file_count": files,
        "total_bytes": files * 1000, "year": 2020, "date_start": "2020-01-01",
        "category": category, "confidence": conf, "classified_by": "vision",
        "reason": "test", "captions": [caption]}, db_path=database.DB_PATH)
    database.save_library_categories(
        [{"name": n, "subcategories": [], "keywords": [], "example_count": 5}
         for n in ("MMA", "Screenshots", "Beach Trips", "Documents")],
        db_path=database.DB_PATH)
    return sid

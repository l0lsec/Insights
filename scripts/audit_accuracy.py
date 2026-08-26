"""Measure how often the assigned category is unsupported by the evidence.

Judges each sampled event on the caption the classifier actually saw. That
measures mapping fidelity -- whether the label follows from the evidence -- and
deliberately not whether the caption itself described the footage. A frame that
missed the subject produces a truthful caption and a wrong label, and that case
is reported separately as weak evidence rather than being scored as a mapping
error.
"""
from __future__ import annotations
import json, os, random, sqlite3, sys
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/sedriclouissaint/tools/Insights")
from dotenv import load_dotenv
load_dotenv("/Users/sedriclouissaint/tools/Insights/.env")
import database, content_library as cl

SAMPLE = int(os.environ.get("AUDIT_SAMPLE", "120"))
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_report.json")

JUDGE = """You are auditing an automatic media classifier.

Category assigned: {cat}
Evidence the classifier saw (image caption): {cap}

Answer ONLY with JSON:
{{"verdict": "supported" | "unsupported" | "weak_evidence", "why": "<8 words>"}}

- "supported": the caption plainly fits the category.
- "unsupported": the caption clearly indicates a DIFFERENT subject. This is a
  mapping error.
- "weak_evidence": the caption is blank, generic, or describes nothing
  identifiable, so no category could be justified from it either way."""

def main() -> int:
    sid = database.latest_library_scan()["id"]
    with sqlite3.connect(database.DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""SELECT event_key, category, captions, file_count, classified_by
                            FROM library_events WHERE scan_id=?
                            AND category IS NOT NULL AND category NOT IN ('','Unsorted')
                            AND captions IS NOT NULL AND captions NOT IN ('','[]')""",
                         (sid,)).fetchall()
    pool = []
    for r in rows:
        try:
            caps = json.loads(r["captions"] or "[]")
        except (TypeError, ValueError):
            continue
        if caps and caps[0].strip():
            pool.append((r["event_key"], r["category"], caps[0].strip(), r["file_count"]))
    random.seed(7)
    sample = random.sample(pool, min(SAMPLE, len(pool)))
    print(f"judging {len(sample)} of {len(pool)} captioned events", flush=True)

    verdicts, per_cat, bad = Counter(), defaultdict(Counter), []
    for i, (key, cat, cap, n) in enumerate(sample, 1):
        try:
            raw = cl._cloud_map(JUDGE.format(cat=cat, cap=cap[:600]))
            d = cl._extract_json(raw) or {}
        except Exception as e:
            print("  judge error:", str(e)[:70], flush=True)
            continue
        v = str(d.get("verdict", "")).strip()
        if v not in ("supported", "unsupported", "weak_evidence"):
            continue
        verdicts[v] += 1
        per_cat[cat][v] += 1
        if v == "unsupported":
            bad.append({"event": key, "category": cat, "files": n,
                        "caption": cap[:160], "why": d.get("why", "")})
        if i % 25 == 0:
            print(f"  {i}/{len(sample)} judged", flush=True)

    total = sum(verdicts.values())
    judged_cats = {k: dict(v) for k, v in per_cat.items()}
    worst = sorted(((c, v.get("unsupported", 0) / max(sum(v.values()), 1), sum(v.values()))
                    for c, v in per_cat.items() if sum(v.values()) >= 3),
                   key=lambda x: -x[1])[:8]
    report = {
        "sampled": total,
        "pool": len(pool),
        "supported": verdicts["supported"],
        "unsupported": verdicts["unsupported"],
        "weak_evidence": verdicts["weak_evidence"],
        "mapping_error_rate": verdicts["unsupported"] / total if total else 0,
        "weak_evidence_rate": verdicts["weak_evidence"] / total if total else 0,
        "worst_categories": [{"category": c, "rate": round(r, 3), "n": n} for c, r, n in worst],
        "examples": bad[:20],
        "per_category": judged_cats,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nsampled {total} | supported {verdicts['supported']} | "
          f"unsupported {verdicts['unsupported']} | weak {verdicts['weak_evidence']}")
    print(f"mapping error rate: {report['mapping_error_rate']:.1%} | "
          f"weak evidence: {report['weak_evidence_rate']:.1%}")
    print("worst categories:", [(c["category"], c["rate"]) for c in report["worst_categories"][:5]])
    print(f"report -> {REPORT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

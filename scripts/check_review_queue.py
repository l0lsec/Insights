"""G2: the review queue ranks a contradicted label above a consistent one."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gate_common import isolated_app, seed
d, database, web, c = isolated_app()
# caption plainly contradicts the category -> must be flagged and ranked high
sid = seed(database, files=12, category="MMA",
           caption="a screenshot of a login page with email and password fields", conf=0.55)
r = c.get(f"/library/scan/{sid}/review")
if r.status_code != 200:
    print("review endpoint failed:", r.status_code); sys.exit(1)
d1 = r.get_json()
if not d1["events"]:
    print("queue empty for an obviously contradicted label"); sys.exit(1)
top = d1["events"][0]
if top["category"] != "MMA":
    print("unexpected top event"); sys.exit(1)
if "caption suggests" not in top["why"] and "low confidence" not in top["why"]:
    print("no explanation for the flag:", top["why"]); sys.exit(1)
# positive control: a well-supported, confident, small event must NOT outrank it
sid2 = seed(database, files=1, category="Screenshots",
            caption="a screenshot of a login page with email and password fields", conf=0.95)
r2 = c.get(f"/library/scan/{sid2}/review").get_json()
flagged2 = [e for e in r2["events"] if e["confidence"] >= 0.9]
if flagged2 and flagged2[0]["score"] >= top["score"]:
    print("consistent confident label scored as suspect as a contradicted one"); sys.exit(1)
print(f"flagged={d1['total_flagged']} top_score={top['score']} why={top['why']!r}")
print("REVIEW_QUEUE_OK")

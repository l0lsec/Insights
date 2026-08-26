"""G1: the accuracy report exists, came from a real sample, and is self-consistent."""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_report.json")
if not os.path.exists(R):
    print("no accuracy_report.json -- audit never ran"); sys.exit(1)
r = json.load(open(R))
n = r.get("sampled", 0)
parts = r.get("supported", 0) + r.get("unsupported", 0) + r.get("weak_evidence", 0)
if n < 100:
    print(f"sample too small to be meaningful: {n}"); sys.exit(1)
if parts != n:
    print(f"verdict counts {parts} do not sum to sampled {n}"); sys.exit(1)
rate = r.get("mapping_error_rate", -1)
if not (0.0 <= rate <= 1.0):
    print(f"implausible rate {rate}"); sys.exit(1)
# recompute rather than trust the stored figure
recomputed = r["unsupported"] / n
if abs(recomputed - rate) > 1e-6:
    print(f"stored rate {rate} != recomputed {recomputed}"); sys.exit(1)
if not r.get("examples"):
    print("no concrete examples captured"); sys.exit(1)
if r.get("pool", 0) < n:
    print("pool smaller than sample"); sys.exit(1)
print(f"sampled={n} pool={r['pool']} mapping_error={rate:.1%} weak={r['weak_evidence_rate']:.1%} "
      f"examples={len(r['examples'])}")
print("ACCURACY_AUDIT_OK")

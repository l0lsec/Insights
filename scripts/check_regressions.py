"""G6: the durable regression suite passes."""
import subprocess, sys, os
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_suite.py")
# Whatever interpreter is running this gate is the one that can import the app.
r = subprocess.run([sys.executable, S], capture_output=True, text=True, timeout=1800)
out = r.stdout + r.stderr
if r.returncode != 0 or "REGRESSIONS_OK" not in r.stdout:
    tail = [l for l in out.splitlines() if l.strip() and not l.startswith(("WARNING", "INFO"))]
    print("regression suite FAILED:"); print("\n".join(tail[-12:])); sys.exit(1)
for line in r.stdout.splitlines():
    if line.strip() and not line.startswith(("WARNING", "INFO")):
        print(line)

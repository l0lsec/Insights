"""The completion gate for multi-account posting and cross-platform publishing.

Run this and the feature is either done or it is not. Each gate below is an
independent claim about the feature, and each prints its own OK token only
after proving it against the real routes and the real database schema. The
runner fails if any gate fails, if a gate crashes, or if a gate exits happily
without printing its token, which is what stops a silently skipped check from
reading as a pass.

    python scripts/run_account_gates.py

Every gate runs in its own throwaway database with fake platform clients, so a
run makes no network calls, needs no credentials, and cannot touch a real
insights.db.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Each gate: the file, the token it must print, and the claim it makes.
GATES = [
    ("check_accounts_model.py", "ACCOUNTS_MODEL_OK",
     "a platform holds several accounts, with defaults that move correctly"),
    ("check_token_isolation.py", "TOKEN_ISOLATION_OK",
     "each account keeps its own credentials on all five platforms"),
    ("check_crosspost.py", "CROSSPOST_OK",
     "one post reaches every platform and account it names, honestly reported"),
    ("check_scheduler_accounts.py", "SCHEDULER_ACCOUNTS_OK",
     "queued posts publish as the account they were queued for"),
    ("check_accounts_ui.py", "ACCOUNTS_UI_OK",
     "the screens expose accounts and cross-posting, and are wired to them"),
    ("check_backcompat.py", "BACKCOMPAT_OK",
     "an existing single-account install migrates and keeps working"),
]

TIMEOUT_SECONDS = 300


def run_gate(filename, token):
    """Run one gate; returns (passed, summary line, failure detail)."""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, filename)],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"timed out after {TIMEOUT_SECONDS}s"

    output = f"{result.stdout}\n{result.stderr}"
    lines = [
        line for line in output.splitlines()
        if line.strip() and not line.startswith(("WARNING", "INFO", "FLASK_SECRET_KEY"))
    ]
    summary = next((line for line in lines if line and line != token), "")

    if result.returncode != 0:
        return False, summary, "\n".join(lines[-12:])
    if token not in result.stdout:
        # A gate that exits 0 without its token did not finish its checks.
        return False, summary, f"did not print {token}\n" + "\n".join(lines[-12:])
    return True, summary, ""


def main():
    print("Multi-account and cross-platform posting: completion gates\n")
    failures = []
    for filename, token, claim in GATES:
        started = time.time()
        passed, summary, detail = run_gate(filename, token)
        elapsed = time.time() - started
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {claim}  ({elapsed:.1f}s)")
        if summary:
            print(f"       {summary}")
        if not passed:
            failures.append((filename, detail))
            print(f"       {detail}".replace("\n", "\n       "))
        print()

    if failures:
        print(f"{len(failures)} of {len(GATES)} gates failed: "
              + ", ".join(name for name, _ in failures))
        return 1

    print(f"All {len(GATES)} gates passed.")
    print("ALL_ACCOUNT_GATES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

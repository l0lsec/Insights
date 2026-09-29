"""Gate: video posting must not change anything that already worked.

Runs the suites that cover the code video posting touches (the shared publisher,
per-account routing, the scheduler, cross-posting, bulk Add Platform and the
Compose page's script) and passes only if every one of them prints its own
success token. A suite that exits 0 without printing its token counts as a
failure, which is what stops a silently skipped check from reading as a pass.

    python scripts/run_video_regressions.py

Each suite runs on its own throwaway database with fake platform clients, so
this makes no network calls and cannot touch a real insights.db or account.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, args, token it must print)
SUITES = [
    ("check_accounts_model.py", [], "ACCOUNTS_MODEL_OK"),
    ("check_token_isolation.py", [], "TOKEN_ISOLATION_OK"),
    ("check_crosspost.py", [], "CROSSPOST_OK"),
    ("check_scheduler_accounts.py", [], "SCHEDULER_ACCOUNTS_OK"),
    ("check_accounts_ui.py", [], "ACCOUNTS_UI_OK"),
    ("check_backcompat.py", [], "BACKCOMPAT_OK"),
    ("check_bulk_platform.py", ["guard"], "SINGLE_CARD_GUARD_OK"),
    ("check_bulk_platform.py", ["add"], "BULK_ADD_OK"),
    ("check_bulk_platform.py", ["queue"], "BULK_QUEUE_OK"),
    ("check_bulk_platform.py", ["select"], "BULK_SELECT_OK"),
    ("check_bulk_platform_ui.py", ["wiring"], "BULK_PLATFORM_UI_OK"),
    ("check_bulk_platform_ui.py", ["syntax"], "COMPOSE_JS_SYNTAX_OK"),
    ("check_compose_js.py", ["duplicates"], "COMPOSE_JS_NO_DUPES_OK"),
    ("check_compose_js.py", ["refreshcard"], "REFRESH_CARD_OK"),
]

TIMEOUT_SECONDS = 300


def main():
    failures = []
    for script, args, token in SUITES:
        label = " ".join([script] + args)
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(HERE, script)] + args,
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=HERE,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{label}: timed out after {TIMEOUT_SECONDS}s")
            continue
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            tail = " | ".join(l for l in output.strip().splitlines()[-3:])
            failures.append(f"{label}: exit {result.returncode}: {tail}")
        elif token not in output:
            failures.append(f"{label}: exited 0 without printing {token}")
        else:
            print(f"  ok  {label}")

    if failures:
        print(f"{len(failures)} of {len(SUITES)} suites failed:")
        for line in failures:
            print(f"  FAIL {line}")
        sys.exit(1)
    print(f"{len(SUITES)} suites passed")
    print("VIDEO_REGRESSIONS_OK")


if __name__ == "__main__":
    main()

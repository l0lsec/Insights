"""Run every completion gate in the repository, and fail if one is unaccounted for.

This is the single entry point for both a local run and CI, so the two cannot
drift apart:

    python scripts/run_all_gates.py

It runs the multi-account gates (defined once, in run_account_gates.py) and the
Content Library gates, each in its own process. A gate passes only if it exits 0
AND prints its own OK token, which is what stops a gate that quietly did nothing
from reading as a pass.

It also checks the gates themselves. Every scripts/check_*.py must be registered
here or listed in NOT_GATES with a reason. A check that exists but is not
registered is worse than no check, because it looks like coverage, so adding one
without wiring it in fails the run instead of being silently skipped.
"""

import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_account_gates import GATES as ACCOUNT_GATES, run_gate  # noqa: E402

# Each gate: the file, the token it must print, the claim it makes, and
# optionally the arguments that select one section of a script holding several.
# The multi-account gates in run_account_gates.py are plain three-item entries.
LIBRARY_GATES = [
    ("check_regressions.py", "REGRESSIONS_OK",
     "the Content Library lifecycle (scan, classify, plan, apply, undo) holds"),
    ("check_relabel_persist.py", "RELABEL_PERSIST_OK",
     "a relabel is written through and reads back"),
    ("check_relabel_pinned.py", "RELABEL_PINNED_OK",
     "a hand-set label survives later classification"),
    ("check_review_queue.py", "REVIEW_QUEUE_OK",
     "the review queue flags the labels most likely to be wrong"),
    ("check_ui_wiring.py", "UI_WIRING_OK",
     "the Library page exposes its review and correction controls"),
    ("check_accuracy_audit.py", "ACCURACY_AUDIT_OK",
     "the committed accuracy report is a real, self-consistent sample"),
]

# The Compose page's gates. Several scripts hold more than one gate and pick one
# by argument, each printing its own token, so one file appears once per section.
# Two of them run the page's JavaScript through node, so CI installs it.
COMPOSE_GATES = [
    ("check_bulk_platform.py", "SINGLE_CARD_GUARD_OK",
     "ticking one platform and queueing a whole card behave as pinned",
     ("guard",)),
    ("check_bulk_platform.py", "BULK_ADD_OK",
     "bulk Add Platform keeps each card's copy, image and brief intact",
     ("add",)),
    ("check_bulk_platform.py", "BULK_QUEUE_OK",
     "bulk add and queue give each row its own slot and leave the rest alone",
     ("queue",)),
    ("check_bulk_platform.py", "BULK_SELECT_OK",
     "bulk add resolves cards the way the page does and refuses bad requests",
     ("select",)),
    ("check_bulk_platform.py", "BULK_REMOVE_OK",
     "bulk Remove Platform keeps queued, published and only-platform rows unless told otherwise",
     ("remove",)),
    ("check_bulk_platform.py", "BULK_REMOVE_SELECT_OK",
     "bulk remove resolves cards the way the page does and refuses bad requests",
     ("removeselect",)),
    ("check_bulk_platform_ui.py", "BULK_PLATFORM_UI_OK",
     "the Compose page exposes Add and Remove Platform and binds them to their endpoints",
     ("wiring",)),
    ("check_bulk_platform_ui.py", "COMPOSE_JS_SYNTAX_OK",
     "every inline script on the Compose page parses",
     ("syntax",)),
    ("check_compose_js.py", "COMPOSE_JS_NO_DUPES_OK",
     "no function on the Compose page is declared twice",
     ("duplicates",)),
    ("check_compose_js.py", "REFRESH_CARD_OK",
     "refreshCard redraws, drops an emptied card, and runs under node",
     ("refreshcard",)),
    ("check_compose_js.py", "BULK_TOTALS_OK",
     "the bulk result's batch totals sum correctly in both directions, under node",
     ("totals",)),
]

# check_*.py files that are deliberately not gates, with the reason. Empty is the
# healthy state: a script that is not a gate should not carry the check_ prefix.
NOT_GATES = {}

ALL_GATES = ACCOUNT_GATES + LIBRARY_GATES + COMPOSE_GATES


def _unpack(entry):
    """(file, token, claim, args) for a gate entry, with or without arguments."""
    filename, token, claim, *rest = entry
    return filename, token, claim, (rest[0] if rest else ())


def gate_inventory_problems():
    """Gates on disk that are not registered, and registered gates that are gone."""
    registered = {_unpack(entry)[0] for entry in ALL_GATES}
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(HERE, "check_*.py"))}
    unregistered = sorted(on_disk - registered - set(NOT_GATES))
    missing = sorted(registered - on_disk)
    return unregistered, missing


def main():
    print("Completion gates\n")

    unregistered, missing = gate_inventory_problems()
    problems = []
    for name in unregistered:
        problems.append(f"{name} exists but is not registered in run_all_gates.py, "
                        f"so it would never run")
    for name in missing:
        problems.append(f"{name} is registered but the file does not exist")

    failures = []
    for entry in ALL_GATES:
        filename, token, claim, args = _unpack(entry)
        if filename in missing:
            continue
        started = time.time()
        passed, summary, detail = run_gate(filename, token, args)
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {claim}  ({time.time() - started:.1f}s)")
        if summary:
            print(f"       {summary}")
        if not passed:
            failures.append(" ".join((filename, *args)))
            print(f"       {detail}".replace("\n", "\n       "))
        print()

    if problems:
        print("Gate inventory:")
        for problem in problems:
            print(f"  - {problem}")
        print()

    if failures or problems:
        if failures:
            print(f"{len(failures)} of {len(ALL_GATES)} gates failed: {', '.join(failures)}")
        if problems:
            print(f"{len(problems)} gate inventory problem(s)")
        return 1

    print(f"All {len(ALL_GATES)} gates passed.")
    print("ALL_GATES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

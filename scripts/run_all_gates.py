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

# Video posting. One script holds a gate per claim and picks one by argument,
# each printing its own token. They use fake platform clients and a real local
# HTTP server, so nothing posts anywhere. Two of them need ffmpeg/ffprobe to
# make and read a real MP4, so CI installs it.
VIDEO_GATES = [
    ("check_video.py", "VIDEO_SCHEMA_OK",
     "a video persists on a saved post, upgrades an old database, and groups cards",
     ("schema",)),
    ("check_video.py", "VIDEO_ROUTER_OK",
     "the shared publisher hands a video to each platform's video method",
     ("router",)),
    ("check_video.py", "VIDEO_LINKEDIN_OK",
     "LinkedIn gets the exact bytes in ordered parts and never degrades to a text post",
     ("linkedin",)),
    ("check_video.py", "VIDEO_TWITTER_OK",
     "X gets the exact bytes in segments and asks for a reconnect without media.write",
     ("twitter",)),
    ("check_video.py", "VIDEO_THREADS_OK",
     "Threads waits for FINISHED before publishing a video; the image path is unchanged",
     ("threads",)),
    ("check_video.py", "VIDEO_FACEBOOK_OK",
     "Facebook posts a video by file_url and a video beats an image",
     ("facebook",)),
    ("check_video.py", "VIDEO_INSTAGRAM_OK",
     "an Instagram card with a video publishes as a Reel and needs no image",
     ("instagram",)),
    ("check_video.py", "VIDEO_ROUTES_OK",
     "the video route attaches, replaces and clears across a whole card and refuses unsafe URLs",
     ("routes",)),
    ("check_video.py", "VIDEO_FANOUT_OK",
     "post-now, post-to-all and the schedule queue each deliver the video",
     ("fanout",)),
    ("check_video.py", "VIDEO_COPY_OK",
     "ticking a platform and bulk Add Platform copy the video and keep one card",
     ("copy",)),
    ("check_video.py", "VIDEO_COMPAT_OK",
     "a real MP4 is probed and each platform's limits hold at the boundary",
     ("compat",)),
    ("check_video.py", "VIDEO_DOWNLOAD_OK",
     "the downloader streams within its cap and refuses unsafe URLs at every hop",
     ("download",)),
    ("check_video.py", "VIDEO_UI_OK",
     "the Compose card and schedule page show video, and the page functions run",
     ("ui",)),
    ("check_video.py", "VIDEO_NEWPOST_OK",
     "a new post takes a video, refuses an unusable one before creating anything",
     ("newpost",)),
    ("check_video.py", "VIDEO_REELOPTS_OK",
     "a Reel's cover frame and collaborators are stored, validated and sent when it publishes",
     ("reelopts",)),
]

# Writing a post by hand, and the model adapters behind generation. One script,
# one gate per claim, picked by argument. They use fake clients and no keys, and
# the wiring gates read the rendered Compose page, one under node.
MANUAL_POST_GATES = [
    ("check_manual_posts.py", "MANUAL_POSTS_ROUTE_OK",
     "a hand-written post writes one row per ticked account, shows as one card, and refusals write nothing",
     ("route",)),
    ("check_manual_posts.py", "MANUAL_POSTS_WIRING_OK",
     "the Write Manually tab is first and default, hides the model controls, and is bound to its endpoint",
     ("wiring",)),
    ("check_manual_posts.py", "MANUAL_POSTS_TARGETS_OK",
     "the composer builds the accounts and platforms the server expects, under node",
     ("targets",)),
    ("check_manual_posts.py", "MANUAL_POSTS_VIDEO_OK",
     "the Write Manually tab attaches a video (file or URL, never both), sends it instead of the image, and the route takes it",
     ("video",)),
    ("check_manual_posts.py", "LLM_ADAPTERS_OK",
     "GPT-6 and Claude 5 requests are translated, older models pass through, and prices resolve",
     ("llm",)),
]

# The accounts and cross-posting JavaScript, run under node rather than read. The
# markup gate in ACCOUNT_GATES only proves the controls exist; these prove what
# they send and what they tell the user, including on every failure path.
ACCOUNT_JS_GATES = [
    ("check_accounts_js.py", "COMPOSE_ACCOUNTS_JS_OK",
     "the Compose cross-posting controls send the right account and report each target honestly",
     ("compose",)),
    ("check_accounts_js.py", "ACCOUNTS_PAGE_JS_OK",
     "the accounts page's default, rename and disconnect controls behave, including on failure",
     ("accounts",)),
    ("check_accounts_js.py", "ACCOUNTS_JS_SYNTAX_OK",
     "every inline script on the accounts page parses",
     ("syntax",)),
]

# The schedule queue shows one row for a post that goes to several platforms.
# One script, one gate per claim, picked by argument. Fake clients, no keys; the
# JavaScript ones lift the page's own functions and run them under node.
QUEUE_GATES = [
    ("check_queue_groups.py", "QUEUE_GROUP_OK",
     "queue entries that are one post collapse; a different moment, copy, image, state or repost does not",
     ("group",)),
    ("check_queue_groups.py", "QUEUE_PAGE_OK",
     "the schedule page and its JSON draw one row per post, carry every member id, and count honestly",
     ("page",)),
    ("check_queue_groups.py", "QUEUE_ORDER_OK",
     "drag-reorder and move to top or bottom never split a post that goes to several platforms",
     ("order",)),
    ("check_queue_groups.py", "QUEUE_EDIT_OK",
     "editing a collapsed row's time moves every platform and nothing outside it",
     ("edit",)),
    ("check_queue_groups.py", "QUEUE_PARITY_OK",
     "the server-rendered queue rows and the script-redrawn rows are identical, under node",
     ("parity",)),
    ("check_queue_groups.py", "QUEUE_JS_OK",
     "a collapsed row's Post Now, Cancel, Retry, Delete, Edit and bulk actions reach every platform, under node",
     ("js",)),
    ("check_queue_groups.py", "QUEUE_SYNTAX_OK",
     "every inline script on the schedule page parses and none is declared twice",
     ("syntax",)),
]

# Loading the rest of the Compose list: the route's batch size, and the Load more,
# Load all and scroll-loading controls, the latter run under node against a fake
# DOM, a fake paging server and a fake IntersectionObserver.
PAGING_GATES = [
    ("check_compose_paging.py", "PAGING_ROUTE_OK",
     "the list route takes a batch size and paging covers each matching card exactly once",
     ("route",)),
    ("check_compose_paging.py", "PAGING_WIRING_OK",
     "the Compose list has Load more, Load all and the scroll switch, and the script redraws them identically",
     ("wiring",)),
    ("check_compose_paging.py", "PAGING_LOADALL_OK",
     "Load all pages through the filtered set, can be stopped, survives failure and stale filters, under node",
     ("loadall",)),
    ("check_compose_paging.py", "PAGING_SCROLL_OK",
     "scrolling loads the next page once at a time, is switchable, and pauses after a failure, under node",
     ("scroll",)),
]

# Instagram collaborators and Reel people-tags. One script, one gate per claim,
# picked by argument. The client ones replace `requests` with a recorder and read
# the HTTP the client would send; the rest drive the real routes, database and
# publisher with a recording client. Nothing posts anywhere. Two run the pages'
# JavaScript under node.
IG_PEOPLE_GATES = [
    ("check_ig_people.py", "IG_PEOPLE_CLIENT_OK",
     "collaborators and Reel tags reach Instagram's request exactly as entered, on a reel, an image and a carousel",
     ("client",)),
    ("check_ig_people.py", "IG_PEOPLE_NONE_OK",
     "a post with no collaborators or tags sends the same requests as before they existed",
     ("none",)),
    ("check_ig_people.py", "IG_PEOPLE_REFUSE_OK",
     "a bad handle or too many collaborators is refused before any request or row is made",
     ("refuse",)),
    ("check_ig_people.py", "IG_PEOPLE_REJECT_OK",
     "Instagram rejecting a collaborator fails the post naming them, publishes nothing, and is not retried without them",
     ("reject",)),
    ("check_ig_people.py", "IG_PEOPLE_PERSIST_OK",
     "the new columns migrate an old database, the routes round-trip, and a card's people stay together",
     ("persist",)),
    ("check_ig_people.py", "IG_PEOPLE_PUBLISH_OK",
     "Post now, the whole-card publish and the scheduler each deliver a post's collaborators and tags",
     ("publish",)),
    ("check_ig_people.py", "IG_PEOPLE_UI_OK",
     "Write Manually, the card and the schedule page show the people controls, the invite copy and a queued post's people",
     ("ui",)),
    ("check_ig_people.py", "IG_PEOPLE_JS_OK",
     "the Compose page's people functions send what the server expects, show a refusal and follow the format, under node",
     ("js",)),
    ("check_ig_people.py", "IG_PEOPLE_QUEUE_OK",
     "the server-rendered queue row and the script redraw show the same collaborators and tags, under node",
     ("queue",)),
]

# Find & Replace results on Compose: "Go to post" and the result's actions menu.
# One script, one gate per claim, picked by argument. Three run the page's own
# functions under node against a fake page; the others drive the real routes.
FIND_RESULTS_GATES = [
    ("check_find_results.py", "FIND_RESULTS_ROUTE_OK",
     "each search result says where its card sits in the list, and the list pages it exactly there",
     ("route",)),
    ("check_find_results.py", "FIND_RESULTS_WIRING_OK",
     "a result has Go to post and a menu of the card's actions, bound to declared functions, under node",
     ("wiring",)),
    ("check_find_results.py", "FIND_JUMP_OK",
     "Go to post pages only as far as the card, waits for a load in flight, and says when it is not there, under node",
     ("jump",)),
    ("check_find_results.py", "FIND_ACTIONS_OK",
     "a result's delete, used, post, schedule and queue reach the whole card and keep page and results in step, under node",
     ("actions",)),
    ("check_find_results.py", "FIND_ACTIONS_SERVER_OK",
     "the result actions' requests reach every row of the card on the real routes",
     ("server",)),
]

# check_*.py files that are deliberately not gates, with the reason. Empty is the
# healthy state: a script that is not a gate should not carry the check_ prefix.
NOT_GATES = {}

ALL_GATES = (ACCOUNT_GATES + ACCOUNT_JS_GATES + LIBRARY_GATES + COMPOSE_GATES
             + VIDEO_GATES + MANUAL_POST_GATES + QUEUE_GATES + PAGING_GATES
             + IG_PEOPLE_GATES + FIND_RESULTS_GATES)


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

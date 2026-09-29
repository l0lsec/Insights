"""Gates for bulk "Add platform" on the Compose page.

One script, one section per claim, picked by argument so each gate reports its
own token only after proving its claim against the real routes and the real
schema:

    python scripts/check_bulk_platform.py add      BULK_ADD_OK
    python scripts/check_bulk_platform.py queue    BULK_QUEUE_OK
    python scripts/check_bulk_platform.py select   BULK_SELECT_OK
    python scripts/check_bulk_platform.py guard    SINGLE_CARD_GUARD_OK
    python scripts/check_bulk_platform.py remove       BULK_REMOVE_OK
    python scripts/check_bulk_platform.py removeselect BULK_REMOVE_SELECT_OK

``guard`` pins the behaviour of the single-card endpoints (tick a platform,
queue a whole card) that the bulk endpoint shares code with, so it is written
to pass on the code as it stood before the bulk action existed.

Every section runs on a throwaway database with fake platform clients, so it
makes no network calls and cannot touch a real insights.db or a real account.
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, connect, check,
)

BULK = "/compose/posts/bulk-add-platform"
BULK_REMOVE = "/compose/posts/bulk-remove-platform"
IMAGE = "https://example.test/pic.jpg"


class Rig:
    """An isolated app with every platform connected and slots to queue into."""

    def __init__(self, platforms=("linkedin", "threads", "twitter", "facebook", "instagram"),
                 slot_platforms=None, second_linkedin=True):
        self.directory, self.database, self.web, self.publisher, self.client = isolated_app()
        self.P = self.database.DB_PATH
        install_fake_clients(self.publisher, self.web)

        # Hermetic: the repo's real .env leaks in through load_dotenv(), so the
        # link-image worker and the stock-image lookup would otherwise reach the
        # network with the user's real keys.
        self.web._maybe_attach_link_image = lambda *a, **k: None
        self.stock_calls = []
        self.stock_url = None
        self.web.get_image_for_post = self._stock

        self.accounts = {}
        ids = {"linkedin": "li-work", "threads": "th-1", "twitter": "tw-1",
               "facebook": "fb-1", "instagram": "ig-1"}
        labels = {"linkedin": "Work", "threads": "brandco", "twitter": "brandco",
                  "facebook": "Brand Page", "instagram": "brandco"}
        for platform in platforms:
            self.accounts[platform] = connect(
                self.database, platform, ids[platform], labels[platform])
        if second_linkedin and "linkedin" in platforms:
            self.studio = connect(self.database, "linkedin", "li-studio", "Studio")
        else:
            self.studio = None

        # Two slots a day, every day, so a batch has room to spread out.
        for hhmm in ("09:00", "18:00"):
            self.database.add_time_slot(
                -1, hhmm, True, list(slot_platforms) if slot_platforms else None,
                db_path=self.P)

    def _stock(self, content):
        self.stock_calls.append(content)
        return self.stock_url

    # --- seeding -----------------------------------------------------------
    def card(self, content, targets, image_url=IMAGE, brief_id=None):
        """One saved row per target, sharing copy and image: one Compose card.

        ``targets`` is a list of ``platform`` or ``(platform, account_id)``.
        Returns ``{platform_or_key: row_id}`` in the order given.
        """
        ids = {}
        for target in targets:
            platform, account_id = target if isinstance(target, tuple) else (target, None)
            row_id = self.database.add_standalone_post(
                source_type="manual", source_content="gate", platform=platform,
                content=content, image_url=image_url, brief_id=brief_id,
                account_id=account_id, db_path=self.P)
            ids[platform if account_id is None else f"{platform}:{account_id}"] = row_id
        return ids

    # --- reading -----------------------------------------------------------
    def rows(self):
        return [dict(r) for r in self.database.list_standalone_posts(db_path=self.P)]

    def cards(self):
        return self.web._group_standalone_posts(
            self.database.list_standalone_posts(db_path=self.P))

    def rows_for(self, content):
        return [r for r in self.rows() if r["content"] == content]

    def targets_of(self, content):
        return sorted((r["platform"], r["account_id"]) for r in self.rows_for(content))

    def pending(self):
        return [dict(s) for s in self.database.list_scheduled_posts(status="pending", db_path=self.P)]

    def pending_for(self, row_id):
        return [s for s in self.pending() if s["standalone_post_id"] == row_id]

    def post(self, payload, url=BULK):
        response = self.client.post(url, json=payload)
        return response.status_code, (response.get_json(silent=True) or {})


def trim(body):
    """A reply for a failure message, without the card HTML that would bury it."""
    if isinstance(body, dict):
        return {k: v for k, v in body.items() if k not in ("html", "results")}
    return body


def by_target(res):
    """The reply's per-target breakdown keyed by (platform, account id)."""
    return {(t["platform"], t["account_id"]): t for t in res["by_target"]}


# ===========================================================================
# guard: the single-card endpoints behave the way they did before bulk existed
# ===========================================================================
def section_guard():
    rig = Rig()
    db, P = rig.database, rig.P
    linkedin_default = rig.accounts["linkedin"]

    brief = db.create_content_brief("Gate brief", "make posts", db_path=P)
    ids = rig.card("Guarded card", ["linkedin", "threads"], brief_id=brief)
    li, th = ids["linkedin"], ids["threads"]
    card_ids = ",".join(str(i) for i in ids.values())

    # -- add a platform: copy, image and brief travel; the card stays one card
    res = rig.client.post(f"/compose/post/{li}/platform", data={
        "platform": "twitter", "action": "add", "post_ids": card_ids}).get_json()
    check(res and res["success"] and res["action"] == "add", f"tick twitter failed: {trim(res)}")
    tw = res["post_id"]
    row = dict(db.get_standalone_post(tw, db_path=P))
    check(row["content"] == "Guarded card" and row["image_url"] == IMAGE,
          f"ticked row did not copy the card's copy/image: {row}")
    check(row["brief_id"] == brief, "ticked row lost the card's brief")
    check(row["platform"] == "twitter" and row["account_id"] == rig.accounts["twitter"],
          f"ticked row is not aimed at twitter's default account: {row}")
    check(sorted(res["post_ids"]) == sorted([li, th, tw]),
          f"reply did not list the card's rows: {res['post_ids']}")
    check('data-platform="twitter"' in res["html"], "reply did not re-render the card")
    check(len(rig.cards()) == 1, "ticking a platform split the card")

    # -- ticking it again is refused, writing nothing
    res = rig.client.post(f"/compose/post/{li}/platform", data={
        "platform": "twitter", "action": "add", "post_ids": card_ids + f",{tw}"})
    check(res.status_code == 400 and "already goes to" in res.get_json()["error"],
          f"a repeat tick was not refused: {res.status_code} {trim(res.get_json())}")
    check(len(rig.rows_for("Guarded card")) == 3, "a refused tick still wrote a row")

    # -- a legacy row (no account recorded) still counts as its platform's tick
    with sqlite3.connect(P) as conn:
        conn.execute("UPDATE standalone_posts SET account_id = NULL WHERE id = ?", (li,))
    res = rig.client.post(f"/compose/post/{th}/platform", data={
        "platform": "linkedin", "action": "add", "post_ids": card_ids + f",{tw}"})
    check(res.status_code == 400 and "already goes to" in res.get_json()["error"],
          f"a legacy (account-less) row was not recognised as the LinkedIn tick: "
          f"{res.status_code} {trim(res.get_json())}")
    with sqlite3.connect(P) as conn:
        conn.execute("UPDATE standalone_posts SET account_id = ? WHERE id = ?",
                     (linkedin_default, li))

    # -- a named second account is its own target
    res = rig.client.post(f"/compose/post/{li}/platform", data={
        "platform": "linkedin", "account_id": str(rig.studio), "action": "add",
        "post_ids": card_ids + f",{tw}"}).get_json()
    check(res and res["success"], f"ticking the second LinkedIn account failed: {trim(res)}")
    studio_row = dict(db.get_standalone_post(res["post_id"], db_path=P))
    check(studio_row["account_id"] == rig.studio, "second account's row has the wrong account")

    # -- an unknown platform and another platform's account are refused
    res = rig.client.post(f"/compose/post/{li}/platform", data={
        "platform": "myspace", "action": "add"})
    check(res.status_code == 400, f"unknown platform was not refused: {res.status_code}")
    res = rig.client.post(f"/compose/post/{li}/platform", data={
        "platform": "twitter", "account_id": str(rig.accounts["threads"]), "action": "add"})
    check(res.status_code == 400, f"another platform's account was not refused: {res.status_code}")

    # -- queue one chip, then unticking it asks first and cleans up on force
    q = rig.client.post(f"/compose/post/{tw}/queue", data={"platform": "twitter"}).get_json()
    check(q and q["success"], f"queueing a single chip failed: {q}")
    check(len(rig.pending_for(tw)) == 1, "single queue wrote no schedule row")

    res = rig.client.post(f"/compose/post/{tw}/platform", data={
        "platform": "twitter", "action": "remove", "post_ids": card_ids + f",{tw}"})
    body = res.get_json()
    check(res.status_code == 409 and body.get("needs_confirm") and body.get("queued"),
          f"removing a queued tick did not ask first: {res.status_code} {body}")
    check(db.get_standalone_post(tw, db_path=P) is not None, "the 409 still deleted the row")

    res = rig.client.post(f"/compose/post/{tw}/platform", data={
        "platform": "twitter", "action": "remove", "force": "1",
        "post_ids": card_ids + f",{tw}"})
    body = res.get_json()
    check(res.status_code == 200 and body["success"] and body["action"] == "remove",
          f"forced remove failed: {res.status_code} {trim(body)}")
    check(db.get_standalone_post(tw, db_path=P) is None, "forced remove kept the row")
    check(not rig.pending_for(tw), "forced remove left a schedule pointing at a deleted post")

    # -- unticking something the card never had is refused
    res = rig.client.post(f"/compose/post/{li}/platform", data={
        "platform": "facebook", "action": "remove", "post_ids": card_ids})
    check(res.status_code == 400, f"removing an absent tick was not refused: {res.status_code}")

    # -- queue the whole card: one slot per row, then a repeat reports "already"
    ids2 = rig.card("Whole card", ["linkedin", "threads"])
    all2 = ",".join(str(i) for i in ids2.values())
    res = rig.client.post(f"/compose/post/{ids2['linkedin']}/queue",
                          data={"all": "1", "post_ids": all2})
    body = res.get_json()
    check(res.status_code == 200 and body["success"] and len(body["queued"]) == 2
          and not body["skipped"], f"queue-whole-card failed: {res.status_code} {body}")
    check(len(rig.pending_for(ids2["linkedin"])) == 1 and len(rig.pending_for(ids2["threads"])) == 1,
          "queue-whole-card did not queue each row exactly once")
    li_entry = rig.pending_for(ids2["linkedin"])[0]
    check(li_entry["account_id"] == linkedin_default and li_entry["platform"] == "linkedin",
          f"queue entry does not carry the row's account: {li_entry}")

    res = rig.client.post(f"/compose/post/{ids2['linkedin']}/queue",
                          data={"all": "1", "post_ids": all2})
    body = res.get_json()
    check(res.status_code == 400 and not body["success"] and len(body["skipped"]) == 2
          and all("Already queued" in s["error"] for s in body["skipped"]),
          f"re-queueing a queued card was not reported as already queued: {res.status_code} {body}")
    check(len(rig.pending_for(ids2["linkedin"])) == 1, "a repeat queue double-booked a row")

    # -- Instagram with no usable media is skipped, not queued, and reported
    ids3 = rig.card("Instagram card", ["linkedin", "instagram"], image_url=None)
    all3 = ",".join(str(i) for i in ids3.values())
    res = rig.client.post(f"/compose/post/{ids3['linkedin']}/queue",
                          data={"all": "1", "post_ids": all3})
    body = res.get_json()
    check(res.status_code == 200 and body["partial"] and len(body["queued"]) == 1
          and body["queued"][0]["platform"] == "linkedin"
          and len(body["skipped"]) == 1 and body["skipped"][0]["platform"] == "instagram",
          f"a card with an image-less Instagram row did not queue partially: {res.status_code} {body}")
    check(not rig.pending_for(ids3["instagram"]), "an image-less Instagram row was queued")

    print("single-card platform tick and queue-whole-card behave as pinned")
    print("SINGLE_CARD_GUARD_OK")


# ===========================================================================
# add: every selected card gets the chosen targets
# ===========================================================================
def section_add():
    rig = Rig()
    db, P = rig.database, rig.P

    brief = db.create_content_brief("Gate brief", "make posts", db_path=P)
    # Three cards: a LinkedIn-only, a LinkedIn+Threads, and one that already has X.
    a = rig.card("Copy A", ["linkedin"], brief_id=brief)
    b = rig.card("Copy B", ["linkedin", "threads"], image_url=None)
    c = rig.card("Copy C", ["linkedin", "twitter"])
    rig.card("Copy D", ["threads"])
    check(len(rig.cards()) == 4, "the seed did not make 4 cards")

    status, res = rig.post({
        "post_ids": [a["linkedin"], b["linkedin"], c["linkedin"]],
        "targets": ["twitter"], "queue": False})
    check(status == 200 and res.get("success"), f"bulk add failed: {status} {trim(res)}")
    check(res["cards"] == 3, f"expected 3 cards considered, got {res['cards']}")
    check(res["added"] == 2, f"expected 2 rows added (C already had X), got {res['added']}")
    check(res["already"] == 1, f"expected 1 already-there, got {res['already']}")
    check(res["queued"] == 0 and res["already_queued"] == 0, "queue=false still queued something")

    # Each new row carries the card's exact copy, image and brief.
    for content, image, has_brief in (("Copy A", IMAGE, True), ("Copy B", None, False)):
        new = [r for r in rig.rows_for(content) if r["platform"] == "twitter"]
        check(len(new) == 1, f"{content}: expected exactly one twitter row, got {len(new)}")
        check(new[0]["content"] == content and new[0]["image_url"] == image,
              f"{content}: new row did not copy copy/image: {new[0]}")
        check(new[0]["account_id"] == rig.accounts["twitter"],
              f"{content}: new row is not aimed at twitter's connected account")
        check((new[0]["brief_id"] == brief) == has_brief, f"{content}: brief not carried")
        check(new[0]["used"] == 0, f"{content}: new row was born used")
    check(len([r for r in rig.rows_for("Copy C") if r["platform"] == "twitter"]) == 1,
          "Copy C already had X and must still have exactly one")
    check([r["platform"] for r in rig.rows_for("Copy D")] == ["threads"],
          "an unselected card was changed")

    # Nothing split: still 4 cards, and each selected one is one card of the right size.
    cards = rig.cards()
    check(len(cards) == 4, f"the bulk add split or merged cards: {len(cards)} cards")
    sizes = sorted(len(g["platforms"]) for g in cards)
    check(sizes == [1, 2, 2, 3], f"unexpected card sizes {sizes}")

    # The same request again adds nothing and says so.
    before = len(rig.rows())
    status, res = rig.post({
        "post_ids": [a["linkedin"], b["linkedin"], c["linkedin"]],
        "targets": ["twitter"], "queue": False})
    check(status == 200 and res["added"] == 0 and res["already"] == 3,
          f"a repeat request was not idempotent: {status} {res}")
    check(len(rig.rows()) == before, "a repeat request wrote rows")

    # Several targets at once, including a named second account on one platform.
    status, res = rig.post({
        "post_ids": [a["linkedin"], b["linkedin"]],
        "targets": ["facebook", f"linkedin:{rig.studio}", "threads"], "queue": False})
    check(status == 200 and res["added"] == 5 and res["already"] == 1,
          f"multi-target add counted wrongly: {status} {res}")
    a_targets = rig.targets_of("Copy A")
    check(("linkedin", rig.studio) in a_targets and ("facebook", rig.accounts["facebook"]) in a_targets,
          f"the named second account / facebook missing from card A: {a_targets}")
    per = by_target(res)
    threads_stats = per[("threads", rig.accounts["threads"])]
    check(per[("linkedin", rig.studio)]["added"] == 2
          and threads_stats["already"] == 1 and threads_stats["added"] == 1,
          f"per-target breakdown is wrong: {res['by_target']}")
    check(len(rig.cards()) == 4, "multi-target add split a card")

    # Instagram can be added without an image, and the reply flags the missing image.
    status, res = rig.post({"post_ids": [b["linkedin"]], "targets": ["instagram"], "queue": False})
    check(status == 200 and res["added"] == 1 and res["needs_image"] == 1,
          f"an image-less Instagram add was not flagged: {status} {res}")
    status, res = rig.post({"post_ids": [a["linkedin"]], "targets": ["instagram"], "queue": False})
    check(status == 200 and res["added"] == 1 and res["needs_image"] == 0,
          f"an Instagram add on a card with an image was flagged: {status} {res}")

    # render=true hands back a fresh card per changed card and none for untouched ones.
    d2 = rig.card("Copy E", ["linkedin"])
    e2 = rig.card("Copy F", ["linkedin", "twitter"])
    status, res = rig.post({
        "post_ids": [d2["linkedin"], e2["linkedin"]], "targets": ["twitter"],
        "queue": False, "render": True})
    check(status == 200 and len(res["results"]) == 1,
          f"render should return only the changed card: {status} {len(res.get('results', []))}")
    only = res["results"][0]
    check(d2["linkedin"] in only["anchor_ids"], "render result is not keyed to the selected row")
    check(sorted(only["post_ids"]) == sorted(r["id"] for r in rig.rows_for("Copy E")),
          "render result does not list the card's rows")
    check('data-platform="twitter"' in only["html"] and "Copy E" in only["html"],
          "render result is not the re-rendered card")
    status, res = rig.post({"post_ids": [d2["linkedin"]], "targets": ["twitter"]})
    check(status == 200 and "results" not in res, "render leaked into a request that did not ask")

    print(f"bulk add gave {len(rig.rows())} rows over {len(rig.cards())} cards with copy, image and brief intact")
    print("BULK_ADD_OK")


# ===========================================================================
# queue: added targets go into the queue, honestly
# ===========================================================================
def section_queue():
    # No slots at all for Facebook, so a batch can hit the "no slots" edge.
    rig = Rig(slot_platforms=("linkedin", "threads", "twitter", "instagram"))
    db, P = rig.database, rig.P
    default_li = rig.accounts["linkedin"]

    A = rig.card("Q-A unqueued", ["linkedin"])
    B = rig.card("Q-B used", ["linkedin"])
    C = rig.card("Q-C queued", ["linkedin"])
    D = rig.card("Q-D posted", ["linkedin"])
    E = rig.card("Q-E threads", ["threads"])
    db.mark_standalone_post_used(B["linkedin"], True, db_path=P)
    db.add_scheduled_post(scheduled_for="2099-01-01T09:00:00", post_type="standalone",
                          standalone_post_id=C["linkedin"], platform="linkedin",
                          account_id=default_li, db_path=P)
    db.add_scheduled_post(scheduled_for="2020-01-01T09:00:00", post_type="standalone",
                          standalone_post_id=D["linkedin"], platform="linkedin",
                          status="posted", account_id=default_li, db_path=P)
    anchors = [A["linkedin"], B["linkedin"], C["linkedin"], D["linkedin"], E["threads"]]

    # -- add Threads + queue: every card gets a Threads row, each in its own slot
    status, res = rig.post({"post_ids": anchors, "targets": ["threads"], "queue": True})
    check(status == 200 and res["success"], f"add+queue failed: {status} {res}")
    check(res["added"] == 4 and res["already"] == 1,
          f"expected 4 added / 1 already (E has Threads): {res}")
    check(res["queued"] == 5 and res["skipped_count"] == 0,
          f"expected all 5 Threads rows queued (4 new + E's unqueued one): {res}")
    threads_rows = [r for r in rig.rows() if r["platform"] == "threads"]
    check(len(threads_rows) == 5, f"expected 5 Threads rows, got {len(threads_rows)}")
    entries = [s for s in rig.pending() if s["platform"] == "threads"]
    check(len(entries) == 5, f"expected 5 pending Threads entries, got {len(entries)}")
    check(len({e["scheduled_for"] for e in entries}) == 5,
          "queued Threads rows share a slot: each needs its own")
    check({e["standalone_post_id"] for e in entries} == {r["id"] for r in threads_rows},
          "queue entries do not point at exactly the Threads rows")
    check(all(e["account_id"] == rig.accounts["threads"] for e in entries),
          "queue entries do not carry the Threads account")
    check(len(rig.cards()) == 5, "queueing changed the number of cards")

    # A repeat is a no-op for the queue: everything is already queued.
    status, res = rig.post({"post_ids": anchors, "targets": ["threads"], "queue": True})
    check(status == 200 and res["added"] == 0 and res["queued"] == 0
          and res["already_queued"] == 5,
          f"a repeat add+queue did not report 5 already queued: {res}")
    check(len([s for s in rig.pending() if s["platform"] == "threads"]) == 5,
          "a repeat add+queue double-booked Threads rows")

    # -- add LinkedIn + queue: existing rows are queued unless used/queued/posted
    status, res = rig.post({"post_ids": anchors, "targets": ["linkedin"], "queue": True})
    check(status == 200 and res["success"], f"linkedin add+queue failed: {status} {res}")
    check(res["added"] == 1 and res["already"] == 4,
          f"expected LinkedIn added only to E: {res}")
    # A (existing unqueued) + E (new) queued; B used, D posted skipped; C already queued.
    check(res["queued"] == 2, f"expected 2 LinkedIn rows queued (A and E): {res}")
    check(res["already_queued"] == 1, f"expected C reported already queued: {res}")
    reasons = sorted(s["reason"] for s in res["skipped"])
    check(res["skipped_count"] == 2 and len(reasons) == 2, f"expected 2 skips: {res}")
    check(any("used" in r for r in reasons) and any("published" in r for r in reasons),
          f"skip reasons do not say used / already published: {reasons}")
    b_row, d_row = B["linkedin"], D["linkedin"]
    check(not rig.pending_for(b_row), "a used row was queued")
    check(not rig.pending_for(d_row), "a published row was re-queued")
    check(len(rig.pending_for(C["linkedin"])) == 1, "an already-queued row was double-booked")
    li_entries = [s for s in rig.pending() if s["platform"] == "linkedin"
                  and s["standalone_post_id"] in (A["linkedin"],)]
    check(len(li_entries) == 1 and li_entries[0]["account_id"] == default_li,
          "queued LinkedIn row does not carry its account")

    # -- a platform with no slots: row is still added, the queue miss is reported
    F = rig.card("Q-F no slots", ["linkedin"])
    status, res = rig.post({"post_ids": [F["linkedin"]], "targets": ["facebook"], "queue": True})
    check(status == 200 and res["added"] == 1 and res["queued"] == 0
          and res["skipped_count"] == 1, f"a no-slot platform was not reported: {status} {res}")
    check("slot" in res["skipped"][0]["reason"].lower(),
          f"the no-slot skip does not say why: {res['skipped'][0]}")
    fb = [r for r in rig.rows_for("Q-F no slots") if r["platform"] == "facebook"]
    check(len(fb) == 1 and not rig.pending_for(fb[0]["id"]),
          "the Facebook row must exist unqueued when there are no slots")
    check(by_target(res)[("facebook", rig.accounts["facebook"])]["skipped"] == 1,
          f"by_target misses the skip: {res['by_target']}")

    # -- Instagram with no image and no stock image: added, queue skipped, reported
    G = rig.card("Q-G no image", ["linkedin"], image_url=None)
    rig.stock_url = None
    status, res = rig.post({"post_ids": [G["linkedin"]], "targets": ["instagram"], "queue": True})
    check(status == 200 and res["added"] == 1 and res["queued"] == 0
          and res["skipped_count"] == 1 and res["needs_image"] == 1,
          f"an image-less Instagram queue was not reported: {status} {res}")
    check("image" in res["skipped"][0]["reason"].lower(),
          f"the Instagram skip does not mention the image: {res['skipped'][0]}")

    # -- Instagram whose stock image is found: the image lands on EVERY row of the
    #    card, or the card would silently split in two.
    H = rig.card("Q-H stock", ["linkedin", "threads"], image_url=None)
    rig.stock_url = "https://stock.test/found.jpg"
    cards_before = len(rig.cards())
    status, res = rig.post({"post_ids": [H["linkedin"]], "targets": ["instagram"], "queue": True})
    check(status == 200 and res["added"] == 1 and res["queued"] == 1,
          f"an Instagram add with a findable stock image did not queue: {status} {res}")
    images = {r["image_url"] for r in rig.rows_for("Q-H stock")}
    check(images == {"https://stock.test/found.jpg"},
          f"the stock image did not reach every row of the card: {images}")
    check(len(rig.cards()) == cards_before, "attaching the Instagram image split the card")

    # -- no platform connected at all: rows added, queue refused per row
    bare = Rig(platforms=("linkedin",), second_linkedin=False)
    K = bare.card("Q-K bare", ["linkedin"])
    status, res = bare.post({"post_ids": [K["linkedin"]], "targets": ["twitter"], "queue": True})
    check(status == 200 and res["added"] == 1 and res["queued"] == 0 and res["skipped_count"] == 1,
          f"an unconnected platform was queued or unreported: {status} {res}")
    check("connect" in res["skipped"][0]["reason"].lower(),
          f"the unconnected skip does not tell the user to connect: {res['skipped'][0]}")
    tw = [r for r in bare.rows_for("Q-K bare") if r["platform"] == "twitter"]
    check(len(tw) == 1 and not bare.pending_for(tw[0]["id"]), "unconnected Twitter row was queued")

    # -- the reported skips are capped but counted in full
    many = Rig(slot_platforms=("linkedin",))
    seeded = [many.card(f"Q-many {i}", ["linkedin"]) for i in range(70)]
    status, res = many.post({"post_ids": [s["linkedin"] for s in seeded],
                             "targets": ["facebook"], "queue": True})
    check(status == 200 and res["added"] == 70 and res["skipped_count"] == 70
          and len(res["skipped"]) <= 50,
          f"skips were not capped while still counted: {status} added={res.get('added')} "
          f"count={res.get('skipped_count')} listed={len(res.get('skipped', []))}")

    print("bulk add+queue: own slots, existing rows queued, used/queued/posted left alone, misses reported")
    print("BULK_QUEUE_OK")


# ===========================================================================
# select: which cards, and what is refused
# ===========================================================================
def section_select():
    rig = Rig()
    X = rig.card("S-X", ["linkedin", "threads"])
    Y = rig.card("S-Y", ["linkedin"])
    Z = rig.card("S-Z", ["threads"])

    # A platform-filtered selection sends one row id per card — not the card's
    # first row — and must still act on the whole card.
    status, res = rig.post({"post_ids": [X["threads"], Y["linkedin"]],
                            "targets": ["twitter"], "queue": False})
    check(status == 200 and res["cards"] == 2 and res["added"] == 2,
          f"a one-row-per-card selection did not resolve to 2 cards: {status} {res}")
    check(rig.targets_of("S-X") == sorted([("linkedin", rig.accounts["linkedin"]),
                                           ("threads", rig.accounts["threads"]),
                                           ("twitter", rig.accounts["twitter"])]),
          f"card X did not gain exactly one twitter row: {rig.targets_of('S-X')}")
    check(len([r for r in rig.rows_for("S-Z") if r["platform"] == "twitter"]) == 0,
          "an unselected card was touched")

    # Two rows of one card, and one id repeated, are still one card processed once.
    status, res = rig.post({"post_ids": [X["linkedin"], X["threads"], X["linkedin"]],
                            "targets": ["facebook"], "queue": False})
    check(status == 200 and res["cards"] == 1 and res["added"] == 1,
          f"two rows of one card were processed twice: {status} {res}")
    check(len([r for r in rig.rows_for("S-X") if r["platform"] == "facebook"]) == 1,
          "card X got more than one facebook row")

    # Stale ids are ignored beside real ones; only-stale is a clear refusal.
    status, res = rig.post({"post_ids": [Z["threads"], 987654], "targets": ["twitter"], "queue": False})
    check(status == 200 and res["cards"] == 1 and res["added"] == 1,
          f"a stale id beside a real one was not ignored: {status} {res}")
    before = len(rig.rows())
    status, res = rig.post({"post_ids": [987654, 987655], "targets": ["twitter"]})
    check(status == 400 and "error" in res, f"an all-stale selection was not refused: {status} {res}")
    check(len(rig.rows()) == before, "an all-stale selection wrote rows")

    # Refusals write nothing.
    before = len(rig.rows())
    other_account = rig.accounts["threads"]
    bad = [
        ("no selection", {"targets": ["twitter"]}),
        ("empty ids", {"post_ids": [], "targets": ["twitter"]}),
        ("no targets", {"post_ids": [Z["threads"]]}),
        ("empty targets", {"post_ids": [Z["threads"]], "targets": []}),
        ("unknown platform", {"post_ids": [Z["threads"]], "targets": ["myspace"]}),
        ("another platform's account", {"post_ids": [Z["threads"]],
                                        "targets": [f"linkedin:{other_account}"]}),
        ("one bad target among good ones", {"post_ids": [Z["threads"]],
                                            "targets": ["twitter", "myspace"]}),
        ("non-numeric ids", {"post_ids": ["abc"], "targets": ["twitter"]}),
        ("filters not an object", {"filters": "unused", "targets": ["twitter"]}),
    ]
    for name, payload in bad:
        status, res = rig.post(payload)
        check(status == 400 and res.get("error"), f"'{name}' was not refused with 400: {status} {res}")
    check(len(rig.rows()) == before, "a refused request still wrote rows")

    # A bare form post (no JSON) is not a supported shape and must not 500.
    response = rig.client.post(BULK, data={"targets": "twitter"})
    check(response.status_code == 400, f"a form-encoded request did not answer 400: {response.status_code}")

    # A string flag is read like the other bulk endpoints read theirs.
    W = rig.card("S-W", ["linkedin"])
    status, res = rig.post({"post_ids": [W["linkedin"]], "targets": ["threads"], "queue": "true"})
    check(status == 200 and res["queued"] == 1, f"queue:'true' was not read as true: {status} {res}")

    # Every Rig repoints the app's default database at its own directory, so the
    # earlier rig is finished with before this one is built.
    # "Select all across every page": the server re-derives the cards from filters.
    rig2 = Rig()
    P2, db2 = rig2.P, rig2.database
    rig2.card("F-1 unused", ["linkedin"])
    rig2.card("F-2 unused", ["linkedin", "threads"])
    used = rig2.card("F-3 used", ["linkedin"])
    rig2.card("F-4 threads", ["threads"])
    db2.mark_standalone_post_used(used["linkedin"], True, db_path=P2)

    expected = {g["head"]["content"] for g in rig2.web._filtered_post_groups({"used": "unused"})[0]}
    check(expected == {"F-1 unused", "F-2 unused", "F-4 threads"},
          f"the filter bar's own answer is not what the seed implies: {expected}")
    status, res = rig2.post({"filters": {"used": "unused"}, "targets": ["twitter"], "queue": False})
    check(status == 200 and res["cards"] == 3 and res["added"] == 3,
          f"filters mode did not act on exactly the 3 unused cards: {status} {res}")
    got = {r["content"] for r in rig2.rows() if r["platform"] == "twitter"}
    check(got == expected, f"filters mode touched {got}, the filter bar shows {expected}")
    check("results" not in res, "filters mode rendered cards nobody asked for")

    # A platform filter chooses cards; it does not narrow what gets added to a card.
    status, res = rig2.post({"filters": {"platform": "threads"}, "targets": ["facebook"], "queue": False})
    check(status == 200 and res["cards"] == 2 and res["added"] == 2,
          f"platform-filtered scope did not resolve to the 2 cards with Threads: {status} {res}")
    check({r["content"] for r in rig2.rows() if r["platform"] == "facebook"}
          == {"F-2 unused", "F-4 threads"}, "platform-filtered scope hit the wrong cards")

    print("bulk add resolves cards from ids and filters the way the page does, and refuses bad requests cleanly")
    print("BULK_SELECT_OK")



# ===========================================================================
# remove: take chosen targets off every selected card, safely
# ===========================================================================
def section_remove():
    rig = Rig()
    db, P = rig.database, rig.P
    default_li = rig.accounts["linkedin"]

    def remove(anchors, targets, **extra):
        return rig.post({"post_ids": anchors, "targets": targets, **extra}, url=BULK_REMOVE)

    def gone(row_id):
        return db.get_standalone_post(row_id, db_path=P) is None

    def orphans():
        """Pending queue entries whose saved post no longer exists."""
        return [s_ for s_ in rig.pending()
                if s_["standalone_post_id"] and gone(s_["standalone_post_id"])]

    A = rig.card("R-A", ["linkedin", "threads", "twitter"])
    B = rig.card("R-B", ["linkedin", "threads"], image_url=None)
    C = rig.card("R-C only", ["linkedin"])
    cards_before = len(rig.cards())

    # -- remove Threads from two cards: exactly those rows go, cards stay whole
    status, res = remove([A["linkedin"], B["linkedin"], C["linkedin"]], ["threads"])
    check(status == 200 and res.get("success"), f"bulk remove failed: {status} {trim(res)}")
    check(res["cards"] == 3 and res["removed"] == 2 and res["absent"] == 1 and res["skipped_count"] == 0,
          f"expected 2 removed, 1 card without Threads, no skips: {trim(res)}")
    check(gone(A["threads"]) and gone(B["threads"]), "the Threads rows were not deleted")
    check(not gone(A["linkedin"]) and not gone(A["twitter"]) and not gone(B["linkedin"]),
          "a row that was not asked for was deleted")
    check(rig.targets_of("R-A") == sorted([("linkedin", default_li), ("twitter", rig.accounts["twitter"])]),
          f"card A does not have exactly LinkedIn and X left: {rig.targets_of('R-A')}")
    check(len(rig.cards()) == cards_before, "removing a platform changed the number of cards")
    check(all(len({(r["content"], r["image_url"]) for r in rig.rows_for(c)}) == 1
              for c in ("R-A", "R-B", "R-C only")), "removing a platform split a card")

    # -- the same request again removes nothing and says so
    before = len(rig.rows())
    status, res = remove([A["linkedin"], B["linkedin"], C["linkedin"]], ["threads"])
    check(status == 200 and res["removed"] == 0 and res["absent"] == 3 and res["skipped_count"] == 0,
          f"a repeat request was not a no-op: {trim(res)}")
    check(len(rig.rows()) == before, "a repeat request deleted rows")

    # -- a post's only platform is never removed: that would delete the post
    status, res = remove([C["linkedin"]], ["linkedin"])
    check(status == 200 and res["removed"] == 0 and res["skipped_count"] == 1
          and res["skipped"][0]["code"] == "last", f"an only platform was not protected: {trim(res)}")
    check(not gone(C["linkedin"]), "the post's only row was deleted")
    check(rig.rows_for("R-C only")[0]["content"] == "R-C only", "the protected post lost its copy")

    # -- asking for every platform of a card keeps the card whole, all reported
    G = rig.card("R-G both", ["linkedin", "threads"])
    status, res = remove([G["linkedin"]], ["linkedin", "threads"])
    check(status == 200 and res["removed"] == 0 and res["skipped_count"] == 2
          and {s_["code"] for s_ in res["skipped"]} == {"last"},
          f"removing every platform of a card was not refused as a whole: {trim(res)}")
    check(not gone(G["linkedin"]) and not gone(G["threads"]), "a card lost rows it should have kept")

    # -- but two of three is fine
    H = rig.card("R-H three", ["linkedin", "threads", "twitter"])
    status, res = remove([H["linkedin"]], ["linkedin", "threads"])
    check(status == 200 and res["removed"] == 2 and res["skipped_count"] == 0
          and rig.targets_of("R-H three") == [("twitter", rig.accounts["twitter"])],
          f"removing two of three platforms failed: {trim(res)}")

    # -- a used row is just a row: used is not a queue state
    F = rig.card("R-F used", ["linkedin", "twitter"])
    db.mark_standalone_post_used(F["twitter"], True, db_path=P)
    status, res = remove([F["linkedin"]], ["twitter"])
    check(status == 200 and res["removed"] == 1 and gone(F["twitter"]), f"a used row was not removable: {trim(res)}")

    # -- a queued row is kept unless the caller says to take it out of the queue
    D = rig.card("R-D queued", ["linkedin", "threads"])
    db.add_scheduled_post(scheduled_for="2099-01-01T09:00:00", post_type="standalone",
                          standalone_post_id=D["threads"], platform="threads",
                          account_id=rig.accounts["threads"], db_path=P)
    status, res = remove([D["linkedin"]], ["threads"])
    check(status == 200 and res["removed"] == 0 and res["skipped_count"] == 1
          and res["skipped"][0]["code"] == "queued", f"a queued row was not kept: {trim(res)}")
    check(not gone(D["threads"]) and len(rig.pending_for(D["threads"])) == 1,
          "an unforced remove touched a queued row or its schedule")
    status, res = remove([D["linkedin"]], ["threads"], force=True)
    check(status == 200 and res["removed"] == 1 and res["unqueued"] == 1 and gone(D["threads"]),
          f"a forced remove did not remove the queued row: {trim(res)}")
    check(not orphans(), "a forced remove left a queue entry pointing at a deleted post")

    # -- a published row is kept unless forced, and its history survives a forced remove
    E = rig.card("R-E posted", ["linkedin", "threads"])
    db.add_scheduled_post(scheduled_for="2020-01-01T09:00:00", post_type="standalone",
                          standalone_post_id=E["threads"], platform="threads", status="posted",
                          account_id=rig.accounts["threads"], db_path=P)
    status, res = remove([E["linkedin"]], ["threads"])
    check(status == 200 and res["removed"] == 0 and res["skipped"][0]["code"] == "published",
          f"a published row was not kept: {trim(res)}")
    check(not gone(E["threads"]), "an unforced remove deleted a published row")
    status, res = remove([E["linkedin"]], ["threads"], force="true")
    check(status == 200 and res["removed"] == 1 and gone(E["threads"]),
          f"a forced remove (string flag) did not remove the published row: {trim(res)}")
    history = [h for h in db.list_scheduled_posts(status="posted", db_path=P)
               if h["standalone_post_id"] == E["threads"]]
    check(len(history) == 1, "removing a published row erased its posting history")

    # -- force never overrides the only-platform rule, and touches no queue entry
    K = rig.card("R-K both queued", ["linkedin", "threads"])
    for platform in ("linkedin", "threads"):
        db.add_scheduled_post(scheduled_for="2099-02-01T09:00:00", post_type="standalone",
                              standalone_post_id=K[platform], platform=platform,
                              account_id=rig.accounts[platform], db_path=P)
    status, res = remove([K["linkedin"]], ["linkedin", "threads"], force=True)
    check(status == 200 and res["removed"] == 0 and res["unqueued"] == 0
          and {s_["code"] for s_ in res["skipped"]} == {"last"},
          f"force overrode the only-platform rule: {trim(res)}")
    check(len(rig.pending_for(K["linkedin"])) == 1 and len(rig.pending_for(K["threads"])) == 1,
          "a refused forced remove still took entries out of the queue")

    # -- a kept (queued) row means the card is not emptied, so the rest can go
    M = rig.card("R-M partly queued", ["linkedin", "threads"])
    db.add_scheduled_post(scheduled_for="2099-03-01T09:00:00", post_type="standalone",
                          standalone_post_id=M["linkedin"], platform="linkedin",
                          account_id=default_li, db_path=P)
    status, res = remove([M["linkedin"]], ["linkedin", "threads"])
    check(status == 200 and res["removed"] == 1 and res["skipped_count"] == 1
          and res["skipped"][0]["code"] == "queued" and gone(M["threads"]) and not gone(M["linkedin"]),
          f"a card with one queued row could not shed its other platform: {trim(res)}")

    # -- one account of two on a platform
    studio = rig.studio
    L = rig.card("R-L accounts", ["linkedin", ("linkedin", studio), "threads"])
    status, res = remove([L["linkedin"]], [f"linkedin:{studio}"])
    check(status == 200 and res["removed"] == 1 and gone(L[f"linkedin:{studio}"])
          and not gone(L["linkedin"]), f"removing one LinkedIn account took the other: {trim(res)}")
    per = by_target(res)[("linkedin", studio)]
    check(per["removed"] == 1 and per["label"] == "Studio", f"per-target breakdown is wrong: {res['by_target']}")

    # -- render hands back the redrawn cards for the changed ones only
    N = rig.card("R-N render", ["linkedin", "threads"])
    O = rig.card("R-O render", ["linkedin"])
    status, res = remove([N["linkedin"], O["linkedin"]], ["threads"], render=True)
    check(status == 200 and len(res["results"]) == 1, f"render should redraw only the changed card: {trim(res)}")
    only = res["results"][0]
    check(N["linkedin"] in only["anchor_ids"] and only["post_ids"] == [N["linkedin"]],
          f"render result does not list the surviving rows: {only['post_ids']}")
    check(f'data-post-ids="{N["linkedin"]}"' in only["html"] and "R-N render" in only["html"],
          "render result is not the redrawn card")
    status, res = remove([O["linkedin"]], ["threads"])
    check("results" not in res, "render leaked into a request that did not ask")

    check(not orphans(), "the run left queue entries pointing at deleted posts")
    print("bulk remove: rows deleted per target, queued/published kept unless forced, an only platform never removed")
    print("BULK_REMOVE_OK")


# ===========================================================================
# removeselect: which cards, and what is refused
# ===========================================================================
def section_removeselect():
    rig = Rig()
    X = rig.card("RS-X", ["linkedin", "threads"])
    Y = rig.card("RS-Y", ["linkedin", "threads"])
    Z = rig.card("RS-Z", ["linkedin", "threads"])

    def remove(payload):
        return rig.post(payload, url=BULK_REMOVE)

    # A platform-filtered selection sends one row id per card, and it need not be
    # the row being removed.
    status, res = remove({"post_ids": [X["linkedin"], Y["threads"]], "targets": ["threads"]})
    check(status == 200 and res["cards"] == 2 and res["removed"] == 2,
          f"a one-row-per-card selection did not resolve to 2 cards: {status} {trim(res)}")
    check([r["platform"] for r in rig.rows_for("RS-X")] == ["linkedin"]
          and [r["platform"] for r in rig.rows_for("RS-Y")] == ["linkedin"],
          "the wrong rows were removed")
    check(sorted(r["platform"] for r in rig.rows_for("RS-Z")) == ["linkedin", "threads"],
          "an unselected card was touched")

    # Two rows of one card and a repeated id are one card, processed once.
    status, res = remove({"post_ids": [Z["linkedin"], Z["threads"], Z["linkedin"]], "targets": ["twitter"]})
    check(status == 200 and res["cards"] == 1 and res["absent"] == 1 and res["removed"] == 0,
          f"one card selected by two rows was counted twice: {status} {trim(res)}")

    # Stale ids beside real ones are ignored; only-stale is refused.
    status, res = remove({"post_ids": [Z["linkedin"], 987654], "targets": ["threads"]})
    check(status == 200 and res["cards"] == 1 and res["removed"] == 1,
          f"a stale id beside a real one was not ignored: {status} {trim(res)}")
    before = len(rig.rows())
    status, res = remove({"post_ids": [987654, 987655], "targets": ["threads"]})
    check(status == 400 and res.get("error"), f"an all-stale selection was not refused: {status} {res}")
    check(len(rig.rows()) == before, "an all-stale selection deleted rows")

    # Refusals delete nothing.
    W = rig.card("RS-W", ["linkedin", "threads"])
    before = len(rig.rows())
    other_account = rig.accounts["threads"]
    bad = [
        ("no selection", {"targets": ["threads"]}),
        ("empty ids", {"post_ids": [], "targets": ["threads"]}),
        ("no targets", {"post_ids": [W["linkedin"]]}),
        ("empty targets", {"post_ids": [W["linkedin"]], "targets": []}),
        ("unknown platform", {"post_ids": [W["linkedin"]], "targets": ["myspace"]}),
        ("another platform's account", {"post_ids": [W["linkedin"]],
                                        "targets": [f"linkedin:{other_account}"]}),
        ("one bad target among good ones", {"post_ids": [W["linkedin"]],
                                            "targets": ["threads", "myspace"]}),
        ("non-numeric ids", {"post_ids": ["abc"], "targets": ["threads"]}),
        ("ids not a list", {"post_ids": "12", "targets": ["threads"]}),
        ("filters not an object", {"filters": "unused", "targets": ["threads"]}),
    ]
    for name, payload in bad:
        status, res = remove(payload)
        check(status == 400 and res.get("error"), f"'{name}' was not refused with 400: {status} {res}")
    check(len(rig.rows()) == before, "a refused request still deleted rows")
    response = rig.client.post(BULK_REMOVE, data={"targets": "threads"})
    check(response.status_code == 400, f"a form-encoded request did not answer 400: {response.status_code}")

    # Every Rig repoints the app's default database at its own directory, so the
    # earlier rig is finished with before this one is built.
    rig2 = Rig()
    db2, P2 = rig2.database, rig2.P
    rig2.card("RF-1 unused", ["linkedin", "threads"])
    rig2.card("RF-2 unused", ["linkedin", "threads"])
    used = rig2.card("RF-3 used", ["linkedin", "threads"])
    rig2.card("RF-4 linkedin only", ["linkedin"])
    db2.mark_standalone_post_used(used["linkedin"], True, db_path=P2)
    db2.mark_standalone_post_used(used["threads"], True, db_path=P2)

    expected = {g["head"]["content"] for g in rig2.web._filtered_post_groups({"used": "unused"})[0]}
    check(expected == {"RF-1 unused", "RF-2 unused", "RF-4 linkedin only"},
          f"the filter bar's own answer is not what the seed implies: {expected}")
    status, res = rig2.post({"filters": {"used": "unused"}, "targets": ["threads"]}, url=BULK_REMOVE)
    check(status == 200 and res["cards"] == 3 and res["removed"] == 2 and res["absent"] == 1,
          f"filters mode did not act on exactly the 3 unused cards: {status} {trim(res)}")
    threads_left = {r["content"] for r in rig2.rows() if r["platform"] == "threads"}
    check(threads_left == {"RF-3 used"}, f"filters mode removed Threads from the wrong cards: {threads_left}")
    check("results" not in res, "filters mode rendered cards nobody asked for")

    print("bulk remove resolves cards from ids and filters the way the page does, and refuses bad requests cleanly")
    print("BULK_REMOVE_SELECT_OK")


SECTIONS = {
    "guard": section_guard,
    "add": section_add,
    "queue": section_queue,
    "select": section_select,
    "remove": section_remove,
    "removeselect": section_removeselect,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_bulk_platform.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()

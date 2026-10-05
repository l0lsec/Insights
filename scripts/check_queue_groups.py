"""Gates for the consolidated schedule queue.

    python scripts/check_queue_groups.py group    QUEUE_GROUP_OK
    python scripts/check_queue_groups.py page     QUEUE_PAGE_OK
    python scripts/check_queue_groups.py order    QUEUE_ORDER_OK
    python scripts/check_queue_groups.py edit     QUEUE_EDIT_OK
    python scripts/check_queue_groups.py parity   QUEUE_PARITY_OK
    python scripts/check_queue_groups.py js       QUEUE_JS_OK
    python scripts/check_queue_groups.py syntax   QUEUE_SYNTAX_OK

The queue stores one entry per platform and account, so a post sent to three
places is three rows. The schedule page shows it as one row with a badge per
platform. These gates prove that rows which are the same post collapse and rows
that only look alike do not; that every control on a collapsed row reaches every
platform it stands for (and only those); and that the two renderers of the page,
the Jinja the server sends and the JavaScript that redraws it on Refresh, draw
the same thing, which is the failure a duplicated renderer invites.

Each runs on a throwaway database with fake platform clients, so none of them
touches a real insights.db or a real account. The JavaScript ones lift the
page's own functions and run them under node with only the boundary stubbed.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, install_fake_clients, connect, check,
)
from check_accounts_js import inline_scripts, node_path  # noqa: E402

T1, T2, T3, T4 = ("2099-03-01T09:00:00", "2099-03-01T10:00:00",
                  "2099-03-01T11:00:00", "2099-03-01T12:00:00")


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

class World:
    """A throwaway app with helpers to put posts into the queue."""

    def __init__(self):
        (self.directory, self.database, self.web,
         self.publisher, self.client) = isolated_app()
        install_fake_clients(self.publisher, self.web)
        self.web._maybe_attach_link_image = lambda *a, **k: None
        self.db = self.database.DB_PATH

    def post(self, content, platform="threads", image=None, video=None):
        return self.database.add_standalone_post(
            "freeform", "gate", platform, content, image_url=image,
            video_url=video, db_path=self.db)

    def queue(self, post_id, platform, when, status="pending", url=None,
              error=None, account_id=None):
        sid = self.database.add_scheduled_post(
            scheduled_for=when, post_type="standalone", standalone_post_id=post_id,
            platform=platform, account_id=account_id, db_path=self.db)
        if status != "pending":
            self.database.update_scheduled_post_status(
                sid, status=status, linkedin_post_urn=url, error_message=error,
                db_path=self.db)
        return sid

    def copies(self, content, platforms, when, **kw):
        """The same post queued to each of ``platforms``; returns the entry ids."""
        return [self.queue(self.post(content, platform), platform, when, **kw)
                for platform in platforms]

    def rows(self, **filters):
        return self.database.list_scheduled_posts(db_path=self.db, **filters)

    def groups(self, **filters):
        return [[r["id"] for r in g]
                for g in self.database.group_scheduled_rows(self.rows(**filters))]

    def times(self, ids):
        return {i: self.database.get_scheduled_post(i, db_path=self.db)["scheduled_for"]
                for i in ids}


def standard_world():
    """A queue that holds one of each case the page has to draw."""
    w = World()
    s = {}
    s["alpha"] = w.copies("Alpha post about heart rate", ["threads", "twitter"], T1)
    s["beta"] = w.copies("Beta post about check ins", ["linkedin", "threads", "twitter"], T2)
    pid = w.post("Gamma post, threads only")
    s["gamma"] = [w.queue(pid, "threads", T3)]
    # Same copy, different moments: two entries the user scheduled apart.
    s["delta"] = [w.queue(w.post("Delta post at two times"), "threads", T4),
                  w.queue(w.post("Delta post at two times", "twitter"), "twitter",
                          "2099-03-01T13:00:00")]
    s["zeta"] = w.copies("Zeta already went out", ["threads", "twitter"], T1,
                         status="posted", url="https://x.test/zeta")
    s["eta"] = w.copies("Eta failed to go out", ["threads", "twitter"], T2,
                        status="failed", error="rate limited")
    s["theta"] = w.copies("Theta was cancelled", ["threads", "twitter"], T3,
                          status="cancelled")
    return w, s


def seed_ordering_world():
    """Three groups of different sizes at T1, T2, T3."""
    w = World()
    x = w.copies("Group X three platforms", ["linkedin", "threads", "twitter"], T1)
    y = w.copies("Group Y alone", ["threads"], T2)
    z = w.copies("Group Z two platforms", ["threads", "twitter"], T3)
    return w, x, y, z


# ---------------------------------------------------------------------------
# group: what counts as the same post
# ---------------------------------------------------------------------------

def section_group():
    w = World()
    db = w.database

    # twins at one moment collapse; the group keeps entry order
    a = w.copies("Same copy", ["threads", "twitter"], T1)
    check(w.groups() == [a], f"twins at one moment must be one group, got {w.groups()}")

    # a different moment, copy, image, video, or type each keep entries apart
    w = World()
    base = dict(content="Looks the same")
    one = w.queue(w.post(base["content"]), "threads", T1)
    later = w.queue(w.post(base["content"], "twitter"), "twitter", T2)
    other_copy = w.queue(w.post("Looks the same!", "linkedin"), "linkedin", T1)
    image = w.queue(w.post(base["content"], "facebook", image="https://img.test/a.png"),
                    "facebook", T1)
    video = w.queue(w.post(base["content"], "instagram", video="https://vid.test/a.mp4"),
                    "instagram", T1)
    groups = w.groups()
    check(len(groups) == 5 and all(len(g) == 1 for g in groups),
          f"a different moment/copy/image/video must not merge, got {groups}")
    check([one] in groups and [later] in groups and [other_copy] in groups
          and [image] in groups and [video] in groups, f"unexpected grouping {groups}")

    # same copy and image: images that are equal DO merge, '' and None are the same
    w = World()
    p1 = w.queue(w.post("Pictured", "threads", image="https://img.test/a.png"), "threads", T1)
    p2 = w.queue(w.post("Pictured", "twitter", image="https://img.test/a.png"), "twitter", T1)
    n1 = w.queue(w.post("Plain", "threads", image=""), "threads", T1)
    n2 = w.queue(w.post("Plain", "twitter", image=None), "twitter", T1)
    check(sorted(map(sorted, w.groups())) == sorted([[p1, p2], [n1, n2]]),
          f"equal copy+image must merge (and '' equals None), got {w.groups()}")

    # same status is part of the key: a posted twin and a pending twin stay apart
    w = World()
    p = w.queue(w.post("Half done"), "threads", T1)
    q = w.queue(w.post("Half done", "twitter"), "twitter", T1, status="posted",
                url="https://x.test/1")
    check(sorted(map(sorted, w.groups())) == sorted([[p], [q]]),
          f"entries in different states must not merge, got {w.groups()}")

    # two entries for the SAME account are a deliberate repost: two groups
    w = World()
    first = w.queue(w.post("Posted twice on purpose"), "threads", T1)
    second = w.queue(w.post("Posted twice on purpose"), "threads", T1)
    twin = w.queue(w.post("Posted twice on purpose", "twitter"), "twitter", T1)
    groups = w.groups()
    check(len(groups) == 2 and sum(len(g) for g in groups) == 3
          and not any(set(g) >= {first, second} for g in groups),
          f"a second entry for the same account must start its own group, got {groups}")
    check(any(twin in g for g in groups), "the other platform joins one of the groups")

    # two accounts on one platform ARE one post going to two places
    w = World()
    work = connect(w.database, "linkedin", "li-work", "Work")
    studio = connect(w.database, "linkedin", "li-studio", "Studio")
    pid = w.post("To both logins", "linkedin")
    pid2 = w.post("To both logins", "linkedin")
    e1 = w.queue(pid, "linkedin", T1, account_id=work)
    e2 = w.queue(pid2, "linkedin", T1, account_id=studio)
    check(w.groups() == [[e1, e2]], f"two accounts must share a group, got {w.groups()}")

    # an article is one post by its article id
    w = World()
    art = w.database.add_article(None, "Topic one", "style", "Body", db_path=w.db)
    other = w.database.add_article(None, "Topic two", "style", "Body", db_path=w.db)
    a1 = w.database.add_scheduled_post(T1, "article", article_id=art, platform="linkedin", db_path=w.db)
    a2 = w.database.add_scheduled_post(T1, "article", article_id=art, platform="threads", db_path=w.db)
    a3 = w.database.add_scheduled_post(T1, "article", article_id=other, platform="threads", db_path=w.db)
    check(sorted(map(sorted, w.groups())) == sorted([[a1, a2], [a3]]),
          f"article entries group by article id, got {w.groups()}")

    # social posts group on their copy too
    w = World()
    art = w.database.add_article(None, "Topic", "style", "Body", db_path=w.db)
    s1 = w.database.add_social_post(art, "linkedin", "Social copy", db_path=w.db)
    s2 = w.database.add_social_post(art, "threads", "Social copy", db_path=w.db)
    s3 = w.database.add_social_post(art, "twitter", "Different social copy", db_path=w.db)
    ids = [w.database.add_scheduled_post(T1, "social", social_post_id=s, platform=p, db_path=w.db)
           for s, p in ((s1, "linkedin"), (s2, "threads"), (s3, "twitter"))]
    check(sorted(map(sorted, w.groups())) == sorted([[ids[0], ids[1]], [ids[2]]]),
          f"social entries group by copy, got {w.groups()}")

    # an entry whose post is gone has nothing to compare and never merges
    w = World()
    gone1 = w.queue(w.post("Soon deleted"), "threads", T1)
    gone2 = w.queue(w.post("Soon deleted", "twitter"), "twitter", T1)
    for entry in (gone1, gone2):
        sp = w.database.get_scheduled_post(entry, db_path=w.db)
        w.database.delete_standalone_post(sp["standalone_post_id"], db_path=w.db)
    check(sorted(map(sorted, w.groups())) == sorted([[gone1], [gone2]]),
          f"entries with no copy must stay separate, got {w.groups()}")

    # order: groups keep first-appearance order in either sort order, and
    # get_scheduled_group_ids agrees with the page's grouping
    w, s = standard_world()
    asc = w.groups(status="pending")
    desc = w.groups(status="pending", sort_order="desc")
    check(asc[0] == s["alpha"] and asc[1] == s["beta"] and asc[2] == s["gamma"],
          f"ascending order is by moment: {asc}")
    check(desc == asc[::-1], f"descending is ascending reversed: {desc} vs {asc}")
    for key, ids in s.items():
        for i in ids:
            got = w.database.get_scheduled_group_ids(i, db_path=w.db)
            check(sorted(got) == sorted(ids) or key == "delta",
                  f"get_scheduled_group_ids({i}) = {got}, expected {ids}")
    check(w.database.get_scheduled_group_ids(999999, db_path=w.db) == [999999],
          "an id that does not exist is its own group")

    # positive control for every negative above: the standard world really does
    # contain multi-entry groups, so "stays apart" is a statement about the key
    sizes = sorted(len(g) for g in w.groups())
    check(max(sizes) == 3 and sizes.count(2) >= 4,
          f"the control world must hold groups of two and three, got {sizes}")

    print("group: twins merge; a different moment, copy, image, video, state or repost does not")
    print("QUEUE_GROUP_OK")


# ---------------------------------------------------------------------------
# HTML parsing shared by the page and parity gates
# ---------------------------------------------------------------------------

def _norm(attrs):
    out = {}
    for key, value in attrs:
        if value is None:
            value = ""
        if key == "class":
            value = " ".join(sorted(value.split()))
        elif key == "style":
            value = re.sub(r"\s+", "", value)
        out[key] = value
    return out


class RowParser(HTMLParser):
    """The rows of a table body: each row's attributes and each cell's text and elements."""

    ELEMENTS = {"span", "button", "a", "input", "strong", "small", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = {"attrs": _norm(attrs), "cells": []}
            self.rows.append(self.row)
        elif tag == "td" and self.row is not None:
            self.cell = {"text": [], "els": []}
            self.row["cells"].append(self.cell)
        elif self.cell is not None and tag in self.ELEMENTS:
            self.cell["els"].append((tag, _norm(attrs)))

    def handle_endtag(self, tag):
        if tag == "td":
            self.cell = None
        elif tag == "tr":
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"].append(data)


def parse_rows(html):
    parser = RowParser()
    parser.feed(html)
    for row in parser.rows:
        for cell in row["cells"]:
            cell["text"] = " ".join("".join(cell["text"]).split())
    return parser.rows


def tbody_of(page):
    match = re.search(r'<tbody id="schedule-tbody">(.*?)</tbody>', page, re.S)
    check(match, "the schedule page has no #schedule-tbody")
    return match.group(1)


def page_rows(client, query=""):
    response = client.get("/schedule" + query)
    check(response.status_code == 200, f"/schedule{query} returned {response.status_code}")
    return parse_rows(tbody_of(response.get_data(as_text=True)))


def group_rows(rows):
    return [r for r in rows if "queue-group" in r["attrs"].get("class", "").split()]


def detail_rows(rows):
    return [r for r in rows if "queue-detail" in r["attrs"].get("class", "").split()]


# ---------------------------------------------------------------------------
# page: what the page and its JSON carry
# ---------------------------------------------------------------------------

def section_page():
    w, s = standard_world()
    client = w.client

    pending = page_rows(client)
    groups = group_rows(pending)
    # alpha, beta, gamma, delta (x2 rows: two moments)
    check(len(groups) == 5, f"5 pending rows expected (alpha, beta, gamma, 2 delta), got {len(groups)}")
    by_first = {int(r["attrs"]["data-post-id"]): r for r in groups}
    for key in ("alpha", "beta", "gamma"):
        ids = s[key]
        row = by_first.get(ids[0])
        check(row, f"{key}: no row for its first entry {ids[0]}")
        check(row["attrs"]["data-post-ids"] == ",".join(map(str, ids)),
              f"{key}: data-post-ids must list every member: {row['attrs']['data-post-ids']}")
        check(sorted(row["attrs"]["class"].split()) == ["queue-group"],
              f"{key}: a pending row is draggable: {row['attrs']['class']}")
        checkbox = [e for e in row["cells"][1]["els"] if e[0] == "input"][0][1]
        check(checkbox["data-post-ids"] == row["attrs"]["data-post-ids"],
              f"{key}: the checkbox must carry every member id: {checkbox}")

    # one platform badge per member, in the platform cell
    def badges(row):
        return [e for e in row["cells"][6]["els"] if e[0] == "span" and "badge" in e[1].get("class", "")]
    check(len(badges(by_first[s["alpha"][0]])) == 2, "alpha shows a badge for threads and twitter")
    check(len(badges(by_first[s["beta"][0]])) == 3, "beta shows a badge for three platforms")
    check(len(badges(by_first[s["gamma"][0]])) == 1, "a lone entry shows one badge")
    names = [t for t in re.findall(r"LinkedIn|Threads|twitter", by_first[s["beta"][0]]["cells"][6]["text"])]
    check(names == ["LinkedIn", "Threads", "twitter"], f"beta platform text {names}")

    # an expander and one detail row per member only on a multi-platform row
    def expander(row):
        return [e for e in row["cells"][6]["els"] if "queue-expand-btn" in e[1].get("class", "")]
    check(expander(by_first[s["alpha"][0]]) and expander(by_first[s["beta"][0]]),
          "multi-platform rows must have an expander")
    check(not expander(by_first[s["gamma"][0]]), "a lone entry has no expander")
    details = detail_rows(pending)
    check(len(details) == 5, f"alpha(2)+beta(3) detail rows expected, got {len(details)}")
    check(all("d-none" in d["attrs"]["class"].split() for d in details),
          "detail rows start collapsed")
    check(sorted(int(d["attrs"]["data-entry-id"]) for d in details)
          == sorted(s["alpha"] + s["beta"]), "detail rows are exactly the members of multi rows")

    # the detail row's own buttons act on that one entry only
    for d in details:
        buttons = [e[1] for e in d["cells"][7]["els"] if e[0] == "button"]
        check(buttons and all(b["data-post-ids"] == d["attrs"]["data-entry-id"] for b in buttons),
              f"a detail row's buttons must act on its own entry only: {buttons}")
        check(not any("edit-content-btn" in b["class"] for b in buttons),
              "a detail row has no Edit Content (the copy belongs to the whole post)")

    # group buttons act on every member; Edit Content carries the saved posts
    alpha_row = by_first[s["alpha"][0]]
    buttons = {b["class"].split()[-1] if "edit" not in b["class"] else b["class"]: b
               for b in (e[1] for e in alpha_row["cells"][7]["els"] if e[0] == "button")}
    classes = " ".join(b["class"] for b in (e[1] for e in alpha_row["cells"][7]["els"] if e[0] == "button"))
    for needed in ("edit-time-btn", "edit-content-btn", "post-now-btn", "cancel-post-btn"):
        check(needed in classes, f"alpha row lacks {needed}")
    for e in alpha_row["cells"][7]["els"]:
        if e[0] == "button":
            check(e[1].get("data-post-ids") if "edit-content-btn" not in e[1]["class"] else True,
                  "every button names its ids")
            if "edit-content-btn" not in e[1]["class"]:
                check(e[1]["data-post-ids"] == ",".join(map(str, s["alpha"])),
                      f"a group button must act on every member: {e[1]}")
                check(e[1]["data-platform"] == "threads,twitter",
                      f"platforms line up with ids: {e[1]}")
    content_btn = [e[1] for e in alpha_row["cells"][7]["els"]
                   if e[0] == "button" and "edit-content-btn" in e[1]["class"]][0]
    sources = [w.database.get_scheduled_post(i, db_path=w.db)["standalone_post_id"] for i in s["alpha"]]
    check(content_btn["data-post-ids"] == ",".join(map(str, sources)),
          f"Edit Content must carry the saved posts behind the row: {content_btn}")

    # state-specific controls
    posted = group_rows(page_rows(client, "?status=posted"))
    check(len(posted) == 1, f"one posted row expected, got {len(posted)}")
    links = [e[1] for e in posted[0]["cells"][7]["els"] if e[0] == "a"]
    check(len(links) == 2 and all(l["href"] == "https://x.test/zeta" for l in links),
          f"a posted multi row links each platform: {links}")
    check("Threads" in posted[0]["cells"][7]["text"] and "twitter" in posted[0]["cells"][7]["text"].lower()
          or "X" in posted[0]["cells"][7]["text"], "view links say which platform")
    check("not-draggable" in posted[0]["attrs"]["class"], "a posted row is not draggable")

    failed = group_rows(page_rows(client, "?status=failed"))
    check(len(failed) == 1, "one failed row expected")
    failed_buttons = {e[1]["class"].split()[-1]: e[1] for e in failed[0]["cells"][7]["els"] if e[0] == "button"}
    check({"retry-post-btn", "delete-post-btn", "show-error-btn"} <= set(failed_buttons),
          f"failed row controls: {sorted(failed_buttons)}")
    check(failed_buttons["retry-post-btn"]["data-post-ids"] == ",".join(map(str, s["eta"])),
          "Retry acts on every failed member")
    check("rate limited" in failed_buttons["show-error-btn"]["data-error"],
          "the error button carries the reason")

    cancelled = group_rows(page_rows(client, "?status=cancelled"))
    check(len(cancelled) == 1 and "delete-post-btn" in cancelled[0]["cells"][7]["els"][0][1]["class"],
          "a cancelled row offers Delete")

    # filters
    only_threads = group_rows(page_rows(client, "?platform=threads"))
    check(len(only_threads) == 4 and all(len(r["attrs"]["data-post-ids"].split(",")) == 1
                                         for r in only_threads),
          "under a platform filter the page shows only that platform's entries, one per row")
    all_status = group_rows(page_rows(client, "?status="))
    check(len(all_status) == 5 + 1 + 1 + 1, f"all statuses: expected 8 rows, got {len(all_status)}")

    # sort: descending lists latest moment first, both draw the same groups
    desc = group_rows(page_rows(client, "?sort=desc"))
    asc = group_rows(page_rows(client, "?sort=asc"))
    check([r["attrs"]["data-post-ids"] for r in desc][::-1] == [r["attrs"]["data-post-ids"] for r in asc]
          and len(asc) > 1, "descending is ascending reversed, over the same groups")
    check(desc[0]["attrs"]["data-scheduled-for"] >= desc[-1]["attrs"]["data-scheduled-for"],
          "descending starts at the latest")

    # the page's badges
    html = client.get("/schedule").get_data(as_text=True)
    total = re.search(r'id="queue-total-badge"[^>]*>([^<]*)<', html).group(1).strip()
    check(total == "5 posts · 8 entries" or total == f"5 posts · {len(w.rows(status='pending'))} entries",
          f"the total badge must count posts and entries: {total!r}")
    threads_badge = re.search(r'id="queue-threads-badge">[^0-9]*(\d+)<', html).group(1)
    check(threads_badge == str(sum(1 for r in w.rows(status="pending") if r["platform"] == "threads")),
          "the Threads badge counts entries, not rows")

    # JSON carries the same groups and the entry/post counts
    data = client.get("/schedule/list-json?status=pending").get_json()
    check(data["success"] and data["total_count"] == 5 and data["entry_count"] == len(w.rows(status="pending")),
          f"list-json counts: {data['total_count']} posts / {data['entry_count']} entries")
    first = {g["id"]: g for g in data["groups"]}[s["beta"][0]]
    check(first["ids"] == s["beta"] and [m["platform"] for m in first["members"]] == ["linkedin", "threads", "twitter"],
          f"a JSON group lists its members: {first['ids']}")
    check(first["platforms"] == ["linkedin", "threads", "twitter"], "platforms line up with ids")
    check(first["source_type"] == "standalone" and len(first["source_ids"]) == 3,
          "a JSON group names the saved posts behind it")
    check(len(data["posts"]) == data["entry_count"]
          and sorted(p["id"] for p in data["posts"]) == sorted(i for g in data["groups"] for i in g["ids"]),
          "the feed still lists every entry flat under posts, for anything that reads entries")

    # an account label tells two accounts on one platform apart
    w2 = World()
    work = connect(w2.database, "linkedin", "li-work", "Work")
    studio = connect(w2.database, "linkedin", "li-studio", "Studio")
    a = w2.queue(w2.post("Both logins", "linkedin"), "linkedin", T1, account_id=work)
    b = w2.queue(w2.post("Both logins", "linkedin"), "linkedin", T1, account_id=studio)
    c = w2.queue(w2.post("Both logins", "threads"), "threads", T1)
    rows = group_rows(page_rows(w2.client))
    check(len(rows) == 1, "two accounts and a platform are still one post")
    text = rows[0]["cells"][6]["text"]
    check("Work" in text and "Studio" in text, f"two accounts on one platform are named: {text!r}")
    check("Threads · " not in text, "a platform with one entry needs no account name")

    # and an empty queue still draws the empty state
    w3 = World()
    check("No scheduled posts" in w3.client.get("/schedule").get_data(as_text=True),
          "an empty queue shows the empty state")

    print("page: one row per post, every member id carried, counts honest, filters and sorts hold")
    print("QUEUE_PAGE_OK")


# ---------------------------------------------------------------------------
# order: reorder and move never split a group
# ---------------------------------------------------------------------------

def intact(w, expected_groups):
    got = sorted(map(sorted, w.groups(status="pending")))
    return got == sorted(map(sorted, expected_groups))


def moments(w, ids):
    return {w.database.get_scheduled_post(i, db_path=w.db)["scheduled_for"] for i in ids}


def section_order():
    # reorder: groups of 3, 1 and 2 swap places without splitting
    w, x, y, z = seed_ordering_world()
    check(w.database.reorder_scheduled_posts([y[0], x[0], z[0]], db_path=w.db), "reorder returned false")
    check(intact(w, [x, y, z]), f"reorder split a group: {w.groups(status='pending')}")
    check(moments(w, y) == {T1} and moments(w, x) == {T2} and moments(w, z) == {T3},
          f"Y,X,Z must take T1,T2,T3: {w.times(x + y + z)}")

    # naming a non-first member moves the same group
    w, x, y, z = seed_ordering_world()
    w.database.reorder_scheduled_posts([z[1], y[0], x[2]], db_path=w.db)
    check(intact(w, [x, y, z]), "reorder by a later member split a group")
    check(moments(w, z) == {T1} and moments(w, y) == {T2} and moments(w, x) == {T3},
          f"Z,Y,X must take T1,T2,T3: {w.times(x + y + z)}")

    # a stale or unknown id is ignored, not mistaken for a position
    w, x, y, z = seed_ordering_world()
    w.database.reorder_scheduled_posts([999999, y[0], x[0]], db_path=w.db)
    check(intact(w, [x, y, z]) and moments(w, y) == {T1} and moments(w, x) == {T2}
          and moments(w, z) == {T3}, "an unknown id must not shift the others")

    # one group named: nothing to reorder
    w, x, y, z = seed_ordering_world()
    before = w.times(x + y + z)
    w.database.reorder_scheduled_posts([x[0], x[1], x[2]], db_path=w.db)
    check(w.times(x + y + z) == before, "reordering one group must change nothing")

    # singletons behave as they always did: times are swapped
    w = World()
    a = w.queue(w.post("Single A"), "threads", T1)
    b = w.queue(w.post("Single B"), "threads", T2)
    c = w.queue(w.post("Single C"), "threads", T3)
    w.database.reorder_scheduled_posts([c, a, b], db_path=w.db)
    t = w.times([a, b, c])
    check(t[c] == T1 and t[a] == T2 and t[b] == T3, f"singleton reorder regressed: {t}")

    # a posted entry is never touched
    w, x, y, z = seed_ordering_world()
    done = w.queue(w.post("Already out"), "threads", T1, status="posted", url="https://x.test/1")
    w.database.reorder_scheduled_posts([y[0], done, x[0]], db_path=w.db)
    check(w.times([done]) == {done: T1}, "a posted entry must keep its time")

    # move to top / bottom
    w, x, y, z = seed_ordering_world()
    w.database.move_posts_to_position([z[1]], "top", db_path=w.db)
    check(intact(w, [x, y, z]) and moments(w, z) == {T1} and moments(w, x) == {T2}
          and moments(w, y) == {T3}, f"move Z to top: {w.times(x + y + z)}")
    # now Z, X, Y: sending X to the bottom leaves Z, Y, X
    w.database.move_posts_to_position([x[0]], "bottom", db_path=w.db)
    check(intact(w, [x, y, z]) and moments(w, z) == {T1} and moments(w, y) == {T2}
          and moments(w, x) == {T3}, f"move X to bottom: {w.times(x + y + z)}")

    w, x, y, z = seed_ordering_world()
    w.database.move_posts_to_position([y[0]], "top", db_path=w.db)
    check(intact(w, [x, y, z]) and moments(w, y) == {T1} and moments(w, x) == {T2}
          and moments(w, z) == {T3},
          f"moving the 1-entry group above a 3-entry group must not split it: {w.times(x + y + z)}")

    # several selected groups keep their relative order
    w, x, y, z = seed_ordering_world()
    w.database.move_posts_to_position([z[0], y[0]], "top", db_path=w.db)
    check(intact(w, [x, y, z]) and moments(w, y) == {T1} and moments(w, z) == {T2}
          and moments(w, x) == {T3}, f"selected groups keep their order: {w.times(x + y + z)}")

    # no selection / unknown selection: no change
    w, x, y, z = seed_ordering_world()
    before = w.times(x + y + z)
    w.database.move_posts_to_position([999999], "top", db_path=w.db)
    check(w.times(x + y + z) == before, "an unknown selection must change nothing")

    # through the routes, exactly as the page sends them
    w, x, y, z = seed_ordering_world()
    reply = w.client.post("/schedule/reorder", json={"post_ids": [y[0], x[0], z[0]]})
    check(reply.status_code == 200 and reply.get_json()["success"], f"/schedule/reorder: {reply.get_data()}")
    check(intact(w, [x, y, z]) and moments(w, y) == {T1}, "the reorder route keeps groups")
    reply = w.client.post("/schedule/move-position", json={"post_ids": [z[0]], "position": "top"})
    check(reply.status_code == 200, f"/schedule/move-position: {reply.get_data()}")
    check(intact(w, [x, y, z]) and moments(w, z) == {T1}, "the move route keeps groups")

    # a filtered view (one platform) moves the other platforms with the post
    w, x, y, z = seed_ordering_world()
    only_threads = [r["id"] for r in w.rows(status="pending", platform="threads")]
    check(len(only_threads) == 3, "control: three threads entries")
    w.database.reorder_scheduled_posts(list(reversed(only_threads)), db_path=w.db)
    check(intact(w, [x, y, z]), "reordering a filtered view must not split a post")
    check(moments(w, z) == {T1} and moments(w, y) == {T2} and moments(w, x) == {T3},
          f"reversing the threads view reverses the groups: {w.times(x + y + z)}")

    print("order: reorder and move keep every group whole, whatever its size")
    print("QUEUE_ORDER_OK")


# ---------------------------------------------------------------------------
# edit: Edit Time applies to a group
# ---------------------------------------------------------------------------

def section_edit():
    NEW = "2099-06-01T08:30:00"
    w, s = standard_world()
    beta = s["beta"]
    stranger = s["gamma"][0]
    posted_twin = s["zeta"][0]

    # the whole group moves
    reply = w.client.post(f"/schedule/{beta[0]}/edit",
                          data={"scheduled_for": NEW, "ids": ",".join(map(str, beta))})
    check(reply.status_code == 200 and reply.get_json()["success"], f"edit: {reply.get_data()}")
    check(sorted(reply.get_json()["updated_ids"]) == sorted(beta), "the reply says which ids moved")
    check(moments(w, beta) == {NEW}, f"every member must move: {w.times(beta)}")

    # ids from another post, or another state, are ignored
    reply = w.client.post(f"/schedule/{s['alpha'][0]}/edit",
                          data={"scheduled_for": NEW,
                                "ids": f"{s['alpha'][0]},{s['alpha'][1]},{stranger},{posted_twin}"})
    check(reply.status_code == 200, f"edit with strangers: {reply.get_data()}")
    check(moments(w, s["alpha"]) == {NEW}, "alpha moves")
    check(moments(w, [stranger]) == {T3}, "an unrelated pending post must not move")
    check(moments(w, [posted_twin]) == {T1}, "a posted entry must not move")

    # ids listing only part of a group move only that part (a filtered view)
    w, s = standard_world()
    alpha = s["alpha"]
    reply = w.client.post(f"/schedule/{alpha[0]}/edit", data={"scheduled_for": NEW, "ids": str(alpha[0])})
    check(reply.status_code == 200 and moments(w, [alpha[0]]) == {NEW} and moments(w, [alpha[1]]) == {T1},
          "naming one id edits that one entry only")

    # no ids: exactly the old single-entry behaviour
    w, s = standard_world()
    reply = w.client.post(f"/schedule/{s['alpha'][0]}/edit", data={"scheduled_for": NEW})
    check(reply.status_code == 200 and moments(w, [s["alpha"][0]]) == {NEW}
          and moments(w, [s["alpha"][1]]) == {T1}, "without ids a request edits one entry")

    # JSON bodies and repeated fields are read the same way the Compose routes read them
    w, s = standard_world()
    reply = w.client.post(f"/schedule/{s['alpha'][0]}/edit",
                          data={"scheduled_for": NEW, "ids": [str(i) for i in s["alpha"]]})
    check(moments(w, s["alpha"]) == {NEW}, "repeated ids fields work")

    # the primary must be pending
    w, s = standard_world()
    reply = w.client.post(f"/schedule/{s['zeta'][0]}/edit",
                          data={"scheduled_for": NEW, "ids": ",".join(map(str, s["zeta"]))})
    check(reply.status_code == 400, f"editing a posted entry must be refused, got {reply.status_code}")
    check(moments(w, s["zeta"]) == {T1}, "a refused edit changes nothing")

    # a time in the past is still refused, for every id
    w, s = standard_world()
    reply = w.client.post(f"/schedule/{s['alpha'][0]}/edit",
                          data={"scheduled_for": "2001-01-01T00:00:00",
                                "ids": ",".join(map(str, s["alpha"]))})
    check(reply.status_code == 400 and moments(w, s["alpha"]) == {T1},
          "a past time is refused and nothing moves")

    # after a group edit the entries are still one row on the page
    w, s = standard_world()
    w.client.post(f"/schedule/{s['beta'][0]}/edit",
                  data={"scheduled_for": NEW, "ids": ",".join(map(str, s["beta"]))})
    rows = group_rows(page_rows(w.client))
    check(any(r["attrs"]["data-post-ids"] == ",".join(map(str, s["beta"])) for r in rows),
          "after editing a group's time it is still one row")

    # an edit that touches only one platform splits the row, which is the point of the expander
    w, s = standard_world()
    w.client.post(f"/schedule/{s['beta'][1]}/edit", data={"scheduled_for": NEW})
    rows = group_rows(page_rows(w.client))
    ids = [r["attrs"]["data-post-ids"] for r in rows]
    check(str(s["beta"][1]) in ids and f"{s['beta'][0]},{s['beta'][2]}" in ids,
          f"scheduling one platform separately splits it off: {ids}")

    print("edit: a group's time moves every member and nothing outside the group")
    print("QUEUE_EDIT_OK")


# ---------------------------------------------------------------------------
# node plumbing
# ---------------------------------------------------------------------------

def schedule_script():
    w = World()
    # one page render is enough: the script is the same whatever the queue holds
    html = w.client.get("/schedule").get_data(as_text=True)
    scripts = inline_scripts(html)
    check(scripts, "the schedule page rendered no inline scripts")
    return w, html, "\n".join(scripts)


def lift(source, names, consts=()):
    """The page's own top-level functions and constants, by name."""
    pieces = []
    for name in consts:
        match = re.search(rf"^(?:const|let) {name}\b.*?^(?:\}};|;)\n|^(?:const|let) {name}\b[^\n]*;\n",
                          source, re.M | re.S)
        check(match, f"could not lift {name} out of the page")
        pieces.append(match.group(0))
    for name in names:
        found = re.findall(rf"^(?:async\s+)?function {name}\(", source, re.M)
        check(len(found) == 1, f"{name} is declared {len(found)} times on the page, expected once")
        match = re.search(rf"^(?:async\s+)?function {name}\(.*?^\}}\n", source, re.M | re.S)
        check(match, f"could not lift {name}'s source out of the page")
        pieces.append(match.group(0))
    return "\n".join(pieces)


def run_node(harness, functions, data=None):
    node = node_path()
    work = tempfile.mkdtemp(prefix="queue_js_")
    try:
        paths = {}
        for name, body in (("functions.js", functions), ("harness.js", harness),
                           ("data.json", json.dumps(data if data is not None else {}))):
            paths[name] = os.path.join(work, name)
            with open(paths[name], "w", encoding="utf-8") as handle:
                handle.write(body)
        run = subprocess.run([node, paths["harness.js"], paths["functions.js"], paths["data.json"]],
                             capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:1200]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


RENDER_FUNCTIONS = ["escapeHtml", "rebuildQueueTable", "getStatusBadge", "getPlatformBadge",
                    "getTypeBadge", "getViewLink", "getActionButtons"]


# ---------------------------------------------------------------------------
# parity: the Jinja rows and the JavaScript rows are the same rows
# ---------------------------------------------------------------------------

PARITY_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const tbody = { innerHTML: '' };
const ctx = {
  document: { getElementById: (id) => id === 'schedule-tbody' ? tbody : null },
  initSortable: () => {},
};
vm.createContext(ctx);
vm.runInContext(source, ctx);
const out = {};
for (const [name, groups] of Object.entries(data)) {
  tbody.innerHTML = '';
  vm.runInContext('rebuildQueueTable(__groups)', Object.assign(ctx, { __groups: groups }));
  out[name] = tbody.innerHTML;
}
console.log(JSON.stringify(out));
"""

PARITY_CASES = [
    ("pending", "?status=pending"),
    ("posted", "?status=posted"),
    ("failed", "?status=failed"),
    ("cancelled", "?status=cancelled"),
    ("everything", "?status="),
    ("threads", "?status=pending&platform=threads"),
    ("descending", "?status=pending&sort=desc"),
]


def section_parity():
    w, s = standard_world()
    # an article, two accounts on one platform, a video and an image, quotes and
    # markup in the copy, so the escaping paths are compared too
    art = w.database.add_article(None, "Topic <b>one</b>", "style", "Body", db_path=w.db)
    for platform in ("linkedin", "threads"):
        w.database.add_scheduled_post(T2, "article", article_id=art, platform=platform, db_path=w.db)
    work = connect(w.database, "linkedin", "li-work", "Work & Co")
    studio = connect(w.database, "linkedin", "li-studio", "Studio")
    w.queue(w.post('He said "hi" & <left> it\'s 100% done', "linkedin", image="https://img.test/a.png?x=1&y=2"),
            "linkedin", T3, account_id=work)
    w.queue(w.post('He said "hi" & <left> it\'s 100% done', "linkedin", image="https://img.test/a.png?x=1&y=2"),
            "linkedin", T3, account_id=studio)
    w.queue(w.post("A video post", "threads", video="https://vid.test/a.mp4"), "threads", T4)
    w.queue(w.post("x" * 140, "threads"), "threads", T4)
    sid = w.queue(w.post("Failed with <b>markup</b> & \"quotes\"", "twitter"), "twitter", T4,
                  status="failed", error='boom <script> & "bang"')
    w.queue(w.post("Posted without a link", "threads"), "threads", T4, status="posted")
    w.queue(w.post("Posted threads link", "threads"), "threads", T3, status="posted", url="not-a-url")

    expected, payload = {}, {}
    for name, query in PARITY_CASES:
        expected[name] = page_rows(w.client, query)
        payload[name] = w.client.get("/schedule/list-json" + query).get_json()["groups"]
        check(expected[name], f"parity case {name} drew no rows, so it proves nothing")

    _, html, script = schedule_script()
    functions = lift(script, RENDER_FUNCTIONS, consts=("QUEUE_PLATFORMS",))
    drawn = run_node(PARITY_HARNESS, functions, payload)

    multi_rows = 0
    for name, _ in PARITY_CASES:
        rebuilt = parse_rows(drawn[name])
        server = expected[name]
        check(len(rebuilt) == len(server),
              f"{name}: the server drew {len(server)} rows and the script drew {len(rebuilt)}")
        for index, (a, b) in enumerate(zip(server, rebuilt)):
            check(a == b, f"{name}: row {index} differs between the page and the redraw:\n"
                          f"  server: {json.dumps(a, ensure_ascii=False)[:900]}\n"
                          f"  script: {json.dumps(b, ensure_ascii=False)[:900]}")
        multi_rows += sum(1 for r in server if "queue-detail" in r["attrs"].get("class", ""))
    check(multi_rows >= 15, f"the parity cases must include multi-platform rows, saw {multi_rows} detail rows")

    # the totals badge: the page and the refresh produce the same words
    html_total = {}
    for name, query in PARITY_CASES:
        page = w.client.get("/schedule" + query).get_data(as_text=True)
        html_total[name] = re.search(r'id="queue-total-badge"[^>]*>([^<]*)<', page).group(1).strip()
    data = {name: w.client.get("/schedule/list-json" + q).get_json() for name, q in PARITY_CASES}
    for name, _ in PARITY_CASES:
        n, entries = data[name]["total_count"], data[name]["entry_count"]
        words = f"{n} {'post' if n == 1 else 'posts'}" + (f" · {entries} entries" if entries != n else "")
        check(words == html_total[name], f"{name}: total badge page={html_total[name]!r} refresh={words!r}")
    check(re.search(r"n === 1 \? 'post' : 'posts'", script) and "entries` : ''" in script.replace("\n", ""),
          "refreshQueue must build the total badge text the way the page does")

    print("parity: the server's rows and the script's rows are identical across "
          f"{len(PARITY_CASES)} views, {multi_rows} per-platform rows compared")
    print("QUEUE_PARITY_OK")


# ---------------------------------------------------------------------------
# js: the controls
# ---------------------------------------------------------------------------

JS_FUNCTIONS = ["idsOf", "platformsOf", "requestEach", "describeFailures", "finishQueueAction",
                "postNow", "cancelPost", "deletePost", "retryPost", "deleteSelected",
                "moveSelectedPosts", "openEditModal", "submitEdit", "openEditContentModal",
                "submitContentEdit", "saveToAllPosts", "toggleQueueGroup", "collapseAllQueueGroups",
                "pendingQueueOrder", "updateCharCount", "resetContentImageUI",
                "loadContentImageLibrary", "updateContentImagePreview"]

JS_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

class FakeFormData {
  constructor() { this.entries = []; }
  append(k, v) { this.entries.push([k, String(v)]); }
}

function world(opts = {}) {
  const log = { fetch: [], alerts: [], confirms: [], reloads: 0, selectors: [], timers: [] };
  const els = {};
  const el = (id) => (els[id] = els[id] || { value: '', textContent: '', classList: { add() {}, remove() {} } });
  const replies = (opts.replies || []).slice();
  const ctx = {
    FormData: FakeFormData,
    location: { reload: () => { log.reloads++; } },
    setTimeout: (fn) => { log.timers.push(fn); },
    confirm: (m) => { log.confirms.push(m); return opts.confirm !== false; },
    alert: (m) => log.alerts.push(m),
    bootstrap: { Modal: Object.assign(function () { return { show() {}, hide() {} }; },
                                    { getInstance: () => ({ hide() { log.hidden = true; } }) }) },
    document: {
      getElementById: (id) => opts.dom && opts.dom[id] ? opts.dom[id] : el(id),
      querySelector: (sel) => (opts.query || {})[sel] || { innerHTML: 'Save', disabled: false },
      querySelectorAll: (sel) => { log.selectors.push(sel); return (opts.all || {})[sel] || []; },
    },
    fetch: async (url, init) => {
      const call = { url, method: init.method,
                     fields: init.body && init.body.entries ? init.body.entries : null,
                     json: init.body && typeof init.body === 'string' ? JSON.parse(init.body) : null };
      log.fetch.push(call);
      if (opts.throwOn && opts.throwOn(url)) throw new Error('offline');
      const reply = replies.shift() || { body: { success: true } };
      return { ok: true, status: 200, json: async () => reply.body };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  return { ctx, log, els, el };
}

const button = (dataset) => ({ dataset, innerHTML: 'ORIGINAL', disabled: false, className: 'x' });
const row = (pending) => ({ classList: { contains: (c) => c === 'not-draggable' && !pending } });
const checkbox = (ids, pending = true) => ({ dataset: { postId: String(ids[0]), postIds: ids.join(',') }, closest: () => row(pending) });
const ok = { body: { success: true } };
const no = (error) => ({ body: { error } });
const out = {};

(async () => {
  let w = world();
  // --- readers ---------------------------------------------------------
  out.idsOf = {
    group: w.ctx.idsOf({ dataset: { postIds: '4,5,6', postId: '4' } }),
    single: w.ctx.idsOf({ dataset: { postId: '7' } }),
    junk: w.ctx.idsOf({ dataset: { postIds: '1,,x,3' } }),
    none: w.ctx.idsOf({ dataset: {} }),
    platforms: w.ctx.platformsOf({ dataset: { platform: 'threads,twitter' } }),
    noplatform: w.ctx.platformsOf({ dataset: {} }),
  };

  // --- Post Now on a group ---------------------------------------------
  w = world({ replies: [ok, ok, ok] });
  let btn = button({});
  await w.ctx.postNow([11, 12, 13], ['linkedin', 'threads', 'twitter'], btn);
  out.postAll = { urls: w.log.fetch.map(f => f.url), methods: w.log.fetch.map(f => f.method),
                  confirm: w.log.confirms, btn: btn.innerHTML, alerts: w.log.alerts,
                  reloadsScheduled: w.log.timers.length, reloads: w.log.reloads };

  w = world({ replies: [ok, no('rate limited'), ok] });
  btn = button({});
  await w.ctx.postNow([11, 12, 13], ['linkedin', 'threads', 'twitter'], btn);
  out.postPartial = { urls: w.log.fetch.map(f => f.url), alerts: w.log.alerts, reloads: w.log.reloads,
                      disabled: btn.disabled, html: btn.innerHTML };

  w = world({ replies: [no('a'), no('b')] });
  btn = button({});
  await w.ctx.postNow([11, 12], ['threads', 'twitter'], btn);
  out.postNone = { alerts: w.log.alerts, reloads: w.log.reloads, disabled: btn.disabled, html: btn.innerHTML };

  w = world({ throwOn: (u) => u.includes('/12/') , replies: [ok, ok] });
  btn = button({});
  await w.ctx.postNow([11, 12, 13], ['linkedin', 'threads', 'twitter'], btn);
  out.postThrows = { urls: w.log.fetch.map(f => f.url), alerts: w.log.alerts, reloads: w.log.reloads };

  w = world({ replies: [ok] });
  btn = button({});
  await w.ctx.postNow([21], ['threads'], btn);
  out.postSingle = { confirm: w.log.confirms, urls: w.log.fetch.map(f => f.url), btn: btn.innerHTML };

  w = world({ confirm: false });
  btn = button({});
  await w.ctx.postNow([11, 12], ['threads', 'twitter'], btn);
  out.postDeclined = { fetches: w.log.fetch.length, html: btn.innerHTML, disabled: btn.disabled };

  // --- Cancel / Delete / Retry -----------------------------------------
  w = world({ replies: [ok, ok] });
  await w.ctx.cancelPost([31, 32], ['threads', 'twitter']);
  out.cancelAll = { urls: w.log.fetch.map(f => f.url), confirm: w.log.confirms, reloads: w.log.reloads, alerts: w.log.alerts };

  w = world({ replies: [no('already posted'), ok] });
  await w.ctx.cancelPost([31, 32], ['threads', 'twitter']);
  out.cancelPartial = { alerts: w.log.alerts, reloads: w.log.reloads };

  w = world({ replies: [no('gone'), no('gone too')] });
  await w.ctx.cancelPost([31, 32], ['threads', 'twitter']);
  out.cancelNone = { alerts: w.log.alerts, reloads: w.log.reloads };

  w = world({ replies: [ok] });
  await w.ctx.cancelPost([33], ['threads']);
  out.cancelSingle = { confirm: w.log.confirms, urls: w.log.fetch.map(f => f.url), reloads: w.log.reloads };

  w = world({ replies: [ok, ok, ok] });
  await w.ctx.deletePost([41, 42, 43], ['linkedin', 'threads', 'twitter']);
  out.deleteAll = { urls: w.log.fetch.map(f => f.url), confirm: w.log.confirms, reloads: w.log.reloads };

  w = world({ replies: [ok, ok] });
  btn = button({});
  await w.ctx.retryPost([51, 52], ['threads', 'twitter'], btn);
  out.retryAll = { urls: w.log.fetch.map(f => f.url), alerts: w.log.alerts, reloads: w.log.reloads };

  w = world({ replies: [no('still failing'), ok] });
  btn = button({});
  await w.ctx.retryPost([51, 52], ['threads', 'twitter'], btn);
  out.retryPartial = { alerts: w.log.alerts, reloads: w.log.reloads, html: btn.innerHTML, disabled: btn.disabled };

  w = world({ replies: [no('still failing'), no('still failing')] });
  btn = button({});
  await w.ctx.retryPost([51, 52], ['threads', 'twitter'], btn);
  out.retryNone = { alerts: w.log.alerts, reloads: w.log.reloads, html: btn.innerHTML, disabled: btn.disabled };

  // --- bulk delete and move --------------------------------------------
  w = world({ all: { '.post-checkbox:checked': [checkbox([1, 2]), checkbox([3]), checkbox([4, 5, 6])] },
              replies: [{ body: { success: true, message: 'Deleted 6 posts' } }] });
  w.ctx.deleteSelected();
  await new Promise(r => setTimeout(r, 20));
  out.bulkDelete = { body: w.log.fetch.map(f => f.json), confirm: w.log.confirms };

  w = world({ all: { '.post-checkbox:checked': [checkbox([3]), checkbox([7])] },
              replies: [{ body: { success: true, message: 'Deleted 2 posts' } }] });
  w.ctx.deleteSelected();
  await new Promise(r => setTimeout(r, 20));
  out.bulkDeleteSingles = { confirm: w.log.confirms };

  const moveBtn = { innerHTML: 'Move', disabled: false };
  w = world({ all: { '.post-checkbox:checked': [checkbox([1, 2]), checkbox([3], false), checkbox([4, 5, 6])] },
              dom: { 'move-top-btn': moveBtn },
              replies: [{ body: { success: true } }] });
  w.ctx.moveSelectedPosts('top');
  await new Promise(r => setTimeout(r, 20));
  out.bulkMove = { body: w.log.fetch.map(f => f.json), confirm: w.log.confirms };

  // --- Edit Time ----------------------------------------------------------
  w = world();
  w.ctx.openEditModal(61, '2099-01-01T09:00:00', [61, 62, 63]);
  out.openEdit = { id: w.el('edit-post-id').value, ids: w.el('edit-post-ids').value };
  w.ctx.openEditModal(64, '2099-01-01T09:00:00');
  out.openEditNoIds = { ids: w.el('edit-post-ids').value };

  w = world({ replies: [{ body: { success: true, scheduled_for_display: 'Monday' } }] });
  w.el('edit-post-id').value = '61';
  w.el('edit-post-ids').value = '61,62,63';
  w.el('edit-datetime').value = '2099-01-01T09:30';
  w.ctx.submitEdit();
  await new Promise(r => setTimeout(r, 20));
  out.submitEdit = { url: w.log.fetch[0].url, fields: w.log.fetch[0].fields };

  // --- Edit Content ------------------------------------------------------
  w = world();
  w.ctx.openEditContentModal(70, 'standalone', 'threads,twitter', 'Copy', '', [70, 71]);
  out.openContent = { ids: w.el('edit-content-post-ids').value, platform: w.el('edit-content-platform').textContent };
  w.ctx.openEditContentModal(72, 'standalone', 'linkedin', 'Copy', '');
  out.openContentSingle = { ids: w.el('edit-content-post-ids').value, platform: w.el('edit-content-platform').textContent };

  const contentWorld = (opts = {}) => {
    const x = world(opts);
    x.el('edit-content-post-id').value = opts.postId || '70';
    x.el('edit-content-post-type').value = opts.type || 'standalone';
    x.el('edit-content-post-ids').value = opts.ids || '70,71';
    x.el('edit-content-textarea').value = '  New copy  ';
    x.el('edit-content-image-url').value = opts.image || '';
    x.el('edit-content-original-image-url').value = opts.originalImage || '';
    return x;
  };

  w = contentWorld({ replies: [ok] });
  await w.ctx.submitContentEdit();
  out.contentStandalone = { calls: w.log.fetch.map(f => ({ url: f.url, fields: f.fields })), reloads: w.log.reloads, alerts: w.log.alerts };

  w = contentWorld({ replies: [ok, ok], image: 'https://img.test/new.png' });
  await w.ctx.submitContentEdit();
  out.contentWithImage = { calls: w.log.fetch.map(f => ({ url: f.url, fields: f.fields })), reloads: w.log.reloads };

  w = contentWorld({ type: 'social', postId: '80', ids: '80,81,82', replies: [ok, ok, ok] });
  await w.ctx.submitContentEdit();
  out.contentSocial = { calls: w.log.fetch.map(f => ({ url: f.url, fields: f.fields })), reloads: w.log.reloads };

  w = contentWorld({ type: 'social', postId: '80', ids: '80,81,82', replies: [ok, no('nope'), ok] });
  await w.ctx.submitContentEdit();
  out.contentSocialPartial = { urls: w.log.fetch.map(f => f.url), alerts: w.log.alerts, reloads: w.log.reloads };

  w = contentWorld({ replies: [no('Content is required')] });
  await w.ctx.submitContentEdit();
  out.contentFails = { calls: w.log.fetch.length, alerts: w.log.alerts, reloads: w.log.reloads };

  w = contentWorld({ replies: [ok, no('bad image')], image: 'https://img.test/new.png' });
  await w.ctx.submitContentEdit();
  out.contentImageFails = { calls: w.log.fetch.length, alerts: w.log.alerts, reloads: w.log.reloads };

  w = contentWorld({ replies: [] });
  w.el('edit-content-textarea').value = '   ';
  await w.ctx.submitContentEdit();
  out.contentEmpty = { calls: w.log.fetch.length, alerts: w.log.alerts };

  // --- expander --------------------------------------------------------
  const detail = () => ({ hidden: true, classList: { toggle(c, force) { this.owner.hidden = force; }, owner: null } });
  const d1 = detail(); d1.classList.owner = d1;
  const d2 = detail(); d2.classList.owner = d2;
  const toggle = { attrs: { 'aria-expanded': 'false' }, textContent: '▸ By platform', dataset: { groupId: '9' },
                   getAttribute(k) { return this.attrs[k]; }, setAttribute(k, v) { this.attrs[k] = v; } };
  w = world({ all: { 'tr.queue-detail[data-group-id="9"]': [d1, d2],
                     '.queue-expand-btn[aria-expanded="true"]': [] } });
  w.ctx.toggleQueueGroup(toggle);
  const opened = { expanded: toggle.attrs['aria-expanded'], text: toggle.textContent, hidden: [d1.hidden, d2.hidden] };
  w = world({ all: { 'tr.queue-detail[data-group-id="9"]': [d1, d2],
                     '.queue-expand-btn[aria-expanded="true"]': [toggle] } });
  w.ctx.collapseAllQueueGroups();
  out.expander = { opened, closed: { expanded: toggle.attrs['aria-expanded'], text: toggle.textContent, hidden: [d1.hidden, d2.hidden] } };

  // --- the order the drag reports ---------------------------------------
  w = world({ dom: { 'schedule-tbody': { querySelectorAll: (sel) => {
    w.log.selectors.push(sel);
    return [{ dataset: { postId: '5' } }, { dataset: { postId: '2' } }, { dataset: { postId: '9' } }];
  } } } });
  out.order = { ids: w.ctx.pendingQueueOrder(), selector: w.log.selectors[0] };

  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e.stack); process.exit(1); });
"""


def section_js():
    w, html, script = schedule_script()
    functions = lift(script, JS_FUNCTIONS, consts=())
    r = run_node(JS_HARNESS, functions)

    check(r["idsOf"] == {"group": [4, 5, 6], "single": [7], "junk": [1, 3], "none": [],
                         "platforms": ["threads", "twitter"], "noplatform": []},
          f"idsOf/platformsOf: {r['idsOf']}")

    # Post Now
    p = r["postAll"]
    check(p["urls"] == ["/schedule/11/post-now", "/schedule/12/post-now", "/schedule/13/post-now"]
          and set(p["methods"]) == {"POST"}, f"Post Now must call every member, in order: {p['urls']}")
    check("linkedin, threads, twitter" in p["confirm"][0], f"the confirm names every platform: {p['confirm']}")
    check(p["btn"] == "✅ Posted!" and p["reloadsScheduled"] == 1 and not p["alerts"],
          f"a fully successful Post Now: {p}")
    p = r["postPartial"]
    check(len(p["urls"]) == 3, "a failure must not stop the remaining platforms")
    check(p["alerts"] and "Posted to 2 of 3" in p["alerts"][0] and "threads: rate limited" in p["alerts"][0],
          f"a partial failure must say what posted and what did not: {p['alerts']}")
    check(p["reloads"] == 1 and p["disabled"] is False and p["html"] == "ORIGINAL",
          f"a partial Post Now reloads (something was posted) and frees the button: {p}")
    p = r["postNone"]
    check(p["reloads"] == 0 and p["disabled"] is False and p["html"] == "ORIGINAL"
          and "threads: a" in p["alerts"][0] and "twitter: b" in p["alerts"][0],
          f"when every platform fails nothing reloads, the button is freed, each reason shows: {p}")
    p = r["postThrows"]
    check(len(p["urls"]) == 3 and "Error: offline" in p["alerts"][0] and p["reloads"] == 1,
          f"a network error on one platform is reported and the others are still sent: {p}")
    p = r["postSingle"]
    check(p["confirm"] == ["Post this threads content immediately?"] and p["urls"] == ["/schedule/21/post-now"],
          f"a lone entry behaves as before: {p}")
    check(r["postDeclined"] == {"fetches": 0, "html": "ORIGINAL", "disabled": False},
          f"declining sends nothing: {r['postDeclined']}")

    # Cancel / Delete / Retry
    c = r["cancelAll"]
    check(c["urls"] == ["/schedule/31/cancel", "/schedule/32/cancel"] and c["reloads"] == 1 and not c["alerts"]
          and "all 2 platforms" in c["confirm"][0], f"cancel a group: {c}")
    c = r["cancelPartial"]
    check("threads: already posted" in c["alerts"][0] and c["reloads"] == 1,
          f"a partly-cancelled group reports and reloads: {c}")
    c = r["cancelNone"]
    check(c["reloads"] == 0 and "threads: gone" in c["alerts"][0] and "twitter: gone too" in c["alerts"][0],
          f"nothing cancelled: report, no reload: {c}")
    c = r["cancelSingle"]
    check(c["confirm"] == ["Cancel this scheduled post?"] and c["urls"] == ["/schedule/33/cancel"]
          and c["reloads"] == 1, f"cancel a lone entry as before: {c}")
    d = r["deleteAll"]
    check(d["urls"] == ["/schedule/41/delete", "/schedule/42/delete", "/schedule/43/delete"]
          and d["reloads"] == 1 and "all 3 platforms" in d["confirm"][0], f"delete a group: {d}")
    t = r["retryAll"]
    check(t["urls"] == ["/schedule/51/retry", "/schedule/52/retry"] and t["alerts"] == ["Post successful!"]
          and t["reloads"] == 1, f"retry a group: {t}")
    t = r["retryPartial"]
    check("threads: still failing" in t["alerts"][0] and t["reloads"] == 1
          and t["disabled"] is False and t["html"] == "ORIGINAL", f"retry partly failing: {t}")
    t = r["retryNone"]
    check(t["reloads"] == 0 and t["disabled"] is False and t["html"] == "ORIGINAL", f"retry all failing: {t}")

    # bulk
    b = r["bulkDelete"]
    check(b["body"] == [{"post_ids": [1, 2, 3, 4, 5, 6]}], f"bulk delete must send every member id: {b['body']}")
    check("3 selected posts (6 platform entries)" in b["confirm"][0], f"bulk delete says both counts: {b['confirm']}")
    check("2 selected posts?" in r["bulkDeleteSingles"]["confirm"][0].replace("Delete 2 selected posts? ", "2 selected posts? ")
          and "platform entries" not in r["bulkDeleteSingles"]["confirm"][0],
          f"singles need no entry count: {r['bulkDeleteSingles']['confirm']}")
    b = r["bulkMove"]
    check(b["body"] == [{"post_ids": [1, 2, 4, 5, 6], "position": "top"}],
          f"bulk move sends every pending member id and skips non-pending rows: {b['body']}")
    check(b["confirm"] == ["Move 2 posts to the top of the queue?"],
          f"bulk move counts posts (rows), not platform entries: {b['confirm']}")

    # Edit Time
    check(str(r["openEdit"]["id"]) == "61" and r["openEdit"]["ids"] == "61,62,63" and r["openEditNoIds"]["ids"] == "64",
          f"the time modal remembers the row's ids: {r['openEdit']} {r['openEditNoIds']}")
    e = r["submitEdit"]
    check(e["url"] == "/schedule/61/edit" and ["ids", "61,62,63"] in e["fields"]
          and ["scheduled_for", "2099-01-01T09:30"] in e["fields"], f"Edit Time sends the ids: {e}")

    # Edit Content
    check(r["openContent"] == {"ids": "70,71", "platform": "🧵 Threads + 𝕏 Twitter"},
          f"the content modal names every platform: {r['openContent']}")
    check(r["openContentSingle"] == {"ids": "72", "platform": "💼 LinkedIn"}, f"single: {r['openContentSingle']}")
    s = r["contentStandalone"]
    check(len(s["calls"]) == 1 and s["calls"][0]["url"] == "/compose/post/70/edit"
          and ["post_ids", "70,71"] in s["calls"][0]["fields"] and ["content", "New copy"] in s["calls"][0]["fields"]
          and s["reloads"] == 1, f"standalone content goes to the card in one request: {s}")
    s = r["contentWithImage"]
    check([c["url"] for c in s["calls"]] == ["/compose/post/70/edit", "/compose/post/70/image"]
          and ["post_ids", "70,71"] in s["calls"][1]["fields"]
          and ["image_url", "https://img.test/new.png"] in s["calls"][1]["fields"] and s["reloads"] == 1,
          f"a changed image is saved to the whole card too: {s}")
    s = r["contentSocial"]
    check([c["url"] for c in s["calls"]] == ["/social/80/edit", "/social/81/edit", "/social/82/edit"]
          and s["reloads"] == 1, f"social posts are edited one at a time: {s}")
    s = r["contentSocialPartial"]
    check(len(s["urls"]) == 3 and "nope" in s["alerts"][0] and s["reloads"] == 0,
          f"a failed social edit says so and does not reload: {s}")
    check(r["contentFails"]["calls"] == 1 and "Content is required" in r["contentFails"]["alerts"][0]
          and r["contentFails"]["reloads"] == 0, f"failed content save: {r['contentFails']}")
    check(r["contentImageFails"]["calls"] == 2 and "bad image" in r["contentImageFails"]["alerts"][0]
          and r["contentImageFails"]["reloads"] == 0, f"failed image save: {r['contentImageFails']}")
    check(r["contentEmpty"]["calls"] == 0 and "cannot be empty" in r["contentEmpty"]["alerts"][0],
          f"empty content sends nothing: {r['contentEmpty']}")

    # expander and drag order
    x = r["expander"]
    check(x["opened"] == {"expanded": "true", "text": "▾ By platform", "hidden": [False, False]}
          and x["closed"] == {"expanded": "false", "text": "▸ By platform", "hidden": [True, True]},
          f"the expander shows and hides the per-platform rows: {x}")
    check(r["order"]["ids"] == [5, 2, 9] and r["order"]["selector"] == "tr.queue-group:not(.not-draggable)",
          f"the drag order is one id per post row: {r['order']}")

    # the wiring those functions depend on is really on the page
    for needle in ("draggable: 'tr.queue-group'", "oldDraggableIndex", "toggleQueueGroup(btn)",
                   "idsOf(btn)", "'edit-post-ids'", "id=\"edit-post-ids\"", "id=\"edit-content-post-ids\""):
        check(needle in html or needle in script, f"the schedule page is missing {needle}")

    print("js: group Post Now, Cancel, Retry, Delete, Edit Time, Edit Content, bulk delete and move "
          "reach every member and report partial failure")
    print("QUEUE_JS_OK")


# ---------------------------------------------------------------------------
# syntax: the scripts parse and nothing is declared twice
# ---------------------------------------------------------------------------

def section_syntax():
    node = node_path()
    w, html, _ = schedule_script()
    scripts = inline_scripts(html)
    work = tempfile.mkdtemp(prefix="queue_syntax_")
    try:
        for index, body in enumerate(scripts):
            path = os.path.join(work, f"script_{index}.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            run = subprocess.run([node, "--check", path], capture_output=True, text=True)
            check(run.returncode == 0,
                  f"inline script {index} on the schedule page does not parse:\n{run.stderr.strip()[:600]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # The page's scripts share one global scope, so a second declaration of the
    # same name silently replaces the first.
    source = "\n".join(scripts)
    names = re.findall(r"^(?:async\s+)?function (\w+)\(", source, re.M)
    names += re.findall(r"^(?:const|let) (\w+)\b", source, re.M)
    duplicated = sorted({n for n in names if names.count(n) > 1})
    check(not duplicated, f"declared more than once on the schedule page: {duplicated}")

    # every function the markup calls from an attribute exists
    called = set(re.findall(r'on(?:click|change|input)="(\w+)\(', html))
    missing = sorted(n for n in called if n not in names)
    check(not missing, f"the page calls functions that are not defined: {missing}")

    print(f"syntax: {len(scripts)} inline scripts parse; {len(names)} top-level names, none repeated")
    print("QUEUE_SYNTAX_OK")


SECTIONS = {
    "group": section_group, "page": section_page, "order": section_order,
    "edit": section_edit, "parity": section_parity, "js": section_js,
    "syntax": section_syntax,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_queue_groups.py {'|'.join(SECTIONS)}")
        sys.exit(2)
    SECTIONS[name]()

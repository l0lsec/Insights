"""Gates for Instagram collaborators and Reel people-tags.

    python scripts/check_ig_people.py client    IG_PEOPLE_CLIENT_OK
    python scripts/check_ig_people.py none      IG_PEOPLE_NONE_OK
    python scripts/check_ig_people.py refuse    IG_PEOPLE_REFUSE_OK
    python scripts/check_ig_people.py reject    IG_PEOPLE_REJECT_OK
    python scripts/check_ig_people.py persist   IG_PEOPLE_PERSIST_OK
    python scripts/check_ig_people.py publish   IG_PEOPLE_PUBLISH_OK
    python scripts/check_ig_people.py ui        IG_PEOPLE_UI_OK
    python scripts/check_ig_people.py js        IG_PEOPLE_JS_OK
    python scripts/check_ig_people.py queue     IG_PEOPLE_QUEUE_OK

Collaborators are Instagram accounts invited onto a post (they appear on it once
they accept) and, on a Reel, people tags are usernames only. Both are typed in
Compose, saved on the post, and have to reach Instagram exactly as entered by
every route a post can leave by: Post now, the whole-card publish and the
background scheduler. A post with none must be untouched.

``client`` and ``none`` replace ``requests`` inside instagram_client with a
recorder and read the HTTP requests the client would have sent. ``refuse`` and
``reject`` prove the two ways a post fails rather than going out without its
people. ``persist``, ``publish`` and ``queue`` drive the real routes, database
and publisher on a throwaway database with a recording Instagram client.
``ui`` and ``js`` read the rendered pages and run the page's own functions under
node. Nothing here posts anything, reads a real token or leaves the machine.
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    isolated_app, connect, check, RecordingClient,
)
import _ig_people_gate as g  # noqa: E402

VIDEO = "https://cdn.test/clip.mp4"
IMAGE = "https://cdn.test/photo.jpg"
IMAGE2 = "https://cdn.test/photo2.jpg"
CAROUSEL = [{"url": IMAGE, "kind": "image"}, {"url": IMAGE2, "kind": "image"}]
ENTERED = "@Amy, bob , AMY"            # what a person types
ENTERED_CLEAN = ["amy", "bob"]         # what must reach Instagram
TAGS_ENTERED = "@Cat,  Dan"
TAGS_CLEAN = ["cat", "dan"]


# ---------------------------------------------------------------------------
# shared rigs
# ---------------------------------------------------------------------------

def rig(accounts=1):
    """An isolated app with Instagram account(s) connected and a recording client.

    The real ``_instagram_publish_for_post`` stays registered, so what the
    recording client sees is what the app really passes to Instagram's client.
    """
    directory, database, web, publisher, client = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None
    web.get_image_for_post = lambda *a, **k: None   # never look for a stock photo
    ids = [connect(database, "instagram", f"ig-{n}", f"brand{n}") for n in range(1, accounts + 1)]
    recorder = RecordingClient("instagram")
    web.get_instagram_client = lambda: recorder
    publisher.get_instagram_client = lambda: recorder
    return database, web, publisher, client, ids, recorder


def token_of(n=1):
    return f"token-instagram-ig-{n}"


def make_row(database, account, fmt, content="Caption", platform="instagram"):
    """One saved Instagram post in the format ``fmt``; returns its id."""
    P = database.DB_PATH
    fields = {"reel": {}, "reel_card": {"video_url": VIDEO}, "feed_video": {"video_url": VIDEO},
              "image": {"image_url": IMAGE}, "carousel": {}, "story": {"image_url": IMAGE}}[fmt]
    pid = database.add_standalone_post("manual", "g", platform, content, db_path=P,
                                       account_id=account, **fields)
    if fmt == "reel":
        database.set_standalone_post_media(pid, "reel", [{"url": VIDEO, "kind": "video"}], db_path=P)
    elif fmt == "reel_card":      # a Reel whose video is the card's, the builder left empty
        database.set_standalone_post_media(pid, "reel", [], db_path=P)
    elif fmt == "carousel":
        database.set_standalone_post_media(pid, "carousel", CAROUSEL, db_path=P)
    elif fmt == "story":
        database.set_standalone_post_media(pid, "story", [{"url": IMAGE, "kind": "image"}], db_path=P)
    return pid


def set_people(client, pid, collaborators=ENTERED, tags=TAGS_ENTERED, post_ids=None):
    """Enter people the way the card does: through the route, as typed."""
    data = {"ig_collaborators": collaborators}
    if tags is not None:
        data["ig_reel_tags"] = tags
    if post_ids:
        data["post_ids"] = ",".join(map(str, post_ids))
    reply = client.post(f"/compose/post/{pid}/ig-people", data=data)
    check(reply.status_code == 200 and reply.get_json().get("success"),
          f"saving people failed: {reply.status_code} {reply.get_data(as_text=True)[:300]}")
    return reply.get_json()


def sig_of(client_call):
    graph, ig, restore = g.install_fake_graph()
    try:
        result = client_call(ig)
    finally:
        restore()
    return graph, result


# ---------------------------------------------------------------------------
# client: people reach the request exactly as entered
# ---------------------------------------------------------------------------

def section_client():
    # --- Reel: collaborators and tags, normalised, in the shape proven live
    graph, result = sig_of(lambda ig: ig.publish_reel_post(
        g.TOKEN, "cap", VIDEO, collaborators=["@ShowUpShowOut", " showupshowoutllc ", "SHOWUPSHOWOUT"],
        user_tags=["@Cat", {"username": "Dan"}, "cat"]))
    check(result.get("success"), f"a reel with people should publish: {result}")
    (created,) = graph.creations()
    check(created["media_type"] == "REELS" and created["video_url"] == VIDEO, f"reel container wrong: {created}")
    check(created["collaborators"] == '["showupshowout","showupshowoutllc"]',
          f"collaborators must be the compact JSON array proven live, deduped and lowercased; got {created.get('collaborators')!r}")
    check(created["user_tags"] == '[{"username":"cat"},{"username":"dan"}]',
          f"reel tags must be username-only objects; got {created.get('user_tags')!r}")
    check("x" not in json.loads(created["user_tags"])[0] and "y" not in json.loads(created["user_tags"])[0],
          "a reel tag must carry no x/y position")
    check(len(graph.publishes()) == 1, "the reel should be published once")
    check(result.get("permalink") == "https://instagram.test/p/x/", f"permalink not returned: {result}")

    # --- Reel: the cover frame rides along, and the people come after it
    graph, result = sig_of(lambda ig: ig.publish_reel_post(
        g.TOKEN, "cap", VIDEO, thumb_offset_ms=3600, collaborators=["amy"], user_tags=["cat"]))
    (created,) = graph.creations()
    check(result.get("success") and list(created)[-3:] == ["thumb_offset", "collaborators", "user_tags"]
          and created["thumb_offset"] == "3600",
          f"a Reel's cover frame, collaborators and tags must all be sent: {list(created.items())}")

    # --- Reel: three collaborators is the limit and is accepted
    graph, result = sig_of(lambda ig: ig.publish_reel_post(
        g.TOKEN, "cap", VIDEO, collaborators=["a.one", "b_two", "c3"]))
    check(result.get("success") and g.collaborators_in(graph.creations()[0]) == ["a.one", "b_two", "c3"],
          "three collaborators (the maximum) must be accepted and sent in order")

    # --- Reel with only tags / only collaborators: the other parameter is absent
    graph, _ = sig_of(lambda ig: ig.publish_reel_post(g.TOKEN, "cap", VIDEO, collaborators=["amy"]))
    check("collaborators" in graph.creations()[0] and "user_tags" not in graph.creations()[0],
          "collaborators alone must not add user_tags")
    graph, _ = sig_of(lambda ig: ig.publish_reel_post(g.TOKEN, "cap", VIDEO, user_tags=["cat"]))
    check("user_tags" in graph.creations()[0] and "collaborators" not in graph.creations()[0],
          "tags alone must not add collaborators")

    # --- Image: collaborators next to the existing photo tags, both intact
    graph, result = sig_of(lambda ig: ig.publish_image_post(
        g.TOKEN, "cap", IMAGE, user_tags=[{"username": "Bob", "x": 0.2, "y": 0.9}],
        collaborators=["@Amy", "bob"]))
    (created,) = graph.creations()
    check(result.get("success") and created["collaborators"] == '["amy","bob"]',
          f"image collaborators wrong: {created}")
    check(json.loads(created["user_tags"]) == [{"username": "Bob", "x": 0.2, "y": 0.9}],
          f"a photo's positioned tags must be unchanged by collaborators: {created.get('user_tags')!r}")

    # --- Carousel: on the parent container only, never on a child
    items = [{"url": IMAGE, "kind": "image"}, {"url": VIDEO, "kind": "video"}]
    graph, result = sig_of(lambda ig: ig.publish_carousel_post(
        g.TOKEN, "cap", items, collaborators=["@Amy", "bob"]))
    created = graph.creations()
    check(result.get("success") and len(created) == 3, f"carousel should make 2 children + a parent: {len(created)}")
    check(all("collaborators" not in c for c in created[:2]),
          "carousel children do not take collaborators; they must go on the parent only")
    check(created[2]["media_type"] == "CAROUSEL" and created[2]["collaborators"] == '["amy","bob"]',
          f"carousel parent must carry the collaborators: {created[2]}")

    # --- A refused reel/carousel/image never reaches the API with the wrong list:
    # positive control that dedupe happens before the limit check
    graph, result = sig_of(lambda ig: ig.publish_reel_post(
        g.TOKEN, "cap", VIDEO, collaborators=["a", "A", "@a", "b", "c", "C"]))
    check(result.get("success") and g.collaborators_in(graph.creations()[0]) == ["a", "b", "c"],
          "duplicates must collapse before the maximum is applied")

    print("client: collaborators and Reel tags reach the request exactly as entered, "
          "on the reel, the image and the carousel parent")
    print("IG_PEOPLE_CLIENT_OK")


# ---------------------------------------------------------------------------
# none: a post with no people sends today's requests, byte for byte
# ---------------------------------------------------------------------------

# Captured from the client before collaborators existed (graph.instagram.com, the
# requests it made, in order, with each query parameter in the order it was sent).
def _base(*creation_params):
    return [["POST", f"{g.API}/me/media", list(creation_params)]]


def _tail(container):
    return [
        ["GET", f"{g.API}/{container}", [["fields", "status_code,status"], ["access_token", "tok-ig"]]],
        ["POST", f"{g.API}/me/media_publish", [["creation_id", container], ["access_token", "tok-ig"]]],
        ["GET", f"{g.API}/p1", [["fields", "permalink,shortcode"], ["access_token", "tok-ig"]]],
    ]


GOLDEN = {
    "image": _base(["caption", "cap"], ["image_url", IMAGE], ["access_token", "tok-ig"]) + _tail("c1"),
    "image_tags": _base(
        ["caption", "cap"], ["image_url", IMAGE], ["access_token", "tok-ig"],
        ["user_tags", '[{"username": "Bob", "x": 0.2, "y": 0.9}]']) + _tail("c1"),
    "carousel": [
        ["POST", f"{g.API}/me/media", [["image_url", IMAGE], ["is_carousel_item", "true"], ["access_token", "tok-ig"]]],
        ["POST", f"{g.API}/me/media", [["media_type", "VIDEO"], ["video_url", VIDEO], ["is_carousel_item", "true"], ["access_token", "tok-ig"]]],
        ["POST", f"{g.API}/me/media", [["media_type", "CAROUSEL"], ["children", "c1,c2"], ["caption", "cap"], ["access_token", "tok-ig"]]],
    ] + _tail("c3"),
    "reel": _base(["media_type", "REELS"], ["video_url", VIDEO], ["caption", "cap"],
                  ["share_to_feed", "true"], ["access_token", "tok-ig"]) + _tail("c1"),
    "reel_nofeed": _base(["media_type", "REELS"], ["video_url", VIDEO], ["caption", "cap"],
                         ["share_to_feed", "false"], ["access_token", "tok-ig"]) + _tail("c1"),
}

NO_PEOPLE = [None, [], "", [" ", ""], "  \n, ,"]


def section_none():
    calls = {
        "image": lambda ig, **kw: ig.publish_image_post(g.TOKEN, "cap", IMAGE, **kw),
        "image_tags": lambda ig, **kw: ig.publish_image_post(
            g.TOKEN, "cap", IMAGE, user_tags=[{"username": "@Bob", "x": 0.2, "y": 0.9}], **kw),
        "carousel": lambda ig, **kw: ig.publish_carousel_post(
            g.TOKEN, "cap", [{"url": IMAGE, "kind": "image"}, {"url": VIDEO, "kind": "video"}], **kw),
        "reel": lambda ig, **kw: ig.publish_reel_post(g.TOKEN, "cap", VIDEO, **kw),
        "reel_nofeed": lambda ig, **kw: ig.publish_reel_post(g.TOKEN, "cap", VIDEO, share_to_feed=False, **kw),
    }
    # the empty spellings each form field can take; reels also take empty tags
    for name, call in calls.items():
        graph, result = sig_of(lambda ig: call(ig))
        check(result.get("success"), f"{name}: the baseline call should publish: {result}")
        check(graph.signature() == GOLDEN[name],
              f"{name}: requests changed for a post with no people:\n  want {GOLDEN[name]}\n  got  {graph.signature()}")
        for empty in NO_PEOPLE:
            graph, result = sig_of(lambda ig: call(ig, collaborators=empty))
            check(graph.signature() == GOLDEN[name],
                  f"{name}: collaborators={empty!r} must send nothing extra, got {graph.signature()}")
        if name.startswith("reel"):
            for empty in NO_PEOPLE:
                graph, _ = sig_of(lambda ig: call(ig, user_tags=empty))
                check(graph.signature() == GOLDEN[name],
                      f"{name}: user_tags={empty!r} must send nothing extra, got {graph.signature()}")

    # positive control: the same call WITH a collaborator is different, so equality above means something
    graph, _ = sig_of(lambda ig: calls["reel"](ig, collaborators=["amy"]))
    check(graph.signature() != GOLDEN["reel"], "control: a collaborator must change the request")

    # the dispatcher: a post without people calls the client the way it always did
    database, web, publisher, client, ids, ig = rig()
    P = database.DB_PATH
    expect = {
        "reel": {"kind": "video", "token": token_of(), "text": "Caption", "video": VIDEO},
        "feed_video": {"kind": "video", "token": token_of(), "text": "Caption", "video": VIDEO},
        "image": {"kind": "image", "token": token_of(), "text": "Caption", "image": IMAGE},
        "carousel": {"kind": "carousel", "token": token_of(), "text": "Caption", "items": CAROUSEL},
    }
    for fmt, want in expect.items():
        pid = make_row(database, ids[0], fmt)
        post = database.get_standalone_post(pid, db_path=P)
        ig.calls.clear()
        result = publisher.publish("instagram", ids[0], content="Caption", image_url=post["image_url"],
                                   video_url=post["video_url"], standalone_post_id=pid)
        check(result["success"], f"{fmt}: publish failed: {result}")
        check(ig.calls == [want], f"{fmt}: a post with no people must call the client as before:\n  want {[want]}\n  got  {ig.calls}")
        row = database.get_standalone_post(pid, db_path=P)
        check(row["ig_collaborators"] is None and row["ig_reel_tags"] is None,
              f"{fmt}: a new post must store no people (NULL), got {row['ig_collaborators']!r}")

    print("none: a post with no collaborators and no tags sends byte-identical requests "
          "(5 request shapes x every empty spelling) and the same client calls")
    print("IG_PEOPLE_NONE_OK")


# ---------------------------------------------------------------------------
# refuse: bad input stops before anything is created
# ---------------------------------------------------------------------------

BAD_HANDLES = [
    "john doe",            # a space: refuse, do not split into two accounts
    "bad!name",
    "a" * 31,              # one over the limit
    "@",                   # nothing after the @
    "a/b",
    "jane@example.com",
    "@@twice",
    "naïve",
]


def section_refuse():
    # --- at the client: refused with the offending text, and not one request made
    for bad in BAD_HANDLES:
        for label, call in (
            ("image", lambda ig, v: ig.publish_image_post(g.TOKEN, "c", IMAGE, collaborators=v)),
            ("carousel", lambda ig, v: ig.publish_carousel_post(
                g.TOKEN, "c", [{"url": IMAGE, "kind": "image"}, {"url": IMAGE2, "kind": "image"}], collaborators=v)),
            ("reel", lambda ig, v: ig.publish_reel_post(g.TOKEN, "c", VIDEO, collaborators=v)),
            ("reel tags", lambda ig, v: ig.publish_reel_post(g.TOKEN, "c", VIDEO, user_tags=v)),
        ):
            graph, result = sig_of(lambda ig: call(ig, ["amy", bad]))
            check(not result.get("success") and result.get("guard_error"),
                  f"{label}: {bad!r} must be refused as permanent: {result}")
            check(bad.strip() in result["friendly"] or repr(bad) in result["friendly"],
                  f"{label}: the refusal must name {bad!r}: {result['friendly']}")
            check(graph.requests == [],
                  f"{label}: {bad!r} was refused but {len(graph.requests)} request(s) were made first")

    # --- too many collaborators / tags
    four = ["a1", "b2", "c3", "d4"]
    for label, call in (
        ("image", lambda ig: ig.publish_image_post(g.TOKEN, "c", IMAGE, collaborators=four)),
        ("carousel", lambda ig: ig.publish_carousel_post(
            g.TOKEN, "c", [{"url": IMAGE, "kind": "image"}, {"url": IMAGE2, "kind": "image"}], collaborators=four)),
        ("reel", lambda ig: ig.publish_reel_post(g.TOKEN, "c", VIDEO, collaborators=four)),
    ):
        graph, result = sig_of(call)
        check(not result.get("success") and "at most 3" in result["friendly"] and "@d4" in result["friendly"],
              f"{label}: four collaborators must be refused saying the maximum and who: {result}")
        check(graph.requests == [], f"{label}: too many collaborators made {len(graph.requests)} request(s)")
    graph, result = sig_of(lambda ig: ig.publish_reel_post(
        g.TOKEN, "c", VIDEO, user_tags=[f"u{n}" for n in range(21)]))
    check(not result.get("success") and "at most 20" in result["friendly"] and graph.requests == [],
          f"21 tags must be refused before any request: {result}")
    # control: exactly 20 tags and exactly 3 collaborators go through
    graph, result = sig_of(lambda ig: ig.publish_reel_post(
        g.TOKEN, "c", VIDEO, collaborators=["a1", "b2", "c3"], user_tags=[f"u{n}" for n in range(20)]))
    check(result.get("success"), f"control: 3 collaborators and 20 tags are allowed: {result}")

    # --- the dispatcher: a stored bad list fails the post before the client is called
    database, web, publisher, client, ids, ig = rig()
    P = database.DB_PATH
    for stored, needle in ((["john doe"], "john doe"), (["a1", "b2", "c3", "d4"], "at most 3")):
        pid = make_row(database, ids[0], "reel")
        database.set_standalone_post_ig_people(pid, stored, None, db_path=P)
        ig.calls.clear()
        post = database.get_standalone_post(pid, db_path=P)
        result = publisher.publish("instagram", ids[0], content="Caption", image_url=None,
                                   video_url=post["video_url"], standalone_post_id=pid)
        check(not result["success"] and result["permanent"] and needle in result["error"],
              f"a stored {stored} must fail the post permanently naming the problem: {result}")
        check(ig.calls == [], f"{stored}: the client was called despite the bad list: {ig.calls}")
    pid = make_row(database, ids[0], "reel")
    database.set_standalone_post_ig_people(pid, ["amy"], ["bad tag"], db_path=P)
    post = database.get_standalone_post(pid, db_path=P)
    result = publisher.publish("instagram", ids[0], content="Caption", image_url=None,
                               video_url=post["video_url"], standalone_post_id=pid)
    check(not result["success"] and "bad tag" in result["error"] and ig.calls == [],
          f"a stored bad reel tag must fail the post before the client: {result} {ig.calls}")

    # control: Reel tags saved on a post that is not a Reel are not sent, so a bad one there
    # does not stop the photo going out (the same bad tag fails a Reel, just above)
    photo = make_row(database, ids[0], "image")
    database.set_standalone_post_ig_people(photo, ["amy"], ["bad tag"], db_path=P)
    ig.calls.clear()
    result = publisher.publish("instagram", ids[0], content="Caption", image_url=IMAGE, standalone_post_id=photo)
    check(result["success"] and ig.calls == [want_call("image", ["amy"], None)],
          f"dormant reel tags must not fail or reach a photo post: {result} {ig.calls}")

    # --- the routes: refuse, say why, and change nothing
    pid = make_row(database, ids[0], "reel")
    set_people(client, pid, "carol", "erin")
    before = dict(database.get_standalone_post(pid, db_path=P))
    for payload, needle in (
        ({"ig_collaborators": "john doe"}, "john doe"),
        ({"ig_collaborators": "a1,b2,c3,d4"}, "at most 3"),
        ({"ig_collaborators": "bad!"}, "bad!"),
        ({"ig_reel_tags": "x y"}, "x y"),
        ({"ig_collaborators": json.dumps(["ok", "no way"])}, "no way"),
    ):
        reply = client.post(f"/compose/post/{pid}/ig-people", data=payload)
        check(reply.status_code == 400 and needle in reply.get_json()["error"],
              f"{payload}: expected a 400 naming {needle!r}, got {reply.status_code} {reply.get_data(as_text=True)[:200]}")
        check(dict(database.get_standalone_post(pid, db_path=P)) == before,
              f"{payload}: a refused request must leave the saved post exactly as it was")
    reply = client.post(f"/compose/post/{pid}/ig-people", data={})
    check(reply.status_code == 400, "a request with neither field is refused")

    # create: a bad handle writes no row at all
    count = lambda: len(database.list_standalone_posts(db_path=P))
    base = count()
    for payload in ({"ig_collaborators": "john doe"}, {"ig_collaborators": "a1,b2,c3,d4"},
                    {"ig_reel_tags": "x y"}):
        reply = client.post("/compose/post/create",
                            data={"content": "Hello", "targets": [f"instagram:{ids[0]}"], **payload})
        check(reply.status_code == 400, f"create with {payload} must be refused, got {reply.status_code}")
        check(count() == base, f"create with {payload} left {count() - base} row(s) behind")
    # collaborators without an Instagram target are refused, not silently dropped
    reply = client.post("/compose/post/create", data={"content": "Hello", "targets": ["threads"],
                                                      "ig_collaborators": "amy"})
    check(reply.status_code == 400 and "Instagram" in reply.get_json()["error"] and count() == base,
          f"collaborators with no Instagram target must be refused: {reply.get_json()}")
    # control: the same create with good input works
    reply = client.post("/compose/post/create", data={"content": "Hello", "targets": [f"instagram:{ids[0]}"],
                                                      "ig_collaborators": "amy"})
    check(reply.status_code == 200 and count() == base + 1, "control: valid people must create the post")

    # a card with no Instagram row has nowhere to put them
    other = database.add_standalone_post("manual", "g", "threads", "Only threads", db_path=P)
    reply = client.post(f"/compose/post/{other}/ig-people", data={"ig_collaborators": "amy"})
    check(reply.status_code == 400 and "Instagram" in reply.get_json()["error"],
          f"people on a card that does not go to Instagram must be refused: {reply.get_json()}")

    # a Story cannot carry collaborators
    story = make_row(database, ids[0], "story")
    reply = client.post(f"/compose/post/{story}/ig-people", data={"ig_collaborators": "amy"})
    check(reply.status_code == 400 and "Stories" in reply.get_json()["error"],
          f"collaborators on a Story must be refused: {reply.get_json()}")
    reel = make_row(database, ids[0], "reel")
    set_people(client, reel, "amy", None)
    fd = {"ig_post_type": "story", "media_items": json.dumps([{"url": IMAGE, "kind": "image"}])}
    reply = client.post(f"/compose/post/{reel}/media", data=fd)
    check(reply.status_code == 400 and "Stories" in reply.get_json()["error"]
          and database.get_standalone_post(reel, db_path=P)["ig_post_type"] == "reel",
          f"switching a post with collaborators to Story must be refused and change nothing: {reply.get_data(as_text=True)[:200]}")

    print("refuse: a handle with spaces, bad characters, too many collaborators or tags is refused "
          "before any request or row, at the client, the dispatcher and the routes")
    print("IG_PEOPLE_REFUSE_OK")


# ---------------------------------------------------------------------------
# reject: Instagram refuses the people -> the post fails, by name, unpublished
# ---------------------------------------------------------------------------

def section_reject():
    def has_people(params):
        return "collaborators" in params

    # --- the payload names one collaborator: only that one is blamed
    graph, ig, restore = g.install_fake_graph(
        reject=lambda p: g.bad_user_error("zed") if has_people(p) else None)
    try:
        result = ig.publish_reel_post(g.TOKEN, "c", VIDEO, collaborators=["amy", "zed"])
    finally:
        restore()
    check(not result["success"], "a rejected collaborator must fail the post")
    check("@zed" in result["friendly"] and "@amy" not in result["friendly"],
          f"the failure must name the offender and not blame the others: {result['friendly']}")
    check(result.get("people_error") and result.get("guard_error"),
          f"a rejected collaborator is a permanent, people-specific failure: {result}")
    check(graph.publishes() == [], "nothing may be published when a collaborator is rejected")
    check(len(graph.creations()) == 1 and has_people(graph.creations()[0]),
          f"the post must not be retried without its collaborators: {graph.creations()}")
    check(not any(u.endswith("/media_publish") for _m, u, _p in graph.requests), "media_publish was called")

    # --- the payload only says a user was invalid: every account sent is named
    graph, ig, restore = g.install_fake_graph(
        reject=lambda p: g.bad_user_error() if has_people(p) else None)
    try:
        result = ig.publish_image_post(g.TOKEN, "c", IMAGE, collaborators=["amy", "zed"])
    finally:
        restore()
    check(not result["success"] and "@amy" in result["friendly"] and "@zed" in result["friendly"]
          and "did not say which" in result["friendly"],
          f"an unattributed rejection must list the candidates: {result['friendly']}")
    check(graph.publishes() == [] and len(graph.creations()) == 1, "an image must not publish or retry")

    # --- a single collaborator is always the named one
    graph, ig, restore = g.install_fake_graph(
        reject=lambda p: g.bad_user_error() if has_people(p) else None)
    try:
        result = ig.publish_reel_post(g.TOKEN, "c", VIDEO, collaborators=["solo"])
    finally:
        restore()
    check("@solo" in result["friendly"] and "did not say which" not in result["friendly"],
          f"with one collaborator the answer is exact: {result['friendly']}")

    # --- a carousel: children succeed, the parent is refused, nothing publishes
    graph, ig, restore = g.install_fake_graph(
        reject=lambda p: g.bad_user_error("zed") if p.get("media_type") == "CAROUSEL" and has_people(p) else None)
    try:
        result = ig.publish_carousel_post(g.TOKEN, "c", CAROUSEL, collaborators=["zed"])
    finally:
        restore()
    check(not result["success"] and "@zed" in result["friendly"] and graph.publishes() == []
          and len(graph.creations()) == 3, f"carousel parent rejection: {result} {len(graph.creations())}")

    # --- a bad reel tag is named as a tagged account
    graph, ig, restore = g.install_fake_graph(
        reject=lambda p: g.bad_user_error("tagbad") if "user_tags" in p else None)
    try:
        result = ig.publish_reel_post(g.TOKEN, "c", VIDEO, user_tags=["ok", "tagbad"])
    finally:
        restore()
    check(not result["success"] and "tagged account @tagbad" in result["friendly"]
          and graph.publishes() == [], f"a rejected tag must be named: {result}")

    # --- control: a media problem with collaborators attached is NOT blamed on them
    media_error = {"error": {"message": "Media download failed", "code": 9004, "error_subcode": 2207003}}
    graph, ig, restore = g.install_fake_graph(reject=lambda p: media_error)
    try:
        result = ig.publish_reel_post(g.TOKEN, "c", VIDEO, collaborators=["amy"])
    finally:
        restore()
    check(not result["success"] and "@amy" not in result["friendly"]
          and "could not download" in result["friendly"] and not result.get("people_error"),
          f"a media failure must keep its own explanation: {result}")
    # ... and a rate limit likewise
    limit = {"error": {"message": "limit", "code": 4}}
    graph, ig, restore = g.install_fake_graph(reject=lambda p: limit)
    try:
        result = ig.publish_reel_post(g.TOKEN, "c", VIDEO, collaborators=["amy"])
    finally:
        restore()
    check("rate limit" in result["friendly"] and "@amy" not in result["friendly"], f"rate limit mis-blamed: {result}")

    # --- end to end: real publisher + dispatcher + client, via Post now and the scheduler
    database, web, publisher, client, ids, _ = rig()
    P = database.DB_PATH
    import instagram_client
    for n, fmt in enumerate(("reel", "image", "carousel")):
        graph, real, restore = g.install_fake_graph(
            reject=lambda p: g.bad_user_error("zed") if has_people(p) else None)
        web.get_instagram_client = lambda real=real: real
        publisher.get_instagram_client = lambda real=real: real
        try:
            pid = make_row(database, ids[0], fmt)
            set_people(client, pid, "amy, zed", None)
            reply = client.post(f"/compose/post/{pid}/instagram")
            body = reply.get_json()
            check(reply.status_code == 400 and not body["success"] and "@zed" in body["error"],
                  f"{fmt}: Post now must report the rejected collaborator by name: {reply.status_code} {body}")
            check(graph.publishes() == [], f"{fmt}: Post now published despite the rejection")
            check(not database.get_standalone_post(pid, db_path=P)["used"],
                  f"{fmt}: a rejected post must not be marked used")

            sid = database.add_scheduled_post(
                scheduled_for="2000-01-01T00:00:00", post_type="standalone", standalone_post_id=pid,
                platform="instagram", account_id=ids[0], db_path=P)
            due = [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid]
            check(due, f"{fmt}: the queued post should be due")
            result = web._publish_scheduled_entry(due[0])
            check(not result["success"] and "@zed" in result["error"] and result["permanent"],
                  f"{fmt}: the scheduler must fail it permanently, naming the collaborator: {result}")
            check(graph.publishes() == [], f"{fmt}: the scheduler published despite the rejection")
        finally:
            restore()

    print("reject: a rejected collaborator fails the post naming the username, publishes nothing, "
          "is never retried without them, and is permanent for the scheduler")
    print("IG_PEOPLE_REJECT_OK")


# ---------------------------------------------------------------------------
# persist: columns, migration, routes, card-wide
# ---------------------------------------------------------------------------

def section_persist():
    import tempfile
    directory, database, web, publisher, client = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None

    # --- an existing database without the columns migrates, keeps every row, repeats cleanly
    path = os.path.join(tempfile.mkdtemp(prefix="ig_people_old_"), "old.db")
    database.init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO standalone_posts (source_type, source_content, platform, content, "
                     "image_url, created_at, used, repost, ig_post_type, media_items, ig_user_tags, video_url) "
                     "VALUES ('manual','g','instagram','Old post', ?, '2026-01-01T00:00:00', 0, 0, "
                     "'reel', ?, ?, ?)", (IMAGE, json.dumps(CAROUSEL), json.dumps([{"username": "old", "x": .5, "y": .5}]), VIDEO))
        # two rows that hold a Reel's collaborators where the first version kept them, inside
        # ig_reel_options: collaborators alone, and collaborators beside a cover frame
        for name, options in (("L1", '{"collaborators": ["Legacy.One", "legacy.two", "@LEGACY.one"]}'),
                              ("L2", '{"thumb_offset_ms": 1200, "collaborators": ["cover.person"]}')):
            conn.execute("INSERT INTO standalone_posts (source_type, source_content, platform, content, "
                         "created_at, used, repost, ig_reel_options) VALUES ('manual','g','instagram', ?, "
                         "'2026-01-01T00:00:00', 0, 0, ?)", (name, options))
        conn.commit()
        columns = [r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")]
        keep = [c for c in columns if c not in ("ig_collaborators", "ig_reel_tags")]
        # rebuild the table as it was before this change: same columns, minus the two
        conn.execute(f"CREATE TABLE standalone_old AS SELECT {', '.join(keep)} FROM standalone_posts")
        conn.execute("DROP TABLE standalone_posts")
        conn.execute("ALTER TABLE standalone_old RENAME TO standalone_posts")
        conn.commit()
        check("ig_collaborators" not in [r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")],
              "the old-schema fixture should lack the new columns")
        before = tuple(conn.execute(f"SELECT {', '.join(keep)} FROM standalone_posts WHERE content = 'Old post'").fetchone())
    database.init_db(path)

    def snapshot():
        with sqlite3.connect(path) as conn:
            return [tuple(r) for r in conn.execute(
                "SELECT content, ig_collaborators, ig_reel_tags, ig_reel_options FROM standalone_posts ORDER BY id")]

    first = snapshot()
    database.init_db(path)   # twice: the migration must be repeatable
    check(snapshot() == first, "running the migration again must change nothing")
    with sqlite3.connect(path) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")]
        check(columns.count("ig_collaborators") == 1 and columns.count("ig_reel_tags") == 1,
              f"each new column must exist exactly once after migrating twice: {columns}")
        after = tuple(conn.execute(f"SELECT {', '.join(keep)} FROM standalone_posts WHERE content = 'Old post'").fetchone())
        check(before == after, "migrating must not change an existing row")
        check(conn.execute("SELECT ig_collaborators, ig_reel_tags FROM standalone_posts WHERE content = 'Old post'")
              .fetchone() == (None, None), "existing rows must start with no people")
    by_name = {row[0]: row for row in first}
    check(json.loads(by_name["L1"][1]) == ["legacy.one", "legacy.two"] and by_name["L1"][3] is None,
          f"collaborators kept in a Reel's options must move to the shared column, cleaned: {by_name['L1']}")
    check(json.loads(by_name["L2"][1]) == ["cover.person"] and json.loads(by_name["L2"][3]) == {"thumb_offset_ms": 1200},
          f"the cover frame must stay where it was: {by_name['L2']}")
    # a row that has both (the two features met): the new column wins and the old copy is dropped
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE standalone_posts SET ig_collaborators = '[\"new.place\"]', "
                     "ig_reel_options = '{\"thumb_offset_ms\": 50, \"collaborators\": [\"old.place\"]}' "
                     "WHERE content = 'L2'")
        conn.commit()
    database.init_db(path)
    both = {row[0]: row for row in snapshot()}["L2"]
    check(json.loads(both[1]) == ["new.place"] and json.loads(both[3]) == {"thumb_offset_ms": 50},
          f"when both hold collaborators the new column must win: {both}")
    # the queue's joined queries still read, and expose the columns
    sid = database.add_scheduled_post("2099-01-01T00:00:00", "standalone", standalone_post_id=1,
                                      platform="instagram", db_path=path)
    for label, row in (("get_scheduled_post", database.get_scheduled_post(sid, db_path=path)),
                       ("list_scheduled_posts", database.list_scheduled_posts(db_path=path)[0])):
        check("standalone_ig_collaborators" in row.keys() and "standalone_ig_reel_tags" in row.keys()
              and "standalone_ig_post_type" in row.keys(), f"{label} must expose the new columns")
    sid_due = database.add_scheduled_post("2000-01-01T00:00:00", "standalone", standalone_post_id=1,
                                          platform="instagram", db_path=path)
    due = database.get_pending_scheduled_posts(db_path=path)
    check(due and "standalone_ig_collaborators" in due[0].keys(),
          "get_pending_scheduled_posts (the scheduler's query) must expose the new columns")

    # --- routes round-trip, as entered -> normalised -> stored
    database, web, publisher, client, ids, ig = rig(accounts=2)
    P = database.DB_PATH
    pid = make_row(database, ids[0], "reel")
    reply = set_people(client, pid)
    check(reply["ig_collaborators"] == ENTERED_CLEAN and reply["ig_reel_tags"] == TAGS_CLEAN,
          f"the route must answer with the cleaned lists: {reply}")
    row = database.get_standalone_post(pid, db_path=P)
    check(json.loads(row["ig_collaborators"]) == ENTERED_CLEAN and json.loads(row["ig_reel_tags"]) == TAGS_CLEAN,
          f"the cleaned lists must be stored as JSON: {row['ig_collaborators']!r} {row['ig_reel_tags']!r}")
    # a field left out is left alone; one sent empty is cleared
    set_people(client, pid, "zoe", None)
    row = database.get_standalone_post(pid, db_path=P)
    check(json.loads(row["ig_collaborators"]) == ["zoe"] and json.loads(row["ig_reel_tags"]) == TAGS_CLEAN,
          "saving only collaborators must leave the tags as they were")
    set_people(client, pid, "", "")
    row = database.get_standalone_post(pid, db_path=P)
    check(row["ig_collaborators"] is None and row["ig_reel_tags"] is None,
          "empty fields must clear both, back to NULL")
    # JSON body and JSON-list string forms are accepted too
    reply = client.post(f"/compose/post/{pid}/ig-people", json={"ig_collaborators": ["@Amy", "bob"]})
    check(reply.status_code == 200 and reply.get_json()["ig_collaborators"] == ENTERED_CLEAN, "a JSON body must work")
    reply = client.post(f"/compose/post/{pid}/ig-people", data={"ig_collaborators": json.dumps(["Carl"])})
    check(reply.status_code == 200 and reply.get_json()["ig_collaborators"] == ["carl"], "a JSON-list string must work")

    # --- card-wide: every Instagram row of the card, no other platform's
    a = make_row(database, ids[0], "image", content="Shared copy")
    b = make_row(database, ids[1], "image", content="Shared copy")
    t = database.add_standalone_post("manual", "g", "threads", "Shared copy", image_url=IMAGE, db_path=P)
    set_people(client, a, ENTERED, None, post_ids=[a, b, t])
    ra, rb, rt = (database.get_standalone_post(i, db_path=P) for i in (a, b, t))
    check(json.loads(ra["ig_collaborators"]) == ENTERED_CLEAN and json.loads(rb["ig_collaborators"]) == ENTERED_CLEAN,
          "both Instagram accounts of a card must get the collaborators")
    check(rt["ig_collaborators"] is None, "another platform's row must not get Instagram collaborators")

    # --- the older Reel-options route writes the SAME collaborators, card-wide, and keeps the
    # cover frame to itself; a key it is not sent stays as it was; an empty body clears both
    c1 = make_row(database, ids[0], "reel_card", content="Cover copy")
    c2 = make_row(database, ids[1], "reel_card", content="Cover copy")
    ct = database.add_standalone_post("manual", "g", "threads", "Cover copy", video_url=VIDEO, db_path=P)
    both_ids = f"{c1},{c2},{ct}"
    reply = client.post(f"/compose/post/{c1}/reel-options", json={
        "thumb_offset_ms": 2500, "collaborators": "Amy, bob", "post_ids": both_ids})
    check(reply.status_code == 200 and reply.get_json()["reel_options"] ==
          {"thumb_offset_ms": 2500, "collaborators": ["amy", "bob"]}, f"reel-options reply: {reply.get_json()}")
    r1, r2, rt = (database.get_standalone_post(i, db_path=P) for i in (c1, c2, ct))
    check(json.loads(r1["ig_collaborators"]) == ["amy", "bob"] and json.loads(r2["ig_collaborators"]) == ["amy", "bob"]
          and rt["ig_collaborators"] is None,
          "reel-options must write the collaborators to every Instagram row of the card and no other")
    check(json.loads(r1["ig_reel_options"]) == {"thumb_offset_ms": 2500} and r2["ig_reel_options"] is None,
          f"the cover frame is stored on the addressed row only, without a second copy of the collaborators: "
          f"{r1['ig_reel_options']!r} {r2['ig_reel_options']!r}")
    card = client.post(f"/compose/post/{c1}/card", data={"post_ids": both_ids}).get_json()["html"]
    check('value="@amy, @bob"' in card, "the card must show collaborators set through reel-options")
    client.post(f"/compose/post/{c1}/reel-options", json={"thumb_offset_ms": 100, "post_ids": both_ids})
    r1 = database.get_standalone_post(c1, db_path=P)
    check(json.loads(r1["ig_collaborators"]) == ["amy", "bob"] and json.loads(r1["ig_reel_options"]) == {"thumb_offset_ms": 100},
          "sending only a cover frame must leave the collaborators alone")
    client.post(f"/compose/post/{c1}/reel-options", json={"collaborators": "", "post_ids": both_ids})
    r1, r2 = (database.get_standalone_post(i, db_path=P) for i in (c1, c2))
    check(r1["ig_collaborators"] is None and r2["ig_collaborators"] is None
          and json.loads(r1["ig_reel_options"]) == {"thumb_offset_ms": 100},
          "sending blank collaborators must clear them card-wide and keep the cover frame")
    set_people(client, c1, "again", None, post_ids=[c1, c2])
    client.post(f"/compose/post/{c1}/reel-options", json={"post_ids": both_ids})
    r1, r2 = (database.get_standalone_post(i, db_path=P) for i in (c1, c2))
    check(r1["ig_collaborators"] is None and r2["ig_collaborators"] is None and r1["ig_reel_options"] is None,
          "an empty body must clear the cover frame and the collaborators")
    reply = client.post(f"/compose/post/{ct}/reel-options", json={"collaborators": "amy"})
    check(reply.status_code == 400 and "Instagram" in reply.get_json()["error"],
          f"reel-options collaborators on a post that is not Instagram must be refused: {reply.get_json()}")
    story = make_row(database, ids[0], "story")
    reply = client.post(f"/compose/post/{story}/reel-options", json={"collaborators": "amy"})
    check(reply.status_code == 400 and "Stories" in reply.get_json()["error"],
          f"reel-options collaborators on a Story must be refused: {reply.get_json()}")
    for bad in ({"collaborators": "john doe"}, {"collaborators": "a1,b2,c3,d4"}):
        reply = client.post(f"/compose/post/{c1}/reel-options", json=bad)
        check(reply.status_code == 400 and database.get_standalone_post(c1, db_path=P)["ig_collaborators"] is None,
              f"reel-options must refuse {bad} and change nothing: {reply.get_json()}")

    # --- ticking a second Instagram account (or bulk Add Platform) carries the card's people
    reply = client.post(f"/compose/post/{a}/platform", data={"platform": "instagram", "action": "add",
                                                              "account_id": str(ids[1]), "post_ids": f"{a},{b}"})
    check(reply.status_code == 400 and "already" in reply.get_json()["error"], "control: that account is already on the card")
    solo = make_row(database, ids[0], "image", content="Solo copy")
    set_people(client, solo, "keep, these")
    reply = client.post(f"/compose/post/{solo}/platform", data={"platform": "instagram", "action": "add",
                                                                 "account_id": str(ids[1]), "post_ids": f"{solo}"})
    check(reply.status_code == 200, f"adding the second account failed: {reply.get_data(as_text=True)[:200]}")
    new_id = reply.get_json()["post_id"]
    nr = database.get_standalone_post(new_id, db_path=P)
    check(json.loads(nr["ig_collaborators"] or "[]") == ["keep", "these"],
          f"a second Instagram account added to a card must carry its collaborators, got {nr['ig_collaborators']!r}")
    # control: Threads added to the same card gets no Instagram people
    th = client.post(f"/compose/post/{solo}/platform", data={"platform": "threads", "action": "add",
                                                              "post_ids": f"{solo}"}).get_json()["post_id"]
    check(database.get_standalone_post(th, db_path=P)["ig_collaborators"] is None,
          "control: a Threads row added to the card must carry no Instagram people")
    # bulk Add Platform
    bulk_src = make_row(database, ids[0], "image", content="Bulk copy")
    set_people(client, bulk_src, "bulky")
    reply = client.post("/compose/posts/bulk-add-platform",
                        json={"post_ids": [bulk_src], "targets": [f"instagram:{ids[1]}"], "queue": False})
    check(reply.status_code == 200 and reply.get_json().get("success"),
          f"bulk add failed: {reply.status_code} {reply.get_data(as_text=True)[:300]}")
    rows = [r for r in database.list_standalone_posts(db_path=P) if r["content"] == "Bulk copy"]
    check(len(rows) == 2 and all(json.loads(r["ig_collaborators"] or "[]") == ["bulky"] for r in rows),
          f"bulk Add Platform must carry collaborators to the new Instagram row: {[r['ig_collaborators'] for r in rows]}")

    # --- Write Manually: saved on the Instagram rows only, echoed back
    reply = client.post("/compose/post/create", data={
        "content": "Fresh post", "targets": [f"instagram:{ids[0]}", f"instagram:{ids[1]}", "threads"],
        "ig_collaborators": ENTERED, "ig_reel_tags": TAGS_ENTERED})
    body = reply.get_json()
    check(reply.status_code == 200 and body["ig_collaborators"] == ENTERED_CLEAN and body["ig_reel_tags"] == TAGS_CLEAN,
          f"create must echo the cleaned people: {body}")
    created = {r["platform"] + str(r["account_id"]): r
               for r in (database.get_standalone_post(i, db_path=P) for i in body["post_ids"])}
    igs = [r for r in created.values() if r["platform"] == "instagram"]
    check(len(igs) == 2 and all(json.loads(r["ig_collaborators"]) == ENTERED_CLEAN
                                and json.loads(r["ig_reel_tags"]) == TAGS_CLEAN for r in igs),
          "every Instagram row created must carry the people")
    check([r["ig_collaborators"] for r in created.values() if r["platform"] == "threads"] == [None],
          "the Threads row created alongside must not")

    # --- the card the page draws shows what was saved
    html = client.post(f"/compose/post/{igs[0]['id']}/card",
                       data={"post_ids": ",".join(str(r["id"]) for r in created.values())}).get_json()["html"]
    check('value="@amy, @bob"' in html, "the card must show the saved collaborators")

    print("persist: the columns migrate an old database without touching its rows, the routes round-trip, "
          "and the people stay card-wide (second account, bulk Add Platform, Write Manually)")
    print("IG_PEOPLE_PERSIST_OK")


# ---------------------------------------------------------------------------
# publish: Post now, whole-card and the scheduler all deliver the people
# ---------------------------------------------------------------------------

def want_call(fmt, collaborators=None, tags=None, n=1):
    base = {
        "reel": {"kind": "video", "token": token_of(n), "text": "Caption", "video": VIDEO},
        "feed_video": {"kind": "video", "token": token_of(n), "text": "Caption", "video": VIDEO},
        "image": {"kind": "image", "token": token_of(n), "text": "Caption", "image": IMAGE},
        "carousel": {"kind": "carousel", "token": token_of(n), "text": "Caption", "items": CAROUSEL},
    }[fmt]
    call = dict(base)
    if collaborators:
        call["collaborators"] = collaborators
    if tags and fmt in ("reel", "feed_video"):
        call["user_tags"] = tags      # tags only ever go on a Reel
    return call


def section_publish():
    database, web, publisher, client, ids, ig = rig()
    P = database.DB_PATH

    def post_now(pid):
        return client.post(f"/compose/post/{pid}/instagram")

    def whole_card(pid):
        return client.post(f"/compose/post/{pid}/publish", data={"post_ids": str(pid)})

    def scheduler(pid):
        sid = database.add_scheduled_post(
            scheduled_for="2000-01-01T00:00:00", post_type="standalone", standalone_post_id=pid,
            platform="instagram", account_id=ids[0], db_path=P)
        due = [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid]
        check(len(due) == 1, "the queued post should be due")
        return web._publish_scheduled_entry(due[0])

    routes = {"Post now": post_now, "whole-card publish": whole_card, "scheduler": scheduler}

    for fmt in ("reel", "feed_video", "image", "carousel"):
        for route_name, route in routes.items():
            for with_people in (True, False):
                pid = make_row(database, ids[0], fmt)
                if with_people:
                    # entered through the route, as typed, and reel tags saved on EVERY format
                    # so the gate can see that only a Reel sends them
                    set_people(client, pid, ENTERED, TAGS_ENTERED)
                ig.calls.clear()
                outcome = route(pid)
                ok = outcome["success"] if isinstance(outcome, dict) else outcome.get_json().get("success")
                check(ok, f"{fmt} via {route_name} (people={with_people}) failed: "
                          f"{outcome if isinstance(outcome, dict) else outcome.get_data(as_text=True)[:300]}")
                want = want_call(fmt, ENTERED_CLEAN if with_people else None,
                                 TAGS_CLEAN if with_people else None)
                check(ig.calls == [want],
                      f"{fmt} via {route_name} (people={with_people}):\n  want {[want]}\n  got  {ig.calls}")

    # a post on the second account is published with that account's token AND its people
    database, web, publisher, client, ids, ig = rig(accounts=2)
    P = database.DB_PATH
    pid = make_row(database, ids[1], "reel")
    set_people(client, pid, "second", None)
    ig.calls.clear()
    client.post(f"/compose/post/{pid}/instagram", data={"account_id": str(ids[1])})
    check(ig.calls == [want_call("reel", ["second"], None, n=2)],
          f"the second account must publish with its own token and the saved people: {ig.calls}")

    # a Reel's cover frame goes out with its people, from every route
    for route_name, route in routes.items():
        pid = make_row(database, ids[0], "reel")
        database.set_standalone_post_reel_options(pid, {"thumb_offset_ms": 2500}, db_path=P)
        set_people(client, pid, ENTERED, TAGS_ENTERED)
        ig.calls.clear()
        outcome = route(pid)
        ok = outcome["success"] if isinstance(outcome, dict) else outcome.get_json().get("success")
        check(ok, f"reel with a cover frame via {route_name} failed")
        want = dict(want_call("reel", ENTERED_CLEAN, TAGS_CLEAN), thumb_offset_ms=2500)
        check(ig.calls == [want], f"{route_name}: the cover frame must travel with the people:\n  want {[want]}\n  got  {ig.calls}")

    # collaborators still kept where the first version stored them are sent for a Reel (until the
    # migration moves them) and never for a photo
    legacy_reel = make_row(database, ids[0], "reel")
    database.set_standalone_post_reel_options(
        legacy_reel, {"thumb_offset_ms": 10, "collaborators": ["Old.Place"]}, db_path=P)
    ig.calls.clear()
    client.post(f"/compose/post/{legacy_reel}/instagram")
    check(ig.calls == [dict(want_call("reel", ["old.place"], None), thumb_offset_ms=10)],
          f"collaborators stored in a Reel's options must still be sent: {ig.calls}")
    legacy_photo = make_row(database, ids[0], "image")
    database.set_standalone_post_reel_options(
        legacy_photo, {"thumb_offset_ms": 77, "collaborators": ["old.place"]}, db_path=P)
    ig.calls.clear()
    client.post(f"/compose/post/{legacy_photo}/instagram")
    check(ig.calls == [want_call("image", None, None)],
          f"Reel options (collaborators, cover frame) must not reach a photo: {ig.calls}")

    # a Story cannot carry collaborators: refused at publish, client untouched
    pid = make_row(database, ids[0], "story")
    database.set_standalone_post_ig_people(pid, ["amy"], None, db_path=P)   # forced past the routes
    for route_name, route in (("Post now", lambda p: client.post(f"/compose/post/{p}/instagram")),):
        ig.calls.clear()
        reply = route(pid)
        check(reply.status_code == 400 and "Stories" in reply.get_json()["error"] and ig.calls == [],
              f"a Story with collaborators must be refused by {route_name} without calling Instagram: "
              f"{reply.status_code} {reply.get_data(as_text=True)[:200]} {ig.calls}")
    sid = database.add_scheduled_post(scheduled_for="2000-01-01T00:00:00", post_type="standalone",
                                      standalone_post_id=pid, platform="instagram", account_id=ids[0], db_path=P)
    due = [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid]
    ig.calls.clear()
    result = web._publish_scheduled_entry(due[0])
    check(not result["success"] and result["permanent"] and "Stories" in result["error"] and ig.calls == [],
          f"the scheduler must fail a Story with collaborators permanently, not post it bare: {result} {ig.calls}")
    # control: the same Story without collaborators publishes
    pid = make_row(database, ids[0], "story")
    ig.calls.clear()
    reply = client.post(f"/compose/post/{pid}/instagram")
    check(reply.status_code == 200 and ig.calls and ig.calls[0]["kind"] == "story",
          f"control: a Story with no collaborators must publish: {reply.get_data(as_text=True)[:200]}")

    print("publish: Post now, the whole-card publish and the scheduler each hand the saved people to the "
          "client for a Reel, a default-format video, an image and a carousel; a Story with collaborators is refused")
    print("IG_PEOPLE_PUBLISH_OK")


# ---------------------------------------------------------------------------
# ui: the controls, the copy, the schedule page
# ---------------------------------------------------------------------------

def element(html, element_id):
    """The opening tag carrying ``id="element_id"``, or None."""
    match = re.search(rf'<[a-z]+\b[^>]*\bid="{re.escape(element_id)}"[^>]*>', html)
    return match.group(0) if match else None


def attr(tag, name):
    match = re.search(rf'\b{name}="([^"]*)"', tag or "")
    return match.group(1) if match else None


def block(html, element_id):
    """The markup from ``id=element_id`` to the end of its parent block (approximate: next 1800 chars)."""
    at = html.find(f'id="{element_id}"')
    return html[at: at + 1800] if at >= 0 else ""


def section_ui():
    database, web, publisher, client, ids, ig = rig()
    P = database.DB_PATH
    reel = make_row(database, ids[0], "reel_card", content="Reel copy")
    feed = make_row(database, ids[0], "image", content="Feed copy")
    vid = make_row(database, ids[0], "feed_video", content="Default format with a card video")
    story = make_row(database, ids[0], "story", content="Story copy")
    for pid in (reel, feed, vid):
        set_people(client, pid, ENTERED, TAGS_ENTERED)
    page = client.get("/compose")
    check(page.status_code == 200, f"/compose returned {page.status_code}")
    html = page.get_data(as_text=True)

    # --- Write Manually
    manual = block(html, "manual-ig-people")
    check(manual, "Write Manually has no Instagram people block")
    check(element(html, "manual-ig-collaborators") and element(html, "manual-ig-reel-tags"),
          "Write Manually needs a Collaborators field and a Tag people field")
    check("d-none" in element(html, "manual-ig-people"),
          "the Instagram people block must start hidden until Instagram is ticked")
    check("receive an invite" in manual and "once they accept" in manual,
          f"the Write Manually copy must say collaborators receive an invite and only appear once they accept: {manual[:700]}")
    check("up to 3" in manual, "the Write Manually field should state the maximum")
    save = re.search(r"function saveManualPost\(\).*?\n}\n", html, re.S).group(0)
    check("ig_collaborators" in save and "ig_reel_tags" in save and "tickedPlatforms().includes('instagram')" in save,
          "saveManualPost must send the people, and only while Instagram is ticked")

    # --- the saved card
    for pid, label, shows_tags in ((reel, "reel", True), (vid, "default format + video", True), (feed, "feed", False)):
        collab = element(html, f"ig-collaborators-{pid}")
        check(collab, f"{label}: the card has no Collaborators field")
        check(attr(collab, "value") == "@amy, @bob", f"{label}: the card must show the saved collaborators, got {attr(collab, 'value')!r}")
        check(f"saveIgPeople({pid})" in collab, f"{label}: the field must save on change")
        tags = element(html, f"ig-reel-tags-{pid}")
        check(tags and attr(tags, "value") == "@cat, @dan", f"{label}: the card must carry the saved Reel tags")
        col = re.search(rf'<div class="ig-reel-tags-col([^"]*)"[^>]*>\s*<label[^>]*for="ig-reel-tags-{pid}"', html)
        check(col, f"{label}: the Reel tag field has no column wrapper")
        hidden = "d-none" in col.group(1)
        check(hidden == (not shows_tags),
              f"{label}: the Tag people field must be {'shown' if shows_tags else 'hidden'} (hidden={hidden})")
    people_block = block(html, f"ig-people-{reel}")
    check("receive an invite" in people_block and "once they accept" in people_block,
          f"the card copy must say collaborators receive an invite and only appear once they accept: {people_block[:900]}")
    check(f'id="ig-people-error-{reel}"' in html, "the card needs somewhere to show a refusal")
    check(re.search(r"Stories can.t have collaborators", people_block), "the card should say Stories take none")

    # the card re-render (after ticking a platform, etc.) keeps it
    again = client.post(f"/compose/post/{reel}/card", data={"post_ids": str(reel)}).get_json()["html"]
    check(attr(element(again, f"ig-collaborators-{reel}"), "value") == "@amy, @bob",
          "a re-rendered card must keep the collaborators")

    # --- the schedule page: a queued post visibly carries them
    when = "2099-03-01T09:00:00"
    th_reel = database.add_standalone_post("manual", "g", "threads", "Reel copy", video_url=VIDEO, db_path=P)
    for pid, platform in ((reel, "instagram"), (th_reel, "threads"), (feed, "instagram"), (story, "instagram")):
        database.add_scheduled_post(when, "standalone", standalone_post_id=pid, platform=platform,
                                    account_id=ids[0] if platform == "instagram" else None, db_path=P)
    # a post whose Threads entry is queued BEFORE its Instagram one: the row is built from
    # the first entry, so the Instagram people must be found on a later member
    first_threads = database.add_standalone_post("manual", "g", "threads", "Order copy", db_path=P)
    later_ig = make_row(database, ids[0], "carousel", content="Order copy")
    set_people(client, later_ig, "zed.last", None)
    database.add_scheduled_post("2099-03-03T09:00:00", "standalone", standalone_post_id=first_threads,
                                platform="threads", db_path=P)
    database.add_scheduled_post("2099-03-03T09:00:00", "standalone", standalone_post_id=later_ig,
                                platform="instagram", account_id=ids[0], db_path=P)
    database.set_standalone_post_ig_people(vid, None, None, db_path=P)
    database.add_scheduled_post("2099-03-02T09:00:00", "standalone", standalone_post_id=vid,
                                platform="instagram", account_id=ids[0], db_path=P)
    page = client.get("/schedule?status=pending").get_data(as_text=True)
    body = re.search(r'<tbody id="schedule-tbody">(.*?)</tbody>', page, re.S).group(1)
    rows = re.split(r'(?=<tr data-post-id)', body)
    reel_row = next(r for r in rows if "Reel copy" in r)
    check("🤝 @amy, @bob" in reel_row, f"the queued Reel's row must show its collaborators: {reel_row[:900]}")
    check("🏷️ @cat, @dan" in reel_row, "the queued Reel's row must show its tags")
    check("Threads" in reel_row and "Instagram" in reel_row,
          "the Reel and its Threads twin are one row, and that row carries the Instagram people")
    feed_row = next(r for r in rows if "Feed copy" in r)
    check("🤝 @amy, @bob" in feed_row and "🏷️" not in feed_row,
          "a feed photo shows its collaborators but not Reel tags (they are not sent on a photo)")
    story_row = next(r for r in rows if "Story copy" in r)
    check("🤝" not in story_row and "🏷️" not in story_row, "control: a post with no people shows none")
    order_row = next(r for r in rows if "Order copy" in r)
    check("🤝 @zed.last" in order_row and "🧵 Threads" in order_row and "📷 Instagram" in order_row
          and order_row.index("🧵 Threads") < order_row.index("📷 Instagram"),
          f"a row whose first entry is Threads must still show its Instagram entry's people: {order_row[:800]}")
    vid_row = next(r for r in rows if "Default format with a card video" in r)
    check("🤝" not in vid_row, "control: people cleared on a post are gone from the queue")
    check("once they accept" in reel_row, "the badge should explain that collaborators appear once they accept")

    # the JSON the refresh button uses carries the same people
    groups = client.get("/schedule/list-json?status=pending").get_json()["groups"]
    by_copy = {g_["content_preview"]: g_ for g_ in groups}
    check(by_copy["Reel copy"]["ig_collaborators"] == ENTERED_CLEAN and by_copy["Reel copy"]["ig_tags"] == TAGS_CLEAN,
          f"list-json must carry the group's people: {by_copy['Reel copy']}")
    check(by_copy["Feed copy"]["ig_collaborators"] == ENTERED_CLEAN and by_copy["Feed copy"]["ig_tags"] == [],
          "list-json must not claim Reel tags for a photo")
    check(by_copy["Story copy"]["ig_collaborators"] == [] and by_copy["Story copy"]["ig_tags"] == [], "control")
    check(by_copy["Order copy"]["ig_collaborators"] == ["zed.last"] and by_copy["Order copy"]["platforms"][0] == "threads",
          f"list-json must carry an Instagram member's people on a group that starts with Threads: {by_copy['Order copy']}")

    print("ui: Write Manually, the saved card and the schedule page show the Collaborators and Tag people "
          "controls with the invite/accept copy, and a queued post carries its people")
    print("IG_PEOPLE_UI_OK")


# ---------------------------------------------------------------------------
# js: the page's own functions under node
# ---------------------------------------------------------------------------

def fn_source(html, name):
    """One top-level function's source, found by matching its braces."""
    match = re.search(rf"(?:async\s+)?function\s+{name}\s*\(", html)
    check(match, f"the page does not define {name}()")
    start = html.index("{", html.index(")", match.start()))
    depth = 0
    for i in range(start, len(html)):
        depth += {"{": 1, "}": -1}.get(html[i], 0)
        if depth == 0:
            return html[match.start(): i + 1]
    check(False, f"{name}() has unbalanced braces")


JS_HARNESS = r"""
const els = {};
const make = (id, extra) => els[id] = Object.assign({
  id, value: '', textContent: '', innerHTML: '', dataset: {}, files: [], disabled: false,
  classes: new Set(), classList: null, focus() {},
}, extra || {});
const wire = (el) => { el.classList = {
  add: (c) => el.classes.add(c), remove: (c) => el.classes.delete(c),
  contains: (c) => el.classes.has(c),
  toggle: (c, on) => { (on === undefined ? !el.classes.has(c) : on) ? el.classes.add(c) : el.classes.delete(c); },
}; return el; };
const el = (id, extra) => wire(make(id, extra));
const toasts = [], alerts = [], sent = [];
const document = {
  getElementById: (id) => els[id] || null,
  querySelector: (sel) => {
    const m = sel.match(/^#ig-people-(\d+) \.ig-reel-tags-col$/);
    return m ? els['reel-col-' + m[1]] || null : null;
  },
};
const showToast = (message, type) => toasts.push([type, message]);
const alert = (m) => alerts.push(m);
const confirm = () => true;
const cardIdsParam = (id) => '7,8';
let reply = { ok: true, body: { success: true } };
const fetch = (url, opts) => {
  sent.push({ url, entries: Array.from(opts.body.entries()) });
  return Promise.resolve({ ok: reply.ok, json: () => Promise.resolve(reply.body) });
};
const flush = () => new Promise((resolve) => setImmediate(resolve));
const reset = () => { toasts.length = 0; alerts.length = 0; sent.length = 0; };
"""


def section_js():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    database, web, publisher, client, ids, ig = rig()
    P = database.DB_PATH
    # a real post so the page renders the real functions
    reel = make_row(database, ids[0], "reel", content="Reel copy")
    html = client.get("/compose").get_data(as_text=True)

    source = "\n".join(fn_source(html, n) for n in (
        "igCurrentType", "igEffectiveFormat", "igRefreshPeople", "igHandleList", "saveIgPeople",
        "switchIgType", "refreshManualIgPeople", "saveManualPost", "manualPostTargets",
        "manualVideoChoice", "postVideoEls", "renderPostVideo"))
    script = JS_HARNESS + r"""
window = {_igLastType: {}, _igMedia: {}, _igTags: {}};
let tickedPlatforms = () => ['threads'];
let selectedAccountTargets = () => [];
const updateManualComposerCount = () => {};
const platformCharLimits = {}; const PLATFORM_LABELS = {};
const location = { reload() {} };
const setTimeout = (fn) => fn();
const igSaveMedia = () => { sent.push({ url: 'igSaveMedia' }); };
const igCheckFeedImage = () => {};
const renderIgStrip = () => {};
""" + source + r"""
(async () => {
  const out = {};
  // --- effective format and the Reel tag column
  el('ig-type-5', { value: 'feed' });
  el('video-section-5', { dataset: { videoUrl: '' } });
  const col = el('reel-col-5'); col.classes.add('d-none');
  out.feedNoVideo = igEffectiveFormat(5);
  igRefreshPeople(5); out.colHiddenForFeed = col.classes.has('d-none');
  els['video-section-5'].dataset.videoUrl = 'https://v.test/a.mp4';
  out.feedVideo = igEffectiveFormat(5);
  igRefreshPeople(5); out.colShownForFeedVideo = !col.classes.has('d-none');
  els['ig-type-5'].value = 'carousel';
  out.carouselVideo = igEffectiveFormat(5);
  igRefreshPeople(5); out.colHiddenForCarousel = col.classes.has('d-none');
  els['ig-type-5'].value = 'reel'; els['video-section-5'].dataset.videoUrl = '';
  out.reel = igEffectiveFormat(5);
  igRefreshPeople(5); out.colShownForReel = !col.classes.has('d-none');
  out.handleList = igHandleList('@a, b\n c ,, d e');

  // --- attaching or removing the card's video, through the page's own renderPostVideo,
  // shows or hides the Reel tag field on a card left on the default format
  els['ig-type-5'].value = 'feed'; els['video-section-5'].dataset.videoUrl = ''; col.classes.add('d-none');
  const node_ = (extra) => Object.assign({ classes: new Set(), dataset: {}, textContent: '', src: '',
    classList: null, querySelector() { return node_(); }, removeAttribute() {}, load() {},
    replaceChildren() {}, appendChild() {} }, extra || {});
  const stub = (id) => wire(Object.assign(node_(), { id }));
  const preview = stub('video-preview-5');
  const player = stub('player'); const label = stub('label');
  preview.querySelector = (sel) => sel === 'video' ? player : label;
  Object.assign(els, { 'video-preview-5': preview, 'video-note-5': stub('n'), 'video-warnings-5': stub('w'),
    'video-edit-form-5': stub('f'), 'add-video-btn-5': stub('a'), 'video-progress-5': stub('p') });
  renderPostVideo(5, 'https://v.test/new.mp4', []);
  out.videoAttached = { effective: igEffectiveFormat(5), colShown: !col.classes.has('d-none') };
  renderPostVideo(5, null, []);
  out.videoRemoved = { effective: igEffectiveFormat(5), colHidden: col.classes.has('d-none') };

  // --- saveIgPeople: what it sends, and what it does with the answer
  el('ig-collaborators-5', { value: '@Amy, bob' }); el('ig-reel-tags-5', { value: '@cat' });
  const err = el('ig-people-error-5'); err.classes.add('d-none');
  reply = { ok: true, body: { success: true, ig_collaborators: ['amy', 'bob'], ig_reel_tags: ['cat'] } };
  reset(); await saveIgPeople(5);
  out.save = { sent: sent.slice(), collab: els['ig-collaborators-5'].value, tags: els['ig-reel-tags-5'].value,
               errHidden: err.classes.has('d-none'), toasts: toasts.slice() };
  els['ig-collaborators-5'].value = 'john doe';
  reply = { ok: false, body: { error: "'john doe' is not a valid Instagram username for a collaborator" } };
  reset(); await saveIgPeople(5);
  out.refused = { collab: els['ig-collaborators-5'].value, errText: err.textContent, errHidden: err.classes.has('d-none') };
  reply = { ok: true, body: { success: true, ig_collaborators: [], ig_reel_tags: [] } };
  els['ig-collaborators-5'].value = '';
  reset(); await saveIgPeople(5);
  out.cleared = { collab: els['ig-collaborators-5'].value, errHidden: err.classes.has('d-none'), toast: toasts.slice() };

  // --- a Story with collaborators typed is refused and the format select reverts
  el('ig-type-6', { value: 'story' }); el('ig-collaborators-6', { value: '@amy' });
  window._igLastType[6] = 'reel';
  reset(); switchIgType(6, 'story');
  out.storyRefused = { select: els['ig-type-6'].value, saved: sent.length, toasts: toasts.slice(), last: window._igLastType[6] };
  els['ig-collaborators-6'].value = '';
  els['ig-type-6'].value = 'story'; el('ig-media-builder-6'); el('ig-feed-block-6');
  reset(); switchIgType(6, 'story');
  out.storyAllowed = { select: els['ig-type-6'].value, last: window._igLastType[6], savedMedia: sent.length };

  // --- Write Manually: people sent only while Instagram is ticked
  el('manual-input', { value: 'A post' }); el('final-image-url', { value: '' });
  el('manual-video-file'); el('manual-video-url'); el('manual-save-btn');
  el('manual-ig-collaborators', { value: ' @Amy, bob ' }); el('manual-ig-reel-tags', { value: '@cat' });
  el('manual-ig-people').classes.add('d-none');
  reply = { ok: true, body: { success: true, post_ids: [1] } };
  tickedPlatforms = () => ['threads'];
  reset(); refreshManualIgPeople(); out.hiddenWithoutIg = els['manual-ig-people'].classes.has('d-none');
  saveManualPost(); await flush();
  out.manualNoIg = sent.map(s => s.entries.map(e => e[0]));
  tickedPlatforms = () => ['instagram', 'threads'];
  reset(); refreshManualIgPeople(); out.shownWithIg = !els['manual-ig-people'].classes.has('d-none');
  saveManualPost(); await flush();
  out.manualWithIg = sent.map(s => s.entries);
  console.log(JSON.stringify(out));
})();
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    check(result.returncode == 0, f"node failed:\n{result.stderr.strip()[:1200]}")
    out = json.loads(result.stdout.strip().splitlines()[-1])

    check(out["feedNoVideo"] == "feed" and out["colHiddenForFeed"], f"a plain feed post has no Reel tag field: {out}")
    check(out["feedVideo"] == "reel" and out["colShownForFeedVideo"], f"a feed post with a card video is a Reel: {out}")
    check(out["carouselVideo"] == "carousel" and out["colHiddenForCarousel"], f"a carousel is not a Reel: {out}")
    check(out["reel"] == "reel" and out["colShownForReel"], f"a Reel shows the tag field: {out}")
    check(out["videoAttached"] == {"effective": "reel", "colShown": True},
          f"attaching a video to a default-format card makes it a Reel and shows the tag field: {out['videoAttached']}")
    check(out["videoRemoved"] == {"effective": "feed", "colHidden": True},
          f"removing the video hides the tag field again: {out['videoRemoved']}")
    check(out["handleList"] == ["@a", "b", "c", "d e"], f"entries split on commas and new lines, never spaces: {out['handleList']}")

    save = out["save"]
    (request,) = save["sent"]
    check(request["url"] == "/compose/post/5/ig-people", f"saveIgPeople posted to the wrong route: {request['url']}")
    fields = dict(request["entries"])
    check(fields == {"ig_collaborators": "@Amy, bob", "ig_reel_tags": "@cat", "post_ids": "7,8"},
          f"saveIgPeople must send exactly what was typed plus the card's rows: {fields}")
    check(save["collab"] == "@amy, @bob" and save["tags"] == "@cat" and save["errHidden"],
          f"the fields must be rewritten with the server's cleaned values: {save}")
    check(save["toasts"] and "invited" in save["toasts"][0][1], f"saving collaborators should say they are invited: {save['toasts']}")
    ref = out["refused"]
    check(ref["collab"] == "john doe" and "john doe" in ref["errText"] and not ref["errHidden"],
          f"a refusal must be shown under the field with the field left as typed: {ref}")
    clr = out["cleared"]
    check(clr["collab"] == "" and clr["errHidden"], f"clearing must clear the error and the field: {clr}")
    st = out["storyRefused"]
    check(st["select"] == "reel" and st["saved"] == 0 and st["toasts"] and "Stories" in st["toasts"][0][1] and st["last"] == "reel",
          f"choosing Story with collaborators typed must revert the select and save nothing: {st}")
    ok = out["storyAllowed"]
    check(ok["select"] == "story" and ok["last"] == "story" and ok["savedMedia"] == 1,
          f"control: choosing Story with no collaborators must go ahead: {ok}")

    check(out["hiddenWithoutIg"] and out["shownWithIg"], f"the Write Manually people block follows the Instagram tick: {out}")
    check(all("ig_collaborators" not in names and "ig_reel_tags" not in names for names in out["manualNoIg"]),
          f"without Instagram ticked the people must not be sent: {out['manualNoIg']}")
    (with_ig,) = out["manualWithIg"]
    sent_fields = dict(with_ig)
    check(sent_fields.get("ig_collaborators") == "@Amy, bob" and sent_fields.get("ig_reel_tags") == "@cat",
          f"with Instagram ticked, Save must send the people as typed (trimmed): {sent_fields}")

    # replay exactly what the page sent against the real routes
    post = client.post("/compose/post/create", data={
        "content": "A post", "targets": [f"instagram:{ids[0]}"],
        "ig_collaborators": sent_fields["ig_collaborators"], "ig_reel_tags": sent_fields["ig_reel_tags"]})
    check(post.status_code == 200 and post.get_json()["ig_collaborators"] == ["amy", "bob"]
          and post.get_json()["ig_reel_tags"] == ["cat"], f"the page's Save, replayed, must be accepted: {post.get_json()}")
    saved = client.post(f"/compose/post/{reel}/ig-people", data=fields)
    check(saved.status_code == 200 and saved.get_json()["ig_collaborators"] == ["amy", "bob"],
          f"the page's save, replayed, must be accepted: {saved.get_data(as_text=True)[:300]}")

    print("js: the page's own functions show the Reel tag field only for a Reel, send the people as typed, "
          "show a refusal, refuse a Story with collaborators, and Save sends them only with Instagram ticked")
    print("IG_PEOPLE_JS_OK")


# ---------------------------------------------------------------------------
# queue: the page and the redraw show the same people
# ---------------------------------------------------------------------------

def section_queue():
    import check_queue_groups as q
    w = q.World()
    db, P = w.database, w.db
    account = connect(db, "instagram", "ig-1", "brand1")
    T = q.T1

    # a Reel with its Threads twin, one feed photo, one post with nothing
    reel = w.post("Reel copy with people", "instagram", video=VIDEO)
    db.set_standalone_post_media(reel, "reel", [], db_path=P)
    db.set_standalone_post_ig_people(reel, ENTERED_CLEAN, TAGS_CLEAN, db_path=P)
    twin = w.post("Reel copy with people", "threads", video=VIDEO)
    w.queue(reel, "instagram", T, account_id=account)
    w.queue(twin, "threads", T)
    photo = w.post("Photo copy with people", "instagram", image=IMAGE)
    db.set_standalone_post_ig_people(photo, ["pia"], ["hidden"], db_path=P)
    w.queue(photo, "instagram", q.T2, account_id=account)
    first = w.post("Order copy", "threads")
    later = w.post("Order copy", "instagram")
    db.set_standalone_post_media(later, "carousel", CAROUSEL, db_path=P)
    db.set_standalone_post_ig_people(later, ["zed.last"], None, db_path=P)
    w.queue(first, "threads", q.T2)
    w.queue(later, "instagram", q.T2, account_id=account)
    bare = w.post("Bare copy", "instagram", image=IMAGE)
    w.queue(bare, "instagram", q.T3, account_id=account)
    posted = w.post("Posted reel copy", "instagram", video=VIDEO)
    db.set_standalone_post_ig_people(posted, ["done"], None, db_path=P)
    w.queue(posted, "instagram", q.T3, status="posted", url="https://instagram.test/p/x/", account_id=account)

    cases = [("pending", "?status=pending"), ("posted", "?status=posted"), ("everything", "?status=")]
    expected, payload = {}, {}
    for name, query in cases:
        expected[name] = q.page_rows(w.client, query)
        payload[name] = w.client.get("/schedule/list-json" + query).get_json()["groups"]
        check(expected[name], f"{name}: drew no rows, so it proves nothing")

    # the server's rows carry the people
    text = " | ".join(" ".join(c["text"] for c in r["cells"]) for r in expected["everything"])
    check("🤝 @amy, @bob" in text and "🏷️ @cat, @dan" in text, f"the page must show the Reel's people: {text[:600]}")
    check("🤝 @pia" in text and "@hidden" not in text, "a photo shows collaborators, never reel tags")
    check("🤝 @done" in text, "a posted post still shows who it went out with")
    check("🤝 @zed.last" in text, "a row that starts with Threads still shows its Instagram entry's people")
    bare_row = next(r for r in expected["everything"] if "Bare copy" in " ".join(c["text"] for c in r["cells"]))
    check("🤝" not in " ".join(c["text"] for c in bare_row["cells"]), "control: a post with no people shows none")

    _, _, script = q.schedule_script()
    functions = q.lift(script, q.RENDER_FUNCTIONS, consts=("QUEUE_PLATFORMS",))
    drawn = q.run_node(q.PARITY_HARNESS, functions, payload)
    compared = 0
    for name, _ in cases:
        rebuilt = q.parse_rows(drawn[name])
        server = expected[name]
        check(len(rebuilt) == len(server),
              f"{name}: the server drew {len(server)} rows and the script drew {len(rebuilt)}")
        for index, (a, b) in enumerate(zip(server, rebuilt)):
            check(a == b, f"{name}: row {index} differs between the page and the redraw:\n"
                          f"  server: {json.dumps(a, ensure_ascii=False)[:900]}\n"
                          f"  script: {json.dumps(b, ensure_ascii=False)[:900]}")
            compared += 1
    redraw_text = " | ".join(" ".join(c["text"] for c in r["cells"]) for r in q.parse_rows(drawn["everything"]))
    check("🤝 @amy, @bob" in redraw_text and "🏷️ @cat, @dan" in redraw_text and "@hidden" not in redraw_text,
          "the script's redraw must show the same people")

    print(f"queue: the server's rows and the script's redraw agree on the people, {compared} rows compared")
    print("IG_PEOPLE_QUEUE_OK")


SECTIONS = {
    "client": section_client, "none": section_none, "refuse": section_refuse,
    "reject": section_reject, "persist": section_persist, "publish": section_publish,
    "ui": section_ui, "js": section_js, "queue": section_queue,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_ig_people.py {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[name]()

"""Gates for sending a LinkedIn post's link as its first comment.

    python scripts/check_link_comment.py client      LINK_COMMENT_CLIENT_OK
    python scripts/check_link_comment.py split       LINK_COMMENT_SPLIT_OK
    python scripts/check_link_comment.py publish     LINK_COMMENT_PUBLISH_OK
    python scripts/check_link_comment.py off         LINK_COMMENT_OFF_OK
    python scripts/check_link_comment.py failure     LINK_COMMENT_FAILURE_OK
    python scripts/check_link_comment.py refuse      LINK_COMMENT_REFUSE_OK
    python scripts/check_link_comment.py retry       LINK_COMMENT_RETRY_OK
    python scripts/check_link_comment.py persist     LINK_COMMENT_PERSIST_OK
    python scripts/check_link_comment.py routes      LINK_COMMENT_ROUTES_OK
    python scripts/check_link_comment.py paths       LINK_COMMENT_PATHS_OK
    python scripts/check_link_comment.py ui          LINK_COMMENT_UI_OK
    python scripts/check_link_comment.py js          LINK_COMMENT_JS_OK
    python scripts/check_link_comment.py docs        LINK_COMMENT_DOCS_OK
    python scripts/check_link_comment.py registered  LINK_COMMENT_REGISTERED_OK

LinkedIn holds back posts that link out, so a LinkedIn post can send its link as
its first comment instead: the post goes out without the link and the link
follows as a comment once the post is live. The choice belongs to the post (or
the whole card), with the LinkedIn account's setting as the default.

The claim that matters most is what happens when it goes wrong. The comment is a
second request made after the post is live, and LinkedIn can refuse it (the
permission it needs is not the one posting needs). A refused comment must never
turn into a failed post, because the scheduler retries failed posts and would post
it again. ``failure`` and ``paths`` are about that; ``off`` is about a post that
does not use the feature being exactly what it was.

``client`` replaces ``requests`` inside linkedin_client with a scripted transport
and reads the HTTP the client would send. The rest drive the real routes,
database, publisher and scheduler entry point on a throwaway database with a
recording LinkedIn client. ``ui`` reads the rendered pages and ``js`` runs the
page's own functions under node. Nothing here posts anything, reads a real token
or leaves the machine.
"""

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

# The failure paths log on purpose (a refused comment, an unwritable record); a gate's
# output is its verdict, so the app's log lines stay out of it.
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import FakeResult, check, connect  # noqa: E402
import _link_comment_gate as g  # noqa: E402
from _link_comment_gate import (  # noqa: E402
    COPY, IMAGE, LINK, VIDEO, author_of, make_post, rig, row_of, state_of, token_of,
)

LEGACY_RESULT_KEYS = {
    "success", "platform", "platform_name", "account_id", "account_label",
    "permalink", "error", "needs_reconnect", "permanent",
}
TOKENS = {
    "client": "LINK_COMMENT_CLIENT_OK", "split": "LINK_COMMENT_SPLIT_OK",
    "publish": "LINK_COMMENT_PUBLISH_OK", "off": "LINK_COMMENT_OFF_OK",
    "failure": "LINK_COMMENT_FAILURE_OK", "refuse": "LINK_COMMENT_REFUSE_OK",
    "retry": "LINK_COMMENT_RETRY_OK", "persist": "LINK_COMMENT_PERSIST_OK",
    "routes": "LINK_COMMENT_ROUTES_OK", "paths": "LINK_COMMENT_PATHS_OK",
    "ui": "LINK_COMMENT_UI_OK", "js": "LINK_COMMENT_JS_OK",
    "docs": "LINK_COMMENT_DOCS_OK", "registered": "LINK_COMMENT_REGISTERED_OK",
}


def publish_row(publisher, database, account, post_id, content, image=None, video=None, **extra):
    """Publish a saved row the way the app does: naming the row, not its mode."""
    return publisher.publish("linkedin", account, content=content, image_url=image,
                             video_url=video, standalone_post_id=post_id,
                             db_path=database.DB_PATH, **extra)


# ---------------------------------------------------------------------------
# client: the HTTP request, and what the client does with each answer
# ---------------------------------------------------------------------------

def section_client():
    import requests
    import linkedin_client
    from linkedin_client import COMMENT_RETRY_DELAY, LINKEDIN_API_VERSION, LinkedInClient
    from _link_comment_gate import FakeResponse, install_http

    client = LinkedInClient("id", "secret")
    ACTOR = "urn:li:person:abc123"
    SHARE = "urn:li:share:7100000000000000001"
    SHARE_ENC = "urn%3Ali%3Ashare%3A7100000000000000001"
    ENDPOINT = "https://api.linkedin.com/rest/socialActions"

    def call(script, post_urn=SHARE, text=LINK):
        http, sleeps, restore = install_http(script)
        try:
            reply = client.create_comment("tok", ACTOR, post_urn, text)
        finally:
            restore()
        return reply, http, sleeps

    # --- the request is exactly the documented one
    reply, http, sleeps = call([FakeResponse(201, headers={"x-restli-id": "6643206422739898368"},
                                             payload={"id": "6643206422739898368"})])
    (sent,) = http.requests
    check(sent["url"] == f"{ENDPOINT}/{SHARE_ENC}/comments",
          f"the comment must go to socialActions/<encoded post urn>/comments, not {sent['url']}")
    check(sent["json"] == {"actor": ACTOR, "object": SHARE, "message": {"text": LINK}},
          f"the body must be actor, object and message.text only: {sent['json']}")
    h = sent["headers"]
    check(h.get("Authorization") == "Bearer tok", f"wrong Authorization: {h}")
    check(h.get("LinkedIn-Version") == LINKEDIN_API_VERSION, f"wrong LinkedIn-Version: {h}")
    check(h.get("X-Restli-Protocol-Version") == "2.0.0", f"wrong protocol version: {h}")
    check(h.get("Content-Type") == "application/json", f"wrong Content-Type: {h}")
    check(reply == {"success": True, "comment_id": "6643206422739898368", "status_code": 201},
          f"a created comment should report its id: {reply}")
    check(sleeps == [], "a comment that worked must not wait")

    # an ugcPost id is encoded the same way, and the comment id may come back in the body
    reply, http, _ = call([FakeResponse(200, payload={"id": "777"})], post_urn="urn:li:ugcPost:456")
    check(http.requests[0]["url"] == f"{ENDPOINT}/urn%3Ali%3AugcPost%3A456/comments",
          f"ugcPost urn not encoded: {http.requests[0]['url']}")
    check(reply["success"] and reply["comment_id"] == "777", f"id from the body: {reply}")

    # --- answers it explains
    reply, http, sleeps = call([
        FakeResponse(403, payload={"message": "Not enough permissions to access: socialActions.CREATE", "status": 403}),
        FakeResponse(201, headers={"x-restli-id": "1"})])
    check(not reply["success"] and reply["status_code"] == 403, f"403 must fail: {reply}")
    check("w_member_social_feed" in reply["friendly"], f"403 must name the permission: {reply['friendly']}")
    check("Not enough permissions" in reply["friendly"], f"403 must pass on what LinkedIn said: {reply['friendly']}")
    check(not reply.get("needs_reconnect"), "a 403 is a missing permission, not an expired login")
    check(len(http.requests) == 1 and len(http.script) == 1 and sleeps == [],
          f"a 403 must not be repeated ({len(http.requests)} requests, {sleeps})")

    reply, http, _ = call([FakeResponse(401, payload={"message": "Invalid access token"}), FakeResponse(201)])
    check(not reply["success"] and reply["needs_reconnect"] is True, f"401 must ask for a reconnect: {reply}")
    check("Reconnect LinkedIn" in reply["friendly"] and len(http.requests) == 1, f"401: {reply}")

    # --- 404 and 429 mean "not created": one more try, after a wait, and no more
    for status, words in ((404, "could not find the post"), (429, "limiting how fast")):
        reply, http, sleeps = call([FakeResponse(status, payload={"message": "x"}),
                                    FakeResponse(201, headers={"x-restli-id": "55"})])
        check(reply["success"] and reply["comment_id"] == "55", f"{status} then 201 should succeed: {reply}")
        check(len(http.requests) == 2 and sleeps == [COMMENT_RETRY_DELAY],
              f"{status} must be retried once after {COMMENT_RETRY_DELAY}s: {len(http.requests)} requests, sleeps {sleeps}")
        check(http.requests[0] == http.requests[1], f"{status}: the retry must be the same request")
        reply, http, sleeps = call([FakeResponse(status, payload={"message": "x"}),
                                    FakeResponse(status, payload={"message": "x"})])
        check(not reply["success"] and words in reply["friendly"], f"{status} twice: {reply}")
        check(len(http.requests) == 2 and sleeps == [COMMENT_RETRY_DELAY],
              f"{status} must be tried exactly twice, never more: {len(http.requests)} requests")

    # --- nothing else is repeated: it might have been created
    for status in (500, 502, 503):
        reply, http, sleeps = call([FakeResponse(status, payload={"message": "down"}), FakeResponse(201)])
        check(not reply["success"] and str(status) in reply["friendly"], f"{status}: {reply}")
        check(len(http.requests) == 1 and sleeps == [], f"a {status} must not be repeated")
    for exc in (requests.ConnectionError("connection dropped"), requests.Timeout("timed out")):
        reply, http, sleeps = call([exc, FakeResponse(201)])
        check(not reply["success"] and reply["friendly"].startswith("Could not reach LinkedIn"),
              f"{type(exc).__name__}: {reply}")
        check(len(http.requests) == 1 and sleeps == [],
              f"after {type(exc).__name__} the comment may exist, so it must not be sent again")

    print("client: the comment request is the documented one; 403/401/404/429/5xx and a dropped "
          "connection are each explained, and only 404 and 429 are tried again, once")
    print("LINK_COMMENT_CLIENT_OK")


# ---------------------------------------------------------------------------
# split: taking the links out of the copy
# ---------------------------------------------------------------------------

SPLIT_CASES = [
    # (name, text, body, links)
    ("a link on its own last line", "Intro\n\nRead it:\nhttps://a.test/x",
     "Intro\n\nRead it:", ["https://a.test/x"]),
    ("a link-only line between paragraphs", "A\n\nhttps://a.test/x\n\nB",
     "A\n\nB", ["https://a.test/x"]),
    ("a link-only line between lines", "A\nhttps://a.test/x\nB", "A\nB", ["https://a.test/x"]),
    ("a link in the middle of a sentence", "See https://a.test/x for more",
     "See for more", ["https://a.test/x"]),
    ("a link at the start of a line", "https://a.test/x is the write-up",
     "is the write-up", ["https://a.test/x"]),
    ("a link at the start of an inner line", "A\nhttps://a.test/x is it\nB",
     "A\nis it\nB", ["https://a.test/x"]),
    ("indentation is kept", "A\n   indented https://a.test/x", "A\n   indented", ["https://a.test/x"]),
    ("punctuation stuck to a link stays", "Read this: https://a.test/x.",
     "Read this: .", ["https://a.test/x"]),
    ("brackets around a link go with it", "Details (https://a.test/x) here",
     "Details here", ["https://a.test/x"]),
    ("a line of only a bracketed link", "(https://a.test/x)", "", ["https://a.test/x"]),
    ("parentheses inside a link stay", "Wiki https://en.wikipedia.org/wiki/Foo_(bar) ok",
     "Wiki ok", ["https://en.wikipedia.org/wiki/Foo_(bar)"]),
    ("a query and a fragment stay", "Go https://a.test/p?x=1&y=2#frag", "Go",
     ["https://a.test/p?x=1&y=2#frag"]),
    ("the same link twice is one", "A https://a.test/x\nB https://a.test/x", "A\nB", ["https://a.test/x"]),
    ("two links keep their order", "https://a.test/2 and https://a.test/1", "and",
     ["https://a.test/2", "https://a.test/1"]),
    ("copy away from the link is not touched", "A  B\n\n\n\nC https://a.test/x", "A  B\n\n\n\nC",
     ["https://a.test/x"]),
    ("a post that is only a link", "https://a.test/x", "", ["https://a.test/x"]),
    ("a trailing newline", "Hello\nhttps://a.test/x\n", "Hello", ["https://a.test/x"]),
    ("windows line endings", "Intro\r\nhttps://a.test/x\r\nOutro", "Intro\r\nOutro", ["https://a.test/x"]),
    ("a link with non-ascii", "Voici https://a.test/é ok", "Voici ok", ["https://a.test/é"]),
]
NO_LINK_TEXTS = [
    "", "Plain copy", "Two  spaces\n\n\n\nand blank lines  ", "email me at a@b.test", "see www.a.test/x",
    "ftp://a.test/x", "http:/broken",
]


def section_split():
    import linkedin_client
    from linkedin_client import LinkedInClient, split_link_for_comment

    def leaks(body):
        """Does ``body`` still carry something LinkedIn would read as a link?"""
        return bool(linkedin_client._URL_RE.search(body) or linkedin_client._LINK_RE.search(body))

    # control: the leak detector really can fail, or "no link remains" proves nothing
    check(leaks("x https://a.test/y") and leaks("https://en.wikipedia.org/wiki/Foo_(bar)"),
          "the leak detector must flag a body that still has a link")

    for name, text, want_body, want_links in SPLIT_CASES:
        body, links = split_link_for_comment(text)
        check(body == want_body and links == want_links,
              f"{name}:\n  text  {text!r}\n  want  {(want_body, want_links)!r}\n  got   {(body, links)!r}")
        check(not leaks(body), f"{name}: a link is still in the body: {body!r}")
        for link in links:
            check(link in text, f"{name}: {link!r} was not in the copy")
        again = split_link_for_comment(body)
        check(again == (body, []), f"{name}: splitting the result again found something: {again!r}")

    for text in NO_LINK_TEXTS:
        body, links = split_link_for_comment(text)
        check(body is text and links == [],
              f"copy with no link must come back as it was, not rewritten: {text!r} -> {(body, links)!r}")
    check(split_link_for_comment(None) == ("", []), "None must be treated as empty copy")

    # the preview-card detection every post already used is unchanged
    check(LinkedInClient.extract_first_url("see https://a.test/x. ok") == "https://a.test/x",
          "extract_first_url changed")
    check(LinkedInClient.extract_first_url("no link here") is None, "extract_first_url changed")
    check(linkedin_client.COMMENT_MAX_CHARS == 1250, "the comment length bound changed")

    print(f"split: {len(SPLIT_CASES)} cases move every link out in order and leave the rest as written; "
          f"{len(NO_LINK_TEXTS)} link-free texts come back untouched")
    print("LINK_COMMENT_SPLIT_OK")


# ---------------------------------------------------------------------------
# publish: the link leaves the post and follows as its comment
# ---------------------------------------------------------------------------

def section_publish():
    database, web, publisher, client, ids, rec, _ = rig()
    P = database.DB_PATH

    cases = [
        ("text", COPY + "\n" + LINK, None, None, "text"),
        ("image", COPY + "\n" + LINK, IMAGE, None, "image"),
        ("video", COPY + "\n" + LINK, None, VIDEO, "video"),
    ]
    for name, content, image, video, kind in cases:
        pid = make_post(database, ids[0], content, image=image, video=video, mode=True)
        rec.calls.clear()
        result = publish_row(publisher, database, ids[0], pid, content, image, video)
        check(result["success"], f"{name}: the post should go out: {result}")
        check(rec.kinds() == [kind, "comment"],
              f"{name}: the post must go first and the comment after it: {rec.kinds()}")
        post, comment = rec.calls
        check(post["text"] == COPY, f"{name}: the post must be the copy without the link: {post['text']!r}")
        check("http" not in post["text"], f"{name}: a link is still in the post")
        if image:
            check(post["image"] == image, f"{name}: the image must still go with the post")
        if video:
            check(post["video"] == video, f"{name}: the video must still go with the post")
        check(comment["text"] == LINK, f"{name}: the comment must be the link: {comment['text']!r}")
        check(comment["actor"] == author_of(1), f"{name}: wrong commenter: {comment['actor']}")
        check(comment["token"] == token_of(1) and post["token"] == token_of(1),
              f"{name}: the comment must use the token the post used: {post['token']} / {comment['token']}")
        check(comment["post"] == "urn:linkedin:1",
              f"{name}: the comment must be on the post that was just made: {comment['post']}")
        check(result["first_comment"]["status"] == "posted" and result["first_comment"]["error"] is None,
              f"{name}: the result must say the comment posted: {result.get('first_comment')}")
        saved = state_of(database, pid)
        check(saved and saved["status"] == "posted" and saved["text"] == LINK
              and saved["post_urn"] == "urn:linkedin:1" and saved["comment_id"] == "c2",
              f"{name}: the row must record the posted comment: {saved}")

    # several links are one comment, once each, in order
    content = "Two things\n\nhttps://a.test/1\nhttps://a.test/2\nhttps://a.test/1"
    pid = make_post(database, ids[0], content, mode=True)
    rec.calls.clear()
    result = publish_row(publisher, database, ids[0], pid, content)
    check(result["success"] and rec.kinds() == ["text", "comment"], f"two links: {rec.kinds()}")
    check(rec.calls[0]["text"] == "Two things" and rec.calls[1]["text"] == "https://a.test/1\nhttps://a.test/2",
          f"two links must be one comment, once each: {rec.calls}")

    # a caller can decide for itself, with no saved row at all
    rec.calls.clear()
    result = publisher.publish("linkedin", ids[0], content=COPY + "\n" + LINK, link_in_comment=True,
                               db_path=P)
    check(result["success"] and rec.kinds() == ["text", "comment"] and rec.calls[0]["text"] == COPY,
          f"an explicit link_in_comment=True needs no row: {rec.calls}")

    # the account's setting applies to a post that has made no choice
    database, web, publisher, client, ids, rec, _ = rig(account_default=True)
    pid = make_post(database, ids[0], COPY + "\n" + LINK)          # no choice of its own
    result = publish_row(publisher, database, ids[0], pid, COPY + "\n" + LINK)
    check(rec.kinds() == ["text", "comment"] and result["first_comment"]["status"] == "posted",
          f"an account set to send links as comments must do so for a post with no choice: {rec.kinds()}")

    print("publish: text, image and video posts go out without the link and the link follows once as the "
          "first comment, by the same author on that post with the same token; several links make one comment")
    print("LINK_COMMENT_PUBLISH_OK")


# ---------------------------------------------------------------------------
# off: a post that does not use it is what it always was
# ---------------------------------------------------------------------------

def section_off():
    content = COPY + "\n" + LINK

    # Each scenario gets a rig of its own and is finished before the next begins:
    # a rig re-points the shared database module at its throwaway file.
    scenarios = [
        ("no setting", {}, None, {}),
        ("account on, post off", {"account_default": True}, False, {}),
        ("caller off, post on", {}, True, {"link_in_comment": False}),
    ]
    for name, rig_args, mode, extra in scenarios:
        database, web, publisher, client, ids, rec, clients = rig(**rig_args)
        pid = make_post(database, ids[0], content, mode=mode)
        for image in (None, IMAGE):
            rec.calls.clear()
            result = publish_row(publisher, database, ids[0], pid, content, image, **extra)
            kind = "image" if image else "text"
            want = {"kind": kind, "token": token_of(1), "text": content}
            want.update({"image": image} if image else {"urn": author_of(1)})
            check(rec.calls == [want],
                  f"{name} / {kind}: the call to LinkedIn must be exactly what it was:\n  want {[want]}\n  got  {rec.calls}")
            check(result["success"] and set(result) == LEGACY_RESULT_KEYS,
                  f"{name} / {kind}: the result must have only the keys it always had: {sorted(result)}")
            check(state_of(database, pid) is None, f"{name}: nothing may be recorded about a comment")

    # other platforms never see it, even when a caller asks
    database, web, publisher, client, ids, rec, clients = rig()
    connect(database, "threads", "th-1", "Thr")
    out = publisher.publish("threads", None, content=content, link_in_comment=True, db_path=database.DB_PATH)
    check(out["success"] and set(out) == LEGACY_RESULT_KEYS, f"threads result changed: {sorted(out)}")
    check(clients["threads"].calls == [{"kind": "text", "token": "token-threads-th-1", "text": content}],
          f"Threads must get the copy untouched, link included: {clients['threads'].calls}")
    check(rec.calls == [], "LinkedIn must not be called for a Threads post")

    # a failed post stays a failed post, with no comment attempt
    database, web, publisher, client, ids, rec, clients = rig()
    rec.fail = True
    pid = make_post(database, ids[0], content, mode=True)
    result = publish_row(publisher, database, ids[0], pid, content)
    check(not result["success"] and "first_comment" not in result and rec.kinds() == ["text"],
          f"a post LinkedIn refused must not try a comment: {result} {rec.kinds()}")
    check(rec.calls[0]["text"] == COPY, "the failed post was still sent without its link")
    check(state_of(database, pid) is None, "a post that never went out has no comment record")

    print("off: with the setting absent, switched off on the post, or overridden by the caller, LinkedIn "
          "gets the same calls and the result the same keys as before; other platforms never see it")
    print("LINK_COMMENT_OFF_OK")


# ---------------------------------------------------------------------------
# failure: a refused comment is not a failed post
# ---------------------------------------------------------------------------

def section_failure():
    content = COPY + "\n" + LINK
    reason = "LinkedIn would not let this connection comment."

    def scenario(setup, expect_error, expect_calls):
        database, web, publisher, client, ids, rec, _ = rig()
        setup(rec)
        pid = make_post(database, ids[0], content, mode=True)
        result = publish_row(publisher, database, ids[0], pid, content)
        check(result["success"] is True,
              f"the post is live, so it must be reported as published: {result}")
        check(result["error"] is None and not result["needs_reconnect"] and not result["permanent"],
              f"a comment problem must not read as a post problem: {result}")
        note = result["first_comment"]
        check(note["status"] == "failed" and expect_error in note["error"],
              f"the result must carry the failure and its reason: {note}")
        check(note["text"] == LINK, f"the link must be kept to paste by hand: {note}")
        check(rec.kinds() == expect_calls, f"calls: {rec.kinds()} (wanted {expect_calls})")
        check(len(rec.posts()) == 1, "the post must have been sent exactly once")
        saved = state_of(database, pid)
        check(saved and saved["status"] == "failed" and expect_error in saved["error"]
              and saved["text"] == LINK,
              f"the row must record the failure, its reason and the link: {saved}")
        return database, web, publisher, client, ids, rec, pid, result

    # LinkedIn refuses it
    database, web, publisher, client, ids, rec, pid, result = scenario(
        lambda r: setattr(r, "comment_error", reason), reason, ["text", "comment"])
    check(state_of(database, pid)["post_urn"] == "urn:linkedin:1", "the row must remember which post to retry on")

    # the client blows up
    scenario(lambda r: setattr(r, "comment_raises", RuntimeError("boom")), "boom", ["text", "comment"])
    # a reply with no reason
    scenario(lambda r: setattr(r, "comment_reply", FakeResult(success=False)),
             "LinkedIn gave no reason", ["text", "comment"])
    # the post comes back with no id: nothing to comment on, and no request made for it
    scenario(lambda r: setattr(r, "no_post_urn", True), "did not say which post", ["text"])

    # the record being unwritable does not undo a live post either
    database, web, publisher, client, ids, rec, _ = rig()
    pid = make_post(database, ids[0], content, mode=True)

    def unwritable(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    original = publisher.database.set_standalone_post_link_comment_state
    publisher.database.set_standalone_post_link_comment_state = unwritable
    try:
        result = publish_row(publisher, database, ids[0], pid, content)
    finally:
        publisher.database.set_standalone_post_link_comment_state = original
    check(result["success"] and result["first_comment"]["status"] == "posted",
          f"a record that could not be saved must not fail a live post: {result}")

    # the scheduler: the entry is posted once and is not left for another pass
    database, web, publisher, client, ids, rec, _ = rig()
    rec.comment_error = reason
    P = database.DB_PATH
    pid = make_post(database, ids[0], content, mode=True)
    sid = database.add_scheduled_post(scheduled_for="2000-01-01T00:00:00", post_type="standalone",
                                      standalone_post_id=pid, platform="linkedin",
                                      account_id=ids[0], db_path=P)
    due = [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid]
    check(len(due) == 1, "the queued post should be due")
    result = web._publish_scheduled_entry(due[0])
    check(result["success"] and result["first_comment"]["status"] == "failed",
          f"the scheduler's entry must be published with a failed comment: {result}")
    entry = database.get_scheduled_post(sid, db_path=P)
    check(entry["status"] == "posted" and (entry["retry_count"] or 0) == 0,
          f"the entry must be posted, not retried: {dict(entry)}")
    check(not [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid],
          "nothing is left pending for the next scheduler pass to post again")
    check(len(rec.posts()) == 1, f"the post went out {len(rec.posts())} times")

    print("failure: a refused, exploding or id-less comment leaves the post reported as published once, "
          "keeps the reason and the link on the row, and the scheduler does not post it again")
    print("LINK_COMMENT_FAILURE_OK")


# ---------------------------------------------------------------------------
# refuse: what cannot carry its comment is refused before anything is sent
# ---------------------------------------------------------------------------

def section_refuse():
    from linkedin_client import COMMENT_MAX_CHARS

    def link_of_length(n, tag="x"):
        base = "https://a.test/"
        return base + tag * (n - len(base))

    database, web, publisher, client, ids, rec, _ = rig()
    P = database.DB_PATH

    # at the limit it goes; one over is refused with nothing sent
    at_limit = link_of_length(COMMENT_MAX_CHARS)
    check(len(at_limit) == COMMENT_MAX_CHARS, "test link has the wrong length")
    content = f"Words\n{at_limit}"
    pid = make_post(database, ids[0], content, mode=True)
    result = publish_row(publisher, database, ids[0], pid, content)
    check(result["success"] and rec.kinds() == ["text", "comment"],
          f"a comment exactly at LinkedIn's limit must go: {result} {rec.kinds()}")

    rec.calls.clear()
    over = link_of_length(COMMENT_MAX_CHARS + 1)
    content = f"Words\n{over}"
    pid = make_post(database, ids[0], content, mode=True)
    result = publish_row(publisher, database, ids[0], pid, content)
    check(not result["success"] and result["permanent"], f"one over must be refused for good: {result}")
    check(str(COMMENT_MAX_CHARS) in result["error"] and str(COMMENT_MAX_CHARS + 1) in result["error"],
          f"the refusal must say the length and the limit: {result['error']}")
    check(rec.calls == [], f"a refused post must send nothing, not even a token refresh: {rec.calls}")
    check(state_of(database, pid) is None, "a refused post leaves no comment record")

    # many links that add up to too much
    rec.calls.clear()
    many = "\n".join(f"https://a.test/{n}/" + "y" * 60 for n in range(25))
    result = publish_row(publisher, database, ids[0], make_post(database, ids[0], "Words\n" + many, mode=True),
                         "Words\n" + many)
    check(not result["success"] and rec.calls == [], f"25 long links must be refused unsent: {result}")

    # the same long link is fine when the feature is off: it was never the comment's business
    rec.calls.clear()
    off_id = make_post(database, ids[0], f"Words\n{over}", mode=False)
    result = publish_row(publisher, database, ids[0], off_id, f"Words\n{over}")
    check(result["success"] and rec.kinds() == ["text"], f"feature off must not refuse: {result}")

    # nothing left to post once the link moves
    for image, video in ((None, None), (IMAGE, None), (None, VIDEO)):
        rec.calls.clear()
        only = f"  \n{LINK}\n"
        pid = make_post(database, ids[0], only, image=image, video=video, mode=True)
        result = publish_row(publisher, database, ids[0], pid, only, image, video)
        check(not result["success"] and result["permanent"] and "Nothing is left" in result["error"],
              f"a post of only a link must be refused (image={bool(image)} video={bool(video)}): {result}")
        check(rec.calls == [], "a post with nothing left must send nothing")
    # ...but a link-only post is fine when the link stays in it
    rec.calls.clear()
    off_only = make_post(database, ids[0], LINK, mode=False)
    result = publish_row(publisher, database, ids[0], off_only, LINK)
    check(result["success"] and rec.calls[0]["text"] == LINK, f"link-only with the feature off: {result}")

    # no link at all: it just posts, as written, with no comment and no record
    rec.calls.clear()
    plain = "No link in this one.\n\nJust words."
    pid = make_post(database, ids[0], plain, mode=True)
    result = publish_row(publisher, database, ids[0], pid, plain)
    check(result["success"] and rec.calls == [{"kind": "text", "token": token_of(1), "urn": author_of(1), "text": plain}],
          f"a post with no link must go out untouched: {rec.calls}")
    check("first_comment" not in result and state_of(database, pid) is None,
          f"a post with no link has no comment to report: {sorted(result)}")

    print("refuse: a comment over the limit and a post with nothing left are refused with nothing sent; "
          "the limit itself goes; the same posts are fine with the feature off; a post with no link just posts")
    print("LINK_COMMENT_REFUSE_OK")


# ---------------------------------------------------------------------------
# retry: post the comment again, only the comment
# ---------------------------------------------------------------------------

def age_token(database, account_id, refresh_token):
    """Expire an account's LinkedIn token, leaving it a refresh token or none."""
    with sqlite3.connect(database.DB_PATH) as conn:
        changed = conn.execute(
            "UPDATE linkedin_tokens SET expires_at = '2000-01-01T00:00:00', refresh_token = ? "
            "WHERE account_id = ?", (refresh_token, account_id)).rowcount
        conn.commit()
    check(changed == 1, f"setup: expected to age one token, aged {changed}")


def section_retry():
    content = COPY + "\n" + LINK
    reason = "LinkedIn would not let this connection comment."

    def failed_card(accounts=1):
        database, web, publisher, client, ids, rec, _ = rig(accounts=accounts)
        rec.comment_error = reason
        pids = []
        for n, account in enumerate(ids, start=1):
            pid = make_post(database, account, content, mode=True)
            result = publish_row(publisher, database, account, pid, content)
            check(result["first_comment"]["status"] == "failed", "setup: the comment should have failed")
            pids.append(pid)
        rec.comment_error = None
        rec.calls.clear()
        return database, web, publisher, client, ids, rec, pids

    def retry(client, pid, pids):
        return client.post(f"/compose/post/{pid}/link-comment/retry", data={"post_ids": ",".join(map(str, pids))})

    # --- a retry posts the stored words on the stored post, with this account's token
    database, web, publisher, client, ids, rec, pids = failed_card()
    pid = pids[0]
    # the copy is edited after the failure; the retry still posts the words kept at the time
    database.update_standalone_post(pid, "Edited since: no link at all now", db_path=database.DB_PATH)
    reply = retry(client, pid, pids)
    body = reply.get_json()
    check(reply.status_code == 200 and body["success"], f"the retry should work: {reply.status_code} {body}")
    check(rec.kinds() == ["comment"], f"a retry must send the comment and nothing else: {rec.kinds()}")
    comment = rec.calls[0]
    check(comment["text"] == LINK and comment["post"] == "urn:linkedin:1" and comment["actor"] == author_of(1)
          and comment["token"] == token_of(1),
          f"the retry must use the words, post and account saved at the failure: {comment}")
    check(state_of(database, pid)["status"] == "posted", "a retry that works must record it")
    check(body["link_comment"]["state"]["status"] == "posted", f"the reply must show it posted: {body}")

    # --- never twice
    rec.calls.clear()
    again = retry(client, pid, pids)
    check(again.status_code == 400 and "no failed link comment" in again.get_json()["error"].lower(),
          f"a second retry must say there is nothing to retry: {again.status_code} {again.get_json()}")
    check(rec.calls == [], "a second retry must not comment again")

    # the publisher refuses on its own as well, not only the route in front of it: a comment
    # already posted, and a post that never had one, are not retried by anything that calls it
    rec.calls.clear()
    for row_id, why in ((pid, "already posted"), (make_post(database, ids[0], "No record\n" + LINK, mode=True), "no record")):
        direct = publisher.retry_first_comment(row_id, db_path=database.DB_PATH)
        check(direct["success"] is False and "no failed link comment" in direct["error"].lower() and rec.calls == [],
              f"retry_first_comment must refuse a post with {why}: {direct} {rec.calls}")

    # --- a post that never failed has nothing to retry
    ok_id = make_post(database, ids[0], "Fresh\n" + LINK, mode=True)
    check(retry(client, ok_id, [ok_id]).status_code == 400 and rec.calls == [], "nothing failed, nothing sent")
    check(client.post("/compose/post/999999/link-comment/retry").status_code == 404, "unknown post -> 404")

    # --- still refused: stays failed, new reason shown, retryable again
    database, web, publisher, client, ids, rec, pids = failed_card()
    rec.comment_error = "LinkedIn is limiting how fast this account can comment."
    reply = retry(client, pids[0], pids)
    body = reply.get_json()
    check(reply.status_code == 400 and not body["success"], f"a retry that fails must say so: {reply.status_code} {body}")
    check("limiting how fast" in body["message"] and
          "limiting how fast" in body["link_comment"]["state"]["error"],
          f"the new reason must replace the old: {body}")
    check(state_of(database, pids[0])["status"] == "failed" and rec.kinds() == ["comment"],
          "a failed retry leaves the comment failed and sends one comment")
    rec.comment_error = None
    rec.calls.clear()
    check(retry(client, pids[0], pids).get_json()["success"] and rec.kinds() == ["comment"],
          "and it can be retried again")

    # --- two accounts: each comment speaks as its own account
    database, web, publisher, client, ids, rec, pids = failed_card(accounts=2)
    reply = retry(client, pids[0], pids)
    check(reply.status_code == 200 and reply.get_json()["success"], f"two-account retry: {reply.get_json()}")
    check(rec.kinds() == ["comment", "comment"], f"one comment per failed row: {rec.kinds()}")
    spoke = sorted((c["token"], c["actor"]) for c in rec.comments())
    check(spoke == sorted([(token_of(1), author_of(1)), (token_of(2), author_of(2))]),
          f"each account's comment must use its own token and author: {spoke}")
    check(all(state_of(database, p)["status"] == "posted" for p in pids), "both rows record it")

    # --- an expired token is refreshed first, like a post would be
    database, web, publisher, client, ids, rec, pids = failed_card()
    age_token(database, ids[0], refresh_token="refresh-li-1")
    reply = retry(client, pids[0], pids)
    check(reply.status_code == 200, f"a retry with an expired token should refresh it: {reply.get_json()}")
    check(rec.kinds() == ["refresh", "comment"] and rec.calls[1]["token"] == "refreshed-refresh-li-1",
          f"the comment must use the refreshed token: {rec.calls}")

    # --- a login that cannot be refreshed asks for a reconnect and changes nothing
    database, web, publisher, client, ids, rec, pids = failed_card()
    age_token(database, ids[0], refresh_token=None)
    reply = retry(client, pids[0], pids)
    check(reply.status_code == 401 and reply.get_json()["needs_reconnect"], f"reconnect expected: {reply.status_code} {reply.get_json()}")
    check(rec.calls == [] and state_of(database, pids[0])["status"] == "failed",
          "an unusable login sends nothing and leaves the comment failed")

    # --- a failure recorded without the post's id cannot be retried, and says why
    database, web, publisher, client, ids, rec, pids = failed_card()
    database.set_standalone_post_link_comment_state(
        pids[0], {"status": "failed", "text": LINK, "post_urn": None, "error": "x"}, db_path=database.DB_PATH)
    reply = retry(client, pids[0], pids)
    check(reply.status_code == 400 and rec.calls == [], f"no post id, no retry: {reply.get_json()}")
    check("by hand" in json.dumps(reply.get_json()), f"it must say what to do instead: {reply.get_json()}")

    print("retry: only the comment is sent, with the saved words on the saved post and each row's own "
          "account; never twice; a refusal stays failed with the new reason; expired and unusable logins handled")
    print("LINK_COMMENT_RETRY_OK")


# ---------------------------------------------------------------------------
# persist: the columns, the upgrade, and the order of precedence
# ---------------------------------------------------------------------------

def section_persist():
    import database as db
    directory = tempfile.mkdtemp(prefix="link_comment_persist_")
    path = os.path.join(directory, "old.db")
    os.chdir(directory)
    db.init_db(path)

    # make it an old database: the new columns never existed
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE standalone_posts DROP COLUMN link_comment")
        conn.execute("ALTER TABLE standalone_posts DROP COLUMN link_comment_state")
        conn.execute("ALTER TABLE social_accounts DROP COLUMN link_in_comment")
        conn.commit()
    with sqlite3.connect(path) as conn:
        post_columns = [r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")]
        account_columns = [r[1] for r in conn.execute("PRAGMA table_info(social_accounts)")]
    check("link_comment" not in post_columns and "link_in_comment" not in account_columns,
          "setup: the old database should not have the new columns")

    acct = db.upsert_social_account("linkedin", "m-1", display_name="Old", db_path=path)
    # written the way the app wrote a post before the columns existed
    old_posts = []
    with sqlite3.connect(path) as conn:
        for n in range(3):
            cur = conn.execute(
                "INSERT INTO standalone_posts (source_type, source_content, platform, content, created_at, "
                "used, account_id) VALUES ('manual', 's', 'linkedin', ?, '2026-01-01T00:00:00', 0, ?)",
                (f"Old post {n}\n{LINK}", acct))
            old_posts.append(cur.lastrowid)
        conn.commit()
    with sqlite3.connect(path) as conn:
        before_rows = conn.execute("SELECT id, platform, content, account_id FROM standalone_posts ORDER BY id").fetchall()
        before_accounts = conn.execute("SELECT id, platform, external_id, label FROM social_accounts ORDER BY id").fetchall()

    db.init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        post_columns = [r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")]
        account_columns = [r[1] for r in conn.execute("PRAGMA table_info(social_accounts)")]
        schema_after_first = (post_columns, account_columns)
        rows = conn.execute("SELECT id, platform, content, account_id, link_comment, link_comment_state "
                            "FROM standalone_posts ORDER BY id").fetchall()
        accounts = conn.execute("SELECT id, platform, external_id, label, link_in_comment "
                                "FROM social_accounts ORDER BY id").fetchall()
    check({"link_comment", "link_comment_state"} <= set(post_columns) and "link_in_comment" in account_columns,
          f"the upgrade must add the columns: {post_columns} / {account_columns}")
    check([(r["id"], r["platform"], r["content"], r["account_id"]) for r in rows] == [tuple(r) for r in before_rows],
          "the upgrade must keep every post exactly as it was")
    check([(r["id"], r["platform"], r["external_id"], r["label"]) for r in accounts] == [tuple(r) for r in before_accounts],
          "the upgrade must keep every account exactly as it was")
    check(all(r["link_comment"] is None and r["link_comment_state"] is None for r in rows),
          "an upgraded post follows its account, with nothing recorded")
    check(all(r["link_in_comment"] == 0 for r in accounts), "an upgraded account keeps links in the post")

    db.init_db(path)            # twice changes nothing
    with sqlite3.connect(path) as conn:
        again = ([r[1] for r in conn.execute("PRAGMA table_info(standalone_posts)")],
                 [r[1] for r in conn.execute("PRAGMA table_info(social_accounts)")])
        rows_again = conn.execute("SELECT id, platform, content, account_id FROM standalone_posts ORDER BY id").fetchall()
    check(again == schema_after_first and rows_again == before_rows, "running the upgrade again changed something")

    # --- the setters round-trip
    pid = old_posts[0]
    for value, stored in ((True, 1), (False, 0), (None, None)):
        db.set_standalone_post_link_comment(pid, value, db_path=path)
        row = db.get_standalone_post(pid, db_path=path)
        check(row["link_comment"] == stored, f"set {value!r} read back {row['link_comment']!r}")
    check(db.set_social_account_link_in_comment(acct, True, db_path=path)
          and db.get_social_account(acct, db_path=path)["link_in_comment"] == 1, "account setter (on)")
    check(db.set_social_account_link_in_comment(acct, False, db_path=path)
          and db.get_social_account(acct, db_path=path)["link_in_comment"] == 0, "account setter (off)")
    check(db.set_social_account_link_in_comment(99999, True, db_path=path) is False, "unknown account -> False")
    new_id = db.add_standalone_post("manual", "s", "linkedin", "New", db_path=path, link_comment=True)
    check(db.get_standalone_post(new_id, db_path=path)["link_comment"] == 1, "add_standalone_post keeps the choice")

    # --- the record reads back, and a damaged one reads as nothing
    note = {"status": "failed", "text": LINK, "error": "e", "post_urn": "urn:li:share:1"}
    db.set_standalone_post_link_comment_state(pid, note, db_path=path)
    check(db.link_comment_state(db.get_standalone_post(pid, db_path=path)) == note, "state round-trip")
    db.set_standalone_post_link_comment_state(pid, None, db_path=path)
    check(db.link_comment_state(db.get_standalone_post(pid, db_path=path)) is None, "clearing the state")
    for damaged in ("{not json", "[1, 2]", '"text"', '{"no_status": 1}', ""):
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE standalone_posts SET link_comment_state = ? WHERE id = ?", (damaged, pid))
            conn.commit()
        check(db.link_comment_state(db.get_standalone_post(pid, db_path=path)) is None,
              f"a damaged record {damaged!r} must read as nothing")

    # --- the post's choice, then the account's, then no
    class Row(dict):
        pass
    table = [  # (post's choice, account's setting, expected)
        (None, 0, False), (None, 1, True), (0, 1, False), (1, 0, True), (1, 1, True), (0, 0, False),
        (1, None, True), (None, None, False), (0, None, False),
    ]
    for post_choice, account_setting, expected in table:
        post = Row(link_comment=post_choice)
        account = None if account_setting is None else Row(link_in_comment=account_setting)
        got = db.link_in_comment_enabled(post, account)
        check(got is expected, f"post {post_choice!r} / account {account_setting!r}: want {expected}, got {got}")
    check(db.link_in_comment_enabled(None, None) is False, "no row and no account: links stay in the post")

    print("persist: an old database upgrades without losing a row and a second upgrade changes nothing; "
          "the setters round-trip; a damaged record reads as nothing; post choice beats account beats off")
    print("LINK_COMMENT_PERSIST_OK")


# ---------------------------------------------------------------------------
# routes: the card's choice, inheritance, and the account's default
# ---------------------------------------------------------------------------

def section_routes():
    database, web, publisher, client, ids, rec, clients = rig(accounts=2)
    P = database.DB_PATH
    a, b = ids
    text = COPY + "\n" + LINK
    li_a = make_post(database, a, text)
    li_b = make_post(database, b, text)
    th = make_post(database, None, text, platform="threads")
    other = make_post(database, a, "A different post\n" + LINK)
    card = f"{li_a},{li_b},{th},{other}"        # `other` is stale: different copy, must be ignored

    def link_of(pid):
        return row_of(database, pid)["link_comment"]

    def choose(pid, mode, post_ids=card, **kw):
        return client.post(f"/compose/post/{pid}/link-comment", data={"mode": mode, "post_ids": post_ids}, **kw)

    # --- on / off / default reach every LinkedIn row of the card and nothing else
    for mode, stored in (("on", 1), ("off", 0), ("default", None)):
        reply = choose(li_a, mode)
        body = reply.get_json()
        check(reply.status_code == 200 and body["success"] and body["mode"] == mode, f"{mode}: {reply.status_code} {body}")
        check(link_of(li_a) == stored and link_of(li_b) == stored, f"{mode}: both LinkedIn rows must change")
        check(link_of(th) is None and link_of(other) is None,
              f"{mode}: the Threads row and a different post must not change")
        check(sorted(body["updated_ids"]) == sorted([li_a, li_b]), f"{mode}: updated ids {body['updated_ids']}")
        check(body["link_comment"]["mode"] == mode, f"{mode}: summary {body['link_comment']}")
    reply = client.post(f"/compose/post/{li_a}/link-comment", json={"mode": "on", "post_ids": [li_a, li_b]})
    check(reply.status_code == 200 and link_of(li_b) == 1, "the JSON form must work too")

    # the summary says what will happen, account default included
    database.set_social_account_link_in_comment(a, True, db_path=P)
    summary = choose(li_a, "default").get_json()["link_comment"]
    check(summary["mode"] == "default" and summary["account_default"] is True and summary["effective"] is True,
          f"following an account that sends links as comments: {summary}")
    summary = choose(li_a, "off").get_json()["link_comment"]
    check(summary["effective"] is False and summary["account_default"] is True, f"a post that says no beats the account: {summary}")
    database.set_social_account_link_in_comment(a, False, db_path=P)

    # --- refusals write nothing
    choose(li_a, "on")
    for bad in ("maybe", "", "ON ", "1", "true", None):
        data = {"post_ids": card}
        if bad is not None:
            data["mode"] = bad
        reply = client.post(f"/compose/post/{li_a}/link-comment", data=data)
        if bad == "ON ":                      # the one forgiving case: case and spacing
            check(reply.status_code == 200, f"{bad!r} should be accepted as on")
            continue
        check(reply.status_code == 400 and "mode must be" in reply.get_json()["error"],
              f"mode {bad!r} must be refused: {reply.status_code} {reply.get_json()}")
        check(link_of(li_a) == 1 and link_of(li_b) == 1, f"mode {bad!r} must write nothing")
    check(choose(th, "on", post_ids=str(th)).status_code == 400 and link_of(th) is None,
          "a card with no LinkedIn row has nothing to choose")
    check(choose(999999, "on").status_code == 404, "unknown post -> 404")

    # --- ticking a LinkedIn account onto a card inherits an explicit choice
    def new_linkedin_row(content, account):
        rows = [r for r in database.list_standalone_posts(db_path=P)
                if r["content"] == content and r["platform"] == "linkedin" and r["account_id"] == account]
        return rows[0] if rows else None

    for explicit, stored in ((None, None), (True, 1), (False, 0)):
        content = f"Tick copy {explicit}\n{LINK}"
        src = make_post(database, a, content, mode=explicit)
        reply = client.post(f"/compose/post/{src}/platform", data={
            "platform": "linkedin", "action": "add", "account_id": str(b), "post_ids": str(src)})
        check(reply.status_code == 200, f"ticking a second account: {reply.status_code} {reply.get_data(as_text=True)[:200]}")
        added = new_linkedin_row(content, b)
        check(added is not None and added["link_comment"] == stored,
              f"a ticked-on LinkedIn row must publish like the card's other one ({explicit!r}): "
              f"{None if added is None else added['link_comment']!r}")
        # a non-LinkedIn target does not take it
        client.post(f"/compose/post/{src}/platform", data={
            "platform": "threads", "action": "add", "post_ids": f"{src},{added['id']}"})
        threads_row = next(r for r in database.list_standalone_posts(db_path=P)
                           if r["content"] == content and r["platform"] == "threads")
        check(threads_row["link_comment"] is None, "a Threads row must not carry the LinkedIn choice")

    content = f"Bulk copy\n{LINK}"
    src = make_post(database, a, content, mode=True)
    reply = client.post("/compose/posts/bulk-add-platform",
                        json={"post_ids": [src], "targets": [f"linkedin:{b}"], "queue": False})
    check(reply.status_code == 200, f"bulk add: {reply.status_code} {reply.get_data(as_text=True)[:200]}")
    added = new_linkedin_row(content, b)
    check(added is not None and added["link_comment"] == 1, "bulk Add Platform must carry the choice too")

    # --- the account's standing choice
    def account(account_id, value, **kw):
        return client.post(f"/accounts/{account_id}/link-in-comment", data={"enabled": value}, **kw)

    def setting(account_id):
        return database.get_social_account(account_id, db_path=P)["link_in_comment"]

    for value, stored in (("true", 1), ("false", 0), ("on", 1), ("off", 0), ("1", 1), ("0", 0)):
        reply = account(a, value)
        check(reply.status_code == 200 and reply.get_json()["link_in_comment"] is bool(stored) and setting(a) == stored,
              f"enabled={value}: {reply.status_code} {reply.get_json()} stored {setting(a)}")
    account(a, "true")
    for bad in ("", "maybe", "2"):
        reply = account(a, bad)
        check(reply.status_code == 400 and setting(a) == 1, f"enabled={bad!r} must be refused and change nothing")
    reply = client.post(f"/accounts/{a}/link-in-comment")
    check(reply.status_code == 400 and setting(a) == 1, "no value must be refused")
    check(client.post(f"/accounts/{a}/link-in-comment", json={"enabled": False}).status_code == 200
          and setting(a) == 0, "the JSON form must work too")
    check(setting(b) == 0, "one account's setting must not touch another")

    threads_account = connect(database, "threads", "th-1", "Thr")
    reply = account(threads_account, "true")
    check(reply.status_code == 400 and "LinkedIn" in reply.get_json()["error"], f"only LinkedIn accounts: {reply.get_json()}")
    check(database.get_social_account(threads_account, db_path=P)["link_in_comment"] in (0, None),
          "a Threads account must not be switched on")
    check(account(999999, "true").status_code == 404, "unknown account -> 404")

    listing = client.get("/accounts/list").get_json()
    summaries = [acct for p in listing["platforms"] if p["platform"] == "linkedin" for acct in p["accounts"]]
    check(summaries and all("link_in_comment" in s for s in summaries), f"the accounts list must carry the setting: {summaries}")

    print("routes: the card route sets every LinkedIn row of the card and nothing else and refuses bad input "
          "unwritten; ticking and bulk-adding a LinkedIn target inherit it; the account route is LinkedIn-only")
    print("LINK_COMMENT_ROUTES_OK")


# ---------------------------------------------------------------------------
# paths: Post now, the whole-card publish and the scheduler
# ---------------------------------------------------------------------------

def section_paths():
    content = COPY + "\n" + LINK
    reason = "LinkedIn would not let this connection comment."

    for failing in (False, True):
        label = "refused" if failing else "accepted"

        def fresh():
            database, web, publisher, client, ids, rec, clients = rig()
            connect(database, "threads", "th-1", "Thr")
            if failing:
                rec.comment_error = reason
            return database, web, publisher, client, ids, rec, clients

        def comment_of(body):
            note = body.get("first_comment")
            check(note and note["status"] == ("failed" if failing else "posted"),
                  f"[{label}] the reply must carry the comment's outcome: {body}")
            if failing:
                check(reason in note["error"], f"[{label}] the reason must reach the page: {note}")
            return note

        # --- Post now
        database, web, publisher, client, ids, rec, clients = fresh()
        pid = make_post(database, ids[0], content, mode=True)
        reply = client.post(f"/compose/post/{pid}/linkedin")
        body = reply.get_json()
        check(reply.status_code == 200 and body["success"], f"[{label}] Post now must report the post as published: {body}")
        comment_of(body)
        check(rec.kinds() == ["text", "comment"] and rec.calls[0]["text"] == COPY and rec.calls[1]["text"] == LINK,
              f"[{label}] Post now: {rec.calls}")
        history = [r for r in database.list_scheduled_posts(db_path=database.DB_PATH) if r["standalone_post_id"] == pid]
        check(len(history) == 1 and history[0]["status"] == "posted", f"[{label}] one history row: {[dict(h) for h in history]}")

        # --- the whole-card publish, LinkedIn beside Threads
        database, web, publisher, client, ids, rec, clients = fresh()
        li = make_post(database, ids[0], content, mode=True)
        th = make_post(database, None, content, platform="threads")
        reply = client.post(f"/compose/post/{li}/publish", data={"post_ids": f"{li},{th}"})
        body = reply.get_json()
        check(reply.status_code == 200 and body["success"] and body["published"] == 2,
              f"[{label}] a lost comment must not make the card publish partial: {body}")
        by_platform = {r["platform"]: r for r in body["results"]}
        comment_of(by_platform["linkedin"])
        check(by_platform["threads"]["first_comment"] is None, f"[{label}] Threads has no comment: {by_platform['threads']}")
        check(rec.kinds() == ["text", "comment"] and rec.calls[0]["text"] == COPY,
              f"[{label}] whole-card LinkedIn calls: {rec.calls}")
        check(clients["threads"].calls[0]["text"] == content,
              f"[{label}] Threads must keep the link in its post: {clients['threads'].calls}")

        # --- the scheduler's entry
        database, web, publisher, client, ids, rec, clients = fresh()
        P = database.DB_PATH
        pid = make_post(database, ids[0], content, mode=True)
        sid = database.add_scheduled_post(scheduled_for="2000-01-01T00:00:00", post_type="standalone",
                                          standalone_post_id=pid, platform="linkedin",
                                          account_id=ids[0], db_path=P)
        due = [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid]
        check(len(due) == 1, "the queued post should be due")
        result = web._publish_scheduled_entry(due[0])
        check(result["success"], f"[{label}] scheduler: {result}")
        comment_of(result)
        check(rec.kinds() == ["text", "comment"] and rec.calls[0]["text"] == COPY,
              f"[{label}] scheduler calls: {rec.calls}")
        check(database.get_scheduled_post(sid, db_path=P)["status"] == "posted", f"[{label}] entry posted")

        # --- Post now from the schedule page says so in its message
        database, web, publisher, client, ids, rec, clients = fresh()
        P = database.DB_PATH
        pid = make_post(database, ids[0], content, mode=True)
        sid = database.add_scheduled_post(scheduled_for="2099-01-01T00:00:00", post_type="standalone",
                                          standalone_post_id=pid, platform="linkedin",
                                          account_id=ids[0], db_path=P)
        reply = client.post(f"/schedule/{sid}/post-now")
        body = reply.get_json()
        check(reply.status_code == 200 and body["success"], f"[{label}] schedule Post now: {body}")
        if failing:
            check("link comment did not post" in body["message"] and reason in body["message"],
                  f"[{label}] the schedule page's message must mention the lost comment: {body['message']}")
        else:
            check("did not post" not in body["message"], f"[{label}] no warning expected: {body['message']}")

    print("paths: Post now, the whole-card publish, the scheduler and the schedule page's Post now each move "
          "the link to the comment, and report a lost comment without calling the post failed")
    print("LINK_COMMENT_PATHS_OK")


# ---------------------------------------------------------------------------
# ui: the card and the accounts page
# ---------------------------------------------------------------------------

def card_soup(client, pid, post_ids):
    from bs4 import BeautifulSoup
    reply = client.post(f"/compose/post/{pid}/card", data={"post_ids": ",".join(map(str, post_ids))})
    check(reply.status_code == 200, f"the card did not render: {reply.status_code}")
    html = reply.get_json()["html"]
    return BeautifulSoup(html, "html.parser"), html


def hidden(node):
    return "d-none" in (node.get("class") or [])


def section_ui():
    database, web, publisher, client, ids, rec, clients = rig(accounts=2)
    P = database.DB_PATH
    text = COPY + "\n" + LINK
    reason = "LinkedIn would not let this connection comment."

    # a LinkedIn card: the control, each state
    pid = make_post(database, ids[0], text)
    soup, html = card_soup(client, pid, [pid])
    box = soup.select_one(f"#li-link-comment-{pid}")
    check(box is not None, "a LinkedIn card must have the link control")
    select = box.select_one(f"#li-link-mode-{pid}")
    options = {o["value"]: o for o in select.find_all("option")}
    check(set(options) == {"default", "on", "off"}, f"the choices are default, on and off: {list(options)}")
    check("selected" in options["default"].attrs and "selected" not in options["on"].attrs, "a new post follows the account")
    check("Account setting (in the post)" in options["default"].get_text(), f"default label: {options['default'].get_text()}")
    check(f"saveLinkComment({pid})" in select.get("onchange", ""), "the select must save on change")
    check("stays in the post" in box.select_one(".li-link-hint").get_text(), "hint for links in the post")
    check(hidden(box.select_one(".li-link-failed")) and hidden(box.select_one(".li-link-posted")), "no state, nothing shown")
    check(hidden(box.select_one(".li-link-error")), "no error shown at first")

    database.set_social_account_link_in_comment(ids[0], True, db_path=P)
    soup, _ = card_soup(client, pid, [pid])
    box = soup.select_one(f"#li-link-comment-{pid}")
    check("Account setting (first comment)" in box.select("option")[0].get_text(),
          "the default option must say what the account does")
    check("first comment, not in the post" in box.select_one(".li-link-hint").get_text(), "hint for the first comment")

    for stored, want in ((1, "on"), (0, "off")):
        database.set_standalone_post_link_comment(pid, bool(stored), db_path=P)
        soup, _ = card_soup(client, pid, [pid])
        selected = [o["value"] for o in soup.select(f"#li-link-mode-{pid} option[selected]")]
        check(selected == [want], f"a post set to {want} must show it: {selected}")
    check("stays in the post" in soup.select_one(".li-link-hint").get_text(), "off beats the account's setting in the hint")

    # failed: the warning, the reason, the link to paste, and Retry
    database.set_standalone_post_link_comment_state(
        pid, {"status": "failed", "text": LINK, "error": reason, "post_urn": "urn:li:share:1"}, db_path=P)
    soup, _ = card_soup(client, pid, [pid])
    failed = soup.select_one(f"#li-link-comment-{pid} .li-link-failed")
    check(not hidden(failed) and reason in failed.get_text() and failed.select_one(".li-link-text").get_text() == LINK,
          f"a failed comment must show its reason and the link: {failed}")
    button = failed.find("button")
    check(button and f"retryLinkComment({pid}" in button.get("onclick", "") and "Retry" in button.get_text(),
          "a failed comment must offer Retry")
    check(hidden(soup.select_one(f"#li-link-comment-{pid} .li-link-posted")), "failed is not posted")

    # a reason from outside is text, never markup
    nasty = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    database.set_standalone_post_link_comment_state(
        pid, {"status": "failed", "text": LINK, "error": nasty, "post_urn": "urn:li:share:1"}, db_path=P)
    _, html = card_soup(client, pid, [pid])
    check("<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
          and "<img src=x" not in html, "a failure reason must be escaped on the card")

    # posted
    database.set_standalone_post_link_comment_state(
        pid, {"status": "posted", "text": LINK, "comment_id": "c1", "post_urn": "urn:li:share:1"}, db_path=P)
    soup, _ = card_soup(client, pid, [pid])
    check(not hidden(soup.select_one(f"#li-link-comment-{pid} .li-link-posted"))
          and hidden(soup.select_one(f"#li-link-comment-{pid} .li-link-failed")), "a posted comment shows the note only")

    # a card with no LinkedIn row has no control
    th = make_post(database, None, text, platform="threads")
    soup, html = card_soup(client, th, [th])
    check(soup.select_one(".li-link-comment") is None and "retryLinkComment" not in html,
          "a card that does not go to LinkedIn must not show the link control")

    # accounts page: the setting only on LinkedIn accounts, and it shows the stored value
    threads_account = connect(database, "threads", "th-1", "Thr")
    database.set_social_account_link_in_comment(ids[1], False, db_path=P)
    from bs4 import BeautifulSoup
    page = client.get("/accounts")
    check(page.status_code == 200, f"the accounts page did not render: {page.status_code}")
    soup = BeautifulSoup(page.get_data(as_text=True), "html.parser")
    linkedin_rows = soup.select('.account-platform[data-platform="linkedin"] .account-row')
    threads_rows = soup.select('.account-platform[data-platform="threads"] .account-row')
    check(len(linkedin_rows) == 2 and threads_rows, "setup: two LinkedIn rows and a Threads row")
    states = {}
    for row in linkedin_rows:
        toggle = row.select_one("input.link-in-comment-toggle")
        check(toggle is not None, "every LinkedIn account must offer the setting")
        check(f"setLinkInComment({row['data-account-id']}" in toggle.get("onchange", ""),
              "the toggle must call setLinkInComment with its account")
        states[int(row["data-account-id"])] = toggle.has_attr("checked")
    check(states == {ids[0]: True, ids[1]: False}, f"the toggles must show each account's setting: {states}")
    check(all(row.select_one("input.link-in-comment-toggle") is None for row in threads_rows),
          "only LinkedIn accounts offer it")

    print("ui: the card shows the choice with the account default spelled out, the failure with Retry and "
          "the link, the posted note, escapes a hostile reason, and only on LinkedIn cards; the Accounts "
          "page offers the setting on LinkedIn accounts only, showing each stored value")
    print("LINK_COMMENT_UI_OK")


# ---------------------------------------------------------------------------
# js: the page's own functions, under node
# ---------------------------------------------------------------------------

def fn_source(html, name):
    """One top-level function's source, found by matching its braces."""
    match = re.search(rf"(?:async\s+)?function\s+{name}\s*\(", html)
    check(match, f"the page does not define {name}()")
    check(len(re.findall(rf"function\s+{name}\s*\(", html)) == 1, f"{name}() is declared more than once")
    start = html.index("{", html.index(")", match.start()))
    depth = 0
    for i in range(start, len(html)):
        depth += {"{": 1, "}": -1}.get(html[i], 0)
        if depth == 0:
            return html[match.start(): i + 1]
    check(False, f"{name}() has unbalanced braces")


JS_FUNCTIONS = (
    "showLinkCommentError", "renderLinkCommentState", "renderLinkComment", "saveLinkComment",
    "retryLinkComment", "noteFirstComment", "publishCardPlatform", "publishCardToAll",
    "markChipPosted", "postedUrlFrom",
)

JS_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

class FakeFormData {
  constructor() { this.entries = []; }
  append(k, v) { this.entries.push([k, String(v)]); }
}
const node = (extra) => {
  const el = Object.assign({ textContent: '', value: '', disabled: false, classes: new Set(), children: {} }, extra || {});
  el.classList = {
    add: (c) => el.classes.add(c), remove: (c) => el.classes.delete(c),
    contains: (c) => el.classes.has(c),
    toggle: (c, on) => { (on === undefined ? !el.classes.has(c) : on) ? el.classes.add(c) : el.classes.delete(c); },
  };
  el.querySelector = (sel) => el.children[sel] || null;
  return el;
};
// one LinkedIn card's control: the same classes the template draws
function makeBox() {
  const box = node();
  box.children = {
    '.li-link-error': node({ classes: new Set(['d-none']) }),
    '.li-link-hint': node({ textContent: 'The link stays in the post.' }),
    '.li-link-failed': node({ classes: new Set(['d-none']) }),
    '.li-link-posted': node({ classes: new Set(['d-none']) }),
    'select': node({ value: 'default' }),
  };
  const failed = box.children['.li-link-failed'];
  failed.children = { '.li-link-error-text': node(), '.li-link-text': node() };
  return box;
}

function world(opts = {}) {
  const log = { fetch: [], alerts: [], toasts: [], used: 0 };
  const box = opts.noBox ? null : makeBox();
  const chip = node();
  chip.appended = [];
  chip.appendChild = (c) => chip.appended.push(c);
  const postItem = { dataset: { postIds: '5,6' },
    querySelector: (sel) => sel === '.li-link-comment' ? box : (sel.startsWith('.platform-chip') ? chip : null) };
  const select = box ? box.children['select'] : null;
  const ctx = {
    FormData: FakeFormData,
    PLATFORM_LABELS: { linkedin: 'LinkedIn', threads: 'Threads' },
    platformConnected: () => true,
    cardIdsParam: () => '5,6',
    alert: (m) => log.alerts.push(m),
    showToast: (m, kind) => log.toasts.push([m, kind]),
    document: {
      getElementById: (id) => id === 'li-link-comment-5' ? box : (id === 'li-link-mode-5' ? select : null),
      createElement: () => node(),
    },
    fetch: async (url, init) => {
      log.fetch.push({ url, method: init.method, fields: init.body.entries });
      if (opts.fetchThrows) throw new Error('offline');
      const reply = (opts.replies || []).shift() || { status: 200, body: {} };
      return { ok: reply.status >= 200 && reply.status < 300, status: reply.status, json: async () => reply.body };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  return { ctx, log, box, postItem, chip, select };
}
const view = (box) => ({
  hint: box.children['.li-link-hint'].textContent,
  errorText: box.children['.li-link-error'].textContent,
  errorHidden: box.children['.li-link-error'].classes.has('d-none'),
  failedHidden: box.children['.li-link-failed'].classes.has('d-none'),
  failedReason: box.children['.li-link-failed'].children['.li-link-error-text'].textContent,
  failedLink: box.children['.li-link-failed'].children['.li-link-text'].textContent,
  postedHidden: box.children['.li-link-posted'].classes.has('d-none'),
  select: box.children['select'].value,
});

const out = {};
(async () => {
  const FAILED = { status: 'failed', error: 'LinkedIn would not let this connection comment.', text: 'https://blog.test/post-1' };
  const POSTED = { status: 'posted', error: null, text: 'https://blog.test/post-1', comment_id: 'c1' };

  // --- saving the choice
  let w = world({ replies: [{ status: 200, body: { success: true,
    link_comment: { mode: 'on', effective: true, account_default: false, state: null } } }] });
  w.select.value = 'on';
  out.saveOk = { returned: await w.ctx.saveLinkComment(5), sent: w.log.fetch, view: view(w.box) };

  w = world({ replies: [{ status: 200, body: { success: true,
    link_comment: { mode: 'off', effective: false, account_default: true, state: null } } }] });
  w.select.value = 'off';
  out.saveOff = { returned: await w.ctx.saveLinkComment(5), view: view(w.box) };

  w = world({ replies: [{ status: 400, body: { error: 'This post does not go to LinkedIn.' } }] });
  w.select.value = 'on';
  out.saveRefused = { returned: await w.ctx.saveLinkComment(5), view: view(w.box) };

  w = world({ fetchThrows: true });
  w.select.value = 'on';
  out.saveOffline = { returned: await w.ctx.saveLinkComment(5), view: view(w.box) };

  w = world({ noBox: true });
  out.saveNoBox = { returned: await w.ctx.saveLinkComment(5), fetches: w.log.fetch.length };

  // a save that shows the post's earlier failure keeps showing it
  w = world({ replies: [{ status: 200, body: { success: true,
    link_comment: { mode: 'on', effective: true, account_default: false, state: FAILED } } }] });
  await w.ctx.saveLinkComment(5);
  out.saveKeepsFailure = view(w.box);

  // --- the state, drawn from the server's record
  w = world();
  w.ctx.renderLinkCommentState(w.box, FAILED);  out.stateFailed = view(w.box);
  w.ctx.renderLinkCommentState(w.box, POSTED);  out.statePosted = view(w.box);
  w.ctx.renderLinkCommentState(w.box, null);    out.stateNone = view(w.box);
  w.ctx.renderLinkCommentState(null, FAILED);   out.stateNoBox = 'no crash';

  // --- retrying
  w = world({ replies: [{ status: 200, body: { success: true,
    link_comment: { mode: 'on', effective: true, account_default: false, state: POSTED } } }] });
  w.ctx.renderLinkCommentState(w.box, FAILED);
  const btn = { disabled: false };
  const pending = w.ctx.retryLinkComment(5, btn);
  out.retryDuring = btn.disabled;
  out.retryOk = { returned: await pending, sent: w.log.fetch, view: view(w.box), toasts: w.log.toasts, btn: btn.disabled };

  w = world({ replies: [{ status: 400, body: { success: false, message: 'The link comment still did not post: no luck',
    link_comment: { mode: 'on', effective: true, account_default: false,
                    state: { status: 'failed', error: 'no luck', text: 'https://blog.test/post-1' } } } }] });
  w.ctx.renderLinkCommentState(w.box, FAILED);
  out.retryRefused = { returned: await w.ctx.retryLinkComment(5, { disabled: false }), view: view(w.box) };

  w = world({ replies: [{ status: 400, body: { error: 'There is no failed link comment to retry for this post' } }] });
  out.retryNothing = { returned: await w.ctx.retryLinkComment(5, null), view: view(w.box) };

  w = world({ fetchThrows: true });
  const btn2 = { disabled: false };
  out.retryOffline = { returned: await w.ctx.retryLinkComment(5, btn2), view: view(w.box), btn: btn2.disabled };

  // --- a publish result's comment
  w = world();
  out.noteNone = { warning: w.ctx.noteFirstComment(w.postItem, 'Work', null), view: view(w.box) };
  out.notePosted = { warning: w.ctx.noteFirstComment(w.postItem, 'Work', POSTED), view: view(w.box) };
  w = world();
  out.noteFailed = { warning: w.ctx.noteFirstComment(w.postItem, 'Work', FAILED), view: view(w.box) };
  w = world({ noBox: true });
  out.noteNoBox = { warning: w.ctx.noteFirstComment(w.postItem, 'Work', FAILED) };

  // --- post now: the post landed, the comment did not
  w = world({ replies: [{ status: 200, body: { success: true, account_label: 'Work', post_urn: 'urn:li:share:1',
                                               first_comment: FAILED } }] });
  out.postNowLost = { returned: await w.ctx.publishCardPlatform(w.postItem, 5, 'linkedin', 1),
                      alerts: w.log.alerts, chip: w.chip.appended.length, view: view(w.box) };
  w = world({ replies: [{ status: 200, body: { success: true, account_label: 'Work', post_urn: 'urn:li:share:1',
                                               first_comment: POSTED } }] });
  out.postNowKept = { returned: await w.ctx.publishCardPlatform(w.postItem, 5, 'linkedin', 1),
                      alerts: w.log.alerts, view: view(w.box) };
  w = world({ replies: [{ status: 200, body: { success: true, account_label: 'Work', post_urn: 'urn:li:share:1' } }] });
  out.postNowPlain = { returned: await w.ctx.publishCardPlatform(w.postItem, 5, 'linkedin', 1), alerts: w.log.alerts };

  // --- post to all: one lands without its comment, one fails outright
  w = world({ replies: [{ status: 200, body: { success: false, partial: true, published: 2, message: 'Posted to 2 of 3 targets',
    results: [
      { platform: 'linkedin', account_id: 1, account_label: 'Work', success: true, post_urn: 'urn:li:share:1', first_comment: FAILED },
      { platform: 'threads', account_id: 3, account_label: 'Threads', success: true, permalink: 'https://th.test/1', first_comment: null },
      { platform: 'twitter', account_id: 4, account_label: 'X', success: false, error: 'rate limit' },
    ] } }] });
  const all = await w.ctx.publishCardToAll(5, [5, 6, 7], w.postItem);
  out.postAll = { published: all && all.published, alerts: w.log.alerts, toasts: w.log.toasts, view: view(w.box) };

  w = world({ replies: [{ status: 200, body: { success: true, published: 1, message: 'Posted to 1 target',
    results: [{ platform: 'linkedin', account_id: 1, account_label: 'Work', success: true, post_urn: 'urn:li:share:1', first_comment: POSTED }] } }] });
  await w.ctx.publishCardToAll(5, [5], w.postItem);
  out.postAllClean = { alerts: w.log.alerts, toasts: w.log.toasts, view: view(w.box) };

  console.log(JSON.stringify(out));
})();
"""


def section_js():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    database, web, publisher, client, ids, rec, clients = rig()
    pid = make_post(database, ids[0], COPY + "\n" + LINK)
    html = client.get("/compose").get_data(as_text=True)
    functions = "\n".join(fn_source(html, name) for name in JS_FUNCTIONS)

    work = tempfile.mkdtemp(prefix="link_comment_js_")
    try:
        fn_path, harness_path = os.path.join(work, "functions.js"), os.path.join(work, "harness.js")
        with open(fn_path, "w", encoding="utf-8") as handle:
            handle.write(functions)
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(JS_HARNESS)
        run = subprocess.run([node, harness_path, fn_path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check(run.returncode == 0, f"the harness crashed:\n{run.stderr.strip()[:1200]}")
    out = json.loads(run.stdout.strip().splitlines()[-1])

    FAILED_REASON = "LinkedIn would not let this connection comment."
    LINK_TEXT = "https://blog.test/post-1"
    on_hint = "The link is posted as the first comment, not in the post."
    off_hint = "The link stays in the post."

    def fields(call):
        return call["fields"]

    # --- saving
    s = out["saveOk"]
    check(s["returned"] is True and len(s["sent"]) == 1, f"saving: {s}")
    check(s["sent"][0]["url"] == "/compose/post/5/link-comment" and s["sent"][0]["method"] == "POST"
          and fields(s["sent"][0]) == [["mode", "on"], ["post_ids", "5,6"]],
          f"saving must send the mode and the whole card to the card route: {s['sent']}")
    check(s["view"]["hint"] == on_hint and s["view"]["select"] == "on" and s["view"]["errorHidden"]
          and s["view"]["failedHidden"] and s["view"]["postedHidden"], f"saving 'on' redraw: {s['view']}")
    check(out["saveOff"]["returned"] is True and out["saveOff"]["view"]["hint"] == off_hint
          and out["saveOff"]["view"]["select"] == "off", f"saving 'off' redraw: {out['saveOff']}")
    r = out["saveRefused"]
    check(r["returned"] is False and r["view"]["errorText"] == "This post does not go to LinkedIn."
          and not r["view"]["errorHidden"] and r["view"]["hint"] == off_hint,
          f"a refusal must be shown and change nothing else: {r}")
    o = out["saveOffline"]
    check(o["returned"] is False and o["view"]["errorText"] == "Could not save" and not o["view"]["errorHidden"],
          f"offline: {o}")
    check(out["saveNoBox"] == {"returned": False, "fetches": 0}, f"a card with no control must send nothing: {out['saveNoBox']}")
    k = out["saveKeepsFailure"]
    check(not k["failedHidden"] and k["failedReason"] == FAILED_REASON and k["failedLink"] == LINK_TEXT,
          f"changing the choice must not hide a failed comment: {k}")

    # --- the state
    f, p, n = out["stateFailed"], out["statePosted"], out["stateNone"]
    check(not f["failedHidden"] and f["failedReason"] == FAILED_REASON and f["failedLink"] == LINK_TEXT
          and f["postedHidden"], f"failed state: {f}")
    check(p["failedHidden"] and not p["postedHidden"] and p["failedReason"] == "" and p["failedLink"] == "",
          f"posted state must hide and clear the failure: {p}")
    check(n["failedHidden"] and n["postedHidden"], f"no state shows neither: {n}")

    # --- retrying
    check(out["retryDuring"] is True, "the Retry button must be disabled while it works")
    r = out["retryOk"]
    check(r["returned"] is True and r["btn"] is False, f"a retry that works: {r}")
    check(len(r["sent"]) == 1 and r["sent"][0]["url"] == "/compose/post/5/link-comment/retry"
          and fields(r["sent"][0]) == [["post_ids", "5,6"]], f"the retry request: {r['sent']}")
    check(r["view"]["failedHidden"] and not r["view"]["postedHidden"] and r["toasts"] == [["Link posted as the first comment", "success"]],
          f"a retry that works must show it posted: {r}")
    r = out["retryRefused"]
    check(r["returned"] is False and r["view"]["failedReason"] == "no luck" and not r["view"]["failedHidden"]
          and "still did not post" in r["view"]["errorText"], f"a retry that is refused again: {r}")
    r = out["retryNothing"]
    check(r["returned"] is False and "no failed link comment" in r["view"]["errorText"], f"nothing to retry: {r}")
    r = out["retryOffline"]
    check(r["returned"] is False and r["btn"] is False and r["view"]["errorText"] == "Could not reach the server",
          f"offline retry must re-enable the button: {r}")

    # --- a publish result's comment
    check(out["noteNone"]["warning"] == "" and out["noteNone"]["view"]["failedHidden"] and out["noteNone"]["view"]["postedHidden"],
          f"no comment, no change: {out['noteNone']}")
    check(out["notePosted"]["warning"] == "" and not out["notePosted"]["view"]["postedHidden"],
          f"a posted comment is shown and not a warning: {out['notePosted']}")
    want = f"Work: posted, but the link comment did not post. {FAILED_REASON}"
    check(out["noteFailed"]["warning"] == want and not out["noteFailed"]["view"]["failedHidden"],
          f"a lost comment is a warning and shown on the card: {out['noteFailed']}")
    check(out["noteNoBox"]["warning"] == want, "a card with no control still gets the warning")

    # --- the publishers
    r = out["postNowLost"]
    check(r["returned"] is True, "a post that landed must still count as posted when its comment is lost")
    check(r["alerts"] == [want] and r["chip"] == 1 and not r["view"]["failedHidden"],
          f"Post now with a lost comment: one alert, the posted chip, the failure on the card: {r}")
    r = out["postNowKept"]
    check(r["returned"] is True and r["alerts"] == [] and not r["view"]["postedHidden"], f"Post now, comment kept: {r}")
    check(out["postNowPlain"] == {"returned": True, "alerts": []}, f"a result with no comment is as before: {out['postNowPlain']}")
    r = out["postAll"]
    check(r["published"] == 2 and r["alerts"] == ["X: rate limit\n" + want],
          f"post to all: one alert listing the failure then the lost comment: {r['alerts']}")
    check(r["toasts"] == [["Posted to 2 of 3 targets", "warning"]] and not r["view"]["failedHidden"],
          f"post to all keeps its own message and shows the failure on the card: {r}")
    r = out["postAllClean"]
    check(r["alerts"] == [] and r["toasts"] == [["Posted to 1 target", "success"]] and not r["view"]["postedHidden"],
          f"post to all with the comment kept: {r}")

    print("js: save sends the mode and the card and redraws or shows a refusal; retry disables its button, "
          "shows the new reason or the posted note; both publishers announce a lost comment without "
          "counting the post as failed")
    print("LINK_COMMENT_JS_OK")


# ---------------------------------------------------------------------------
# docs, and registration
# ---------------------------------------------------------------------------

DOC_REQUIRED = (
    "first comment",
    "/compose/post/<id>/link-comment",
    "/compose/post/<id>/link-comment/retry",
    "/accounts/<id>/link-in-comment",
    "w_member_social_feed",
    "LINKEDIN_SCOPES",
)


def missing_from(text, required):
    return [item for item in required if item not in text]


def section_docs():
    readme = open(os.path.join(g.REPO, "README.md"), encoding="utf-8").read()
    # control: the check can fail
    check(missing_from("nothing relevant here", DOC_REQUIRED) == list(DOC_REQUIRED),
          "the documentation check must flag a README that says nothing")
    absent = missing_from(readme, DOC_REQUIRED)
    check(not absent, f"README.md does not mention: {absent}")
    print(f"docs: README.md covers the feature, its three routes, the permission and LINKEDIN_SCOPES")
    print("LINK_COMMENT_DOCS_OK")


def registry_problems(entries, sections):
    """Sections of this script that the gate runner does not run, or runs under the wrong token."""
    registered = {}
    for entry in entries:
        filename, token, _claim, *rest = entry
        if filename == "check_link_comment.py" and rest and rest[0]:
            registered[rest[0][0]] = token
    problems = [f"{name} is not registered" for name in sections if name not in registered]
    problems += [f"{name} is registered with {registered[name]}, expected {TOKENS[name]}"
                 for name in sections if name in registered and registered[name] != TOKENS[name]]
    problems += [f"{name} is registered but is not a section here" for name in registered if name not in sections]
    return problems


def section_registered():
    import run_all_gates
    # control: a registry missing a section, and one with a wrong token, are both caught
    fake = [("check_link_comment.py", TOKENS[n], "claim", (n,)) for n in TOKENS if n != "split"]
    check(registry_problems(fake, TOKENS) == ["split is not registered"], "the registry check cannot see a missing gate")
    fake = [("check_link_comment.py", TOKENS[n] if n != "js" else "WRONG", "claim", (n,)) for n in TOKENS]
    check(len(registry_problems(fake, TOKENS)) == 1, "the registry check cannot see a wrong token")

    problems = registry_problems(run_all_gates.ALL_GATES, TOKENS)
    check(not problems, f"the gate runner does not run this check properly: {problems}")
    print(f"registered: all {len(TOKENS)} sections of this check are in the repository's gate runner under their tokens")
    print("LINK_COMMENT_REGISTERED_OK")


SECTIONS = {
    "client": section_client, "split": section_split, "publish": section_publish, "off": section_off,
    "failure": section_failure, "refuse": section_refuse, "retry": section_retry,
    "persist": section_persist, "routes": section_routes, "paths": section_paths, "ui": section_ui,
    "js": section_js, "docs": section_docs, "registered": section_registered,
}
assert set(SECTIONS) == set(TOKENS), "every section needs a token"

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_link_comment.py {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[name]()

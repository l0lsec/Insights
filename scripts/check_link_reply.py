"""Gates for sending a Threads or X post's link as its first reply.

    python scripts/check_link_reply.py client      LINK_REPLY_CLIENT_OK
    python scripts/check_link_reply.py publish     LINK_REPLY_PUBLISH_OK
    python scripts/check_link_reply.py failure     LINK_REPLY_FAILURE_OK
    python scripts/check_link_reply.py block       LINK_REPLY_BLOCK_OK
    python scripts/check_link_reply.py refuse      LINK_REPLY_REFUSE_OK
    python scripts/check_link_reply.py retry       LINK_REPLY_RETRY_OK
    python scripts/check_link_reply.py routes      LINK_REPLY_ROUTES_OK
    python scripts/check_link_reply.py ui          LINK_REPLY_UI_OK
    python scripts/check_link_reply.py js          LINK_REPLY_JS_OK
    python scripts/check_link_reply.py docs        LINK_REPLY_DOCS_OK
    python scripts/check_link_reply.py registered  LINK_REPLY_REGISTERED_OK

Threads and X show posts that link out to fewer people, the way LinkedIn does,
so a post there can send its link as its first reply: the post goes out without
the link and the link follows as a reply to it once it is live. It is the
LinkedIn first comment (check_link_comment.py) on two more platforms, sharing
its columns, its routes and its one rule: a refused reply never turns into a
failed post, because the scheduler retries failed posts and would post it again.

What is new here is ``block``. When a platform refuses a link reply for want of a
permission (Threads needs threads_manage_replies, which plain Connect does not
ask for), every later post would go out with its link nowhere. So the account is
blocked: its links stay in the post until it is reconnected, a retried reply
posts, or its setting is saved again. That holds for LinkedIn too.

``client`` replaces ``requests`` inside threads_client and twitter_client with a
scripted transport. The rest drive the real routes, database and publisher on a
throwaway database with recording clients. Nothing here posts anything, reads a
real token or leaves the machine.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import (  # noqa: E402
    FakeResult, RecordingClient, check, connect, install_fake_clients, isolated_app,
)
import _link_comment_gate as g  # noqa: E402
from _link_comment_gate import COPY, IMAGE, LINK, VIDEO, FakeResponse, make_post, row_of, state_of  # noqa: E402
from check_link_comment import card_soup, fn_source, hidden  # noqa: E402

LEGACY_RESULT_KEYS = {
    "success", "platform", "platform_name", "account_id", "account_label",
    "permalink", "error", "needs_reconnect", "permanent",
}
TOKENS = {
    "client": "LINK_REPLY_CLIENT_OK", "publish": "LINK_REPLY_PUBLISH_OK",
    "failure": "LINK_REPLY_FAILURE_OK", "block": "LINK_REPLY_BLOCK_OK",
    "refuse": "LINK_REPLY_REFUSE_OK", "retry": "LINK_REPLY_RETRY_OK",
    "routes": "LINK_REPLY_ROUTES_OK", "ui": "LINK_REPLY_UI_OK", "js": "LINK_REPLY_JS_OK",
    "docs": "LINK_REPLY_DOCS_OK", "registered": "LINK_REPLY_REGISTERED_OK",
}
PLATFORMS = ("threads", "twitter")
EXTERNAL = {"threads": "th-1", "twitter": "x-1"}
# What a Threads permission refusal looks like (Graph API error code 10).
THREADS_NO_PERMISSION = {"error": {"message": "Application does not have permission for this action",
                                   "type": "OAuthException", "code": 10}}


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class ReplyRecorder(RecordingClient):
    """A recording Threads or X client whose posts carry an id and that records replies.

    ``reply_answer`` overrides the reply's answer (a refusal, say), ``reply_raises``
    makes it blow up, and ``no_post_id`` makes a post succeed without saying
    what it is called.
    """

    def __init__(self, platform):
        super().__init__(platform)
        self.reply_answer = None
        self.reply_raises = None
        self.no_post_id = False

    def _record(self, kind, token, **extra):
        reply = super()._record(kind, token, **extra)
        if reply.get("success") and not self.no_post_id:
            reply["post_id" if self.name == "threads" else "tweet_id"] = f"{self.name}-{len(self.calls)}"
        return reply

    def _reply(self, token, post_id, text):
        self.calls.append({"kind": "reply", "token": token, "post": post_id, "text": text})
        if self.reply_raises is not None:
            raise self.reply_raises
        if self.reply_answer is not None:
            return self.reply_answer
        key = "post_id" if self.name == "threads" else "tweet_id"
        return FakeResult(success=True, **{key: f"r{len(self.calls)}"})

    # Threads
    def publish_reply(self, access_token, reply_to_id, text):
        return self._reply(access_token, reply_to_id, text)

    # X
    def create_reply(self, access_token, text, reply_to):
        return self._reply(access_token, reply_to, text)

    # X image and video posts (the shared recorder only knows its text post)
    def create_image_post(self, access_token, author_urn=None, text=None, image_url=None):
        return self._record("image", access_token, text=text, image=image_url)

    def kinds(self):
        return [call["kind"] for call in self.calls]

    def replies(self):
        return [call for call in self.calls if call["kind"] == "reply"]


def rig(account_default=False):
    """An isolated app with a Threads and an X account and recording clients.

    Returns ``(database, web, publisher, client, ids, recorders)``; ``ids`` and
    ``recorders`` are keyed by platform. Each account's token is
    ``token-<platform>-<external id>``.
    """
    directory, database, web, publisher, client = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None
    web.get_image_for_post = lambda *a, **k: None
    ids = {p: connect(database, p, EXTERNAL[p], f"{p} account") for p in PLATFORMS}
    recorders = {p: ReplyRecorder(p) for p in PLATFORMS}
    install_fake_clients(publisher, web, **recorders)
    if account_default:
        for account_id in ids.values():
            database.set_social_account_link_in_comment(account_id, True, db_path=database.DB_PATH)
    return database, web, publisher, client, ids, recorders


def token_of(platform):
    return f"token-{platform}-{EXTERNAL[platform]}"


def publish_row(publisher, database, platform, account, post_id, content, image=None, video=None, **extra):
    return publisher.publish(platform, account, content=content, image_url=image, video_url=video,
                             standalone_post_id=post_id, db_path=database.DB_PATH, **extra)


def blocked_of(database, account_id):
    return database.link_reply_blocked(database.get_social_account(account_id, db_path=database.DB_PATH))


# ---------------------------------------------------------------------------
# client: the HTTP each client sends for a reply
# ---------------------------------------------------------------------------

class FakeHTTP:
    """Scripted ``requests.post``/``get``; running out of steps fails loudly."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def _step(self, method, url, **kw):
        self.requests.append({"method": method, "url": url, **kw})
        if not self.script:
            raise AssertionError(f"unexpected request #{len(self.requests)}: {method} {url}")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def post(self, url, params=None, json=None, headers=None, timeout=None, **_kw):
        return self._step("POST", url, params=params, json=json, headers=dict(headers or {}))

    def get(self, url, params=None, timeout=None, **_kw):
        return self._step("GET", url, params=params)


def section_client():
    import threads_client
    import twitter_client

    # --- Threads: container with reply_to_id, wait, publish, read the permalink
    http = FakeHTTP([
        FakeResponse(200, payload={"id": "container-1"}),
        FakeResponse(200, payload={"status": "FINISHED"}),
        FakeResponse(200, payload={"id": "reply-9"}),
        FakeResponse(200, payload={"permalink": "https://threads.test/r9", "shortcode": "r9"}),
    ])
    saved = (threads_client.requests.post, threads_client.requests.get, threads_client._sleep)
    threads_client.requests.post, threads_client.requests.get = http.post, http.get
    threads_client._sleep = lambda seconds: None
    try:
        tc = threads_client.ThreadsClient()
        long_link = "https://blog.test/" + "a" * 600
        reply = tc.publish_reply("tok", "post-1", long_link)
        create = http.requests[0]
        check(create["method"] == "POST" and create["url"].endswith("/me/threads"), f"container: {create}")
        check(create["params"] == {"media_type": "TEXT", "text": long_link, "reply_to_id": "post-1",
                                   "access_token": "tok"},
              f"the reply container must name the post it answers and keep the words whole: {create['params']}")
        check(http.requests[2]["url"].endswith("/me/threads_publish")
              and http.requests[2]["params"]["creation_id"] == "container-1", "the container is then published")
        check(reply["success"] and reply["post_id"] == "reply-9" and reply["permalink"] == "https://threads.test/r9",
              f"a published reply reports its id: {reply}")

        http.script = [FakeResponse(400, payload=THREADS_NO_PERMISSION)]
        http.requests.clear()
        refused = tc.publish_reply("tok", "post-1", LINK)
        check(not refused["success"] and refused["status_code"] == 400
              and refused["error"]["error"]["code"] == 10 and len(http.requests) == 1,
              f"a refused container is reported with Threads' error and nothing else is sent: {refused}")

        http.requests.clear()
        for missing in ("", None, "  "):
            out = tc.publish_reply("tok", missing, LINK)
            check(not out["success"], f"no post to reply to must be refused: {out}")
        check(http.requests == [], "a reply with no target must send nothing (it would become a new post)")
    finally:
        threads_client.requests.post, threads_client.requests.get, threads_client._sleep = saved

    # --- X: one POST /2/tweets with reply.in_reply_to_tweet_id
    http = FakeHTTP([FakeResponse(201, payload={"data": {"id": "1999", "text": "x"}})])
    saved = twitter_client.requests.post
    twitter_client.requests.post = http.post
    try:
        xc = twitter_client.TwitterClient()
        long_text = "x" * 300
        reply = xc.create_reply("tok", long_text, "1234")
        sent = http.requests[0]
        check(sent["url"] == twitter_client.TWITTER_TWEETS_URL, f"wrong endpoint: {sent['url']}")
        check(sent["json"] == {"text": long_text, "reply": {"in_reply_to_tweet_id": "1234"}},
              f"the reply body must name the post and keep the words whole: {sent['json']}")
        check(sent["headers"].get("Authorization") == "Bearer tok", "the reply uses the account's token")
        check(reply["success"] and reply["tweet_id"] == "1999", f"a posted reply reports its id: {reply}")

        http.script = [FakeResponse(403, payload={"title": "Forbidden", "detail": "not allowed"})]
        refused = xc.create_reply("tok", LINK, "1234")
        check(not refused["success"] and refused["status_code"] == 403, f"a 403 is reported as such: {refused}")

        http.requests.clear()
        check(not xc.create_reply("tok", LINK, "")["success"] and http.requests == [],
              "a reply with no target must send nothing")

        # a plain post is what it always was
        http.script = [FakeResponse(201, payload={"data": {"id": "5"}})]
        http.requests.clear()
        xc.create_post("tok", "hello")
        check(http.requests[0]["json"] == {"text": "hello"}, f"a post's body changed: {http.requests[0]['json']}")
    finally:
        twitter_client.requests.post = saved

    print("client: a Threads reply is a container naming reply_to_id, published; an X reply is one tweet "
          "naming in_reply_to_tweet_id; neither trims the words, both refuse a missing target unsent")
    print("LINK_REPLY_CLIENT_OK")


# ---------------------------------------------------------------------------
# publish: the link leaves the post and follows as its first reply
# ---------------------------------------------------------------------------

def section_publish():
    content = COPY + "\n" + LINK
    for platform in PLATFORMS:
        for kind, image, video in (("text", None, None), ("image", IMAGE, None), ("video", None, VIDEO)):
            database, web, publisher, client, ids, recs = rig()
            rec = recs[platform]
            pid = make_post(database, ids[platform], content, image=image, video=video, mode=True,
                            platform=platform)
            result = publish_row(publisher, database, platform, ids[platform], pid, content, image, video)
            name = f"{platform}/{kind}"
            check(result["success"], f"{name}: the post must succeed: {result}")
            check(rec.kinds() == [kind, "reply"], f"{name}: one post then one reply: {rec.kinds()}")
            post, reply = rec.calls
            check(post["text"] == COPY, f"{name}: the post must go out without its link: {post['text']!r}")
            check(reply["text"] == LINK and reply["token"] == post["token"] == token_of(platform),
                  f"{name}: the reply must be the link, with the post's token: {reply}")
            check(reply["post"] == f"{platform}-1", f"{name}: the reply must answer the post just made: {reply}")
            note = result["first_comment"]
            check(note["status"] == "posted" and note["comment_id"] == "r2" and note["post_urn"] == f"{platform}-1",
                  f"{name}: the result reports the reply: {note}")
            saved = state_of(database, pid)
            check(saved and saved["status"] == "posted" and saved["text"] == LINK,
                  f"{name}: the row records the reply: {saved}")
            check(blocked_of(database, ids[platform]) is None, f"{name}: a reply that posts blocks nothing")

    # the account's setting decides when the post makes no choice; the post's choice beats it
    database, web, publisher, client, ids, recs = rig(account_default=True)
    pid = make_post(database, ids["threads"], content, platform="threads")
    publish_row(publisher, database, "threads", ids["threads"], pid, content)
    check(recs["threads"].kinds() == ["text", "reply"], f"the account default sends the reply: {recs['threads'].kinds()}")
    pid = make_post(database, ids["twitter"], content, mode=False, platform="twitter")
    out = publish_row(publisher, database, "twitter", ids["twitter"], pid, content)
    check(recs["twitter"].calls == [{"kind": "text", "token": token_of("twitter"), "text": content}]
          and set(out) == LEGACY_RESULT_KEYS,
          f"a post that says no keeps its link and its old result: {recs['twitter'].calls} {sorted(out)}")

    # with the setting off, a post is exactly what it was
    database, web, publisher, client, ids, recs = rig()
    for platform in PLATFORMS:
        pid = make_post(database, ids[platform], content, platform=platform)
        out = publish_row(publisher, database, platform, ids[platform], pid, content)
        check(recs[platform].kinds() == ["text"] and recs[platform].calls[0]["text"] == content
              and set(out) == LEGACY_RESULT_KEYS and state_of(database, pid) is None,
              f"{platform}: off must be untouched: {recs[platform].calls} {sorted(out)}")

    # a caller's explicit answer beats the setting either way, with no saved row
    for platform in PLATFORMS:
        recs[platform].calls.clear()
        out = publisher.publish(platform, ids[platform], content=content, link_in_comment=True,
                                db_path=database.DB_PATH)
        check(recs[platform].kinds() == ["text", "reply"] and recs[platform].calls[0]["text"] == COPY
              and out["first_comment"]["status"] == "posted",
              f"{platform}: link_in_comment=True must send the reply: {recs[platform].calls}")
    database, web, publisher, client, ids, recs = rig(account_default=True)
    for platform in PLATFORMS:
        publisher.publish(platform, ids[platform], content=content, link_in_comment=False,
                          db_path=database.DB_PATH)
        check(recs[platform].calls == [{"kind": "text", "token": token_of(platform), "text": content}],
              f"{platform}: link_in_comment=False must keep the link: {recs[platform].calls}")

    # two links make one reply, once each; a post with no link just posts
    database, web, publisher, client, ids, recs = rig(account_default=True)
    two = f"{COPY}\n{LINK}\nAlso https://other.test/x and {LINK}"
    publisher.publish("threads", ids["threads"], content=two, db_path=database.DB_PATH)
    check(recs["threads"].replies()[0]["text"] == f"{LINK}\nhttps://other.test/x",
          f"several links make one reply: {recs['threads'].replies()}")
    out = publisher.publish("twitter", ids["twitter"], content="No link here", db_path=database.DB_PATH)
    check(recs["twitter"].kinds() == ["text"] and "first_comment" not in out, "a post with no link has no reply")

    print("publish: Threads and X text, image and video posts go out without the link and the link follows "
          "once as a reply to that post with its token; account default, post choice and off all hold")
    print("LINK_REPLY_PUBLISH_OK")


# ---------------------------------------------------------------------------
# failure: a refused reply is not a failed post
# ---------------------------------------------------------------------------

def section_failure():
    content = COPY + "\n" + LINK
    scenarios = (
        ("refused", lambda r: setattr(r, "reply_answer", FakeResult(
            success=False, status_code=429, error={"title": "Too Many Requests", "detail": "slow down"}))),
        ("exploding", lambda r: setattr(r, "reply_raises", RuntimeError("socket closed"))),
        ("no reason", lambda r: setattr(r, "reply_answer", FakeResult(success=False))),
        ("no post id", lambda r: setattr(r, "no_post_id", True)),
    )
    for platform in PLATFORMS:
        for name, setup in scenarios:
            database, web, publisher, client, ids, recs = rig()
            rec = recs[platform]
            setup(rec)
            pid = make_post(database, ids[platform], content, mode=True, platform=platform)
            result = publish_row(publisher, database, platform, ids[platform], pid, content)
            label = f"{platform}/{name}"
            check(result["success"] is True and result["error"] is None,
                  f"{label}: a reply problem must not read as a post problem: {result}")
            note = result["first_comment"]
            check(note["status"] == "failed" and note["text"] == LINK and note["error"],
                  f"{label}: the reply is reported failed with a reason and the link: {note}")
            check(rec.kinds().count("text") == 1, f"{label}: the post went out once: {rec.kinds()}")
            check(state_of(database, pid)["status"] == "failed", f"{label}: the row keeps the failure")
            check(blocked_of(database, ids[platform]) is None, f"{label}: only a permission refusal blocks")
            if name == "no post id":
                check(rec.replies() == [] and "did not say which post" in note["error"],
                      f"{label}: with no id there is nothing to reply to: {note}")
            if name == "refused" and platform == "twitter":
                check(note["error"] == "slow down", f"X's detail is the reason: {note['error']}")
            if name == "no reason":
                check(note["error"].endswith("gave no reason"), f"{label}: {note['error']}")

    # the scheduler's entry is published once, its reply lost
    database, web, publisher, client, ids, recs = rig()
    recs["threads"].reply_answer = FakeResult(success=False, error={"error": {"message": "boom", "code": 1}})
    P = database.DB_PATH
    pid = make_post(database, ids["threads"], content, mode=True, platform="threads")
    sid = database.add_scheduled_post(scheduled_for="2000-01-01T00:00:00", post_type="standalone",
                                      standalone_post_id=pid, platform="threads",
                                      account_id=ids["threads"], db_path=P)
    due = [r for r in database.get_pending_scheduled_posts(db_path=P) if r["id"] == sid]
    check(len(due) == 1, "the queued post should be due")
    result = web._publish_scheduled_entry(due[0])
    check(result["success"] and result["first_comment"]["status"] == "failed"
          and recs["threads"].kinds() == ["text", "reply"],
          f"the scheduler's entry is published with a failed reply: {result} {recs['threads'].kinds()}")

    print("failure: a refused, exploding, reasonless or id-less reply leaves the post reported as published "
          "once and keeps the reason and link on the row; none of them blocks the account")
    print("LINK_REPLY_FAILURE_OK")


# ---------------------------------------------------------------------------
# block: a permission refusal keeps the account's next links in the post
# ---------------------------------------------------------------------------

def section_block():
    content = COPY + "\n" + LINK
    refusals = {
        "threads": FakeResult(success=False, status_code=400, error=THREADS_NO_PERMISSION),
        "twitter": FakeResult(success=False, status_code=403, error={"title": "Forbidden", "detail": "not allowed"}),
    }
    for platform in PLATFORMS:
        database, web, publisher, client, ids, recs = rig()
        P, rec, acct = database.DB_PATH, recs[platform], ids[platform]
        rec.reply_answer = refusals[platform]
        first = make_post(database, acct, content, mode=True, platform=platform)
        result = publish_row(publisher, database, platform, acct, first, content)
        check(result["success"] and result["first_comment"]["status"] == "failed", f"{platform}: {result}")
        block = blocked_of(database, acct)
        check(block and block["error"] == result["first_comment"]["error"],
              f"{platform}: a permission refusal must block the account with its reason: {block}")
        if platform == "threads":
            check("threads_manage_replies" in block["error"] and "Reconnect with reply permission" in block["error"],
                  f"the Threads reason must name the permission and the fix: {block['error']}")

        # the next post keeps its link and sends no reply
        rec.calls.clear()
        second = make_post(database, acct, content, mode=True, platform=platform)
        result = publish_row(publisher, database, platform, acct, second, content)
        check(result["success"] and rec.calls == [{"kind": "text", "token": token_of(platform), "text": content}],
              f"{platform}: a blocked account posts its link in the post and nothing else: {rec.calls}")
        note = result["first_comment"]
        check(note["status"] == "kept" and note["text"] == LINK and note["error"] == block["error"],
              f"{platform}: the result says the link was kept and why: {note}")
        check(state_of(database, second)["status"] == "kept", f"{platform}: the row records it was kept")

        # a retry that posts lifts the block
        rec.reply_answer = None
        rec.calls.clear()
        retried = publisher.retry_first_comment(first, db_path=P)
        check(retried["success"] and rec.kinds() == ["reply"] and blocked_of(database, acct) is None,
              f"{platform}: a retried reply that posts must lift the block: {retried} {rec.kinds()}")
        rec.calls.clear()
        third = make_post(database, acct, content, mode=True, platform=platform)
        publish_row(publisher, database, platform, acct, third, content)
        check(rec.kinds() == ["text", "reply"] and rec.calls[0]["text"] == COPY,
              f"{platform}: after that the link goes to the reply again: {rec.calls}")

        # saving the account's setting lifts it; so does reconnecting
        database.set_social_account_link_reply_blocked(acct, {"error": "x", "at": "t"}, db_path=P)
        reply = client.post(f"/accounts/{acct}/link-in-comment", data={"enabled": "true"})
        check(reply.status_code == 200 and blocked_of(database, acct) is None,
              f"{platform}: saving the setting must lift the block: {reply.get_json()}")
        database.set_social_account_link_reply_blocked(acct, {"error": "x", "at": "t"}, db_path=P)
        connect(database, platform, EXTERNAL[platform], "reconnected")
        check(blocked_of(database, acct) is None, f"{platform}: reconnecting the account must lift the block")

    # LinkedIn's 403 blocks the same way (the comment permission is partner-gated)
    _directory, database, web, publisher, client = isolated_app()
    li = connect(database, "linkedin", "li-1", "Member")
    recorder = g.LinkedInRecorder()
    install_fake_clients(publisher, web, linkedin=recorder)
    recorder.comment_reply = FakeResult(success=False, status_code=403,
                                        friendly="LinkedIn refused: Not enough permissions")
    pid = make_post(database, li, content, mode=True)
    publisher.publish("linkedin", li, content=content, standalone_post_id=pid, db_path=database.DB_PATH)
    check(blocked_of(database, li) is not None, "a LinkedIn 403 must block the account too")
    recorder.calls.clear()
    pid = make_post(database, li, content, mode=True)
    out = publisher.publish("linkedin", li, content=content, standalone_post_id=pid, db_path=database.DB_PATH)
    check(recorder.kinds() == ["text"] and recorder.calls[0]["text"] == content and out["first_comment"]["status"] == "kept",
          f"a blocked LinkedIn account keeps its link in the post: {recorder.calls}")

    # which refusals count as "this connection may not": each sign on its own
    cases = (
        ("threads", {"status_code": 400, "error": {"error": {"code": 10, "message": "Unsupported"}}}, True),
        ("threads", {"status_code": 400, "error": {"error": {"code": 200, "message": "x"}}}, True),
        ("threads", {"status_code": 400, "error": {"error": {"code": 1, "message": "Missing Permission"}}}, True),
        ("threads", {"status_code": 400, "error": {"error": {"code": 1, "message": "Unknown error"}}}, False),
        ("threads", {"status_code": 429, "error": {"error": {"code": 4, "message": "rate"}}}, False),
        ("twitter", {"status_code": 403, "error": {"detail": "x"}}, True),
        ("twitter", {"status_code": 400, "error": {"detail": "permission"}}, False),
        ("linkedin", {"status_code": 403, "friendly": "x"}, True),
        ("linkedin", {"status_code": 500, "friendly": "x"}, False),
    )
    for platform, reply, want in cases:
        check(publisher._refused_for_permission(platform, reply) is want,
              f"{platform} {reply} should {'' if want else 'not '}count as a permission refusal")

    # a damaged block reads as none
    database.set_social_account_link_reply_blocked(li, None, db_path=database.DB_PATH)
    import sqlite3
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute("UPDATE social_accounts SET link_reply_blocked = 'not json' WHERE id = ?", (li,))
    check(blocked_of(database, li) is None, "a damaged block must read as none")

    print("block: a permission refusal on Threads, X or LinkedIn keeps that account's next links in the post "
          "(noted as kept, no reply sent) until a retry posts, the setting is saved, or it is reconnected")
    print("LINK_REPLY_BLOCK_OK")


# ---------------------------------------------------------------------------
# refuse: what cannot carry its reply is refused before anything is sent
# ---------------------------------------------------------------------------

def section_refuse():
    database, web, publisher, client, ids, recs = rig(account_default=True)
    P = database.DB_PATH
    links = " ".join(f"https://blog.test/{'p' * 40}{n}" for n in range(6))     # > 280 joined
    out = publisher.publish("twitter", ids["twitter"], content=f"{COPY} {links}", db_path=P)
    check(not out["success"] and out["permanent"] and "X/Twitter allows 280" in out["error"]
          and recs["twitter"].calls == [], f"an X reply over 280 must be refused unsent: {out}")
    links = " ".join(f"https://blog.test/{'p' * 90}{n}" for n in range(6))     # > 500 joined
    out = publisher.publish("threads", ids["threads"], content=f"{COPY} {links}", db_path=P)
    check(not out["success"] and "Threads allows 500" in out["error"] and recs["threads"].calls == [],
          f"a Threads reply over 500 must be refused unsent: {out}")
    for platform in PLATFORMS:
        out = publisher.publish(platform, ids[platform], content=LINK, db_path=P)
        check(not out["success"] and "Nothing is left to post once the link moves to the reply" in out["error"]
              and recs[platform].calls == [], f"{platform}: a link-only post must be refused unsent: {out}")
    print("refuse: a reply longer than the platform takes and a post left with no words are refused with "
          "nothing sent")
    print("LINK_REPLY_REFUSE_OK")


# ---------------------------------------------------------------------------
# retry: only the reply, as the row's own account
# ---------------------------------------------------------------------------

def section_retry():
    content = COPY + "\n" + LINK
    for platform in PLATFORMS:
        database, web, publisher, client, ids, recs = rig()
        rec = recs[platform]
        rec.reply_answer = FakeResult(success=False, error={"message": "try later"})
        pid = make_post(database, ids[platform], content, mode=True, platform=platform)
        publish_row(publisher, database, platform, ids[platform], pid, content)
        rec.reply_answer = None
        rec.calls.clear()
        reply = client.post(f"/compose/post/{pid}/link-comment/retry",
                            data={"platform": platform, "post_ids": str(pid)})
        body = reply.get_json()
        check(reply.status_code == 200 and body["success"] and body["message"] == "The link reply was posted",
              f"{platform}: the retry route: {reply.status_code} {body}")
        check(rec.calls == [{"kind": "reply", "token": token_of(platform), "post": f"{platform}-1", "text": LINK}],
              f"{platform}: the retry sends only the reply, with the saved words on the saved post: {rec.calls}")
        check(body["link_comment"]["state"]["status"] == "posted", f"{platform}: summary {body['link_comment']}")
        again = client.post(f"/compose/post/{pid}/link-comment/retry", data={"platform": platform, "post_ids": str(pid)})
        check(again.status_code == 400 and "no failed link reply" in again.get_json()["error"] and len(rec.calls) == 1,
              f"{platform}: a second retry must send nothing: {again.get_json()}")
        # asking about LinkedIn (the default) finds nothing to retry on a Threads/X card
        other = client.post(f"/compose/post/{pid}/link-comment/retry", data={"post_ids": str(pid)})
        check(other.status_code == 400, "the retry route left without a platform means LinkedIn")
    print("retry: the route retries only the named platform's failed reply, once, with the saved words on the "
          "saved post and the row's own token")
    print("LINK_REPLY_RETRY_OK")


# ---------------------------------------------------------------------------
# routes: the card's choice per platform, inheritance, and the account's setting
# ---------------------------------------------------------------------------

def section_routes():
    database, web, publisher, client, ids, recs = rig()
    P = database.DB_PATH
    li_account = connect(database, "linkedin", "li-1", "Member")
    text = COPY + "\n" + LINK
    th = make_post(database, ids["threads"], text, platform="threads")
    x = make_post(database, ids["twitter"], text, platform="twitter")
    li = make_post(database, li_account, text)
    card = f"{th},{x},{li}"

    def link_of(pid):
        return row_of(database, pid)["link_comment"]

    def choose(platform, mode, **data):
        return client.post(f"/compose/post/{th}/link-comment",
                           data={"mode": mode, "platform": platform, "post_ids": card, **data})

    reply = choose("threads", "on")
    body = reply.get_json()
    check(reply.status_code == 200 and body["platform"] == "threads" and body["updated_ids"] == [th]
          and link_of(th) == 1 and link_of(x) is None and link_of(li) is None,
          f"choosing for Threads changes only the Threads row: {body}")
    check(body["link_comment"]["noun"] == "reply" and body["link_comment"]["platform"] == "threads",
          f"the summary speaks of a reply: {body['link_comment']}")
    reply = choose("twitter", "off")
    check(reply.status_code == 200 and link_of(x) == 0 and link_of(th) == 1 and link_of(li) is None,
          "choosing for X changes only the X row")
    reply = client.post(f"/compose/post/{th}/link-comment", data={"mode": "on", "post_ids": card})
    check(reply.status_code == 200 and link_of(li) == 1 and link_of(th) == 1 and link_of(x) == 0,
          "with no platform the route means LinkedIn, as it always did")
    for bad in ("facebook", "instagram", "nope"):
        reply = choose(bad, "on")
        check(reply.status_code == 400 and "platform must be" in reply.get_json()["error"],
              f"platform {bad!r} must be refused: {reply.get_json()}")
    alone = make_post(database, ids["threads"], "Only Threads\n" + LINK, platform="threads")
    reply = client.post(f"/compose/post/{alone}/link-comment",
                        data={"mode": "on", "platform": "twitter", "post_ids": str(alone)})
    check(reply.status_code == 400 and "does not go to X/Twitter" in reply.get_json()["error"],
          f"a card with no X row has nothing to choose for X: {reply.get_json()}")

    # ticking a second Threads account onto a card inherits the card's Threads choice
    other_threads = connect(database, "threads", "th-2", "Second")
    src = make_post(database, ids["threads"], "Tick copy\n" + LINK, mode=True, platform="threads")
    reply = client.post(f"/compose/post/{src}/platform", data={
        "platform": "threads", "action": "add", "account_id": str(other_threads), "post_ids": str(src)})
    check(reply.status_code == 200, f"ticking a second Threads account: {reply.status_code}")
    added = [r for r in database.list_standalone_posts(db_path=P)
             if r["content"] == "Tick copy\n" + LINK and r["account_id"] == other_threads]
    check(added and added[0]["link_comment"] == 1, "a ticked-on Threads row must publish like the card's other one")
    client.post(f"/compose/post/{src}/platform", data={
        "platform": "twitter", "action": "add", "post_ids": f"{src},{added[0]['id']}"})
    x_row = next(r for r in database.list_standalone_posts(db_path=P)
                 if r["content"] == "Tick copy\n" + LINK and r["platform"] == "twitter")
    check(x_row["link_comment"] is None, "an X row must not carry the Threads choice")

    # the account route takes Threads and X accounts
    for platform in PLATFORMS:
        reply = client.post(f"/accounts/{ids[platform]}/link-in-comment", data={"enabled": "true"})
        body = reply.get_json()
        check(reply.status_code == 200 and body["message"].endswith("will post links as the first reply")
              and database.get_social_account(ids[platform], db_path=P)["link_in_comment"] == 1,
              f"{platform}: the account route: {body}")

    # Post now through the card's own route reports the reply
    recs["threads"].calls.clear()
    reply = client.post(f"/compose/post/{th}/threads", data={"account_id": str(ids["threads"])})
    body = reply.get_json()
    check(reply.status_code == 200 and body["success"] and body["first_comment"]["status"] == "posted"
          and recs["threads"].kinds() == ["text", "reply"],
          f"Post now on Threads sends the link as the reply: {reply.status_code} {body}")

    # the reply-permission reconnect asks for exactly one more scope; plain Connect is unchanged
    import threads_client
    web.get_threads_client = lambda: threads_client.ThreadsClient(app_id="app", app_secret="s",
                                                                  redirect_uri="https://cb.test/")
    plain = client.get("/threads/auth")
    extra = client.get("/threads/auth?replies=1")
    if plain.status_code == 302 and extra.status_code == 302:
        from urllib.parse import parse_qs, urlparse
        plain_scope = parse_qs(urlparse(plain.headers["Location"]).query).get("scope", [""])[0]
        extra_scope = parse_qs(urlparse(extra.headers["Location"]).query).get("scope", [""])[0]
        check(plain_scope == threads_client.THREADS_SCOPES, f"plain Connect must ask what it always did: {plain_scope}")
        check(extra_scope.split(",") == threads_client.THREADS_SCOPES.split(",") + ["threads_manage_replies"],
              f"the reply reconnect adds threads_manage_replies and nothing else: {extra_scope}")
    else:
        check(False, f"the Threads auth start did not redirect: {plain.status_code} {extra.status_code}")

    print("routes: the card route sets one platform's rows (LinkedIn when unnamed) and refuses others; ticking "
          "inherits per platform; the account route takes Threads and X; Post now reports the reply; the "
          "reply reconnect asks for threads_manage_replies only")
    print("LINK_REPLY_ROUTES_OK")


# ---------------------------------------------------------------------------
# ui: a control per platform on the card, and the accounts page
# ---------------------------------------------------------------------------

def section_ui():
    from bs4 import BeautifulSoup
    database, web, publisher, client, ids, recs = rig()
    P = database.DB_PATH
    text = COPY + "\n" + LINK
    th = make_post(database, ids["threads"], text, platform="threads")
    x = make_post(database, ids["twitter"], text, platform="twitter")
    soup, html = card_soup(client, th, [th, x])
    for platform, label in (("threads", "Threads"), ("twitter", "X/Twitter")):
        box = soup.select_one(f"#link-comment-{platform}-{th}")
        check(box is not None and box.get("data-platform") == platform and box.get("data-noun") == "reply",
              f"the card must have a {platform} control that speaks of replies: {box}")
        check(f"{label} link" in box.get_text(), f"the control names its platform: {box.get_text()[:80]}")
        select = box.select_one(f"#link-mode-{platform}-{th}")
        check(f"saveLinkComment({th}, '{platform}')" in select.get("onchange", ""), "the select saves its platform")
        check("In the first reply" in select.get_text(), f"the choice says reply: {select.get_text()}")
        button = box.select_one(".link-failed button")
        check(f"retryLinkComment({th}, '{platform}'" in button.get("onclick", "") and "Retry reply" in button.get_text(),
              "Retry names its platform")
        check(hidden(box.select_one(".link-kept")) and box.select_one(".link-blocked") is None,
              "nothing kept and no block at first")
    check(soup.select_one(".link-comment[data-platform='linkedin']") is None, "no LinkedIn row, no LinkedIn control")

    # kept, and a block in force
    database.set_standalone_post_link_comment_state(th, {"status": "kept", "text": LINK, "error": "no"}, db_path=P)
    database.set_standalone_post_link_comment(th, True, db_path=P)
    database.set_social_account_link_reply_blocked(ids["threads"], {"error": "<b>needs permission</b>", "at": "t"},
                                                   db_path=P)
    soup, html = card_soup(client, th, [th])
    box = soup.select_one(f"#link-comment-threads-{th}")
    check(not hidden(box.select_one(".link-kept")), "a kept link is shown")
    blocked = box.select_one(".link-blocked")
    check(blocked is not None and "needs permission" in blocked.get_text() and "<b>needs" not in html,
          f"a block in force is shown, escaped: {blocked}")

    # accounts page
    page = BeautifulSoup(client.get("/accounts").get_data(as_text=True), "html.parser")
    for platform in PLATFORMS:
        rows = page.select(f'.account-platform[data-platform="{platform}"] .account-row')
        toggle = rows[0].select_one("input.link-in-comment-toggle") if rows else None
        check(toggle is not None and "first reply" in toggle.parent.get_text(),
              f"a {platform} account offers posting links as the first reply")
    threads_row = page.select_one('.account-platform[data-platform="threads"] .account-row')
    check(threads_row.select_one('a[href="/threads/auth?replies=1"]') is not None,
          "a Threads account offers the reconnect with reply permission")
    check(threads_row.select_one(".link-reply-blocked") is not None, "a blocked account says so")
    x_row = page.select_one('.account-platform[data-platform="twitter"] .account-row')
    check(x_row.select_one(".link-reply-blocked") is None and x_row.select_one('a[href="/threads/auth?replies=1"]') is None,
          "an X account that is not blocked shows no warning and no Threads reconnect")
    print("ui: a Threads or X card shows its own control with reply wording, kept and blocked notes escaped; "
          "the Accounts page offers the setting, the reply reconnect on Threads, and the block")
    print("LINK_REPLY_UI_OK")


# ---------------------------------------------------------------------------
# js: the page's functions with a Threads box
# ---------------------------------------------------------------------------

JS_FUNCTIONS = ("linkCommentBox", "linkCommentNoun", "showLinkCommentError", "renderLinkCommentState",
                "renderLinkComment", "saveLinkComment", "retryLinkComment", "noteFirstComment")

JS_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
class FakeFormData { constructor() { this.entries = []; } append(k, v) { this.entries.push([k, String(v)]); } }
const node = (extra) => {
  const el = Object.assign({ textContent: '', value: '', classes: new Set(), children: {} }, extra || {});
  el.classList = { toggle: (c, on) => on ? el.classes.add(c) : el.classes.delete(c),
                   contains: (c) => el.classes.has(c) };
  el.querySelector = (sel) => el.children[sel] || null;
  return el;
};
function makeBox(platform, noun) {
  const box = node({ dataset: { platform, noun } });
  box.children = {
    '.link-error': node({ classes: new Set(['d-none']) }), '.link-hint': node(),
    '.link-failed': node({ classes: new Set(['d-none']) }), '.link-posted': node({ classes: new Set(['d-none']) }),
    '.link-kept': node({ classes: new Set(['d-none']) }), 'select': node({ value: 'on' }),
  };
  box.children['.link-failed'].children = { '.link-error-text': node(), '.link-text': node() };
  return box;
}
const out = {};
(async () => {
  const boxes = { threads: makeBox('threads', 'reply'), linkedin: makeBox('linkedin', 'comment') };
  const log = { fetch: [], toasts: [] };
  const replies = [];
  const ctx = {
    FormData: FakeFormData, cardIdsParam: () => '5',
    showToast: (m, k) => log.toasts.push([m, k]),
    document: { getElementById: (id) => ({ 'link-comment-threads-5': boxes.threads, 'link-mode-threads-5': boxes.threads.children['select'],
                                           'link-comment-linkedin-5': boxes.linkedin })[id] || null },
    fetch: async (url, init) => { log.fetch.push({ url, fields: init.body.entries });
      const r = replies.shift(); return { ok: r.status < 300, json: async () => r.body }; },
  };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  const postItem = { querySelector: (sel) => ({ '.link-comment[data-platform="threads"]': boxes.threads,
                                               '.link-comment[data-platform="linkedin"]': boxes.linkedin })[sel] || null };
  const FAILED = { status: 'failed', error: 'no permission', text: 'https://blog.test/post-1' };
  out.note = ctx.noteFirstComment(postItem, 'Thr', FAILED, 'threads');
  out.threadsFailedShown = !boxes.threads.children['.link-failed'].classes.has('d-none');
  out.linkedinUntouched = boxes.linkedin.children['.link-failed'].classes.has('d-none');
  out.kept = (ctx.noteFirstComment(postItem, 'Thr', { status: 'kept', text: 'x', error: 'e' }, 'threads'),
              !boxes.threads.children['.link-kept'].classes.has('d-none'));
  replies.push({ status: 200, body: { success: true, link_comment: { mode: 'on', effective: true, platform: 'threads', state: null } } });
  out.saved = await ctx.saveLinkComment(5, 'threads');
  out.saveFields = log.fetch[0].fields;
  out.hint = boxes.threads.children['.link-hint'].textContent;
  replies.push({ status: 200, body: { success: true, link_comment: { mode: 'on', effective: true, state: { status: 'posted' } } } });
  out.retried = await ctx.retryLinkComment(5, 'threads', null);
  out.retryFields = log.fetch[1].fields;
  out.toasts = log.toasts;
  console.log(JSON.stringify(out));
})();
"""


def section_js():
    node = shutil.which("node")
    check(node, "node is required for this gate and was not found on PATH")
    database, web, publisher, client, ids, recs = rig()
    make_post(database, ids["threads"], COPY + "\n" + LINK, platform="threads")
    html = client.get("/compose").get_data(as_text=True)
    functions = "\n".join(fn_source(html, name) for name in JS_FUNCTIONS)
    work = tempfile.mkdtemp(prefix="link_reply_js_")
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
    check(out["note"] == "Thr: posted, but the link reply did not post. no permission",
          f"a lost Threads reply is warned about as a reply: {out['note']}")
    check(out["threadsFailedShown"] and out["linkedinUntouched"],
          "the failure lands on the Threads box, not the LinkedIn one")
    check(out["kept"] is True, "a kept link is shown on the card")
    check(out["saved"] is True and out["saveFields"] == [["mode", "on"], ["platform", "threads"], ["post_ids", "5"]],
          f"saving names the platform: {out['saveFields']}")
    check(out["hint"] == "The link is posted as the first reply, not in the post.", f"hint: {out['hint']}")
    check(out["retried"] is True and out["retryFields"] == [["platform", "threads"], ["post_ids", "5"]]
          and out["toasts"] == [["Link posted as the first reply", "success"]],
          f"retry names the platform and speaks of a reply: {out}")
    print("js: with a Threads box the page's functions save and retry for that platform, word it as a reply, "
          "and put a lost or kept reply on that platform's box only")
    print("LINK_REPLY_JS_OK")


# ---------------------------------------------------------------------------
# docs, and registration
# ---------------------------------------------------------------------------

DOC_REQUIRED = ("first reply", "threads_manage_replies", "Reconnect with reply permission",
                "in_reply_to_tweet_id", "links stay in the post")


def section_docs():
    readme = open(os.path.join(g.REPO, "README.md"), encoding="utf-8").read()
    check([item for item in DOC_REQUIRED if item not in "nothing"] == list(DOC_REQUIRED), "control")
    absent = [item for item in DOC_REQUIRED if item not in readme]
    check(not absent, f"README.md does not mention: {absent}")
    print("docs: README.md covers the first reply, the Threads permission and its reconnect, and the block")
    print("LINK_REPLY_DOCS_OK")


def section_registered():
    import run_all_gates
    registered = {}
    for entry in run_all_gates.ALL_GATES:
        filename, token, _claim, *rest = entry
        if filename == "check_link_reply.py" and rest and rest[0]:
            registered[rest[0][0]] = token
    problems = [n for n in TOKENS if registered.get(n) != TOKENS[n]]
    problems += [n for n in registered if n not in TOKENS]
    check(not problems, f"the gate runner does not run these sections properly: {problems}")
    print(f"registered: all {len(TOKENS)} sections of this check are in the gate runner under their tokens")
    print("LINK_REPLY_REGISTERED_OK")


SECTIONS = {
    "client": section_client, "publish": section_publish, "failure": section_failure,
    "block": section_block, "refuse": section_refuse, "retry": section_retry,
    "routes": section_routes, "ui": section_ui, "js": section_js, "docs": section_docs,
    "registered": section_registered,
}
assert set(SECTIONS) == set(TOKENS), "every section needs a token"

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_link_reply.py {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[name]()

"""Shared fakes and rigs for the LinkedIn "link in the first comment" gates.

Two kinds of fake, for two kinds of claim:

* ``install_http`` replaces ``requests.post`` inside linkedin_client with a
  scripted transport, so a gate can assert the HTTP request the client would have
  sent (method, URL, headers, body) and how it reacts to each answer LinkedIn can
  give, including a connection that drops. Nothing leaves the machine.
* ``LinkedInRecorder`` stands in for the whole client when a gate is about what
  the app decides (which words go in the post, which in the comment, who is told
  what) rather than what bytes go out. It records every call, comments included.

Every rig runs on a throwaway database, so a gate can seed rows directly and then
drive the real routes, the real publisher and the real scheduler entry point.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from _accounts_gate import (  # noqa: E402
    FakeResult, RecordingClient, connect, install_fake_clients, isolated_app,
)

LINK = "https://blog.test/post-1"
IMAGE = "https://cdn.test/photo.jpg"
VIDEO = "https://cdn.test/clip.mp4"
COPY = "Intro line\n\nFull write-up below."


# ---------------------------------------------------------------------------
# the HTTP seam
# ---------------------------------------------------------------------------

class FakeResponse:
    """The parts of ``requests.Response`` the client reads."""

    def __init__(self, status_code=201, headers=None, payload=None, text=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeLinkedInHTTP:
    """Records every POST and answers from a script, one step per request.

    A step is a ``FakeResponse`` or an exception to raise. Running out of steps
    fails loudly: an extra request is exactly the bug a retry rule is there to
    prevent, so it must not be answered politely.
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def post(self, url, json=None, headers=None, timeout=None, **_kw):
        self.requests.append({"url": url, "json": json, "headers": dict(headers or {})})
        if not self.script:
            raise AssertionError(f"unexpected request #{len(self.requests)} to {url}")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def install_http(script):
    """Point linkedin_client at a FakeLinkedInHTTP; returns (http, sleeps, restore)."""
    import linkedin_client
    http = FakeLinkedInHTTP(script)
    sleeps = []
    original = (linkedin_client.requests.post, linkedin_client._sleep)
    linkedin_client.requests.post = http.post
    linkedin_client._sleep = lambda seconds: sleeps.append(seconds)

    def restore():
        linkedin_client.requests.post, linkedin_client._sleep = original

    return http, sleeps, restore


# ---------------------------------------------------------------------------
# the client seam
# ---------------------------------------------------------------------------

class LinkedInRecorder(RecordingClient):
    """A recording LinkedIn client that also records, and can refuse, comments.

    ``comment_error`` is a sentence: the comment is refused with it, the way the
    real client refuses with ``friendly``. ``comment_raises`` makes the call blow
    up, ``comment_reply`` overrides the whole reply, and ``no_post_urn`` makes a
    post succeed without saying what it is called.
    """

    def __init__(self, **kwargs):
        super().__init__("linkedin", **kwargs)
        self.comment_error = None
        self.comment_raises = None
        self.comment_reply = None
        self.no_post_urn = False

    def _record(self, kind, token, **extra):
        reply = super()._record(kind, token, **extra)
        if self.no_post_urn and reply.get("success"):
            reply["post_urn"] = ""
        return reply

    def create_comment(self, access_token, actor_urn, post_urn, text):
        self.calls.append({"kind": "comment", "token": access_token, "actor": actor_urn,
                           "post": post_urn, "text": text})
        if self.comment_raises is not None:
            raise self.comment_raises
        if self.comment_reply is not None:
            return self.comment_reply
        if self.comment_error:
            return FakeResult(success=False, friendly=self.comment_error)
        return FakeResult(success=True, comment_id=f"c{len(self.calls)}")

    # --- what a gate reads back ---------------------------------------------
    def kinds(self):
        return [call["kind"] for call in self.calls]

    def comments(self):
        return [call for call in self.calls if call["kind"] == "comment"]

    def posts(self):
        return [call for call in self.calls if call["kind"] in ("text", "image", "video")]


# ---------------------------------------------------------------------------
# the app rig
# ---------------------------------------------------------------------------

def rig(accounts=1, account_default=False):
    """An isolated app with LinkedIn account(s) connected and a recording client.

    Returns ``(database, web, publisher, client, account_ids, recorder, clients)``.
    ``account_default`` turns the accounts' own link-in-comment setting on.
    Account ``n`` has the token ``token-linkedin-li-n`` and the author
    ``urn:li:person:li-n``, so a gate can tell which account spoke.
    """
    directory, database, web, publisher, client = isolated_app()
    web._maybe_attach_link_image = lambda *a, **k: None
    web.get_image_for_post = lambda *a, **k: None
    ids = [connect(database, "linkedin", f"li-{n}", f"Member {n}") for n in range(1, accounts + 1)]
    recorder = LinkedInRecorder()
    clients = install_fake_clients(publisher, web, linkedin=recorder)
    if account_default:
        for account_id in ids:
            database.set_social_account_link_in_comment(account_id, True, db_path=database.DB_PATH)
    return database, web, publisher, client, ids, recorder, clients


def token_of(n=1):
    return f"token-linkedin-li-{n}"


def author_of(n=1):
    return f"urn:li:person:li-{n}"


def make_post(database, account, content, image=None, video=None, mode=None, platform="linkedin"):
    """One saved post row; ``mode`` is the post's own choice (None follows the account)."""
    return database.add_standalone_post(
        "manual", "gate", platform, content, image_url=image, video_url=video,
        account_id=account, link_comment=mode, db_path=database.DB_PATH)


def row_of(database, post_id):
    return database.get_standalone_post(post_id, db_path=database.DB_PATH)


def state_of(database, post_id):
    return database.link_comment_state(row_of(database, post_id))

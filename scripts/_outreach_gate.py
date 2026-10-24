"""Shared fakes and rigs for the Outreach gates.

Nothing here reaches Twilio or a model:

* ``FakeTwilioHTTP`` replaces the ``requests`` module inside twilio_client. It
  answers reads (the Messages list, one Message) from an in-memory phone log and
  answers each send from a script, one step per POST. A POST with no step left
  fails the gate: an extra send is exactly the bug the sender's rules exist to
  prevent. ``refuse_posts`` makes every POST fail the gate, for the claim that
  the background worker never sends.
* ``FakeLLM`` stands in for ``insights._get_llm_client`` and answers from a
  script of replies (or exceptions), recording every prompt it was given.

Every rig runs on a throwaway database with a logged-in test client.
"""

import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import requests as real_requests  # noqa: E402

OUR_NUMBER = "+13055550100"
SID = "ACgate000000000000000000000000000"
TOKEN = "gate-token"
XHR = {"X-Requested-With": "XMLHttpRequest"}

# Inside the default 8am-8pm window in Miami and Chicago: 15:00 UTC.
NOON = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)
# 2am in Miami.
NIGHT = datetime(2026, 10, 21, 6, 0, tzinfo=timezone.utc)

MIAMI = "+13055550123"
MIAMI_2 = "+13055550124"
CHICAGO = "+13125550188"
STRANGER = "+17865550199"


def utc(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def parse_dt(value):
    """A stored UTC time back as an aware datetime."""
    return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=201, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def sent_ok(sid="SMok", body=""):
    return FakeResponse(201, {"sid": sid, "status": "queued", "body": body})


def twilio_error(code, status=400, message="refused"):
    return FakeResponse(status, {"code": code, "message": message, "status": status})


def twilio_message(sid, from_, to, body, when, direction="inbound", status="received",
                   error_code=None):
    stamp = format_datetime(when.astimezone(timezone.utc))
    return {"sid": sid, "from": from_, "to": to, "body": body, "status": status,
            "direction": direction, "date_sent": stamp, "date_created": stamp,
            "error_code": error_code, "error_message": None, "num_segments": "1"}


class FakeTwilioHTTP:
    """The parts of ``requests`` twilio_client uses, answering like Twilio."""

    exceptions = real_requests.exceptions

    def __init__(self, post_script=(), messages=(), refuse_posts=False, page_size=None,
                 post_delay=0.0):
        self.post_script = list(post_script)
        self.messages = list(messages)
        self.refuse_posts = refuse_posts
        self.page_size = page_size
        self.post_delay = post_delay
        self.posts = []
        self.gets = []
        self._lock = threading.Lock()

    # --- sending ---------------------------------------------------------
    def post(self, url, data=None, auth=None, timeout=None, **_kw):
        if self.refuse_posts:
            raise AssertionError(f"a POST to Twilio was made from a place that must never send: {url}")
        with self._lock:
            self.posts.append({"url": url, "data": dict(data or {}), "auth": auth, "timeout": timeout})
            if not self.post_script:
                raise AssertionError(f"unexpected send #{len(self.posts)}: {data}")
            step = self.post_script.pop(0)
        if self.post_delay:
            import time
            time.sleep(self.post_delay)
        if isinstance(step, Exception):
            raise step
        return step

    # --- reading ---------------------------------------------------------
    def get(self, url, params=None, auth=None, timeout=None, **_kw):
        self.gets.append({"url": url, "params": dict(params or {})})
        if url.endswith("/Messages.json") or "/Messages.json?" in url:
            if "?" in url:  # a next_page_uri
                from urllib.parse import parse_qs, urlsplit
                params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
            params = params or {}
            found = [m for m in self.messages
                     if (not params.get("To") or m["to"] == params["To"])
                     and (not params.get("From") or m["from"] == params["From"])]
            found.sort(key=lambda m: m["date_created"], reverse=True)
            page = int(params.get("Page", 0))
            size = self.page_size or len(found) or 1
            chunk = found[page * size:(page + 1) * size]
            more = (page + 1) * size < len(found)
            next_uri = None
            if more:
                from urllib.parse import urlencode
                query = urlencode({**{k: v for k, v in params.items() if k != "Page"}, "Page": page + 1})
                next_uri = f"/2010-04-01/Accounts/{SID}/Messages.json?{query}"
            return FakeResponse(200, {"messages": chunk, "next_page_uri": next_uri})
        if "/Messages/" in url:
            sid = url.rsplit("/", 1)[-1].replace(".json", "")
            match = next((m for m in self.messages if m["sid"] == sid), None)
            if match is None:
                return FakeResponse(404, {"code": 20404, "message": "not found"})
            return FakeResponse(200, match)
        raise AssertionError(f"unexpected GET {url}")


def install_twilio(http):
    """Point twilio_client at ``http`` and at fake credentials; returns a restore function."""
    import twilio_client
    original = (twilio_client.requests, twilio_client.get_twilio_client)
    twilio_client.requests = http
    twilio_client.get_twilio_client = lambda: twilio_client.TwilioClient(
        SID, TOKEN, OUR_NUMBER, "")

    def restore():
        twilio_client.requests, twilio_client.get_twilio_client = original

    return restore


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self, prompt_tokens=200, completion_tokens=80):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Reply:
    def __init__(self, text):
        message = type("Message", (), {"content": text})()
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = _Usage()


class FakeLLM:
    """Answers chat calls from a script; a step may be text, a dict (sent as JSON) or an exception.

    ``responder(prompt)`` answers instead when the script is empty, for gates
    whose order of calls depends on what the worker read.
    """

    def __init__(self, script=(), default=None, responder=None):
        self.script = list(script)
        self.default = default
        self.responder = responder
        self.calls = []
        outer = self

        class _Completions:
            def create(self, model=None, messages=None, **kwargs):
                outer.calls.append({"model": model, "messages": messages, **kwargs})
                if outer.script:
                    step = outer.script.pop(0)
                elif outer.responder is not None:
                    step = outer.responder(messages[-1]["content"])
                elif outer.default is not None:
                    step = outer.default
                else:
                    raise AssertionError(f"unexpected model call #{len(outer.calls)}")
                if isinstance(step, Exception):
                    raise step
                if isinstance(step, dict):
                    step = json.dumps(step)
                return _Reply(step)

        self.chat = type("Chat", (), {"completions": _Completions()})()

    def prompts(self):
        return [c["messages"][-1]["content"] for c in self.calls]


def install_llm(llm):
    import insights
    original = insights._get_llm_client
    insights._get_llm_client = lambda *a, **k: (llm, "gate-model", "openai")

    def restore():
        insights._get_llm_client = original

    return restore


def draft_reply(text="Hey, where are you starting from?", intent="question", offer=None,
                reason="fits"):
    return {"reply": text, "intent": intent, "offer_key": offer, "reason": reason}


# ---------------------------------------------------------------------------
# the app rig
# ---------------------------------------------------------------------------

# A made-up business for every gate: the shipped defaults are empty, and the
# agent drafts nothing without a business context.
TEST_CONTEXT = {
    "business_name": "Northwind Mentoring",
    "business_field": "career mentoring for people moving into tech",
    "program_summary": "A twelve-week mentorship with weekly calls, a study plan and mock interviews.",
    "ideal_lead": "Career changers and IT support staff who want a first technical role.",
    "voice_notes": "Warm, direct, short sentences.",
    "sms_signature": "Northwind Mentoring",
    "offers": [
        {"key": "playbook", "name": "Starter Playbook", "url": "https://northwind.test/playbook",
         "when_to_use": "Someone new asking where to start.", "match": "playbook"},
        {"key": "career_quiz", "name": "Career quiz", "url": "https://northwind.test/quiz",
         "when_to_use": "Someone unsure which role fits them.", "match": "quiz, career path"},
        {"key": "apply", "name": "Intro call", "url": "",
         "when_to_use": "Someone who wants a plan.", "match": ""},
    ],
}


def rig(now=NOON, context=True):
    """An isolated app, a logged-in client, and one clock fixed at ``now``.

    The sender's "now" and the time the database stamps on rows both read the
    clock, so a gate that moves it (``set_now``) moves the whole app, the way
    real time would. Returns ``(database, web, client, set_now)``.
    """
    os.environ.pop("OUTREACH_MODEL", None)
    from _gate_common import isolated_app
    _directory, database, web, client = isolated_app()
    if context:
        database.update_outreach_profile(**TEST_CONTEXT)
    import outreach_sender
    clock = {"now": now}
    outreach_sender.utc_now = lambda: clock["now"]
    database.outreach_now = lambda: utc(clock["now"])
    web.outreach_utc_now = lambda: clock["now"]

    def set_now(moment):
        clock["now"] = moment

    return database, web, client, set_now


def lead(database, phone=MIAMI, name="Sam Rivera", sms=True, call=True, **extra):
    fields = dict(name=name, phone=phone, sms_consent=sms, call_consent=call,
                  magnets=["playbook"], source="kartra")
    fields.update(extra)
    return database.create_lead([("phone", phone)], **fields)


def draft(database, lead_id, body="Hey Sam, saw you grabbed the playbook. Where are you starting from?",
          kind="opener", based_on=None, created_at=None):
    return database.add_outreach_message(
        "sms", "out", "draft", body, lead_id=lead_id, kind=kind,
        context={"by": "agent", "reason": "gate"}, based_on_message_id=based_on,
        created_at=created_at)


def inbound(database, lead_id, body, when, sid=None, peer=MIAMI):
    return database.add_outreach_message(
        "sms", "in", "received", body, lead_id=lead_id, kind="inbound",
        external_id=sid or f"SMin{body[:6]}{when.timestamp()}", peer=peer, sent_at=utc(when))


def sent(database, lead_id, body, when, sid=None):
    message_id = database.add_outreach_message(
        "sms", "out", "sent", body, lead_id=lead_id, kind="opener",
        external_id=sid or f"SMout{when.timestamp()}{lead_id}", sent_at=utc(when),
        created_at=utc(when))
    return message_id


def send(client, message_id, body=None, database=None):
    if body is None and database is not None:
        body = database.get_outreach_message(message_id)["body"]
    response = client.post(f"/outreach/messages/{message_id}/send", json={"body": body or ""},
                           headers=XHR)
    return response.status_code, response.get_json() or {}


def status_of(database, message_id):
    return database.get_outreach_message(message_id)["status"]


def logged_in_client(web, database):
    """A second logged-in client (another browser tab with its own cookie jar)."""
    from werkzeug.security import generate_password_hash
    uid = database.create_user(f"tab{datetime.now().timestamp()}", generate_password_hash("p"),
                               db_path=database.DB_PATH)
    client = web.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = uid
    return client


def minutes(n):
    return timedelta(minutes=n)


# ---------------------------------------------------------------------------
# the social platforms
# ---------------------------------------------------------------------------

def social_reply(text="Good question. Most people start with networking basics and a home lab.",
                 should=True, is_lead=True, fit=70, intent="question", offer=None, reason="asked for help"):
    return {"should_reply": should, "is_lead": is_lead, "fit": fit, "reply": text,
            "intent": intent, "offer_key": offer, "reason": reason}


def meta_error(code, message="error", status=400):
    return FakeResponse(status, {"error": {"message": message, "type": "OAuthException", "code": code}})


PERMISSION_ERROR = (10, "(#10) Application does not have permission for this action")


class FakeSocialHTTP:
    """The parts of ``requests`` outreach_social uses, answering like the four platforms.

    ``routes`` maps a URL fragment to a response, a list of responses (one per
    call, the last repeated), or a function ``(url, params) -> response``. GETs
    are matched against them, longest fragment first; an unrouted GET fails the
    gate. POSTs answer from ``post_script`` in order, each step a pair of the URL
    fragment it must hit and the response (or exception) to give.
    """

    exceptions = real_requests.exceptions

    def __init__(self, routes=None, post_script=(), refuse_posts=False):
        self.routes = dict(routes or {})
        self.post_script = list(post_script)
        self.refuse_posts = refuse_posts
        self.gets = []
        self.posts = []

    def get(self, url, params=None, headers=None, timeout=None, **_kw):
        self.gets.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        for fragment in sorted(self.routes, key=len, reverse=True):
            if fragment in url:
                answer = self.routes[fragment]
                if callable(answer) and not isinstance(answer, FakeResponse):
                    answer = answer(url, dict(params or {}))
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, params=None, data=None, json=None, headers=None, timeout=None, **_kw):
        if self.refuse_posts:
            raise AssertionError(f"a write request was made from a place that must never write: {url}")
        self.posts.append({"url": url, "params": dict(params or {}), "data": dict(data or {}),
                           "json": json, "headers": dict(headers or {})})
        if not self.post_script:
            raise AssertionError(f"unexpected POST #{len(self.posts)} to {url}")
        fragment, step = self.post_script.pop(0)
        if fragment not in url:
            raise AssertionError(f"POST went to {url}, expected one containing {fragment}")
        if isinstance(step, Exception):
            raise step
        return step

    def gets_to(self, fragment):
        return [g for g in self.gets if fragment in g["url"]]


def install_social(http):
    import outreach_social
    original = (outreach_social.requests, outreach_social._sleep)
    outreach_social.requests = http
    outreach_social._sleep = lambda seconds: None

    def restore():
        outreach_social.requests, outreach_social._sleep = original

    return restore


def connect_social(database):
    """Threads, Instagram, a Facebook Page and X, each connected as the owner.

    Returns ``{platform: account_id}``. Handles: Threads @northwind, Instagram
    @northwind.ig, the Page "PAGE1" (id PAGE1), X @northwind_x (id 42).
    """
    from _accounts_gate import connect
    return {
        "threads": connect(database, "threads", "th-1", "northwind"),
        "instagram": connect(database, "instagram", "ig-1", "northwind.ig"),
        "facebook": connect(database, "facebook", "PAGE1", "PAGE1"),
        "twitter": connect(database, "twitter", "42", "northwind_x"),
    }


def iso(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+0000")


def threads_post(pid="T1", text="What was your first cert?", when=None, has_replies=True):
    return {"id": pid, "text": text, "permalink": f"https://www.threads.net/@northwind/post/{pid}",
            "timestamp": iso(when or NOON - timedelta(days=1)), "username": "northwind",
            "has_replies": has_replies}


def threads_reply(rid, text, username, parent="T1", root="T1", when=None, own=False):
    return {"id": rid, "text": text, "username": username,
            "permalink": f"https://www.threads.net/@{username}/post/{rid}",
            "timestamp": iso(when or NOON - timedelta(hours=2)), "replied_to": {"id": parent},
            "root_post": {"id": root}, "is_reply_owned_by_me": own}


def data(*items, next_url=None):
    payload = {"data": list(items)}
    if next_url:
        payload["paging"] = {"next": next_url}
    return FakeResponse(200, payload)

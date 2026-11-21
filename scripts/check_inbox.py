"""Gates for the Inbox: one conversation per person, and Instagram and Facebook DMs.

    python scripts/check_inbox.py schema    INBOX_SCHEMA_OK
    python scripts/check_inbox.py state     INBOX_STATE_OK
    python scripts/check_inbox.py reply     INBOX_REPLY_OK
    python scripts/check_inbox.py dm_read   INBOX_DM_READ_OK
    python scripts/check_inbox.py dm_send   INBOX_DM_SEND_OK
    python scripts/check_inbox.py dm_auth   INBOX_DM_AUTH_OK
    python scripts/check_inbox.py canned    INBOX_CANNED_OK
    python scripts/check_inbox.py ui        INBOX_UI_OK
    python scripts/check_inbox.py docs      INBOX_DOCS_OK
    python scripts/check_inbox.py dashes    INBOX_NO_DASHES_OK

The claims that matter most: a reply from the Inbox leaves only through
outreach_sender, with every Outreach rule still applied (``reply``, ``dm_send``);
the background worker reads DMs and has no way to send one (``dm_read``); and a
conversation keeps its status, labels and notes when the person behind it
becomes a lead or two leads merge (``schema``).

Every section runs on a throwaway database. Twilio and Meta are scripted fakes
standing in for ``requests``; a request nobody scripted fails the gate, so an
extra send is caught rather than made.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import timedelta

import logging

logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import check  # noqa: E402
import _outreach_gate as g  # noqa: E402
from _outreach_gate import MIAMI, MIAMI_2, CHICAGO, NOON, STRANGER, XHR, minutes  # noqa: E402

REPO = g.REPO
TOKENS = {
    "schema": "INBOX_SCHEMA_OK", "state": "INBOX_STATE_OK", "reply": "INBOX_REPLY_OK",
    "dm_read": "INBOX_DM_READ_OK", "dm_send": "INBOX_DM_SEND_OK", "dm_auth": "INBOX_DM_AUTH_OK",
    "canned": "INBOX_CANNED_OK", "ui": "INBOX_UI_OK", "docs": "INBOX_DOCS_OK",
    "dashes": "INBOX_NO_DASHES_OK",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def comment(database, platform, external, text, author, *, account_id, when, lead_id=None,
            author_id="", triage=None, root="P1"):
    context = {"source": "comment", "account_id": account_id, "root_id": root, "parent_id": root,
               "permalink": f"https://example.test/{root}", "post_text": "My post", "author": author,
               "author_id": author_id, "author_name": ""}
    if triage:
        context["triage"] = triage
    return database.add_outreach_message(
        platform, "in", "received", text, lead_id=lead_id, kind="inbound", context=context,
        external_id=external, peer=f"@{author}", sent_at=g.utc(when))


def dm(database, channel, external, text, *, account_id, when, author="sam.sec", author_id="901",
       author_name="", lead_id=None, thread="IGC1"):
    context = {"source": "dm", "account_id": account_id, "root_id": thread, "author": author,
               "author_id": author_id, "author_name": author_name}
    return database.add_outreach_message(
        channel, "in", "received", text, lead_id=lead_id, kind="inbound", context=context,
        external_id=external, peer=f"@{author}" if author else author_name, sent_at=g.utc(when))


def conversations(client, **params):
    from urllib.parse import urlencode
    response = client.get("/inbox/conversations?" + urlencode({"status": "all", **params}))
    check(response.status_code == 200, f"the list answers: {response.status_code} {response.get_data(as_text=True)[:200]}")
    return response.get_json()


def by_name(listing, name):
    found = [c for c in listing["conversations"] if c["name"] == name]
    check(len(found) == 1, f"one conversation named {name!r}: {[c['name'] for c in listing['conversations']]}")
    return found[0]


def post(client, path, payload=None):
    response = client.post(path, json=payload or {}, headers=XHR)
    return response.status_code, response.get_json() or {}


def detail(client, conversation_id):
    response = client.get(f"/inbox/conversations/{conversation_id}")
    check(response.status_code == 200, f"conversation {conversation_id} opens: {response.get_json()}")
    return response.get_json()


def reply(client, conversation_id, channel, body, seen=None):
    if seen is None:
        seen = detail(client, conversation_id)["last_in_id"]
    return post(client, f"/inbox/conversations/{conversation_id}/reply",
                {"channel": channel, "body": body, "seen_id": seen})


def out_rows(database, lead_id, channel):
    return [m for m in database.list_lead_messages(lead_id, channel=channel) if m["direction"] == "out"]


# ---------------------------------------------------------------------------
# schema: tables, grouping, and state that follows the person
# ---------------------------------------------------------------------------

def section_schema():
    import sqlite3
    import inbox
    database, web, client, set_now = g.rig()
    with sqlite3.connect(database.DB_PATH) as conn:
        for table in ("inbox_conversations", "inbox_notes", "canned_responses"):
            conn.execute(f"DROP TABLE {table}")
    database.init_db(database.DB_PATH)
    database.init_db(database.DB_PATH)
    with sqlite3.connect(database.DB_PATH) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    check({"inbox_conversations", "inbox_notes", "canned_responses"} <= tables,
          f"the inbox tables come back on an old database: {tables}")

    ids = g.connect_social(database)
    sam = g.lead(database)
    g.inbound(database, sam, "Is the course open?", NOON - minutes(50))
    comment(database, "instagram", "C1", "Saw your post!", "sam.ig", account_id=ids["instagram"],
            when=NOON - minutes(40), lead_id=sam)
    comment(database, "instagram", "C2", "How do I start?", "tom.it", account_id=ids["instagram"],
            when=NOON - minutes(30))
    comment(database, "instagram", "C3", "Earlier question", "tom.it", account_id=ids["instagram"],
            when=NOON - minutes(35))
    comment(database, "instagram", "C4", "cheap followers", "spammer", account_id=ids["instagram"],
            when=NOON - minutes(20), triage="skip")
    comment(database, "facebook", "F1", "Remote?", "Jane Doe", account_id=ids["facebook"],
            when=NOON - minutes(25), author_id="555")
    database.add_outreach_message("sms", "in", "received", "who is this", kind="inbound",
                                  external_id="SMstr", peer=STRANGER, sent_at=g.utc(NOON - minutes(10)))
    listing = conversations(client)
    keys = sorted(c["key"] for c in listing["conversations"])
    check(keys == sorted([f"lead:{sam}", "instagram:tom.it", "facebook:555", f"phone:{STRANGER}"]),
          f"one conversation per person, and a skipped stranger opens none: {keys}")
    sam_conv = by_name(listing, "Sam Rivera")
    check(sam_conv["channels"] == ["instagram", "sms"], f"a lead's texts and comments are one conversation: {sam_conv}")
    for message_id in range(1, 8):
        message = database.get_outreach_message(message_id)
        if message:
            sql_key = next((c["key"] for c in listing["conversations"]
                            if message_id in [m["id"] for m in database.inbox_messages_for_key(c["key"])]), None)
            if sql_key:
                check(database.inbox_message_key(message) == sql_key,
                      f"the Python key and the SQL key agree for message {message_id}: "
                      f"{database.inbox_message_key(message)} vs {sql_key}")

    # The stranger on Instagram gets a status, labels and a note, then becomes a lead.
    tom = by_name(listing, "@tom.it")
    check(post(client, f"/inbox/conversations/{tom['id']}/labels", {"labels": ["hot", "course"]})[0] == 200, "labels save")
    check(post(client, f"/inbox/conversations/{tom['id']}/notes", {"body": "Asked twice"})[0] == 200, "a note saves")
    check(post(client, f"/inbox/conversations/{tom['id']}/status", {"status": "resolved"})[0] == 200, "resolve saves")
    tom_lead = database.create_lead([("instagram", "tom.it")], name="Tom Lee")
    for message in database.inbox_messages_for_key("instagram:tom.it"):
        database.update_outreach_message(message["id"], lead_id=tom_lead)
    listing = conversations(client)
    moved = by_name(listing, "Tom Lee")
    check(moved["key"] == f"lead:{tom_lead}" and moved["labels"] == ["hot", "course"]
          and moved["status"] == "resolved" and moved["id"] == tom["id"],
          f"becoming a lead keeps the conversation's status and labels: {moved}")
    notes = [t for t in detail(client, moved["id"])["timeline"] if t["type"] == "note"]
    check([n["body"] for n in notes] == ["Asked twice"], f"and its notes: {notes}")

    # Two leads merge: labels join, notes move, one conversation is left.
    other = g.lead(database, phone=MIAMI_2, name="Sam R (2)")
    g.inbound(database, other, "Also me", NOON - minutes(5))
    listing = conversations(client)
    sam_id, other_id = by_name(listing, "Sam Rivera")["id"], by_name(listing, "Sam R (2)")["id"]
    post(client, f"/inbox/conversations/{sam_id}/labels", {"labels": ["vip"]})
    post(client, f"/inbox/conversations/{other_id}/labels", {"labels": ["texted"]})
    post(client, f"/inbox/conversations/{other_id}/notes", {"body": "Second number"})
    database.merge_leads(sam, other)
    listing = conversations(client)
    merged = by_name(listing, "Sam Rivera")
    check(merged["id"] == sam_id and merged["labels"] == ["vip", "texted"],
          f"after a merge the kept conversation has both labels: {merged}")
    check([t["body"] for t in detail(client, sam_id)["timeline"] if t["type"] == "note"] == ["Second number"],
          "and the merged lead's note")
    check(not any(c["name"] == "Sam R (2)" for c in listing["conversations"])
          and database.get_inbox_state(conversation_id=other_id) is None,
          "and the dropped lead's conversation is gone")
    check(inbox.effective_status(None, {"last_in_id": 3}, NOON) == "open", "a conversation with no state is open")
    print("schema: the inbox tables upgrade an old database; messages group into one conversation per person; "
          "status, labels and notes follow a person who becomes a lead, and two leads that merge")
    print(TOKENS["schema"])


# ---------------------------------------------------------------------------
# state: filters, status rules, read marks, refusals
# ---------------------------------------------------------------------------

def section_state():
    database, web, client, set_now = g.rig()
    from werkzeug.security import generate_password_hash
    with client.session_transaction() as session:
        me = session["user_id"]
    teammate = database.create_user("teammate", generate_password_hash("p"), db_path=database.DB_PATH)
    a = g.lead(database, phone=MIAMI, name="Ana")
    b = g.lead(database, phone=MIAMI_2, name="Ben")
    c = g.lead(database, phone=CHICAGO, name="Cy")
    g.inbound(database, a, "Question about pricing", NOON - minutes(60))
    g.inbound(database, b, "Thanks for the call", NOON - minutes(50))
    g.sent(database, b, "Talk soon", NOON - minutes(45))
    g.inbound(database, c, "Is it remote?", NOON - minutes(40))
    listing = conversations(client)
    ids = {n: by_name(listing, n)["id"] for n in ("Ana", "Ben", "Cy")}

    check(post(client, f"/inbox/conversations/{ids['Ana']}/status", {"status": "resolved"})[0] == 200, "resolve")
    status, data = post(client, f"/inbox/conversations/{ids['Ben']}/status", {"status": "snoozed", "preset": "1h"})
    check(status == 200 and data["snoozed_until"] == g.utc(NOON + timedelta(hours=1)), f"snooze for an hour: {data}")
    check(post(client, f"/inbox/conversations/{ids['Cy']}/assign", {"assignee_id": teammate})[0] == 200, "assign")
    check(post(client, f"/inbox/conversations/{ids['Ana']}/assign", {"assignee_id": me})[0] == 200, "assign to me")
    check(post(client, f"/inbox/conversations/{ids['Cy']}/labels", {"labels": ["Remote", " remote ", "q4"]})[1]["labels"]
          == ["Remote", "q4"], "labels are trimmed and kept once")

    def names(**params):
        return sorted(c["name"] for c in conversations(client, **params)["conversations"])

    check(names(status="open") == ["Cy"], f"open: {names(status='open')}")
    check(names(status="snoozed") == ["Ben"] and names(status="resolved") == ["Ana"], "snoozed and resolved tabs")
    counts = conversations(client, status="open")["counts"]
    check(counts == {"open": 1, "snoozed": 1, "resolved": 1, "all": 3}, f"tab counts: {counts}")
    check(names(assignee="mine") == ["Ana"] and names(assignee="unassigned") == ["Ben"], "mine and unassigned")
    check(names(label="q4") == ["Cy"], "by label")
    check(names(needs_reply="1") == ["Ana", "Cy"], f"needs reply: {names(needs_reply='1')}")
    check(names(q="pricing") == ["Ana"] and names(q="ben") == ["Ben"], "search by words and by name")
    check(names(channel="sms") == ["Ana", "Ben", "Cy"] and names(channel="instagram_dm") == [], "by channel")

    # People reopen what the owner closed, and snoozes end on their own.
    g.inbound(database, a, "One more thing", NOON - minutes(1))
    check("Ana" in names(status="open"), "a new message reopens a resolved conversation")
    set_now(NOON + timedelta(hours=2))
    check("Ben" in names(status="open"), "a snooze ends at its time")
    set_now(NOON)
    post(client, f"/inbox/conversations/{ids['Ben']}/status", {"status": "snoozed", "preset": "tomorrow"})
    check("Ben" in names(status="snoozed"), "snoozed until tomorrow")
    g.inbound(database, b, "Actually, one question", NOON - minutes(2))
    check("Ben" in names(status="open"), "a new message wakes a snoozed conversation early")

    # Unread counts clear on opening and come back on Mark unread.
    ana = by_name(conversations(client), "Ana")
    check(ana["unread"] == 2, f"two unread from Ana: {ana}")
    post(client, f"/inbox/conversations/{ids['Ana']}/read")
    check(by_name(conversations(client), "Ana")["unread"] == 0, "opened: none unread")
    post(client, f"/inbox/conversations/{ids['Ana']}/read", {"unread": True})
    check(by_name(conversations(client), "Ana")["unread"] == 1, "marked unread: the latest is unread again")

    for path, payload in ((f"/inbox/conversations/{ids['Ana']}/status", {"status": "bogus"}),
                          (f"/inbox/conversations/{ids['Ana']}/status", {"status": "snoozed", "until": "2020-01-01T09:00"}),
                          (f"/inbox/conversations/{ids['Ana']}/status", {"status": "snoozed"}),
                          (f"/inbox/conversations/{ids['Ana']}/assign", {"assignee_id": 9999}),
                          (f"/inbox/conversations/{ids['Ana']}/labels", {"labels": ["<b>bold</b>"]}),
                          (f"/inbox/conversations/{ids['Ana']}/labels", {"labels": [f"l{i}" for i in range(11)]}),
                          (f"/inbox/conversations/{ids['Ana']}/labels", {"labels": "notalist"})):
        status, data = post(client, path, payload)
        check(status == 400 and data.get("error"), f"{path} {payload} is refused: {status} {data}")
    check(client.post(f"/inbox/conversations/{ids['Ana']}/status", json={"status": "resolved"}).status_code == 400,
          "a write without the page's XHR header is refused")
    check(client.get("/inbox/conversations?status=nope").status_code == 400, "an unknown filter is refused")
    check(post(client, "/inbox/conversations/9999/status", {"status": "open"})[0] == 404, "an unknown conversation is 404")
    status, data = post(client, f"/inbox/conversations/{ids['Ana']}/status", {"status": "snoozed", "until": "2026-10-22T09:00"})
    check(status == 200 and data["snoozed_until"] == "2026-10-22 13:00:00",
          f"a custom snooze is read in the owner's time zone (New York): {data}")
    print("state: status tabs and counts, mine/unassigned, label, channel, needs-reply and search filters; new messages "
          "reopen, snoozes end on time; unread clears and comes back; bad values are refused with 400")
    print(TOKENS["state"])


# ---------------------------------------------------------------------------
# reply: the Inbox sends through outreach_sender
# ---------------------------------------------------------------------------

def section_reply():
    import inbox
    database, web, client, set_now = g.rig()
    twilio = g.FakeTwilioHTTP(post_script=[g.sent_ok("SM1"), g.sent_ok("SM2")])
    g.install_twilio(twilio)
    social = g.FakeSocialHTTP({})
    g.install_social(social)
    ids = g.connect_social(database)

    sam = g.lead(database)
    g.inbound(database, sam, "Is the course still open?", NOON - minutes(30))
    conv = by_name(conversations(client), "Sam Rivera")
    seen = detail(client, conv["id"])["last_in_id"]
    status, data = reply(client, conv["id"], "sms", "Yes, it is!", seen)
    check(status == 200 and data["status"] == "sent" and len(twilio.posts) == 1, f"an SMS reply is sent once: {status} {data}")
    check(twilio.posts[0]["data"]["To"] == MIAMI and twilio.posts[0]["data"]["Body"].startswith("Yes, it is!"),
          f"to Sam, with the owner's words: {twilio.posts[0]['data']}")
    sent = out_rows(database, sam, "sms")
    check(len(sent) == 1 and sent[0]["status"] == "sent" and sent[0]["external_id"] == "SM1"
          and sent[0]["context"].get("via") == "inbox", f"recorded as sent from the Inbox: {sent}")
    check(by_name(conversations(client), "Sam Rivera")["unread"] == 0, "replying marks the conversation read")
    status, data = reply(client, conv["id"], "sms", "Yes, it is!", seen)
    check(status == 409 and len(twilio.posts) == 1, f"a second press sends nothing: {status} {data}")

    # Two presses at the same moment still send once.
    g.inbound(database, sam, "Great, how much?", NOON - minutes(1))
    seen = detail(client, conv["id"])["last_in_id"]
    results = []

    def press():
        tab = g.logged_in_client(web, database)
        results.append(reply(tab, conv["id"], "sms", "It is $300.", seen))

    threads = [threading.Thread(target=press) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(sorted(r[0] for r in results) == [200, 409] and len(twilio.posts) == 2,
          f"two simultaneous presses send once: {results}")

    # Refusals keep nothing behind.
    quiet = g.lead(database, phone=CHICAGO, name="Quinn", sms=False)
    g.inbound(database, quiet, "hello?", NOON - timedelta(hours=30))
    qconv = by_name(conversations(client), "Quinn")
    status, data = reply(client, qconv["id"], "sms", "Hi Quinn")
    check(status == 403 and "consent" in data["error"].lower() and len(twilio.posts) == 2,
          f"no consent and no recent text: refused, nothing sent: {status} {data}")
    check(out_rows(database, quiet, "sms") == [] and database.list_outreach_queue() == [],
          "and no draft is left in the Outreach queue")
    status, data = reply(client, qconv["id"], "sms", "Hi Quinn", seen=0)
    check(status == 409 and data.get("reload") and "wrote again" in data["error"],
          f"a reply written before their newest message is held: {status} {data}")
    database.set_lead_opt_out(sam, "manual")
    status, data = reply(client, conv["id"], "sms", "One more thing")
    check(status == 403 and len(twilio.posts) == 2, f"after an opt-out nothing is sent: {status} {data}")

    # The agent's open draft on that channel is reused, not doubled.
    dana = g.lead(database, phone=MIAMI_2, name="Dana")
    g.inbound(database, dana, "Interested", NOON - minutes(20))
    agent_draft = g.draft(database, dana, body="Agent words", kind="continuation")
    twilio.post_script.append(g.sent_ok("SM3"))
    dconv = by_name(conversations(client), "Dana")
    check(detail(client, dconv["id"])["drafts"]["sms"]["body"] == "Agent words", "the composer is offered the agent's draft")
    status, data = reply(client, dconv["id"], "sms", "My own words")
    rows = out_rows(database, dana, "sms")
    check(status == 200 and len(rows) == 1 and rows[0]["id"] == agent_draft and rows[0]["body"].startswith("My own words"),
          f"the agent's draft is the one sent, with the owner's words: {status} {rows}")

    # A comment reply answers that person's latest comment, and makes them a lead.
    comment(database, "instagram", "C1", "First question", "sam.sec", account_id=ids["instagram"],
            when=NOON - minutes(50))
    comment(database, "instagram", "C2", "Second question", "sam.sec", account_id=ids["instagram"],
            when=NOON - minutes(40))
    stranger = by_name(conversations(client), "@sam.sec")
    post(client, f"/inbox/conversations/{stranger['id']}/labels", {"labels": ["ig"]})
    options = {o["channel"]: o for o in detail(client, stranger["id"])["reply_options"]}
    check(list(options) == ["instagram"] and options["instagram"]["enabled"], f"only the comment channel is offered: {options}")
    social.post_script = [("/C2/replies", g.FakeResponse(200, {"id": "R9"}))]
    status, data = reply(client, stranger["id"], "instagram", "Start with the basics.")
    check(status == 200 and data["status"] == "sent", f"the comment reply goes out: {status} {data}")
    check(social.posts[0]["data"]["message"] == "Start with the basics." and "C2" in social.posts[0]["url"],
          f"it answers their latest comment: {social.posts}")
    lead_id = database.find_lead_by_identity("instagram", "sam.sec")
    check(lead_id and all(m["lead_id"] == lead_id for m in database.inbox_messages_for_key(f"lead:{lead_id}"))
          and len(database.inbox_messages_for_key(f"lead:{lead_id}")) == 3,
          "the commenter became a lead, with both comments and the reply")
    moved = by_name(conversations(client), "@sam.sec")
    check(moved["key"] == f"lead:{lead_id}" and moved["labels"] == ["ig"] and data["conversation_id"] == moved["id"],
          f"the conversation followed them, labels and all: {moved} {data}")

    # A private note sends nothing anywhere.
    twilio.refuse_posts = True
    social.refuse_posts = True
    before = len(database.list_lead_messages(dana))
    status, data = post(client, f"/inbox/conversations/{dconv['id']}/notes", {"body": "Call her Friday"})
    check(status == 200 and len(database.list_lead_messages(dana)) == before, f"a note adds no message: {status}")
    check(post(client, f"/inbox/conversations/{dconv['id']}/notes", {"body": "   "})[0] == 400, "an empty note is refused")
    timeline = detail(client, dconv["id"])["timeline"]
    check(timeline[-1]["type"] == "note" and timeline[-1]["body"] == "Call her Friday", "the note shows in the thread")
    status, data = reply(client, dconv["id"], "instagram", "wrong channel")
    check(status == 409, f"a channel they never used is refused: {status} {data}")
    check(post(client, f"/inbox/conversations/{dconv['id']}/reply", {"channel": "sms", "body": " "})[0] == 400,
          "an empty reply is refused")
    check(client.post(f"/inbox/conversations/{dconv['id']}/reply", json={"channel": "sms", "body": "x"}).status_code == 400,
          "a reply without the page's XHR header is refused")
    check(inbox.REPLY_CHANNELS[0] == "sms", "texts come first in the channel picker")
    print("reply: an Inbox reply goes through outreach_sender once (double and simultaneous presses send once); "
          "refusals send nothing and leave no draft; the agent's draft is reused; a comment reply answers the latest "
          "comment and makes the person a lead; notes send nothing")
    print(TOKENS["reply"])


# ---------------------------------------------------------------------------
# dm_read: the worker reads DMs and never sends
# ---------------------------------------------------------------------------

_SENDING_CALLS = ("send_sms", "post", "reply", "reply_to_comment", "_threads_reply", "_x_reply",
                  "_simple_reply", "publish", "publish_targets", "direct_message", "_dm_post")


def _sends_in(path):
    tree = ast.parse(open(path).read())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name in ("outreach_sender", "inbox")]
        elif isinstance(node, ast.ImportFrom) and node.module in ("outreach_sender", "inbox"):
            found.append(f"from {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr in _SENDING_CALLS:
            found.append(node.attr)
    return found


def _ig_message(mid, text, when, *, own=False, username="sam.sec", user_id="901"):
    sender = {"id": "ig-1", "username": "northwind.ig"} if own else {"id": user_id, "username": username}
    to = {"id": user_id, "username": username} if own else {"id": "ig-1", "username": "northwind.ig"}
    return {"id": mid, "created_time": g.iso(when), "from": sender, "to": {"data": [to]}, "message": text}


def _dm_routes(ig_thread, fb_thread, ig_updated, fb_updated="2026-10-21T13:00:00+0000"):
    return {
        "/me/conversations": lambda url, params: g.data({"id": "IGC1", "updated_time": ig_updated[0]}),
        "/IGC1": lambda url, params: g.FakeResponse(200, {"messages": {"data": list(reversed(ig_thread))}}),
        "/PAGE1/conversations": g.data({"id": "FBC1", "updated_time": fb_updated}),
        "/FBC1": lambda url, params: g.FakeResponse(200, {"messages": {"data": fb_thread}}),
    }


def section_dm_read():
    import outreach_worker
    database, web, client, set_now = g.rig()
    ids = g.connect_social(database)
    g.install_twilio(g.FakeTwilioHTTP(refuse_posts=True))
    ig_thread = [_ig_message("m1", "Hi! Is the mentorship open?", NOON - minutes(30)),
                 _ig_message("m2", "Thanks for asking, yes", NOON - minutes(25), own=True),
                 _ig_message("m3", "How much is it?", NOON - minutes(20))]
    fb_thread = [{"id": "f1", "created_time": g.iso(NOON - minutes(15)), "message": "Is it remote?",
                  "from": {"id": "555", "name": "Jane Doe"}, "to": {"data": [{"id": "PAGE1", "name": "PAGE1"}]}}]
    updated = ["2026-10-21T14:40:00+0000"]
    http = g.FakeSocialHTTP(_dm_routes(ig_thread, fb_thread, updated), refuse_posts=True)
    g.install_social(http)

    summary = {"errors": []}
    outreach_worker.dm_pass(database.get_outreach_profile(), NOON, summary, force=True)
    check(summary.get("dm_inbound") == 3 and not summary["errors"], f"three DMs read: {summary}")
    ig_lead = database.find_lead_by_identity("instagram", "sam.sec")
    fb_lead = database.find_lead_by_identity("facebook", "555")
    check(ig_lead and fb_lead, "each person became a lead on their first DM")
    ig_rows = database.list_lead_messages(ig_lead, channel="instagram_dm")
    check([(m["direction"], m["status"], m["body"]) for m in ig_rows] == [
        ("in", "received", "Hi! Is the mentorship open?"),
        ("out", "external", "Thanks for asking, yes"),
        ("in", "received", "How much is it?")], f"the Instagram thread, the owner's own message included: {ig_rows}")
    check(ig_rows[0]["context"]["root_id"] == "IGC1" and ig_rows[0]["context"]["author_id"] == "901"
          and ig_rows[0]["context"]["account_id"] == ids["instagram"], f"rows carry the thread and the sender: {ig_rows[0]}")
    fb_rows = database.list_lead_messages(fb_lead, channel="facebook_dm")
    check(len(fb_rows) == 1 and database.get_lead(fb_lead)["name"] == "Jane Doe", f"the Messenger thread: {fb_rows}")
    listing = conversations(client)
    check(by_name(listing, "@sam.sec")["channels"] == ["instagram_dm"] and by_name(listing, "Jane Doe"),
          "both show in the Inbox")
    reads = len(http.gets_to("/IGC1"))

    set_now(NOON + minutes(10))
    summary = {"errors": []}
    outreach_worker.dm_pass(database.get_outreach_profile(), NOON + minutes(10), summary)
    check(not summary.get("dm_inbound") and len(http.gets_to("/IGC1")) == reads,
          f"an unchanged thread is not fetched again and nothing is stored twice: {summary}")
    check(len(database.list_lead_messages(ig_lead, channel="instagram_dm")) == 3, "still three rows")
    listed = len(http.gets_to("/me/conversations"))
    outreach_worker.dm_pass(database.get_outreach_profile(), NOON + minutes(12), {"errors": []})
    check(len(http.gets_to("/me/conversations")) == listed,
          "within OUTREACH_DM_MINUTES of the last read, DMs are not listed again")

    # A DM Insights sent whose outcome was unknown is settled when it shows up.
    unknown = database.add_outreach_message(
        "instagram_dm", "out", "unknown", "On my way", lead_id=ig_lead, kind="continuation",
        context={"account_id": ids["instagram"], "root_id": "IGC1", "recipient_id": "901"},
        sent_at=g.utc(NOON + minutes(11)))
    lost = database.add_outreach_message(
        "instagram_dm", "out", "unknown", "Never arrived", lead_id=ig_lead, kind="continuation",
        context={"account_id": ids["instagram"], "root_id": "IGC1", "recipient_id": "901"},
        sent_at=g.utc(NOON - minutes(70)))
    ig_thread.append(_ig_message("m4", "On my way", NOON + minutes(11), own=True))
    set_now(NOON + minutes(16))
    outreach_worker.dm_pass(database.get_outreach_profile(), NOON + minutes(16), {"errors": []})
    check(database.get_outreach_message(unknown)["status"] == "sent"
          and database.get_outreach_message(unknown)["external_id"] == "m4",
          "a thread with an unsettled DM is fetched even though Meta's update time did not change, and the DM is settled")
    check(database.get_outreach_message(lost)["status"] == "failed",
          "an unknown DM still missing an hour later is marked not sent")
    check(not database.find_outreach_message("instagram_dm", "m4") or
          database.find_outreach_message("instagram_dm", "m4")["id"] == unknown, "the settled DM is not stored twice")

    # A missing messaging permission rests the source instead of asking every pass.
    http.routes["/me/conversations"] = g.meta_error(*g.PERMISSION_ERROR)
    later = NOON + timedelta(hours=1)
    set_now(later)
    outreach_worker.dm_pass(database.get_outreach_profile(), later, {"errors": []}, force=True)
    sources = database.get_outreach_profile()["runtime_state"]["sources"]
    check(sources[f"instagram_dm:{ids['instagram']}"]["state"] == "needs_permission"
          and sources[f"facebook_dm:{ids['facebook']}"]["state"] == "ok", f"source states: {sources}")
    asked = len(http.gets_to("/me/conversations"))
    outreach_worker.dm_pass(database.get_outreach_profile(), later + minutes(30), {"errors": []})
    check(len(http.gets_to("/me/conversations")) == asked, "a refused source rests")
    html = client.get("/inbox").get_data(as_text=True)
    check("Needs permission" in html and 'href="/instagram/auth?engage=dm"' in html,
          "the Channels dialog says so, with the reconnect link")

    check(http.posts == [], "the worker made no write request to Meta")
    check(_sends_in(os.path.join(REPO, "outreach_worker.py")) == [],
          f"the worker has no path to the sender or a send call: {_sends_in(os.path.join(REPO, 'outreach_worker.py'))}")
    status = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "check_outreach.py"), "worker"],
                            capture_output=True, text=True)
    check(status.returncode == 0 and "OUTREACH_WORKER_OK" in status.stdout,
          f"the Outreach worker gate still passes: {status.stdout[-300:]} {status.stderr[-300:]}")
    print("dm_read: Instagram and Facebook DMs become rows tied to a lead made once per person; unchanged threads are "
          "not refetched; the owner's own messages settle or join the thread; a lost DM fails after an hour; a missing "
          "permission rests the source; the worker makes no write request and cannot reach the sender")
    print(TOKENS["dm_read"])


# ---------------------------------------------------------------------------
# dm_send: the Send API calls and the 24-hour window
# ---------------------------------------------------------------------------

def section_dm_send():
    import requests
    import outreach_worker
    database, web, client, set_now = g.rig()
    ids = g.connect_social(database)
    g.install_twilio(g.FakeTwilioHTTP(refuse_posts=True))
    http = g.FakeSocialHTTP({})
    g.install_social(http)

    def person(channel, external, author, author_id, name, thread, when, account):
        item = {"id": external, "text": "hello", "author": author, "author_id": author_id,
                "author_name": name, "created": g.utc(when), "root_id": thread}
        outreach_worker.ingest_dm(item, channel, {"id": account}, {})
        kind = "instagram" if channel == "instagram_dm" else "facebook"
        return database.find_lead_by_identity(kind, author.lower() if kind == "instagram" else author_id)

    sam = person("instagram_dm", "m1", "sam.sec", "901", "", "IGC1", NOON - minutes(30), ids["instagram"])
    jane = person("facebook_dm", "f1", "", "555", "Jane Doe", "FBC1", NOON - minutes(30), ids["facebook"])
    listing = conversations(client)
    sam_conv, jane_conv = by_name(listing, "@sam.sec")["id"], by_name(listing, "Jane Doe")["id"]

    http.post_script = [("/me/messages", g.FakeResponse(200, {"recipient_id": "901", "message_id": "mid.1"}))]
    status, data = reply(client, sam_conv, "instagram_dm", "Hi Sam, yes \u2014 it is open")
    check(status == 200 and data["status"] == "sent", f"an Instagram DM goes out: {status} {data}")
    sent = http.posts[-1]
    check("graph.instagram.com" in sent["url"] and sent["json"] == {"recipient": {"id": "901"},
          "message": {"text": "Hi Sam, yes, it is open"}}
          and sent["headers"]["Authorization"] == "Bearer token-instagram-ig-1",
          f"Instagram: POST /me/messages to the sender's id, as the owner, in plain punctuation: {sent}")
    row = out_rows(database, sam, "instagram_dm")[-1]
    check(row["status"] == "sent" and row["external_id"] == "mid.1", f"recorded with Meta's message id: {row}")

    http.post_script = [("/PAGE1/messages", g.FakeResponse(200, {"recipient_id": "555", "message_id": "mid.2"}))]
    status, data = reply(client, jane_conv, "facebook_dm", "Yes, fully remote.")
    sent = http.posts[-1]
    check(status == 200 and sent["json"]["messaging_type"] == "RESPONSE" and sent["json"]["recipient"] == {"id": "555"}
          and sent["params"]["access_token"] == "token-facebook-PAGE1",
          f"Facebook: POST /<page>/messages as a RESPONSE with the page token: {status} {sent}")

    def refused(conversation, channel, body, code, words, seen=None):
        before = len(http.posts)
        status, data = reply(client, conversation, channel, body, seen)
        check(status == code and words.lower() in (data.get("error") or "").lower(),
              f"expected {code} '{words}', got {status}: {data}")
        check(len(http.posts) == before, f"refused ({words}) with no request to Meta")
        check(not [m for m in database.list_outreach_queue() if m["channel"] in ("instagram_dm", "facebook_dm")],
              "and no DM is left waiting in the Outreach queue")

    refused(sam_conv, "instagram_dm", "x" * 1001, 400, "at most 1000")
    set_now(NOON + timedelta(hours=25))
    options = {o["channel"]: o for o in detail(client, sam_conv)["reply_options"]}
    check(not options["instagram_dm"]["enabled"] and "24 hours" in options["instagram_dm"]["reason"],
          f"past the window the channel is offered disabled, with the reason: {options}")
    refused(sam_conv, "instagram_dm", "Still there?", 403, "24 hours")
    set_now(NOON)
    database.update_lead(jane, do_not_contact=1)
    refused(jane_conv, "facebook_dm", "One more", 403, "asked not to be contacted")
    database.update_lead(jane, do_not_contact=0)
    import sqlite3
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute("UPDATE social_accounts SET status = 'removed' WHERE id = ?", (ids["facebook"],))
        conn.commit()
    refused(jane_conv, "facebook_dm", "Hello again", 409, "no longer connected")

    # The guard itself holds a DM draft written before their newest message.
    draft = database.add_outreach_message(
        "instagram_dm", "out", "draft", "Old answer", lead_id=sam, kind="continuation",
        context={"account_id": ids["instagram"], "root_id": "IGC1", "recipient_id": "901"},
        based_on_message_id=1)
    outreach_worker.ingest_dm({"id": "m2", "text": "wait", "author": "sam.sec", "author_id": "901",
                               "created": g.utc(NOON - minutes(5)), "root_id": "IGC1"},
                              "instagram_dm", {"id": ids["instagram"]}, {})
    before = len(http.posts)
    status, data = g.send(client, draft, "Old answer")
    check(status == 409 and "wrote again" in data["error"] and len(http.posts) == before,
          f"a DM draft older than their newest message is held: {status} {data}")
    database.update_outreach_message(draft, status="skipped")

    # A lost answer is unknown and never retried.
    http.post_script = [("/me/messages", requests.exceptions.ReadTimeout("lost"))]
    status, data = reply(client, sam_conv, "instagram_dm", "Here is the link")
    check(status == 502 and data["status"] == "unknown", f"a lost answer is unknown: {status} {data}")
    before = len(http.posts)
    status, data = reply(client, sam_conv, "instagram_dm", "Here is the link")
    check(status == 409 and "may already have reached" in data["error"] and len(http.posts) == before,
          f"and the same words are not sent again: {status} {data}")
    check(not [m for m in database.list_outreach_queue() if m["channel"] == "instagram_dm"],
          "an unknown DM is the Inbox's to show, not the Outreach queue's")

    # Meta's own window refusal is explained, not shown raw.
    http.post_script = [("/me/messages", g.FakeResponse(400, {"error": {
        "message": "This message is sent outside of allowed window.", "code": 10, "error_subcode": 2534022}}))]
    status, data = reply(client, sam_conv, "instagram_dm", "Another try")
    check(status == 502 and data["status"] == "failed" and "24 hours" in data["error"],
          f"Meta's window refusal is said plainly: {status} {data}")
    check(http.posts and all("/messages" in p["url"] for p in http.posts), "every write went to a Send API")
    print("dm_send: Instagram and Facebook DMs match each Send API; refused past the 24-hour window, after an opt-out, "
          "past the length limit, on a removed account and when they wrote since; a lost answer is unknown and never "
          "resent; Meta's window refusal is explained")
    print(TOKENS["dm_send"])


# ---------------------------------------------------------------------------
# dm_auth: the reconnect asks for messaging; plain Connect is unchanged
# ---------------------------------------------------------------------------

def section_dm_auth():
    from urllib.parse import parse_qs, urlsplit
    for key in ("THREADS_APP_ID", "THREADS_APP_SECRET", "INSTAGRAM_APP_ID", "INSTAGRAM_APP_SECRET",
                "FACEBOOK_APP_ID", "FACEBOOK_APP_SECRET"):
        os.environ[key] = "gate"
    database, web, client, set_now = g.rig()
    import instagram_client
    import facebook_client
    import outreach_social

    def scopes(url):
        response = client.get(url)
        check(response.status_code == 302, f"{url} redirects to the platform: {response.status_code}")
        return parse_qs(urlsplit(response.headers["Location"]).query)["scope"][0].split(",")

    for platform, configured in (("instagram", instagram_client.INSTAGRAM_SCOPES),
                                 ("facebook", facebook_client.FACEBOOK_SCOPES)):
        base = configured.split(",")
        check(scopes(f"/{platform}/auth") == base, f"{platform}: plain Connect is unchanged")
        engage = scopes(f"/{platform}/auth?engage=1")
        check(engage == base + [s for s in outreach_social.ENGAGE_SCOPES[platform] if s not in base],
              f"{platform}: engagement permissions are unchanged: {engage}")
        check(not set(outreach_social.DM_SCOPES[platform]) & set(engage) - set(base),
              f"{platform}: engagement alone does not ask for messaging")
        dm = scopes(f"/{platform}/auth?engage=dm")
        check(dm[:len(base)] == base and all(s in dm for s in outreach_social.ENGAGE_SCOPES[platform])
              and all(s in dm for s in outreach_social.DM_SCOPES[platform]),
              f"{platform}: DM permissions add messaging to the engagement scopes: {dm}")
        check(platform in database.get_outreach_profile()["runtime_state"]["engage_requested"],
              f"{platform}: the reconnect is remembered, so a resting source is asked again")
    check(set(outreach_social.DM_SCOPES) == {"instagram", "facebook"}
          and outreach_social.DM_SCOPES["instagram"] == ["instagram_business_manage_messages"]
          and outreach_social.DM_SCOPES["facebook"] == ["pages_messaging"], "the messaging scopes are Meta's names")
    status = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "check_outreach.py"), "engage_auth"],
                            capture_output=True, text=True)
    check(status.returncode == 0 and "OUTREACH_ENGAGE_AUTH_OK" in status.stdout,
          f"the Outreach engagement gate still passes: {status.stdout[-300:]}")
    print("dm_auth: Reconnect with DM permissions asks for the configured, engagement and messaging scopes; plain "
          "Connect and Reconnect with engagement permissions are unchanged")
    print(TOKENS["dm_auth"])


# ---------------------------------------------------------------------------
# canned: canned replies
# ---------------------------------------------------------------------------

def section_canned():
    database, web, client, set_now = g.rig()
    status, data = post(client, "/inbox/canned", {"short_code": "/pricing", "content": "It is $300 for twelve weeks."})
    check(status == 200 and data["canned"]["short_code"] == "pricing", f"a canned reply is added, slash dropped: {data}")
    pricing = data["canned"]["id"]
    check(post(client, "/inbox/canned", {"short_code": "PRICING", "content": "dup"})[0] == 409,
          "short codes are unique whatever the case")
    for payload in ({"short_code": "has space", "content": "x"}, {"short_code": "", "content": "x"},
                    {"short_code": "ok", "content": "   "}, {"short_code": "a" * 31, "content": "x"},
                    {"short_code": "long", "content": "x" * 2001}):
        check(post(client, "/inbox/canned", payload)[0] == 400, f"refused: {payload['short_code'][:10]!r}")
    post(client, "/inbox/canned", {"short_code": "apply", "content": "Apply here."})
    post(client, "/inbox/canned", {"short_code": "Hours", "content": "We answer 9 to 5."})
    listed = client.get("/inbox/canned").get_json()["canned"]
    check([c["short_code"] for c in listed] == ["apply", "Hours", "pricing"], f"sorted for the picker: {listed}")
    status, data = post(client, f"/inbox/canned/{pricing}", {"short_code": "price", "content": "Now $250."})
    check(status == 200 and data["canned"]["id"] == pricing and data["canned"]["short_code"] == "price"
          and data["canned"]["content"] == "Now $250.", f"edited in place: {data}")
    listed = client.get("/inbox/canned").get_json()["canned"]
    check([c["short_code"] for c in listed] == ["apply", "Hours", "price"], f"an edit adds no new reply: {listed}")
    check(post(client, f"/inbox/canned/{pricing}", {"short_code": "apply", "content": "x"})[0] == 409,
          "an edit cannot take another reply's short code")
    check(post(client, f"/inbox/canned/{pricing}/delete")[0] == 200, "deleted")
    check(post(client, f"/inbox/canned/{pricing}/delete")[0] == 404, "a second delete is 404")
    check(post(client, "/inbox/canned/9999", {"short_code": "zz", "content": "x"})[0] == 404, "editing a missing one is 404")
    check(client.post("/inbox/canned", json={"short_code": "x", "content": "y"}).status_code == 400,
          "a write without the page's XHR header is refused")
    html = client.get("/inbox").get_data(as_text=True)
    check('"short_code": "apply"' in html, "the page starts with the canned replies for the picker")
    print("canned: canned replies are added, edited and deleted over XHR; short codes are unique, case-insensitive "
          "and validated; empty or overlong replies are refused; the picker gets them sorted")
    print(TOKENS["canned"])


# ---------------------------------------------------------------------------
# ui: the page, its wiring and its script
# ---------------------------------------------------------------------------

def section_ui():
    database, web, client, set_now = g.rig()
    response = client.get("/inbox")
    check(response.status_code == 200, f"/inbox renders: {response.status_code}")
    html = response.get_data(as_text=True)
    check('href="/inbox"' in html and ">Inbox<" in html, "the nav has an Inbox entry")
    for element in ("conversation-list", "inbox-search", "filter-assignee", "filter-channel", "filter-label",
                    "filter-needs-reply", "thread", "thread-name", "resolve-btn", "snooze-btn", "snooze-custom",
                    "assign-select", "labels-btn", "label-new", "unread-btn", "mode-reply", "mode-note",
                    "reply-channel", "composer-text", "composer-send", "composer-status", "canned-picker",
                    "canned-modal", "canned-form", "canned-list", "channels-modal", "check-now-btn", "back-btn"):
        check(f'id="{element}"' in html, f"the page has #{element}")
    for status in ("open", "snoozed", "resolved", "all"):
        check(f'data-status="{status}"' in html, f"a {status} tab")
    for preset in ("1h", "tomorrow", "next_week"):
        check(f'data-snooze="{preset}"' in html, f"a {preset} snooze")
    template = open(os.path.join(REPO, "templates", "inbox.html")).read()
    rules = {str(r) for r in web.app.url_map.iter_rules()}
    for fragment, rule in (("'/inbox/conversations?'", "/inbox/conversations"),
                           ("'/inbox/conversations/' + id", "/inbox/conversations/<int:conversation_id>"),
                           ("'/reply'", "/inbox/conversations/<int:conversation_id>/reply"),
                           ("'/notes'", "/inbox/conversations/<int:conversation_id>/notes"),
                           ("'/read'", "/inbox/conversations/<int:conversation_id>/read"),
                           ("'/inbox/notes/'", "/inbox/notes/<int:note_id>/delete"),
                           ("'/labels'", "/inbox/conversations/<int:conversation_id>/labels"),
                           ("change('status'", "/inbox/conversations/<int:conversation_id>/status"),
                           ("change('assign'", "/inbox/conversations/<int:conversation_id>/assign"),
                           ("'/inbox/canned'", "/inbox/canned"),
                           ("'/inbox/canned/' + id", "/inbox/canned/<int:canned_id>"),
                           ("'/delete'", "/inbox/canned/<int:canned_id>/delete"),
                           ("'/inbox/check'", "/inbox/check")):
        check(fragment in template, f"the script calls {fragment}")
        check(rule in rules, f"and {rule} exists")

    scripts = re.findall(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)[^>]*>(.*?)</script>", html, re.S)
    workdir = tempfile.mkdtemp(prefix="inbox_js_")
    for index, script in enumerate(scripts):
        path = os.path.join(workdir, f"script{index}.js")
        with open(path, "w") as handle:
            handle.write(script)
        result = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        check(result.returncode == 0, f"inline script {index} does not parse: {result.stderr[:400]}")
    check(any("submitComposer" in s for s in scripts), "the Inbox script is among them")

    check("alert(" not in template and "confirm(" not in template, "the Inbox adds no alert( or confirm( call")
    check(not re.search(r"#[0-9a-fA-F]{3,8}\b", re.sub(r'(href|id|data-bs-target)="#[^"]*"', "", template)),
          "the Inbox hard-codes no colors (it uses base.html tokens)")
    check(template.count("btn-primary") == 3 and 'id="resolve-btn"' in template,
          "one primary action per section: Resolve, Send, and Add canned reply")
    check("bootstrap.bundle" not in template, "the page does not load Bootstrap a second time")
    print(f"ui: /inbox renders with its nav entry and every control; the script's {13} endpoints exist; "
          f"{len(scripts)} inline scripts parse; no alert(, no hard-coded colors, one primary action per section")
    print(TOKENS["ui"])


# ---------------------------------------------------------------------------
# docs: registered, and written down
# ---------------------------------------------------------------------------

def section_docs():
    import run_all_gates
    registered = {(entry[0], entry[1], tuple(entry[3]) if len(entry) > 3 else ())
                  for entry in run_all_gates.ALL_GATES}
    for name, token in TOKENS.items():
        check(("check_inbox.py", token, (name,)) in registered,
              f"section {name} ({token}) is not registered in run_all_gates.py")
    readme = open(os.path.join(REPO, "README.md")).read()
    env = open(os.path.join(REPO, ".env.example")).read()
    for words in ("## Inbox", "Reconnect with DM permissions", "instagram_business_manage_messages",
                  "pages_messaging", "24 hours", "Canned replies", "Private note", "Snooze", "/inbox"):
        check(words in readme, f"the README's Inbox section should mention {words!r}")
    check("OUTREACH_DM_MINUTES" in env, ".env.example should document OUTREACH_DM_MINUTES")
    print("docs: every Inbox section is registered; the README and .env.example describe the Inbox and DM permissions")
    print(TOKENS["docs"])


def section_dashes():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout
    base = git("merge-base", "HEAD", "origin/main").strip() or "HEAD"
    added = [line[1:] for line in git("diff", base, "--unified=0", "--no-color").splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    new_files = [f for f in git("ls-files", "--others", "--exclude-standard").splitlines()]
    new_files += ["inbox.py", "templates/inbox.html", "scripts/check_inbox.py"]
    for name in sorted(set(new_files)):
        path = os.path.join(REPO, name)
        if os.path.isfile(path):
            added += open(path, encoding="utf-8", errors="replace").read().splitlines()
    dash = "\u2014"
    found = [line.strip()[:80] for line in added if dash in line]
    control = ["a line with \u2014 in it"]
    check(any(dash in line for line in control), "the dash check finds a known dash (positive control)")
    check(len(added) > 100, f"the check read the change ({len(added)} lines)")
    check(not found, f"em dashes in the change: {found[:5]}")
    print(f"dashes: {len(added)} added or new lines carry no em dash")
    print(TOKENS["dashes"])


SECTIONS = {
    "schema": section_schema, "state": section_state, "reply": section_reply, "dm_read": section_dm_read,
    "dm_send": section_dm_send, "dm_auth": section_dm_auth, "canned": section_canned, "ui": section_ui,
    "docs": section_docs, "dashes": section_dashes,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SECTIONS:
        print(f"usage: check_inbox.py {{{'|'.join(SECTIONS)}}}")
        sys.exit(2)
    SECTIONS[name]()

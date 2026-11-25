"""The Inbox: every person's messages as one conversation, whatever the channel.

Texts, logged calls, comments on the owner's posts and Instagram and Facebook
DMs all live in ``outreach_messages``. The Inbox groups them by person (by lead
once the person is one, by their platform identity before that) and adds what
a shared inbox needs on top: a status (open, snoozed, resolved), an assignee,
labels, private notes, a read mark and canned replies.

Status follows the people, not only the owner: a new message reopens a
resolved or snoozed conversation, and a snooze ends on its own at its time.
That is worked out when the list is read (``effective_status``), so nothing
that stores messages has to know the Inbox exists.

Replies leave through ``outreach_sender.send`` and nowhere else. ``prepare_reply``
only writes the draft the sender is handed (or reuses the agent's open draft on
that channel), so every rule the Outreach queue enforces holds here too.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import database
import outreach_social
from outreach_rules import fmt_utc, hours_between, parse_utc, plain_text, pretty_phone

CHANNEL_NAMES = {
    "sms": "Text", "call": "Call", "threads": "Threads comment",
    "instagram": "Instagram comment", "facebook": "Facebook comment", "twitter": "X reply",
    "instagram_dm": "Instagram DM", "facebook_dm": "Messenger",
}
# The short name the channel picker and filters use.
CHANNEL_SHORT = {
    "sms": "Text", "call": "Call", "threads": "Threads", "instagram": "IG comment",
    "facebook": "FB comment", "twitter": "X", "instagram_dm": "IG DM", "facebook_dm": "Messenger",
}
COMMENT_CHANNELS = outreach_social.SOCIAL_CHANNELS
DM_CHANNELS = outreach_social.DM_CHANNELS
REPLY_CHANNELS = ("sms",) + DM_CHANNELS + COMMENT_CHANNELS

STATUS_FILTERS = ("open", "snoozed", "resolved", "all")
ASSIGNEE_FILTERS = ("all", "mine", "unassigned")
SNOOZE_PRESETS = ("1h", "tomorrow", "next_week")
LABEL_RE = re.compile(r"^[\w][\w \-]{0,29}$")
SHORT_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,30}$")
MAX_LABELS = 10
MAX_REPLY = 4000
MAX_NOTE = 4000
MAX_CANNED = 2000
LIST_LIMIT = 200


class InboxError(Exception):
    """A request the Inbox refuses, with the HTTP status and a reason to show."""

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra


# ── Conversations ───────────────────────────────────────────────────────────

def effective_status(state: dict | None, group: dict, now) -> str:
    """The status the owner sees: stored, unless the person wrote since or a snooze ran out."""
    if not state:
        return "open"
    status = state.get("status") or "open"
    last_in = int(group.get("last_in_id") or 0)
    mark = int(state.get("status_mark") or 0)
    if status == "snoozed":
        until = parse_utc(state.get("snoozed_until"))
        if last_in > mark or (until is not None and until <= now):
            return "open"
    if status == "resolved" and last_in > mark:
        return "open"
    return status


def load_groups(now) -> list:
    """Every conversation, each with stored state (created where missing).

    States left behind by a person who has since become a lead move onto the
    lead's conversation first, so labels, notes and assignment follow them.
    """
    groups = database.inbox_groups()
    if database.adopt_orphan_inbox_states([g["key"] for g in groups]):
        groups = database.inbox_groups()
    missing = [g["key"] for g in groups if g["state"] is None]
    for key in missing:
        database.ensure_inbox_state(key)
    if missing:
        groups = database.inbox_groups()
    for group in groups:
        group["status"] = effective_status(group["state"], group, now)
        group["id"] = group["state"]["id"]
    return groups


def _display_name(lead: dict | None, message: dict | None, key: str) -> str:
    if lead and lead.get("name"):
        return lead["name"]
    context = (message or {}).get("context") or {}
    if context.get("author_name"):
        return context["author_name"]
    if context.get("author"):
        return "@" + context["author"].lstrip("@")
    if message and message.get("peer"):
        return pretty_phone(message["peer"]) if message["channel"] == "sms" else message["peer"]
    if lead and lead.get("phone"):
        return pretty_phone(lead["phone"])
    if lead and lead.get("email"):
        return lead["email"]
    return key.split(":", 1)[-1] or "Unknown"


def list_conversations(user_id: int | None, now, status: str = "open", assignee: str = "all",
                       channel: str | None = None, label: str | None = None,
                       needs_reply: bool = False, search: str | None = None) -> dict:
    """The conversation list for the page, with counts for the status tabs."""
    groups = load_groups(now)
    leads = {}
    for group in groups:
        if group.get("lead_id") and group["lead_id"] not in leads:
            leads[group["lead_id"]] = database.get_lead(group["lead_id"])
    last = database.outreach_messages_by_ids([g["last_id"] for g in groups])
    search = (search or "").strip()
    matching = database.inbox_keys_matching(search) if search else None

    def keep(group, by_status=True):
        state = group["state"]
        if by_status and status != "all" and group["status"] != status:
            return False
        if assignee == "mine" and state.get("assignee_id") != user_id:
            return False
        if assignee == "unassigned" and state.get("assignee_id"):
            return False
        if channel and channel not in group["channels"]:
            return False
        if label and label not in (state.get("labels") or []):
            return False
        if needs_reply and int(group.get("last_in_id") or 0) <= int(group.get("last_out_id") or 0):
            return False
        if search:
            lead = leads.get(group.get("lead_id")) or {}
            haystack = " ".join(str(lead.get(k) or "") for k in ("name", "email", "phone")).lower()
            if group["key"] not in matching and search.lower() not in haystack \
                    and search.lower() not in group["key"].lower():
                return False
        return True

    counts = {s: 0 for s in STATUS_FILTERS}
    items = []
    for group in groups:
        if not keep(group, by_status=False):
            continue
        counts[group["status"]] = counts.get(group["status"], 0) + 1
        counts["all"] += 1
        if not keep(group):
            continue
        if len(items) >= LIST_LIMIT:
            continue
        lead = leads.get(group.get("lead_id"))
        message = last.get(group["last_id"]) or {}
        state = group["state"]
        items.append({
            "id": group["id"], "key": group["key"], "name": _display_name(lead, message, group["key"]),
            "lead_id": group.get("lead_id"), "status": group["status"],
            "channels": group["channels"], "unread": int(group.get("unread") or 0),
            "last_at": group.get("last_at"),
            "preview": (message.get("body") or "")[:140],
            "preview_from_us": message.get("direction") == "out",
            "preview_channel": message.get("channel"),
            "needs_reply": int(group.get("last_in_id") or 0) > int(group.get("last_out_id") or 0),
            "assignee_id": state.get("assignee_id"), "labels": state.get("labels") or [],
            "snoozed_until": state.get("snoozed_until") if group["status"] == "snoozed" else None,
        })
    return {"conversations": items, "counts": counts,
            "unread_total": sum(int(g.get("unread") or 0) for g in groups if g["status"] == "open")}


def get_group(conversation_id: int, now) -> dict:
    """One conversation's group row and state, or InboxError 404."""
    state = database.get_inbox_state(conversation_id=conversation_id)
    if not state:
        raise InboxError(404, "That conversation no longer exists")
    group = next((g for g in database.inbox_groups() if g["key"] == state["key"]), None)
    if group is None:
        # Its messages moved (the person became a lead, or leads merged).
        moved = database.adopt_orphan_inbox_states([g["key"] for g in database.inbox_groups()])
        raise InboxError(404 if not moved else 410,
                         "This conversation moved; reload the list to see it under the person's name")
    group["state"] = state
    group["id"] = state["id"]
    group["status"] = effective_status(state, group, now)
    return group


# ── One conversation ────────────────────────────────────────────────────────

def _latest(messages, channel, direction="in"):
    for message in reversed(messages):
        if message["channel"] == channel and message["direction"] == direction \
                and (direction != "in" or message["status"] == "received"):
            return message
    return None


def reply_options(lead: dict | None, messages: list, now, twilio_ready: bool) -> list:
    """Each channel this person can be answered on, enabled or with the reason it is not.

    Only the hard limits are decided here (no channel, a closed DM window, an
    opt-out). Consent, hours, caps and links are checked by the sender at Send,
    whatever the page shows.
    """
    present = {m["channel"] for m in messages}
    options = []
    blocked = None
    if lead and (lead.get("opted_out_at") or lead.get("do_not_contact")):
        blocked = f"{lead.get('name') or 'This person'} asked not to be contacted."
    for channel in REPLY_CHANNELS:
        if channel == "sms":
            if not (lead and lead.get("phone")) and "sms" not in present:
                continue
            reason = blocked
            if not reason and not twilio_ready:
                reason = "Set up Twilio in Outreach Setup to text from here."
            options.append({"channel": channel, "name": CHANNEL_NAMES[channel],
                            "short": CHANNEL_SHORT[channel], "enabled": not reason,
                            "reason": reason or "", "limit": 1600})
            continue
        latest = _latest(messages, channel)
        if not latest:
            continue
        reason = blocked
        if not reason and channel in DM_CHANNELS:
            at = parse_utc(latest.get("sent_at") or latest.get("created_at"))
            if not at or hours_between(now, at) > outreach_social.DM_WINDOW_HOURS:
                reason = (f"{outreach_social.DM_NAMES[channel]} only allows a reply within "
                          f"{outreach_social.DM_WINDOW_HOURS} hours of their last message.")
        limit = (outreach_social.DM_LIMITS.get(channel)
                 or outreach_social.REPLY_LIMITS.get(channel) or 2000)
        options.append({"channel": channel, "name": CHANNEL_NAMES[channel],
                        "short": CHANNEL_SHORT[channel], "enabled": not reason,
                        "reason": reason or "", "limit": limit,
                        "answers": (latest.get("body") or "")[:120]})
    return options


def conversation_detail(conversation_id: int, now, twilio_ready: bool) -> dict:
    group = get_group(conversation_id, now)
    state = group["state"]
    messages = database.inbox_messages_for_key(group["key"])
    lead = database.get_lead(group["lead_id"]) if group.get("lead_id") else None
    timeline = []
    for m in messages:
        context = m.get("context") or {}
        timeline.append({
            "type": "message", "id": m["id"], "channel": m["channel"],
            "channel_name": CHANNEL_NAMES.get(m["channel"], m["channel"]),
            "direction": m["direction"], "status": m["status"], "body": m.get("body") or "",
            "at": m.get("sent_at") or m.get("created_at"), "error": m.get("error") or "",
            "permalink": context.get("permalink"), "note": context.get("note") or "",
            "post_text": (context.get("post_text") or "")[:160],
            "outcome": context.get("outcome") or "",
        })
    for note in database.list_inbox_notes(state["id"]):
        timeline.append({"type": "note", "id": note["id"], "body": note["body"],
                         "at": note["created_at"], "author": note.get("username") or ""})
    timeline.sort(key=lambda t: (t["at"] or "", 0 if t["type"] == "message" else 1, t["id"]))
    options = reply_options(lead, messages, now, twilio_ready)
    drafts = {d["channel"]: {"id": d["id"], "body": d.get("body") or "", "status": d["status"],
                             "error": d.get("error") or ""}
              for d in database.inbox_open_drafts_for_key(group["key"])}
    last_in = next((m for m in reversed(messages) if m["direction"] == "in"), None)
    enabled = [o["channel"] for o in options if o["enabled"]]
    default = last_in["channel"] if last_in and last_in["channel"] in enabled else \
        (enabled[0] if enabled else None)
    identities = database.list_lead_identities(lead["id"]) if lead else []
    contact = None
    if lead:
        contact = {k: lead.get(k) for k in ("id", "name", "email", "stage", "source", "notes",
                                            "sms_consent", "opted_out_at", "do_not_contact",
                                            "last_inbound_at", "last_contact_at")}
        contact["phone"] = pretty_phone(lead.get("phone"))
        contact["identities"] = [{"kind": i["kind"], "value": i["value"]} for i in identities
                                 if i["kind"] not in ("phone", "email")]
    return {
        "id": state["id"], "key": group["key"],
        "name": _display_name(lead, last_in or (messages[-1] if messages else None), group["key"]),
        "status": group["status"], "stored_status": state.get("status"),
        "snoozed_until": state.get("snoozed_until"), "assignee_id": state.get("assignee_id"),
        "labels": state.get("labels") or [], "unread": int(group.get("unread") or 0),
        "last_id": group["last_id"], "last_in_id": group.get("last_in_id") or 0,
        "timeline": timeline, "reply_options": options, "default_channel": default,
        "drafts": drafts, "contact": contact, "channels": group["channels"],
    }


# ── Changing a conversation ─────────────────────────────────────────────────

def snooze_until(preset: str | None, custom: str | None, now, zone: str) -> str:
    """The UTC time a snooze ends, from a preset or the owner's local date-time."""
    try:
        tz = ZoneInfo(zone or "America/New_York")
    except Exception:
        tz = ZoneInfo("America/New_York")
    local = now.astimezone(tz)
    if preset == "1h":
        return fmt_utc(now + timedelta(hours=1))
    if preset == "tomorrow":
        moment = datetime.combine(local.date() + timedelta(days=1), time(9, 0), tz)
        return fmt_utc(moment)
    if preset == "next_week":
        days = 7 - local.weekday()  # next Monday
        moment = datetime.combine(local.date() + timedelta(days=days), time(9, 0), tz)
        return fmt_utc(moment)
    if custom:
        try:
            moment = datetime.fromisoformat(custom.strip())
        except ValueError:
            raise InboxError(400, "Pick a date and time to snooze until")
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=tz)
        if moment <= now:
            raise InboxError(400, "Pick a time in the future to snooze until")
        return fmt_utc(moment)
    raise InboxError(400, "Choose how long to snooze")


def set_status(conversation_id: int, status: str, now, *, preset=None, custom=None,
               zone: str = "America/New_York") -> dict:
    if status not in database.INBOX_STATUSES:
        raise InboxError(400, "Unknown status")
    group = get_group(conversation_id, now)
    changes = {"status": status, "status_mark": group["last_id"], "snoozed_until": None}
    if status == "snoozed":
        changes["snoozed_until"] = snooze_until(preset, custom, now, zone)
    database.update_inbox_state(conversation_id, **changes)
    return {"status": status, "snoozed_until": changes["snoozed_until"]}


def set_assignee(conversation_id: int, assignee_id, now) -> dict:
    get_group(conversation_id, now)
    if assignee_id in (None, "", 0, "0"):
        database.update_inbox_state(conversation_id, assignee_id=None)
        return {"assignee_id": None}
    try:
        assignee_id = int(assignee_id)
    except (TypeError, ValueError):
        raise InboxError(400, "Pick someone to assign")
    if not database.get_user_by_id(assignee_id):
        raise InboxError(400, "That person is not a user of this Insights")
    database.update_inbox_state(conversation_id, assignee_id=assignee_id)
    return {"assignee_id": assignee_id}


def clean_labels(labels) -> list:
    if not isinstance(labels, list):
        raise InboxError(400, "Labels must be a list")
    out = []
    for raw in labels:
        label = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not label:
            continue
        if not LABEL_RE.match(label):
            raise InboxError(400, f"\"{label[:40]}\" is not a usable label: use up to 30 letters, "
                                  "digits, spaces, dashes or underscores")
        if label.lower() not in (l.lower() for l in out):
            out.append(label)
    if len(out) > MAX_LABELS:
        raise InboxError(400, f"A conversation can have at most {MAX_LABELS} labels")
    return out


def set_labels(conversation_id: int, labels, now) -> dict:
    get_group(conversation_id, now)
    cleaned = clean_labels(labels)
    database.update_inbox_state(conversation_id, labels=cleaned)
    return {"labels": cleaned}


def mark_read(conversation_id: int, now, unread: bool = False) -> dict:
    group = get_group(conversation_id, now)
    if unread:
        mark = max(0, int(group.get("last_in_id") or 0) - 1)
    else:
        mark = int(group["last_id"])
    database.update_inbox_state(conversation_id, read_mark=mark)
    return {"read_mark": mark}


def add_note(conversation_id: int, user_id: int, body: str, now) -> dict:
    get_group(conversation_id, now)
    body = (body or "").strip()
    if not body:
        raise InboxError(400, "The note is empty")
    if len(body) > MAX_NOTE:
        raise InboxError(400, f"A note can be at most {MAX_NOTE} characters")
    note_id = database.add_inbox_note(conversation_id, user_id, body)
    return {"note_id": note_id}


# ── Replying ────────────────────────────────────────────────────────────────

def _lead_for_conversation(group: dict, messages: list) -> int:
    """The conversation's lead, created from the person's identity if they have none yet."""
    if group.get("lead_id"):
        return group["lead_id"]
    kind, _sep, value = group["key"].partition(":")
    if kind == "message" or not value:
        raise InboxError(409, "Insights cannot tell who this message is from, so it cannot reply")
    lead_id = database.find_lead_by_identity(kind, value)
    if lead_id is None:
        last_in = next((m for m in reversed(messages) if m["direction"] == "in"), None)
        fields = {"name": _display_name(None, last_in, group["key"]),
                  "source": (last_in or {}).get("channel") or kind, "stage": "replied",
                  "last_inbound_at": (last_in or {}).get("sent_at") or (last_in or {}).get("created_at")}
        if kind == "phone":
            fields["phone"] = value
        lead_id = database.create_lead([(kind, value)], **fields)
    database.link_inbox_key_to_lead(group["key"], lead_id)
    return lead_id


def _reply_context(channel: str, messages: list) -> tuple[dict, int | None]:
    """What the sender needs to address a reply on ``channel``, and the message it answers."""
    if channel == "sms":
        latest = _latest(messages, "sms")
        return {"by": "owner", "via": "inbox"}, (latest["id"] if latest else None)
    latest = _latest(messages, channel)
    if not latest:
        raise InboxError(409, f"There is no {CHANNEL_NAMES[channel].lower()} from them to answer")
    context = latest.get("context") or {}
    if channel in DM_CHANNELS:
        return ({"by": "owner", "via": "inbox", "account_id": context.get("account_id"),
                 "root_id": context.get("root_id"), "recipient_id": context.get("author_id")},
                latest["id"])
    return ({"by": "owner", "via": "inbox", "in_reply_to": latest.get("external_id"),
             "account_id": context.get("account_id"), "root_id": context.get("root_id"),
             "permalink": context.get("permalink"), "source": context.get("source")},
            latest["id"])


REPEAT_SECONDS = 120


def _refuse_repeat(lead_id: int, channel: str, body: str, now) -> None:
    """A second press of Send (a double click, a retry after a slow answer) must
    not send the same words twice. A reply whose outcome is unknown counts: it
    may already be with them.

    This early check gives the quick answer; ``repeat_guard`` makes the same
    check again inside the sender's claim, where it cannot race."""
    _refuse_repeat_in(database.list_lead_messages(lead_id, channel=channel), body, now)


def repeat_guard(body: str, now):
    """The repeat check, for ``outreach_sender.send(also=...)``.

    It runs in the same ``BEGIN IMMEDIATE`` transaction that moves the draft to
    sending. Two presses at the same moment each pass ``prepare_reply``'s early
    check, and when the first one's draft is already sent by the time the second
    looks for an open draft, the second writes a draft of its own; only this
    check, made under the claim's lock, sees the first one's claim and refuses.
    Raises ``InboxError``, which rolls the claim back with nothing written.
    """
    def check(conn, message, lead):
        rows = conn.execute(
            "SELECT direction, status, sent_at, body FROM outreach_messages "
            "WHERE lead_id = ? AND channel = ? AND id != ?",
            (message["lead_id"], message["channel"], message["id"])).fetchall()
        _refuse_repeat_in([dict(r) for r in rows], body, now)
        return None
    return check


def _refuse_repeat_in(messages, body: str, now) -> None:
    words = plain_text(body)
    recent = fmt_utc(now - timedelta(seconds=REPEAT_SECONDS))
    today = fmt_utc(now - timedelta(hours=24))
    for message in messages:
        if message["direction"] != "out" or message["status"] not in ("sending", "sent", "delivered", "unknown"):
            continue
        since = today if message["status"] == "unknown" else recent
        if (message.get("sent_at") or "") < since:
            continue
        if (message.get("body") or "").startswith(words):
            if message["status"] == "unknown":
                raise InboxError(409, "This reply may already have reached them (Insights could not "
                                      "confirm it). It is not sent again; check the conversation first.")
            raise InboxError(409, "That reply was just sent.", reload=True)


def prepare_reply(conversation_id: int, channel: str, body: str, seen_id, now) -> dict:
    """Write the draft Send will be handed, after the checks only the Inbox can make.

    Returns ``{message_id, created}``. ``created`` says the row is new (so a
    refused send can remove it); otherwise it is the agent's open draft on that
    channel, which the owner's words replace when it is sent.
    """
    if channel not in REPLY_CHANNELS:
        raise InboxError(400, "Pick a channel to reply on")
    body = (body or "").strip()
    if not body:
        raise InboxError(400, "Write a reply first")
    if len(body) > MAX_REPLY:
        raise InboxError(400, f"A reply can be at most {MAX_REPLY} characters")
    group = get_group(conversation_id, now)
    messages = database.inbox_messages_for_key(group["key"])
    try:
        seen = int(seen_id or 0)
    except (TypeError, ValueError):
        seen = 0
    newest_in = max((m["id"] for m in messages if m["direction"] == "in"), default=0)
    if newest_in > seen:
        raise InboxError(409, "They wrote again while you were typing. Read it first; your reply "
                              "is still in the box.", reload=True)
    lead_id = _lead_for_conversation(group, messages)
    lead = database.get_lead(lead_id)
    if channel != "sms" and not _latest(messages, channel):
        raise InboxError(409, f"They have not written to you as a {CHANNEL_NAMES[channel].lower()}")
    if channel == "sms" and not lead.get("phone"):
        raise InboxError(409, f"{lead.get('name') or 'This person'} has no phone number on file")
    _refuse_repeat(lead_id, channel, body, now)
    existing = database.open_outreach_item(lead_id, channel)
    if existing:
        if existing["status"] == "sending":
            raise InboxError(409, "A message to them on this channel is being sent right now")
        return {"message_id": existing["id"], "created": False}
    context, based_on = _reply_context(channel, messages)
    message_id = database.add_outreach_message(
        channel, "out", "draft", body, lead_id=lead_id, kind="continuation", context=context,
        based_on_message_id=based_on)
    if message_id is None:
        existing = database.open_outreach_item(lead_id, channel)
        if not existing or existing["status"] == "sending":
            raise InboxError(409, "A message to them on this channel is being sent right now")
        return {"message_id": existing["id"], "created": False}
    return {"message_id": message_id, "created": True}


# ── Canned replies ──────────────────────────────────────────────────────────

def save_canned(short_code: str, content: str, canned_id: int | None = None) -> dict:
    short_code = (short_code or "").strip().lstrip("/")
    content = (content or "").strip()
    if not SHORT_CODE_RE.match(short_code):
        raise InboxError(400, "Use 1 to 30 letters, digits, dashes or underscores for the short code")
    if not content:
        raise InboxError(400, "Write the reply this short code inserts")
    if len(content) > MAX_CANNED:
        raise InboxError(400, f"A canned reply can be at most {MAX_CANNED} characters")
    if canned_id and not database.get_canned_response(canned_id):
        raise InboxError(404, "That canned reply no longer exists")
    saved = database.save_canned_response(short_code, content, canned_id)
    if saved is None:
        raise InboxError(409, f"/{short_code} is already used by another canned reply")
    return database.get_canned_response(saved)

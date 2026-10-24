"""Background outreach: read replies, settle sends, write drafts. Never sends.

One pass (``run_once``) does, in order:

1. reads the texts that arrived at the owner's Twilio number, and the texts
   that left it from other tools (Kartra, say), so every conversation is
   complete; a STOP opts the person out on the spot;
2. settles sends: a send that never finished becomes ``unknown``, an unknown
   one is looked up at Twilio (found means sent, missing after ten minutes means
   not sent), and recent sends pick up their delivery result;
3. reads the replies, comments and mentions on the owner's recent social posts
   (every ``OUTREACH_SOCIAL_MINUTES``) and runs Threads keyword search (every
   ``OUTREACH_SEARCH_MINUTES``, within a daily budget); the owner's own replies
   seen there settle unconfirmed replies and close drafts they answered by hand;
4. drafts, within the day's caps: public replies to the social messages worth
   answering, replies to people who texted back, nudges to people who went
   quiet, and first texts to new leads who agreed to texts.

Drafts go to the Outreach queue. This module has no way to send them: it never
imports ``outreach_sender``, only ever reads from Twilio and the social
platforms, and calls none of the reply functions, which a gate checks. ``worker_loop`` runs ``run_once`` every ``OUTREACH_TICK_SECONDS``
when ``OUTREACH_ENABLED`` is on; the Setup tab's Run now runs one pass.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import timedelta

import database
import outreach_agent
import outreach_social
import social_publisher
import twilio_client
import usage_meter
from outreach_rules import (
    context_missing, fmt_utc, keyword_of, lead_score, parse_utc, start_of_day_utc, utc_now,
)

logger = logging.getLogger(__name__)

STALE_SENDING_MINUTES = 10
UNKNOWN_GIVE_UP_MINUTES = 10
DELIVERY_WINDOW_HOURS = 48
MAX_STATUS_CHECKS = 50
REVIEW_RETRY_MINUTES = 30
MAX_DRAFT_ATTEMPTS = 2
# Social listening: how recent a post must be to be watched, how many posts per
# account, how long a refused source rests before it is asked again, and how
# long an unconfirmed reply waits to be seen on the post before it counts as not sent.
WATCH_DAYS = 14
WATCH_POSTS = 10
REFUSED_SOURCE_HOURS = 6
SOCIAL_UNKNOWN_GIVE_UP_MINUTES = 60

_run_lock = threading.Lock()


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _flag("OUTREACH_ENABLED")


def x_mentions_enabled() -> bool:
    """X bills every read, so reading mentions is opt-in."""
    return _flag("OUTREACH_X_MENTIONS")


def social_minutes() -> int:
    return max(1, int(os.getenv("OUTREACH_SOCIAL_MINUTES", "30") or 30))


def search_minutes() -> int:
    return max(1, int(os.getenv("OUTREACH_SEARCH_MINUTES", "60") or 60))


def run_once(now=None, client=None) -> dict:
    """One full pass. Returns a summary; a pass already in progress is not doubled."""
    if not _run_lock.acquire(blocking=False):
        return {"skipped": "A pass is already running"}
    try:
        return _run(now or utc_now(), client or twilio_client.get_twilio_client())
    finally:
        _run_lock.release()


def worker_loop() -> None:
    """The daemon thread started by ``start_workers`` when outreach is enabled."""
    tick = int(os.getenv("OUTREACH_TICK_SECONDS", "300") or 300)
    with usage_meter.usage_context("proactive"):
        while True:
            time.sleep(tick)
            try:
                run_once()
            except Exception:
                logger.exception("outreach worker pass failed")


def _run(now, client) -> dict:
    summary = {"at": fmt_utc(now), "inbound": 0, "unmatched": 0, "opted_out": 0,
               "settled": 0, "delivered": 0, "drafts": 0, "needs_review": 0,
               "social_inbound": 0, "search_matches": 0, "skipped_by_agent": 0, "errors": []}
    if client.can_read():
        poll_inbound(client, now, summary)
        reconcile(client, now, summary)
    elif database.list_leads(limit=1):
        summary["errors"].append("Twilio is not set up for reading replies: set "
                                 + ", ".join(client.missing()))
    profile = database.get_outreach_profile()
    social_pass(profile, now, summary)
    search_pass(profile, now, summary)
    draft_pass(database.get_outreach_profile(), now, summary)
    database.update_outreach_profile(
        last_run_at=summary["at"], last_run_summary=summary,
        last_poll_error="; ".join(summary["errors"]) or None)
    return summary


# ── 1. Reading the number's messages ────────────────────────────────────────

def poll_inbound(client, now, summary) -> None:
    since = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    received = client.list_messages(to=client.from_number, sent_on_or_after=since)
    if received["success"]:
        for message in sorted(received["messages"],
                              key=lambda m: (m["date_sent"] or m["date_created"] or "", m["sid"])):
            if message["direction"] == "inbound":
                ingest_inbound(message, now, summary)
    else:
        summary["errors"].append(f"Reading replies: {received['friendly']}")
    sent = client.list_messages(from_=client.from_number, sent_on_or_after=since)
    if sent["success"]:
        for message in sent["messages"]:
            if message["direction"] != "inbound":
                ingest_external(message)
    else:
        summary["errors"].append(f"Reading sent texts: {sent['friendly']}")


def ingest_inbound(message: dict, now, summary: dict) -> int | None:
    """Store one text that arrived. Returns the new row id, or None if already stored."""
    sid, peer = message["sid"], message["from"]
    if database.find_outreach_message("sms", sid):
        return None
    keyword = keyword_of(message["body"])
    when = message["date_sent"] or message["date_created"] or fmt_utc(now)
    lead_id = database.find_lead_by_identity("phone", peer)
    if keyword == "stop" and lead_id is None:
        # Remember the opt-out even for a stranger, so a later import of this
        # number arrives already opted out.
        lead_id = database.create_lead([("phone", peer)], phone=peer, source="sms",
                                       opted_out_at=fmt_utc(now), opt_out_source="keyword")
    row_id = database.add_outreach_message(
        "sms", "in", "received", message["body"], lead_id=lead_id, kind="inbound",
        context={"keyword": keyword} if keyword else {}, external_id=sid, peer=peer, sent_at=when)
    if row_id is None:
        return None
    summary["inbound"] += 1
    if lead_id is None:
        summary["unmatched"] += 1
        return row_id
    lead = database.get_lead(lead_id)
    changes = {"last_inbound_at": when}
    if keyword == "stop":
        database.set_lead_opt_out(lead_id, "keyword")
        summary["opted_out"] += 1
    else:
        changes.update(follow_ups_sent=0, next_follow_up_at=None)
        if lead.get("stage") in (None, "", "new", "contacted"):
            changes["stage"] = "replied"
        if keyword == "start" and lead.get("opted_out_at") \
                and lead.get("opt_out_source") in ("keyword", "carrier"):
            database.clear_lead_opt_out(lead_id)
            lead = database.get_lead(lead_id)
    changes["score"] = lead_score({**lead, **changes}, now)
    database.update_lead(lead_id, **changes)
    return row_id


def ingest_external(message: dict) -> int | None:
    """Store a text another tool sent from the number to a known lead, as context."""
    sid = message["sid"]
    if database.find_outreach_message("sms", sid):
        return None
    lead_id = database.find_lead_by_identity("phone", message["to"])
    if lead_id is None:
        return None
    return database.add_outreach_message(
        "sms", "out", "external", message["body"], lead_id=lead_id, kind="external",
        context={"note": "Sent from your number outside Insights"}, external_id=sid,
        peer=message["to"], sent_at=message["date_sent"] or message["date_created"])


# ── 2. Settling sends ───────────────────────────────────────────────────────

def reconcile(client, now, summary) -> None:
    stale = fmt_utc(now - timedelta(minutes=STALE_SENDING_MINUTES))
    for message in database.outreach_messages_with_status("sending", older_than=stale):
        database.update_outreach_message(
            message["id"], status="unknown",
            error="Insights stopped before Twilio answered. Checking Twilio for this text.")
    for message in database.outreach_messages_with_status("unknown"):
        if settle_unknown(message, client, now).get("settled"):
            summary["settled"] += 1
    checked = 0
    recent = fmt_utc(now - timedelta(hours=DELIVERY_WINDOW_HOURS))
    for message in database.outreach_messages_with_status("sent", newer_than=recent):
        if not message.get("external_id") or checked >= MAX_STATUS_CHECKS:
            continue
        checked += 1
        if check_delivery(message, client) == "delivered":
            summary["delivered"] += 1


def settle_unknown(message: dict, client, now) -> dict:
    """Find out whether a send Twilio never confirmed actually went out.

    Twilio is asked for the latest texts to that person. One with the same words,
    created after the attempt, is this text, so it is marked sent. Nothing after
    ten minutes means it never went out, and it is marked failed (safe to send
    again). Returns ``{"settled": "sent" | "not_sent" | None, "friendly": ...}``.
    """
    lead = database.get_lead(message["lead_id"]) if message.get("lead_id") else None
    peer = message.get("peer") or (lead or {}).get("phone")
    attempted = parse_utc(message.get("sent_at")) or parse_utc(message.get("updated_at")) or now
    if not peer:
        return {"settled": None, "friendly": "No phone number to look up"}
    found = client.list_messages(to=peer, from_=client.from_number, max_pages=1)
    if not found["success"]:
        return {"settled": None, "friendly": found["friendly"]}
    floor = fmt_utc(attempted - timedelta(minutes=2))
    body = (message.get("body") or "").strip()
    for candidate in found["messages"]:
        if (candidate["body"] or "").strip() == body and (candidate["date_created"] or "") >= floor:
            database.record_outreach_sent(message["id"], candidate["sid"], "sent")
            return {"settled": "sent", "friendly": "Twilio has it: the text went out."}
    if now - attempted >= timedelta(minutes=UNKNOWN_GIVE_UP_MINUTES):
        database.update_outreach_message(
            message["id"], status="failed",
            error="Twilio has no record of this text, so it did not go out. Safe to send again.")
        return {"settled": "not_sent", "friendly": "Twilio has no record of it, so it did not go out."}
    return {"settled": None, "friendly": "Not at Twilio yet. Check again in a few minutes."}


def check_delivery(message: dict, client) -> str | None:
    """Pick up a sent text's delivery result. Returns the new status, if it changed."""
    result = client.get_message(message["external_id"])
    if not result["success"]:
        return None
    state = (result["message"] or {}).get("status")
    code = (result["message"] or {}).get("error_code")
    if state == "delivered":
        database.update_outreach_message(message["id"], status="delivered")
        return "delivered"
    if state in ("undelivered", "failed"):
        friendly = twilio_client.friendly_error(code, (result["message"] or {}).get("error_message") or "")
        database.update_outreach_message(message["id"], status="failed", error=friendly)
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        if code in twilio_client.PAUSE_ERRORS:
            database.update_outreach_profile(sms_paused_reason=friendly)
        if code == twilio_client.ERROR_OPTED_OUT and message.get("lead_id"):
            database.set_lead_opt_out(message["lead_id"], "carrier")
        return "failed"
    return None


# ── 2b. Social conversations on the owner's own posts ───────────────────────

def _due(last: str | None, minutes: int, now) -> bool:
    moment = parse_utc(last)
    return moment is None or now - moment >= timedelta(minutes=minutes)


def _source_key(platform: str, account_id) -> str:
    return f"{platform}:{account_id}"


def _set_source(platform, account, state, friendly="", now=None) -> None:
    database.update_outreach_runtime(sources={_source_key(platform, account["id"]): {
        "state": state, "friendly": friendly, "at": fmt_utc(now or utc_now()),
        "handle": account["handle"] or account["display_name"] or account["label"] or ""}})


def _source_resting(runtime: dict, source_name: str, platform: str, account_id, now) -> bool:
    """A source that refused (no permission, dead login) rests until it is reconnected
    or a few hours pass, so a missing permission does not cost a request every pass."""
    source = (runtime.get("sources") or {}).get(_source_key(source_name, account_id)) or {}
    if source.get("state") not in ("needs_permission", "needs_reconnect"):
        return False
    requested = parse_utc((runtime.get("engage_requested") or {}).get(platform))
    checked = parse_utc(source.get("at"))
    if requested and checked and requested > checked:
        return False
    return bool(checked) and now - checked < timedelta(hours=REFUSED_SOURCE_HOURS)


def social_pass(profile: dict, now, summary: dict) -> None:
    """Read replies, comments and mentions on the owner's recent posts. Reads only."""
    runtime = profile.get("runtime_state") or {}
    if not _due(runtime.get("last_social_at"), social_minutes(), now):
        return
    fetched_roots = set()
    for account in database.list_social_accounts():
        platform = account["platform"]
        if platform not in outreach_social.SOCIAL_CHANNELS:
            continue
        if platform == "twitter" and not x_mentions_enabled():
            _set_source(platform, account, "off", "Reading X mentions is off "
                        "(X bills every read). Set OUTREACH_X_MENTIONS=1 to turn it on.", now)
            continue
        if _source_resting(runtime, platform, platform, account["id"], now):
            continue
        client, token, _account, failure = social_publisher.session(platform, account["id"])
        if failure:
            _set_source(platform, account, "needs_reconnect",
                        failure.get("error") or "Reconnect this account", now)
            continue
        try:
            outcome = _read_account(platform, account, token, now, summary, fetched_roots)
        except Exception as exc:  # noqa: BLE001 - one account must not stop the rest
            logger.exception("social read failed for %s account %s", platform, account["id"])
            outcome = {"ok": False, "state": "error", "friendly": f"{exc.__class__.__name__}"}
        _set_source(platform, account, "ok" if outcome["ok"] else outcome["state"],
                    outcome.get("friendly") or "", now)
        if not outcome["ok"]:
            summary["errors"].append(f"{outreach_social.PLATFORM_NAMES[platform]}: {outcome['friendly']}")
    _give_up_unseen_replies(fetched_roots, now)
    database.update_outreach_runtime(last_social_at=fmt_utc(now))


def _recent(item: dict, now) -> bool:
    created = parse_utc(item.get("created"))
    return created is None or now - created <= timedelta(days=WATCH_DAYS)


def _read_account(platform, account, token, now, summary, fetched_roots) -> dict:
    handle = (account["handle"] or "").lstrip("@")
    account_id = account["id"]
    if platform == "twitter":
        runtime = database.get_outreach_profile()["runtime_state"]
        since = (runtime.get("x_since") or {}).get(str(account_id))
        result = outreach_social.x_mentions(token["access_token"], account["external_id"] or "", since)
        if result["ok"]:
            for item in result["items"]:
                if not item["own"]:
                    ingest_social(item, platform, account, "mention", "", summary)
            if result.get("newest_id"):
                database.update_outreach_runtime(x_since={str(account_id): result["newest_id"]})
        return result

    if platform == "threads":
        posts = outreach_social.threads_own_posts(token["access_token"], WATCH_POSTS)
    elif platform == "instagram":
        posts = outreach_social.instagram_own_posts(token["access_token"], WATCH_POSTS)
    else:
        posts = outreach_social.facebook_own_posts(token["page_access_token"], token["page_id"], WATCH_POSTS)
    if not posts["ok"]:
        return posts
    for post in posts["items"]:
        if not _recent(post, now):
            continue
        count = post.get("comment_count")
        if platform == "threads" and post.get("has_replies") is False:
            continue
        if count is not None:
            watched = database.get_watch(platform, account_id, post["id"])
            if count == 0 or (watched and watched["seen_count"] == count):
                continue
        if platform == "threads":
            thread = outreach_social.threads_conversation(token["access_token"], post["id"], handle)
        elif platform == "instagram":
            thread = outreach_social.instagram_comments(token["access_token"], post["id"], handle,
                                                        post.get("permalink"))
        else:
            thread = outreach_social.facebook_comments(token["page_access_token"], post["id"],
                                                       token["page_id"])
        if not thread["ok"]:
            return thread
        fetched_roots.add((platform, str(post["id"])))
        for item in thread["items"]:
            source = "reply" if platform == "threads" else "comment"
            ingest_social(item, platform, account, source, post.get("text") or "", summary)
        if count is not None:
            database.set_watch(platform, account_id, post["id"], count)
    if platform == "threads":
        mentions = outreach_social.threads_mentions(token["access_token"], handle)
        if mentions["ok"]:
            for item in mentions["items"]:
                if not item["own"]:
                    ingest_social(item, platform, account, "mention", "", summary)
        elif mentions["state"] != "needs_permission":
            return mentions
    return {"ok": True, "state": "ok", "friendly": ""}


def _identity_for(platform: str, item: dict) -> tuple[str, str]:
    """How a person on a platform is recognised again: their username where it is
    stable enough (Threads, Instagram), their numeric id where names repeat."""
    if platform in ("facebook", "twitter") and item.get("author_id"):
        return platform, str(item["author_id"])
    return platform, (item.get("author") or "").lower()


def ingest_social(item: dict, platform: str, account, source: str, post_text: str,
                  summary: dict) -> int | None:
    """Store one message from a conversation. The owner's own messages settle
    replies Insights sent and mark what the owner answered by hand."""
    if not item.get("id"):
        return None
    if item.get("own"):
        _own_message_seen(item, platform, account)
        return None
    if database.find_outreach_message(platform, str(item["id"])):
        return None
    kind, value = _identity_for(platform, item)
    lead_id = database.find_lead_by_identity(kind, value) if value else None
    context = {"source": source, "account_id": account["id"], "root_id": str(item.get("root_id") or ""),
               "parent_id": str(item.get("parent_id") or ""), "permalink": item.get("permalink"),
               "post_text": (post_text or "")[:600], "author": item.get("author") or "",
               "author_id": str(item.get("author_id") or ""), "author_name": item.get("author_name") or ""}
    author = (item.get("author") or "").lstrip("@")
    peer = (author if platform == "facebook" else f"@{author}") if author else ""
    row_id = database.add_outreach_message(
        platform, "in", "received", item.get("text") or "", lead_id=lead_id, kind="inbound",
        context=context, external_id=str(item["id"]), peer=peer, sent_at=item.get("created"))
    if row_id:
        summary["social_inbound"] = summary.get("social_inbound", 0) + 1
        if source == "search":
            summary["search_matches"] = summary.get("search_matches", 0) + 1
        if lead_id:
            database.update_lead(lead_id, last_inbound_at=item.get("created") or database.outreach_now())
    return row_id


def _own_message_seen(item: dict, platform: str, account) -> None:
    """A message by the owner showed up in a conversation.

    If it is a reply Insights sent whose outcome was unknown, it is now known.
    If the owner answered a comment by hand, that comment is answered and any
    draft for it closes, so the owner is never offered a second answer.
    """
    item_id = str(item["id"])
    if database.find_outreach_message(platform, item_id):
        return
    body = (item.get("text") or "").strip()
    root = str(item.get("root_id") or "")
    for message in database.outreach_messages_in_conversation(platform, root):
        if message["direction"] == "out" and message["status"] in ("unknown", "sending") \
                and (message.get("body") or "").strip() == body:
            database.record_outreach_sent(message["id"], item_id, "sent")
            return
    answered = database.find_outreach_message(platform, str(item.get("parent_id") or ""))
    if not answered or answered["direction"] != "in":
        return
    database.update_outreach_message(answered["id"], context={**(answered.get("context") or {}),
                                                             "triage": "answered"})
    for message in database.outreach_messages_in_conversation(platform, root):
        if message["direction"] == "out" and message["status"] in ("draft", "needs_review") \
                and message.get("based_on_message_id") == answered["id"]:
            database.update_outreach_message(
                message["id"], status="skipped",
                error=f"You replied on {outreach_social.PLATFORM_NAMES[platform]} directly.")
    database.add_outreach_message(
        platform, "out", "external", item.get("text") or "", lead_id=answered.get("lead_id"),
        kind="external", context={"root_id": root, "parent_id": str(item.get("parent_id") or ""),
                                  "permalink": item.get("permalink"), "note": "You replied directly"},
        external_id=item_id, peer=answered.get("peer"), sent_at=item.get("created"))


def _give_up_unseen_replies(fetched_roots: set, now) -> None:
    """An unconfirmed reply that is still not on its post an hour later never went out."""
    cutoff = fmt_utc(now - timedelta(minutes=SOCIAL_UNKNOWN_GIVE_UP_MINUTES))
    for platform in ("threads", "instagram", "facebook"):
        for message in database.outreach_messages_with_status("unknown", channel=platform):
            root = str((message.get("context") or {}).get("root_id") or "")
            if (platform, root) in fetched_roots and (message.get("sent_at") or "") < cutoff:
                database.update_outreach_message(
                    message["id"], status="failed",
                    error="The reply is not on the post an hour later, so it did not go out. "
                          "Safe to send again.")


# ── 2c. Threads keyword search ──────────────────────────────────────────────

def search_pass(profile: dict, now, summary: dict) -> None:
    """Search Threads for people asking how to get into the field. Reads only."""
    keywords = [k.strip() for k in profile.get("search_keywords") or [] if k and k.strip()]
    if not keywords:
        return
    runtime = profile.get("runtime_state") or {}
    search = dict(runtime.get("search") or {})
    if not _due(search.get("last_at"), search_minutes(), now):
        return
    account = database.get_default_social_account("threads")
    if not account:
        return
    if _source_resting(runtime, "search", "threads", account["id"], now):
        return
    today = start_of_day_utc(now, profile.get("timezone"))
    if search.get("day") != today:
        search = {"day": today, "used": 0, "last_at": search.get("last_at"),
                  "since_unix": search.get("since_unix")}
    budget = int(profile.get("search_daily_queries") or 0)
    client, token, _account, failure = social_publisher.session("threads", account["id"])
    if failure:
        _set_source("search", account, "needs_reconnect", failure.get("error") or "", now)
        return
    since = search.get("since_unix") or int((now - timedelta(days=1)).timestamp())
    handle = (account["handle"] or "").lstrip("@")
    state, friendly = "ok", ""
    for query in keywords:
        if search["used"] >= budget:
            friendly = f"Used today's {budget} searches"
            break
        result = outreach_social.threads_keyword_search(token["access_token"], query, since, 25, handle)
        search["used"] += 1
        if not result["ok"]:
            state, friendly = result["state"], result["friendly"]
            break
        for item in result["items"]:
            ingest_social(item, "threads", account, "search", "", summary)  # drops the owner's own
    search.update(last_at=fmt_utc(now), since_unix=int(now.timestamp()) if state == "ok" else since)
    database.update_outreach_runtime(search=search)
    _set_source("search", account, state, friendly, now)


# ── 3. Drafting ─────────────────────────────────────────────────────────────

def draft_pass(profile: dict, now, summary: dict) -> None:
    missing = context_missing(profile)
    if missing:
        # Nothing is written in the owner's name before Setup says who they are.
        # Replies and texts that arrive are kept and drafted once it does.
        summary["context_missing"] = missing
        summary["errors"].append("Fill in your business context in Setup (missing: "
                                 + ", ".join(missing) + "); nothing is drafted until then.")
        return

    def capped() -> bool:
        caps = outreach_agent.caps_status(profile, now)
        if caps["hit"]:
            summary["capped"] = caps["reason"]
        return caps["hit"]

    for item in database.social_items_to_triage():
        if capped():
            return
        triage_social(item, profile, now, summary)

    for lead in database.leads_awaiting_reply("sms"):
        if capped():
            return
        draft_for_lead(lead, profile, now, summary, kind="continuation",
                       based_on=lead.get("last_in_id"))

    retry_before = fmt_utc(now - timedelta(minutes=REVIEW_RETRY_MINUTES))
    for message in database.outreach_messages_with_status("needs_review", older_than=retry_before):
        if int(message.get("attempts") or 0) >= MAX_DRAFT_ATTEMPTS or not message.get("lead_id"):
            continue
        if capped():
            return
        lead = database.get_lead(message["lead_id"])
        if lead and message["channel"] in outreach_social.SOCIAL_CHANNELS:
            item = database.get_outreach_message(message.get("based_on_message_id") or 0)
            if item:
                draft_social(item, profile, now, summary, existing=message, force=True)
        elif lead:
            draft_for_lead(lead, profile, now, summary, kind=message.get("kind") or "continuation",
                           based_on=message.get("based_on_message_id"), existing=message,
                           follow_up=int((message.get("context") or {}).get("follow_up") or 0))

    for lead in database.leads_due_follow_up(fmt_utc(now), int(profile.get("max_follow_ups") or 0)):
        # Whatever happens next, this nudge is now spoken for: a skipped draft
        # must not come back on the next pass.
        database.update_lead(lead["id"], next_follow_up_at=None)
        if not lead.get("sms_consent"):
            continue
        if capped():
            return
        draft_for_lead(lead, profile, now, summary, kind="follow_up",
                       follow_up=int(lead.get("follow_ups_sent") or 0) + 1)

    caps = outreach_agent.caps_status(profile, now)
    if caps["hit"]:
        summary["capped"] = caps["reason"]
        return
    room = max(0, caps["draft_cap"] - caps["drafts"]) if caps["draft_cap"] else 25
    for lead in database.leads_needing_opener(room):
        if capped():
            return
        draft_for_lead(lead, profile, now, summary, kind="opener")


def draft_for_lead(lead: dict, profile: dict, now, summary: dict | None = None, *, kind: str,
                   based_on: int | None = None, follow_up: int = 0, existing: dict | None = None,
                   instruction: str | None = None, after_call: dict | None = None,
                   by: str = "agent") -> dict:
    """Write one draft for a lead and put it in the queue.

    ``existing`` rewrites that queue item in place (Regenerate, or a retry of a
    draft that needed review). Returns ``{"status": "draft" | "needs_review" |
    "opted_out" | "exists", "message_id": ..., "error": ...}``.
    """
    summary = summary if summary is not None else {"drafts": 0, "needs_review": 0, "opted_out": 0}
    if context_missing(profile):
        return _needs_context(profile)
    thread = database.list_lead_messages(lead["id"])
    context = {"by": by, "follow_up": follow_up}
    if after_call:
        context["after_call"] = after_call
    try:
        if kind == "opener":
            result = outreach_agent.draft_opener(lead, profile, instruction=instruction)
        else:
            result = outreach_agent.draft_continuation(
                lead, thread, profile, follow_up=follow_up if kind == "follow_up" else 0,
                instruction=instruction, after_call=after_call)
    except Exception as exc:  # noqa: BLE001 - a refusal, bad JSON, or a provider error
        if not isinstance(exc, outreach_agent.DraftError):
            logger.exception("outreach draft failed for lead %s", lead.get("id"))
        error = str(exc) if isinstance(exc, outreach_agent.DraftError) else \
            f"Could not write a draft ({exc.__class__.__name__}). Write it yourself or try again."
        summary["needs_review"] = summary.get("needs_review", 0) + 1
        if existing:
            database.update_outreach_message(
                existing["id"], status="needs_review", error=error,
                attempts=int(existing.get("attempts") or 0) + 1)
            return {"status": "needs_review", "message_id": existing["id"], "error": error}
        message_id = database.add_outreach_message(
            "sms", "out", "needs_review", "", lead_id=lead["id"], kind=kind, context=context,
            based_on_message_id=based_on, error=error, attempts=1)
        return {"status": "needs_review" if message_id else "exists", "message_id": message_id,
                "error": error}

    context.update(reason=result["reason"], intent=result["intent"], offer_key=result["offer_key"])
    if kind == "continuation" and result["intent"] == "opt_out" and not after_call:
        database.set_lead_opt_out(lead["id"], "detected")
        summary["opted_out"] = summary.get("opted_out", 0) + 1
        note = ("They asked not to be texted, so Insights opted them out. If that is wrong, "
                "undo it in the lead's details.")
        if existing:
            database.update_outreach_message(existing["id"], status="skipped", error=note,
                                             context=context)
        else:
            database.add_outreach_message("sms", "out", "skipped", "", lead_id=lead["id"],
                                          kind=kind, context=context,
                                          based_on_message_id=based_on, error=note)
        return {"status": "opted_out", "message_id": None, "error": note}

    summary["drafts"] = summary.get("drafts", 0) + 1
    if existing:
        database.update_outreach_message(
            existing["id"], status="draft", body=result["reply"], context=context, error=None,
            kind=kind, attempts=int(existing.get("attempts") or 0) + 1,
            based_on_message_id=based_on or existing.get("based_on_message_id"))
        return {"status": "draft", "message_id": existing["id"], "error": None}
    message_id = database.add_outreach_message(
        "sms", "out", "draft", result["reply"], lead_id=lead["id"], kind=kind, context=context,
        based_on_message_id=based_on)
    return {"status": "draft" if message_id else "exists", "message_id": message_id, "error": None}


def _needs_context(profile: dict) -> dict:
    """A draft asked for before Setup says who the owner is: nothing is written."""
    return {"status": "needs_context", "message_id": None,
            "error": "Fill in your business context in Setup first (missing: "
                     + ", ".join(context_missing(profile)) + ")."}


def _lead_for_item(item: dict, fit: int = 0) -> int:
    """The lead behind a social message: found by their handle, or created."""
    if item.get("lead_id"):
        return item["lead_id"]
    channel = item["channel"]
    context = item.get("context") or {}
    kind, value = _identity_for(channel, {"author": context.get("author"),
                                          "author_id": context.get("author_id")})
    lead_id = database.find_lead_by_identity(kind, value) if value else None
    if lead_id is None:
        if context.get("author_name"):
            name = context["author_name"]
        elif context.get("author"):
            name = context["author"] if channel == "facebook" else "@" + context["author"]
        else:
            name = f"Someone on {outreach_social.PLATFORM_NAMES.get(channel, channel)}"
        source = "threads_search" if context.get("source") == "search" else channel
        lead_id = database.create_lead(
            [(kind, value)] if value else [], name=name, source=source, stage="replied",
            last_inbound_at=item.get("sent_at") or item.get("created_at"), score=fit)
    database.update_outreach_message(item["id"], lead_id=lead_id)
    return lead_id


def triage_social(item: dict, profile: dict, now, summary: dict) -> dict:
    """Decide about one new social message: skip it, or draft a reply to it."""
    lead = database.get_lead(item["lead_id"]) if item.get("lead_id") else None
    if lead and (lead.get("do_not_contact") or lead.get("opted_out_at")):
        database.update_outreach_message(item["id"], context={**(item.get("context") or {}),
                                                              "triage": "skip",
                                                              "triage_reason": "Do not contact"})
        return {"status": "skipped"}
    if lead and database.open_outreach_item(lead["id"], item["channel"]):
        return {"status": "waiting"}  # one open draft per lead; this one waits its turn
    return draft_social(item, profile, now, summary)


def draft_social(item: dict, profile: dict, now, summary: dict | None = None, *,
                 existing: dict | None = None, instruction: str | None = None,
                 force: bool = False, by: str = "agent") -> dict:
    """Write (or rewrite) the public reply to one social message."""
    summary = summary if summary is not None else {}
    if context_missing(profile):
        return _needs_context(profile)
    channel = item["channel"]
    context_in = item.get("context") or {}
    lead = database.get_lead(item["lead_id"]) if item.get("lead_id") else None
    # The model sees this person's exchange with the owner on the post, not
    # every stranger's comment under it.
    conversation = [m for m in database.outreach_messages_in_conversation(channel, context_in.get("root_id") or "")
                    if item.get("lead_id") and m.get("lead_id") == item.get("lead_id")]
    context = {"by": by, "in_reply_to": item.get("external_id"),
               "account_id": context_in.get("account_id"), "root_id": context_in.get("root_id"),
               "permalink": context_in.get("permalink"), "source": context_in.get("source")}
    try:
        result = outreach_agent.draft_social_reply(item, conversation, profile, lead,
                                                   instruction=instruction, force=force)
    except Exception as exc:  # noqa: BLE001 - a refusal, bad JSON, or a provider error
        if not isinstance(exc, outreach_agent.DraftError):
            logger.exception("social draft failed for message %s", item.get("id"))
        error = str(exc) if isinstance(exc, outreach_agent.DraftError) else \
            f"Could not write a reply ({exc.__class__.__name__}). Write it yourself or try again."
        summary["needs_review"] = summary.get("needs_review", 0) + 1
        if existing:
            database.update_outreach_message(existing["id"], status="needs_review", error=error,
                                             attempts=int(existing.get("attempts") or 0) + 1)
            return {"status": "needs_review", "message_id": existing["id"], "error": error}
        lead_id = _lead_for_item(item)
        message_id = database.add_outreach_message(
            channel, "out", "needs_review", "", lead_id=lead_id, kind="continuation",
            context=context, based_on_message_id=item["id"], error=error, attempts=1)
        if message_id:
            database.update_outreach_message(item["id"], context={**context_in, "triage": "drafted"})
        return {"status": "needs_review" if message_id else "exists", "message_id": message_id,
                "error": error}

    context.update(reason=result["reason"], intent=result["intent"],
                   offer_key=result["offer_key"], fit=result["fit"], is_lead=result["is_lead"])
    if result["intent"] == "opt_out" and not force:
        # "Leave me alone" in a public thread: no reply, today or later.
        result.update(should_reply=False, reason=result["reason"] or "They asked not to be contacted")
        if item.get("lead_id"):
            database.update_lead(item["lead_id"], do_not_contact=1)
    if not result["should_reply"]:
        summary["skipped_by_agent"] = summary.get("skipped_by_agent", 0) + 1
        database.update_outreach_message(item["id"], context={**context_in, "triage": "skip",
                                                              "triage_reason": result["reason"]})
        return {"status": "skipped", "message_id": None, "error": result["reason"]}

    summary["drafts"] = summary.get("drafts", 0) + 1
    if existing:
        database.update_outreach_message(existing["id"], status="draft", body=result["reply"],
                                         context={**(existing.get("context") or {}), **context},
                                         error=None, attempts=int(existing.get("attempts") or 0) + 1)
        return {"status": "draft", "message_id": existing["id"], "error": None}
    lead_id = _lead_for_item(item, result["fit"])
    message_id = database.add_outreach_message(
        channel, "out", "draft", result["reply"], lead_id=lead_id, kind="continuation",
        context=context, based_on_message_id=item["id"])
    if message_id:
        database.update_outreach_message(item["id"], context={**context_in, "triage": "drafted"})
        lead = database.get_lead(lead_id)
        if lead and result["fit"] > int(lead.get("score") or 0):
            database.update_lead(lead_id, score=result["fit"])
    return {"status": "draft" if message_id else "exists", "message_id": message_id, "error": None}


def latest_inbound_id(lead_id: int) -> int | None:
    message = database.last_lead_message(lead_id, "sms", "in")
    return message["id"] if message else None

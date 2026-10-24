"""The only way an outreach message leaves Insights.

``send`` is called by the Send button's route and by nothing else: it refuses
to run outside a logged-in browser request, so the background worker (which
drafts around the clock) has no path to a person's phone. Pressing Send is the
owner's approval.

Every rule is checked here, on the server, whatever the page showed. For a
text:

* the person agreed to texts, or texted in within the last 24 hours;
* they have not opted out and are not marked do-not-contact;
* it is inside the texting window wherever their number says they may be;
* they have had fewer than 3 texts or calls in 24 hours (Florida's FTSA limit);
* today's send cap is not reached and texting is not paused;
* they have not replied since the draft was written;
* every link is one of the owner's offers.

A public reply on Threads, Instagram, Facebook or X (``_social_guard``) has its
own rules: no consent or texting hours, but never after an opt-out, never
without the comment it answers, never past a newer message in that
conversation, at most 3 a day to one person, and only the owner's links.

The checks and the draft -> sending claim happen in one database transaction
(``database.claim_outreach_message``), so a double click or a second tab sends
once. A send whose outcome Twilio did not confirm is marked ``unknown`` and is
never sent again automatically; the worker settles it by reading Twilio.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import database
import outreach_social
import twilio_client
from outreach_rules import (
    MAX_SEGMENTS, foreign_links, fmt_utc, hours_between, lead_score, lead_zones, parse_utc,
    plain_text, send_window_check, sms_segments, start_of_day_utc, utc_now, with_first_text_footer,
)

logger = logging.getLogger(__name__)

PER_LEAD_DAILY_LIMIT = 3
REPLY_WINDOW_HOURS = 24


class NotFromSendButton(RuntimeError):
    """``send`` was called somewhere other than the Send route."""


def _require_send_request(user_id) -> None:
    from flask import has_request_context, request, session
    if not has_request_context():
        raise NotFromSendButton("Outreach messages are only sent from the Send button")
    if not user_id or session.get("user_id") != user_id:
        raise NotFromSendButton("Outreach messages are only sent by the logged-in owner")
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        raise NotFromSendButton("Outreach messages are only sent from the Outreach page")


def final_text(body: str, lead: dict | None, profile: dict, first: bool) -> str:
    """The exact text that goes out: plain punctuation, and on a first text who
    it is from and how to opt out."""
    text = plain_text(body)
    if first and text:
        text = with_first_text_footer(text, profile.get("sms_signature") or "")
    return text


def _refusal(code: int, reason: str) -> dict:
    return {"ok": False, "http_status": code, "error": reason}


PER_LEAD_SOCIAL_DAILY_LIMIT = 3


def _social_guard(conn, message: dict, lead: dict | None, *, text: str, profile: dict, now) -> tuple | None:
    """Why a public reply may not go out now, as ``(http_status, reason)``, or None.

    A public reply to someone who wrote to the owner (or whom the owner found
    asking for help) needs no texting consent and no texting hours. What still
    holds: never after an opt-out or do-not-contact, never without a target to
    reply to, never twice for one draft, never past a newer message in the same
    conversation, never more than 3 replies to one person a day, and only the
    owner's own links.
    """
    platform = message["channel"]
    name = outreach_social.PLATFORM_NAMES[platform]
    context = message.get("context") or {}
    if lead is None:
        return 409, "This reply has no lead"
    who = lead.get("name") or "This person"
    if lead.get("opted_out_at") or lead.get("do_not_contact"):
        return 403, f"{who} asked not to be contacted. Insights will not reply to them."
    if not str(context.get("in_reply_to") or "").strip():
        return 400, "This draft does not say which message it answers, so it cannot be sent."
    account = conn.execute("SELECT platform, status FROM social_accounts WHERE id = ?",
                           (context.get("account_id"),)).fetchone()
    if not account or account[0] != platform or account[1] == "removed":
        return 409, f"The {name} account this conversation is on is no longer connected."
    if not text.strip():
        return 400, "The reply is empty."
    limit = outreach_social.REPLY_LIMITS[platform]
    if len(text) > limit:
        return 400, f"{name} replies can be at most {limit} characters; this one is {len(text)}."
    newer = conn.execute(
        "SELECT MAX(id) FROM outreach_messages WHERE lead_id = ? AND channel = ? "
        "AND direction = 'in' AND status = 'received' AND json_extract(context, '$.root_id') = ?",
        (lead["id"], platform, str(context.get("root_id") or ""))).fetchone()[0]
    if newer and newer > max(message["id"], message.get("based_on_message_id") or 0):
        return 409, f"{who} wrote again in this conversation since the draft. Regenerate it first."
    since = fmt_utc(now - timedelta(hours=24))
    replies = conn.execute(
        "SELECT COUNT(*) FROM outreach_messages WHERE lead_id = ? AND direction = 'out' "
        "AND id != ? AND channel IN ('threads', 'instagram', 'facebook', 'twitter') "
        "AND status IN ('sending', 'sent', 'delivered', 'unknown') AND sent_at >= ?",
        (lead["id"], message["id"], since)).fetchone()[0]
    if replies >= PER_LEAD_SOCIAL_DAILY_LIMIT:
        return 429, f"{who} has had {replies} replies from you in 24 hours. Give it a day."
    cap = int(profile.get("daily_send_cap") or 0)
    if cap:
        sent_today = conn.execute(
            "SELECT COUNT(*) FROM outreach_messages WHERE channel = ? AND direction = 'out' "
            "AND id != ? AND status IN ('sending', 'sent', 'delivered', 'unknown') AND sent_at >= ?",
            (platform, message["id"], start_of_day_utc(now, profile.get("timezone")))).fetchone()[0]
        if sent_today >= cap:
            return 429, f"Reached today's limit of {cap} {name} replies. Change it in Setup."
    foreign = foreign_links(text, profile.get("offers"))
    if foreign:
        return 400, (f"This reply links to {', '.join(foreign)}, which is not one of your offers. "
                     "Add it to your offers in Setup, or take it out.")
    return None


def _guard(conn, message: dict, lead: dict | None, *, text: str, profile: dict, now) -> tuple | None:
    """Why this message may not go out now, as ``(http_status, reason)``, or None."""
    if message["direction"] == "out" and message["channel"] in outreach_social.SOCIAL_CHANNELS:
        return _social_guard(conn, message, lead, text=text, profile=profile, now=now)
    if message["channel"] != "sms" or message["direction"] != "out":
        return 400, "Only texts and social replies can be sent from here"
    if profile.get("sms_paused_reason"):
        return 423, (f"Texting is paused: {profile['sms_paused_reason']} "
                     "Resume it in Setup once that is fixed.")
    if lead is None:
        return 409, "Link this conversation to a lead before replying"
    who = lead.get("name") or "This person"
    if lead.get("opted_out_at"):
        return 403, f"{who} opted out on {lead['opted_out_at'][:10]}. Insights will not text them."
    if lead.get("do_not_contact"):
        return 403, f"{who} is marked do-not-contact."
    if not lead.get("phone"):
        return 400, f"{who} has no phone number on file."
    if not text.strip():
        return 400, "The text is empty."

    # Ids grow in the order Insights learned of things, so an inbound text with a
    # higher id than the draft (or than the text the draft answers) is one the
    # draft was written without.
    last_in = conn.execute(
        "SELECT id, COALESCE(sent_at, created_at) FROM outreach_messages WHERE lead_id = ? "
        "AND channel = 'sms' AND direction = 'in' AND status = 'received' "
        "ORDER BY id DESC LIMIT 1",
        (lead["id"],),
    ).fetchone()
    last_in_at = parse_utc(last_in[1]) if last_in else None
    replying = bool(last_in_at and hours_between(now, last_in_at) <= REPLY_WINDOW_HOURS)
    if not lead.get("sms_consent") and not replying:
        return 403, (f"There is no text consent on record for {who}. You can reply within 24 hours "
                     "of a text from them, or call if they agreed to calls.")
    if last_in and last_in[0] > max(message["id"], message.get("based_on_message_id") or 0):
        return 409, f"{who} replied since this was drafted. Regenerate it to answer their latest text."

    zones, _source = lead_zones(lead["phone"], lead.get("timezone"), profile.get("timezone"))
    ok, why = send_window_check(now, zones, profile.get("send_window_start"),
                                profile.get("send_window_end"))
    if not ok:
        return 423, why

    since = fmt_utc(now - timedelta(hours=24))
    reached = conn.execute(
        "SELECT COUNT(*) FROM outreach_messages WHERE lead_id = ? AND direction = 'out' "
        "AND id != ? AND status IN (%s) AND COALESCE(sent_at, created_at) >= ?"
        % ", ".join("?" for _ in database.OUTREACH_REACHED_STATUSES),
        (lead["id"], message["id"], *database.OUTREACH_REACHED_STATUSES, since),
    ).fetchone()[0]
    if reached >= PER_LEAD_DAILY_LIMIT:
        return 429, (f"{who} has had {reached} texts or calls in the last 24 hours, the most "
                     "Florida's telemarketing law allows. Try again tomorrow.")

    cap = int(profile.get("daily_send_cap") or 0)
    if cap:
        sent_today = conn.execute(
            "SELECT COUNT(*) FROM outreach_messages WHERE channel = 'sms' AND direction = 'out' "
            "AND id != ? AND status IN ('sending', 'sent', 'delivered', 'unknown') AND sent_at >= ?",
            (message["id"], start_of_day_utc(now, profile.get("timezone"))),
        ).fetchone()[0]
        if sent_today >= cap:
            return 429, f"Reached today's limit of {cap} texts. Change it in Setup."

    foreign = foreign_links(text, profile.get("offers"))
    if foreign:
        return 400, (f"This text links to {', '.join(foreign)}, which is not one of your offers. "
                     "Add it to your offers in Setup, or take it out.")
    if sms_segments(text)["segments"] > MAX_SEGMENTS:
        return 400, f"This text is longer than {MAX_SEGMENTS} SMS segments. Shorten it."
    return None


def _after_sent(lead: dict, message: dict, profile: dict, now) -> None:
    """Move the lead along once a text has gone out."""
    changes = {"last_contact_at": fmt_utc(now)}
    if lead.get("stage") in (None, "", "new"):
        changes["stage"] = "contacted"
    follow_ups = int(lead.get("follow_ups_sent") or 0)
    if message.get("kind") == "follow_up":
        follow_ups += 1
        changes["follow_ups_sent"] = follow_ups
    days = int(profile.get("follow_up_days") or 0)
    stage = changes.get("stage", lead.get("stage"))
    if days and follow_ups < int(profile.get("max_follow_ups") or 0) \
            and stage not in ("booked", "enrolled", "not_fit"):
        changes["next_follow_up_at"] = fmt_utc(now + timedelta(days=days))
    else:
        changes["next_follow_up_at"] = None
    changes["score"] = lead_score({**lead, **changes}, now)
    database.update_lead(lead["id"], **changes)


def preview_text(message: dict, lead: dict | None, profile: dict, body: str | None = None) -> dict:
    """What Send would send for this draft, and what it costs in segments."""
    first = bool(lead) and not database.lead_reached_before(lead["id"], "sms")
    text = final_text(message.get("body") if body is None else body, lead, profile, first)
    return {"text": text, "first": first, **sms_segments(text)}


def send(message_id: int, body: str, user_id: int, now=None, client=None) -> dict:
    """Send one approved draft. Returns ``{ok, status, error?, http_status?, text?}``."""
    _require_send_request(user_id)
    now = now or utc_now()
    profile = database.get_outreach_profile()
    message = database.get_outreach_message(message_id)
    if message is None:
        return _refusal(404, "That message no longer exists")
    lead = database.get_lead(message["lead_id"]) if message.get("lead_id") else None
    social = message["channel"] in outreach_social.SOCIAL_CHANNELS
    if social:
        text = plain_text(body)
    else:
        first = bool(lead) and not database.lead_reached_before(lead["id"], "sms")
        if first and not (profile.get("sms_signature") or "").strip():
            # A first text must say who it is from; carriers and the TCPA expect it.
            return _refusal(400, "Add the name your texts come from (Setup, Your business "
                                 "context) before the first text to anyone.")
        text = final_text(body, lead, profile, first)

    def check(conn, current, current_lead):
        return _guard(conn, current, current_lead, text=text, profile=profile, now=now)

    message, lead, refusal = database.claim_outreach_message(message_id, text, user_id, check)
    if refusal:
        return _refusal(*refusal)
    if social:
        return _send_social(message, lead, text, now)

    client = client or twilio_client.get_twilio_client()
    result = client.send_sms(lead["phone"], text)
    if result.get("success"):
        database.record_outreach_sent(message_id, result.get("sid"), "sent")
        _after_sent(lead, message, profile, now)
        return {"ok": True, "status": "sent", "sid": result.get("sid"), "text": text}
    friendly = result.get("friendly") or "Twilio did not send it"
    if result.get("ambiguous"):
        database.record_outreach_sent(message_id, None, "unknown", error=friendly)
        _after_sent(lead, message, profile, now)
        return {"ok": False, "status": "unknown", "http_status": 502, "error": friendly}
    database.record_outreach_sent(message_id, None, "failed", error=friendly)
    if message.get("kind") == "follow_up":
        # The failed card offers Edit & retry; the worker must not draft another.
        database.update_lead(lead["id"], next_follow_up_at=None)
    if result.get("opted_out"):
        database.set_lead_opt_out(lead["id"], "carrier")
    if result.get("error_code") in twilio_client.PAUSE_ERRORS:
        database.update_outreach_profile(sms_paused_reason=friendly)
    return {"ok": False, "status": "failed", "http_status": 502, "error": friendly}


def _send_social(message: dict, lead: dict, text: str, now) -> dict:
    """Post a claimed reply into its conversation and record what happened."""
    context = message.get("context") or {}
    platform = message["channel"]
    result = outreach_social.reply(platform, context.get("account_id"), context.get("in_reply_to"), text)
    if result.get("success"):
        database.record_outreach_sent(message["id"], str(result.get("id") or ""), "sent")
        if result.get("permalink"):
            database.update_outreach_message(message["id"],
                                             context={**context, "permalink": result["permalink"]})
        changes = {"last_contact_at": fmt_utc(now)}
        changes["score"] = lead_score({**lead, **changes}, now)
        database.update_lead(lead["id"], **changes)
        return {"ok": True, "status": "sent", "id": result.get("id"),
                "permalink": result.get("permalink"), "text": text}
    friendly = result.get("friendly") or "The platform did not post it"
    if result.get("ambiguous"):
        database.record_outreach_sent(message["id"], None, "unknown", error=friendly)
        return {"ok": False, "status": "unknown", "http_status": 502, "error": friendly}
    if result.get("open_url"):
        database.update_outreach_message(message["id"], context={**context, "open_url": result["open_url"]})
    database.record_outreach_sent(message["id"], None, "failed", error=friendly)
    return {"ok": False, "status": "failed", "http_status": 502, "error": friendly,
            "open_url": result.get("open_url")}

"""The rules outreach messages are held to, as pure functions.

Everything here is a decision about a piece of text, a phone number or a time:
what counts as an opt-out, how many SMS segments a text costs, which links it
may carry, whether it is a sensible hour for the person receiving it, and how
promising a lead looks. Nothing here reads the database or the network, so the
sender, the worker, the routes and the gates all share one answer to each
question.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import phonenumbers
from phonenumbers import timezone as phone_timezone

# ── Keywords ────────────────────────────────────────────────────────────────
# Carriers and Twilio treat these single-word replies as opt-out, opt-in and
# help requests. Twilio already blocks texts to a number that sent one of the
# opt-out words; recording it here stops Insights drafting to that person too.
OPT_OUT_KEYWORDS = {"STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT", "REVOKE", "OPTOUT"}
OPT_IN_KEYWORDS = {"START", "UNSTOP", "YES"}
HELP_KEYWORDS = {"HELP", "INFO"}


def keyword_of(body: str) -> str | None:
    """'stop', 'start' or 'help' when the whole message is one of those words."""
    word = re.sub(r"[^A-Za-z]", "", body or "").upper()
    if word in OPT_OUT_KEYWORDS:
        return "stop"
    if word in OPT_IN_KEYWORDS:
        return "start"
    if word in HELP_KEYWORDS:
        return "help"
    return None


# ── Text ────────────────────────────────────────────────────────────────────
# One character outside the GSM-7 alphabet makes a carrier send the whole text
# as UCS-2, which cuts a segment from 160 characters to 70 and roughly doubles
# the cost. Typographic punctuation is the usual culprit, so it is swapped for
# the plain equivalent before anything is counted or sent.
_GSM7_BASIC = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
_GSM7_EXTENDED = set("^{}\\[~]|€")

_PLAIN_PUNCTUATION = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "−": "-", "…": "...", " ": " ",
    "•": "-", "·": "-", "​": "", "‌": "", "‍": "", "﻿": "",
}


def plain_text(text: str) -> str:
    """Typographic punctuation replaced with plain characters; em dashes become commas."""
    text = re.sub(r"\s*—\s*", ", ", text or "")
    for fancy, plain in _PLAIN_PUNCTUATION.items():
        text = text.replace(fancy, plain)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def sms_segments(text: str) -> dict:
    """How a carrier will split ``text``: encoding, length in units, segments."""
    text = text or ""
    if all(ch in _GSM7_BASIC or ch in _GSM7_EXTENDED for ch in text):
        units = sum(2 if ch in _GSM7_EXTENDED else 1 for ch in text)
        single, multi, encoding = 160, 153, "GSM-7"
    else:
        units = len(text.encode("utf-16-le")) // 2
        single, multi, encoding = 70, 67, "UCS-2"
    if units == 0:
        segments = 0
    elif units <= single:
        segments = 1
    else:
        segments = -(-units // multi)
    return {"encoding": encoding, "units": units, "segments": segments}


MAX_SEGMENTS = 6

STOP_LINE = "Reply STOP to opt out"


def with_first_text_footer(body: str, signature: str) -> str:
    """The first text to a person says who it is from and how to stop more."""
    body = (body or "").rstrip()
    if re.search(r"\bSTOP\b", body):
        return body
    lines = [body, ""]
    signature = (signature or "").strip()
    if signature and signature.lower() not in body.lower():
        lines.append(f"- {signature}")
    lines.append(STOP_LINE)
    return "\n".join(lines).strip()


# ── Links ───────────────────────────────────────────────────────────────────
_TLDS = "com|co|org|net|io|ly|me|gg|app|dev|link|ai|us|info|biz|xyz|page|site|to|cc|tv"
_URL_RE = re.compile(
    r"(?:https?://[^\s<>\"']+)"
    r"|(?:\bwww\.[^\s<>\"']+)"
    # A bare domain, but not the domain half of an email address.
    rf"|(?:(?<![@\w.-])[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*\.(?:{_TLDS})\b(?:/[^\s<>\"']*)?)",
    re.IGNORECASE,
)


def _bare(url: str) -> str:
    url = url.strip().rstrip(".,;:!?)]}'\"").lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.rstrip("/")


def find_links(text: str) -> list[str]:
    """Every link-looking thing in ``text``, as written."""
    return [m.group(0).rstrip(".,;:!?)]}'\"") for m in _URL_RE.finditer(text or "")]


def allowed_links(offers) -> list[str]:
    """The links the owner has put in their offers, in comparable form."""
    return [_bare(o.get("url") or "") for o in (offers or []) if (o.get("url") or "").strip()]


def foreign_links(text: str, offers) -> list[str]:
    """Links in ``text`` that are not one of the owner's offer links (or a page under one)."""
    allowed = allowed_links(offers)
    out = []
    for link in find_links(text):
        bare = _bare(link)
        if not any(bare == a or bare.startswith(a + "/") or bare.startswith(a + "?")
                   for a in allowed):
            out.append(link)
    return out


def strip_foreign_links(text: str, offers) -> tuple[str, list[str]]:
    """``text`` without the links that are not the owner's, and the links removed."""
    removed = foreign_links(text, offers)
    for link in removed:
        text = text.replace(link, "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    if removed:
        # "It lays out the order: <link>" should not end on the colon.
        text = re.sub(r"[ \t]*:\s*$", ".", text.strip())
    return text.strip(), removed


# ── Phone numbers and time ──────────────────────────────────────────────────

def normalize_phone(raw: str, region: str = "US", country_code: str | None = None) -> str | None:
    """``raw`` as an E.164 number (``+13055550123``), or None if it is not a real number."""
    raw = (raw or "").strip()
    if not raw:
        return None
    cc = re.sub(r"\D", "", country_code or "")
    candidate = raw
    if cc and not raw.startswith("+"):
        candidate = f"+{cc}{re.sub(r'[^0-9]', '', raw)}"
    try:
        number = phonenumbers.parse(candidate, None if candidate.startswith("+") else region)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


def pretty_phone(e164: str | None) -> str:
    if not e164:
        return ""
    try:
        number = phonenumbers.parse(e164, None)
        if number.country_code == 1:
            return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.NATIONAL)
        return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    except phonenumbers.NumberParseException:
        return e164


# A number that maps to more zones than this (a toll-free number maps to every
# zone in North America) says nothing useful about where the person is.
_MAX_ZONES = 4


def lead_zones(phone: str | None, lead_timezone: str | None, owner_timezone: str) -> tuple[list[str], str]:
    """The time zones a lead may be in, and where that answer came from.

    The lead's own setting wins. Otherwise the phone number's zones are used, all
    of them, because a texting window has to hold wherever the person might be.
    A number that could be anywhere falls back to the owner's zone.
    """
    if lead_timezone:
        return [lead_timezone], "set on the lead"
    zones = []
    if phone:
        try:
            zones = [z for z in phone_timezone.time_zones_for_number(phonenumbers.parse(phone, None))
                     if z and z != "Etc/Unknown"]
        except phonenumbers.NumberParseException:
            zones = []
    if 1 <= len(zones) <= _MAX_ZONES:
        return list(zones), "from the phone number"
    return [owner_timezone or "America/New_York"], "yours (their number does not say)"


def _hhmm(value: str, default: time) -> time:
    try:
        hours, minutes = (int(p) for p in (value or "").split(":")[:2])
        return time(hours, minutes)
    except Exception:
        return default


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def send_window_check(now_utc: datetime, zones: list[str], start: str, end: str) -> tuple[bool, str]:
    """Is it inside the texting window in every one of ``zones``? With the reason if not."""
    start_t = _hhmm(start, time(8, 0))
    end_t = _hhmm(end, time(20, 0))
    for zone in zones:
        try:
            local = now_utc.astimezone(ZoneInfo(zone))
        except Exception:
            continue
        if not (start_t <= local.time() < end_t):
            return False, (
                f"It is {local.strftime('%-I:%M %p')} for them ({zone}). Texts go out between "
                f"{start_t.strftime('%-I:%M %p')} and {end_t.strftime('%-I:%M %p')} their time."
            )
    return True, ""


def start_of_day_utc(now_utc: datetime, zone: str) -> str:
    """Midnight today in ``zone``, as a UTC string in the outreach tables' format."""
    try:
        tz = ZoneInfo(zone or "America/New_York")
    except Exception:
        tz = ZoneInfo("America/New_York")
    local = now_utc.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return fmt_utc(midnight)


def fmt_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            moment = datetime.fromisoformat(value)
            return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def hours_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier) / timedelta(hours=1)


# ── Business context ────────────────────────────────────────────────────────

# What the agent must know before it writes anything in the owner's name, and
# how Setup names each one.
CONTEXT_REQUIRED = {
    "business_name": "Business name",
    "business_field": "What field you work in",
    "program_summary": "What you offer",
    "ideal_lead": "Who it is for",
}


def context_missing(profile: dict) -> list[str]:
    """The Setup labels of the business-context fields still empty."""
    return [label for key, label in CONTEXT_REQUIRED.items()
            if not str((profile or {}).get(key) or "").strip()]


def offer_names(profile: dict) -> dict:
    """``{offer key: offer name}`` from the owner's offers."""
    return {o.get("key"): o.get("name") or o.get("key")
            for o in (profile or {}).get("offers") or [] if o.get("key")}


# ── Lead score ──────────────────────────────────────────────────────────────

def lead_score(lead: dict, now_utc: datetime | None = None) -> int:
    """How promising a lead looks, 0 to 100, from what they have done.

    No model is involved: an opt-in already says the person is interested, so
    the score only orders the list by what they took, whether they answered and
    how recently they raised their hand.
    """
    if lead.get("opted_out_at") or lead.get("do_not_contact") or lead.get("stage") == "not_fit":
        return 0
    now_utc = now_utc or utc_now()
    magnets = set(lead.get("magnets") or [])
    score = 10
    # Each of the owner's offers a lead took, a little more for taking several.
    score += 18 * min(len(magnets), 2)
    if len(magnets) > 1:
        score += 5
    if lead.get("quiz_result"):
        score += 10
    if lead.get("last_inbound_at"):
        score += 25
    stage = lead.get("stage")
    if stage == "booked":
        score += 20
    elif stage == "enrolled":
        score += 25
    joined = parse_utc(lead.get("opted_in_at")) or parse_utc(lead.get("created_at"))
    if joined:
        age_days = (now_utc - joined).days
        if age_days <= 14:
            score += 15
        elif age_days <= 60:
            score += 5
    return max(0, min(100, score))

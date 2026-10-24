"""Twilio SMS client for outreach: send a text, read the number's messages.

Built on ``requests`` like the platform clients, without the Twilio SDK. Every
method returns a dict and none raises, because the callers decide what a
failure means.

The answer that matters most is whether a text might have gone out. A send
that Twilio refused, or that never reached Twilio, is safe to try again; one
whose answer was lost (a read timeout, a dropped connection, a 5xx) may already
be on the person's phone. ``send_sms`` reports the second kind as
``ambiguous`` so the caller marks it unknown rather than sending it twice.

Configuration (``.env``):

* ``TWILIO_ACCOUNT_SID`` and ``TWILIO_AUTH_TOKEN``
* ``TWILIO_FROM_NUMBER``: the number texts come from and replies arrive at, in
  E.164 form. Reading replies needs it.
* ``TWILIO_MESSAGING_SERVICE_SID`` (optional): send through a Messaging
  Service instead of straight from the number.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from email.utils import parsedate_to_datetime
from datetime import timezone

import requests

logger = logging.getLogger(__name__)

TWILIO_API_HOST = "https://api.twilio.com"
TWILIO_API = f"{TWILIO_API_HOST}/2010-04-01"
# (connect, read). A connect timeout means the request never reached Twilio.
TIMEOUT = (5, 20)
MAX_PAGES = 5

# Twilio error codes the outreach code acts on.
ERROR_OPTED_OUT = 21610          # the recipient replied STOP
ERROR_NOT_REGISTERED = 30034     # the number is not registered for A2P 10DLC
ERROR_CARRIER_FILTERED = 30007   # the carrier filtered the message as spam
PAUSE_ERRORS = {ERROR_NOT_REGISTERED, ERROR_CARRIER_FILTERED}

_FRIENDLY = {
    20003: "Twilio rejected the account SID or auth token. Check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.",
    21211: "That phone number is not valid.",
    21408: "Your Twilio account is not allowed to text that country.",
    21606: "TWILIO_FROM_NUMBER is not an SMS-capable number on your Twilio account.",
    21610: "This person replied STOP to your number, so carriers will not deliver texts to them.",
    21612: "Twilio cannot send between these two numbers.",
    21614: "That number cannot receive texts (it may be a landline).",
    30003: "The phone was unreachable.",
    30005: "The number does not exist or is no longer in service.",
    30006: "That number is a landline or cannot receive texts.",
    30007: "The carrier filtered this text as spam. Insights has paused SMS so you can check your registration and wording.",
    30034: "Carriers blocked this text because the number is not registered for A2P 10DLC. Insights has paused SMS until you fix the registration.",
}


def friendly_error(code, message: str = "") -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    if code in _FRIENDLY:
        return _FRIENDLY[code]
    if code:
        return f"Twilio error {code}: {message or 'see the Twilio console'}"
    return message or "Twilio did not say what went wrong"


def _utc(value: str | None) -> str | None:
    """A Twilio RFC 2822 date as a UTC string in the outreach tables' format."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _message(item: dict) -> dict:
    """The fields of a Twilio Message resource the outreach code reads."""
    return {
        "sid": item.get("sid"),
        "from": item.get("from"),
        "to": item.get("to"),
        "body": item.get("body") or "",
        "status": item.get("status"),
        "direction": item.get("direction") or "",
        "date_sent": _utc(item.get("date_sent")),
        "date_created": _utc(item.get("date_created")),
        "error_code": item.get("error_code"),
        "error_message": item.get("error_message"),
        "num_segments": item.get("num_segments"),
    }


class TwilioClient:
    def __init__(self, account_sid: str | None = None, auth_token: str | None = None,
                 from_number: str | None = None, messaging_service_sid: str | None = None):
        self.account_sid = (account_sid if account_sid is not None
                            else os.getenv("TWILIO_ACCOUNT_SID", "")).strip()
        self.auth_token = (auth_token if auth_token is not None
                           else os.getenv("TWILIO_AUTH_TOKEN", "")).strip()
        self.from_number = (from_number if from_number is not None
                            else os.getenv("TWILIO_FROM_NUMBER", "")).strip()
        self.messaging_service_sid = (messaging_service_sid if messaging_service_sid is not None
                                      else os.getenv("TWILIO_MESSAGING_SERVICE_SID", "")).strip()

    # --- configuration -------------------------------------------------
    def can_send(self) -> bool:
        return bool(self.account_sid and self.auth_token
                    and (self.from_number or self.messaging_service_sid))

    def can_read(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number)

    def missing(self) -> list[str]:
        out = []
        if not self.account_sid:
            out.append("TWILIO_ACCOUNT_SID")
        if not self.auth_token:
            out.append("TWILIO_AUTH_TOKEN")
        if not self.from_number:
            out.append("TWILIO_FROM_NUMBER")
        return out

    @property
    def _auth(self):
        return (self.account_sid, self.auth_token)

    # --- sending -------------------------------------------------------
    def send_sms(self, to: str, body: str) -> dict:
        """Send one text. Never raises.

        Returns ``{success, sid, status, friendly, ambiguous, opted_out,
        error_code, status_code}``. ``ambiguous`` is True when the text may have
        been sent even though no success came back.
        """
        result = {"success": False, "sid": None, "status": None, "friendly": "",
                  "ambiguous": False, "opted_out": False, "error_code": None,
                  "status_code": None}
        if not self.can_send():
            result["friendly"] = "Twilio is not configured: set " + ", ".join(self.missing())
            return result
        data = {"To": to, "Body": body}
        if self.messaging_service_sid:
            data["MessagingServiceSid"] = self.messaging_service_sid
        else:
            data["From"] = self.from_number
        url = f"{TWILIO_API}/Accounts/{self.account_sid}/Messages.json"
        try:
            response = requests.post(url, data=data, auth=self._auth, timeout=TIMEOUT)
        except requests.exceptions.ConnectTimeout:
            result["friendly"] = "Could not reach Twilio (connection timed out). Nothing was sent."
            return result
        except requests.exceptions.RequestException as exc:
            # A read timeout or a connection dropped mid-request: Twilio may
            # have accepted the text and the answer was lost on the way back.
            logger.warning("Twilio send ended without an answer: %s", exc)
            result["ambiguous"] = True
            result["friendly"] = ("Twilio did not answer, so this text may or may not have gone out. "
                                  "Insights will check rather than send it again.")
            return result

        result["status_code"] = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code in (200, 201) and payload.get("sid"):
            result.update(success=True, sid=payload["sid"], status=payload.get("status"))
            return result
        if response.status_code >= 500:
            result["ambiguous"] = True
            result["friendly"] = (f"Twilio had a problem ({response.status_code}), so this text may or "
                                  "may not have gone out. Insights will check rather than send it again.")
            return result
        code = payload.get("code")
        result["error_code"] = code
        result["opted_out"] = code == ERROR_OPTED_OUT
        result["friendly"] = friendly_error(code, payload.get("message") or response.text[:200])
        return result

    # --- reading -------------------------------------------------------
    def list_messages(self, to: str | None = None, from_: str | None = None,
                      sent_on_or_after: str | None = None, max_pages: int = MAX_PAGES) -> dict:
        """Messages to or from a number, newest first. Never raises.

        ``sent_on_or_after`` is a ``YYYY-MM-DD`` date; Twilio filters by day.
        Returns ``{success, messages, friendly}``.
        """
        if not (self.account_sid and self.auth_token):
            return {"success": False, "messages": [],
                    "friendly": "Twilio is not configured: set " + ", ".join(self.missing())}
        params = {"PageSize": 100}
        if to:
            params["To"] = to
        if from_:
            params["From"] = from_
        if sent_on_or_after:
            params["DateSent>"] = sent_on_or_after
        url = f"{TWILIO_API}/Accounts/{self.account_sid}/Messages.json"
        messages = []
        for _page in range(max(1, max_pages)):
            try:
                response = requests.get(url, params=params, auth=self._auth, timeout=TIMEOUT)
            except requests.exceptions.RequestException as exc:
                return {"success": False, "messages": messages,
                        "friendly": f"Could not reach Twilio: {exc.__class__.__name__}"}
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.status_code != 200:
                return {"success": False, "messages": messages,
                        "friendly": friendly_error(payload.get("code"), payload.get("message", ""))}
            messages.extend(_message(item) for item in payload.get("messages") or [])
            next_uri = payload.get("next_page_uri")
            if not next_uri:
                break
            url, params = f"{TWILIO_API_HOST}{next_uri}", None
        return {"success": True, "messages": messages, "friendly": ""}

    def get_message(self, sid: str) -> dict:
        """One message's current state. Returns ``{success, message, friendly}``."""
        if not (self.account_sid and self.auth_token):
            return {"success": False, "message": None, "friendly": "Twilio is not configured"}
        url = f"{TWILIO_API}/Accounts/{self.account_sid}/Messages/{sid}.json"
        try:
            response = requests.get(url, auth=self._auth, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            return {"success": False, "message": None,
                    "friendly": f"Could not reach Twilio: {exc.__class__.__name__}"}
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code != 200:
            return {"success": False, "message": None,
                    "friendly": friendly_error(payload.get("code"), payload.get("message", ""))}
        return {"success": True, "message": _message(payload), "friendly": ""}


def validate_signature(url: str, params, signature: str | None, auth_token: str) -> bool:
    """Is ``signature`` (the X-Twilio-Signature header) Twilio's signature of this request?

    Twilio signs the full URL followed by every POST parameter, sorted by name,
    each written as name then value, with HMAC-SHA1 keyed by the auth token.
    """
    if not signature or not auth_token:
        return False
    if hasattr(params, "items"):
        if hasattr(params, "lists"):
            pairs = [(k, v) for k, values in params.lists() for v in values]
        else:
            pairs = list(params.items())
    else:
        pairs = list(params or [])
    payload = url + "".join(f"{k}{v}" for k, v in sorted(pairs, key=lambda kv: (kv[0], kv[1])))
    digest = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature.strip())


def get_twilio_client() -> TwilioClient:
    return TwilioClient()

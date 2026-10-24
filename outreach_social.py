"""Conversations on the owner's own social accounts: read them, and answer in-thread.

The platform clients publish; this module is the other direction. It reads the
replies, comments and mentions on the owner's recent posts (Threads, Instagram,
a Facebook Page, X), runs Threads keyword search, and sends one reply into a
conversation. Tokens come from ``social_publisher.session``, the same refresh
path publishing uses.

Every read returns ``{ok, items, state, friendly}``. ``state`` says what the
owner can do about a failure: ``needs_permission`` (the app was not granted the
engagement permission: reconnect with engagement permissions),
``needs_reconnect`` (the login itself is dead), ``rate_limited`` or ``error``.
Items are normalised to one shape whatever the platform::

    {"id", "text", "author", "author_id", "author_name", "created", "permalink",
     "parent_id", "root_id", "own"}

``reply`` is the only write here, and only ``outreach_sender`` calls it. Like
the Twilio client it reports a request whose answer was lost as ``ambiguous``,
so a reply that may already be public is never sent twice.
"""

from __future__ import annotations

import logging
import time
from datetime import timezone, datetime
from urllib.parse import quote

import requests

import social_publisher
from facebook_client import GRAPH_API_BASE as FACEBOOK_API

logger = logging.getLogger(__name__)

THREADS_API = "https://graph.threads.net/v1.0"
INSTAGRAM_API = "https://graph.instagram.com/v21.0"
X_API = "https://api.x.com/2"
TIMEOUT = (5, 20)
MAX_PAGES = 3

SOCIAL_CHANNELS = ("threads", "instagram", "facebook", "twitter")
PLATFORM_NAMES = {"threads": "Threads", "instagram": "Instagram", "facebook": "Facebook",
                  "twitter": "X"}
# The longest reply each platform takes.
REPLY_LIMITS = {"threads": 500, "instagram": 2200, "facebook": 8000, "twitter": 280}

# What "Reconnect with engagement permissions" asks for on top of the scopes a
# platform is already configured with. X needs nothing new: tweet.read and
# users.read already cover reading mentions.
ENGAGE_SCOPES = {
    "threads": ["threads_read_replies", "threads_manage_replies",
                "threads_manage_mentions", "threads_keyword_search"],
    "instagram": ["instagram_business_manage_comments"],
    "facebook": ["pages_manage_engagement", "pages_read_user_content"],
    "twitter": [],
}

_sleep = time.sleep


def merge_scopes(configured: str, extra) -> str:
    """The configured scope string with ``extra`` added once each, in order."""
    sep = " " if " " in (configured or "") and "," not in configured else ","
    scopes = [s.strip() for s in (configured or "").replace(",", " ").split() if s.strip()]
    for scope in extra or []:
        if scope not in scopes:
            scopes.append(scope)
    return sep.join(scopes)


# ── Answers ─────────────────────────────────────────────────────────────────

def _payload(response):
    try:
        return response.json()
    except ValueError:
        return {}


def classify_error(platform: str, status_code: int, payload: dict) -> tuple[str, str]:
    """``(state, friendly)`` for a refused request."""
    name = PLATFORM_NAMES.get(platform, platform)
    if platform == "twitter":
        detail = payload.get("detail") or payload.get("title") or ""
        if status_code == 401:
            return "needs_reconnect", f"{name} rejected the login. Reconnect {name}."
        if status_code == 429:
            return "rate_limited", f"{name} is limiting requests; Insights will try later."
        if status_code == 403:
            return "needs_permission", f"{name} refused this: {detail or 'not allowed for this app'}"
        return "error", f"{name} error {status_code}: {detail or 'no detail'}"
    error = payload.get("error") or {}
    if not isinstance(error, dict):
        error = {"message": str(error)}
    code = error.get("code")
    message = error.get("message") or ""
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    if code == 190 or status_code == 401:
        return "needs_reconnect", f"{name} rejected the login. Reconnect {name}."
    if code in (4, 17, 32, 613) or status_code == 429:
        return "rate_limited", f"{name} is limiting requests; Insights will try later."
    if code == 10 or (code is not None and 200 <= code < 300) or "permission" in message.lower():
        return "needs_permission", (f"{name} has not granted Insights permission for this "
                                    f"({message or 'missing permission'}). Use "
                                    f"\"Reconnect with engagement permissions\" in Setup.")
    return "error", f"{name} error{f' {code}' if code else ''}: {message or status_code}"


def _failed(platform, response) -> dict:
    state, friendly = classify_error(platform, response.status_code, _payload(response))
    return {"ok": False, "items": [], "state": state, "friendly": friendly}


def _unreachable(platform, exc) -> dict:
    return {"ok": False, "items": [], "state": "error",
            "friendly": f"Could not reach {PLATFORM_NAMES.get(platform, platform)}: "
                        f"{exc.__class__.__name__}"}


def _utc(value: str | None) -> str | None:
    """A platform timestamp (ISO 8601, maybe +0000) as a UTC string in the outreach format."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    if len(text) > 5 and text[-5] in "+-" and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _get_paged(platform, url, params, headers=None, key="data") -> dict:
    items, next_url, page_params = [], url, params
    for _page in range(MAX_PAGES):
        try:
            response = requests.get(next_url, params=page_params, headers=headers, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            return _unreachable(platform, exc) if not items else \
                {"ok": True, "items": items, "state": "ok", "friendly": ""}
        if response.status_code != 200:
            return _failed(platform, response)
        payload = _payload(response)
        items.extend(payload.get(key) or [])
        next_url = (payload.get("paging") or {}).get("next")
        if not next_url:
            break
        page_params = None  # the next link carries its own query
    return {"ok": True, "items": items, "state": "ok", "friendly": ""}


def _item(**fields) -> dict:
    out = {"id": None, "text": "", "author": "", "author_id": "", "author_name": "",
           "created": None, "permalink": None, "parent_id": None, "root_id": None, "own": False}
    out.update(fields)
    out["author"] = (out["author"] or "").lstrip("@")
    return out


# ── Threads ─────────────────────────────────────────────────────────────────

THREADS_FIELDS = "id,text,username,permalink,timestamp,replied_to,root_post,is_reply_owned_by_me"


def threads_own_posts(token: str, limit: int = 10) -> dict:
    result = _get_paged("threads", f"{THREADS_API}/me/threads", {
        "fields": "id,text,permalink,timestamp,username,has_replies", "limit": limit,
        "access_token": token})
    if result["ok"]:
        result["items"] = [_item(id=p.get("id"), text=p.get("text") or "", author=p.get("username"),
                                 created=_utc(p.get("timestamp")), permalink=p.get("permalink"),
                                 root_id=p.get("id"), own=True,
                                 has_replies=p.get("has_replies", True))
                           for p in result["items"][:limit]]
    return result


def threads_conversation(token: str, post_id: str, own_username: str = "") -> dict:
    """Every reply under one of the owner's posts, flattened, oldest first."""
    result = _get_paged("threads", f"{THREADS_API}/{post_id}/conversation", {
        "fields": THREADS_FIELDS, "reverse": "false", "access_token": token})
    if result["ok"]:
        items = []
        for r in result["items"]:
            own = bool(r.get("is_reply_owned_by_me")) or \
                (bool(own_username) and (r.get("username") or "").lower() == own_username.lower())
            items.append(_item(id=r.get("id"), text=r.get("text") or "", author=r.get("username"),
                               created=_utc(r.get("timestamp")), permalink=r.get("permalink"),
                               parent_id=(r.get("replied_to") or {}).get("id") or post_id,
                               root_id=(r.get("root_post") or {}).get("id") or post_id, own=own))
        result["items"] = items
    return result


def threads_mentions(token: str, own_username: str = "") -> dict:
    result = _get_paged("threads", f"{THREADS_API}/me/mentions", {
        "fields": "id,text,username,permalink,timestamp,root_post", "access_token": token})
    if result["ok"]:
        result["items"] = [_item(id=m.get("id"), text=m.get("text") or "", author=m.get("username"),
                                 created=_utc(m.get("timestamp")), permalink=m.get("permalink"),
                                 parent_id=m.get("id"),
                                 root_id=(m.get("root_post") or {}).get("id") or m.get("id"),
                                 own=bool(own_username) and
                                 (m.get("username") or "").lower() == own_username.lower())
                           for m in result["items"]]
    return result


def threads_keyword_search(token: str, query: str, since_unix: int | None = None,
                           limit: int = 25, own_username: str = "") -> dict:
    """Recent public posts matching ``query`` (only the owner's own until Meta approves)."""
    params = {"q": query, "search_type": "RECENT", "limit": limit,
              "fields": "id,text,username,permalink,timestamp,is_reply,has_replies",
              "access_token": token}
    if since_unix:
        params["since"] = int(since_unix)
    try:
        response = requests.get(f"{THREADS_API}/keyword_search", params=params, timeout=TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return _unreachable("threads", exc)
    if response.status_code != 200:
        return _failed("threads", response)
    items = []
    for p in _payload(response).get("data") or []:
        items.append(_item(id=p.get("id"), text=p.get("text") or "", author=p.get("username"),
                           created=_utc(p.get("timestamp")), permalink=p.get("permalink"),
                           parent_id=p.get("id"), root_id=p.get("id"),
                           own=bool(own_username) and
                           (p.get("username") or "").lower() == own_username.lower()))
    return {"ok": True, "items": items, "state": "ok", "friendly": ""}


def _threads_reply(token: str, reply_to_id: str, text: str) -> dict:
    """Create a reply container, wait for it, publish it."""
    try:
        created = requests.post(f"{THREADS_API}/me/threads", params={
            "media_type": "TEXT", "text": text, "reply_to_id": reply_to_id,
            "access_token": token}, timeout=TIMEOUT)
    except requests.exceptions.RequestException as exc:
        # Nothing is public until the publish step, so a lost container is safe.
        return {"success": False, "friendly": f"Could not reach Threads ({exc.__class__.__name__}). "
                                              "Nothing was posted."}
    if created.status_code != 200 or not _payload(created).get("id"):
        state, friendly = classify_error("threads", created.status_code, _payload(created))
        return {"success": False, "friendly": friendly, "state": state}
    container = _payload(created)["id"]
    for _attempt in range(5):
        try:
            status = requests.get(f"{THREADS_API}/{container}", params={
                "fields": "status,error_message", "access_token": token}, timeout=TIMEOUT)
            state = _payload(status).get("status")
        except requests.exceptions.RequestException:
            state = None
        if state == "FINISHED":
            break
        if state in ("ERROR", "EXPIRED"):
            return {"success": False, "friendly": "Threads could not prepare the reply. Nothing was posted."}
        _sleep(1)
    try:
        published = requests.post(f"{THREADS_API}/me/threads_publish", params={
            "creation_id": container, "access_token": token}, timeout=TIMEOUT)
    except requests.exceptions.RequestException:
        return {"success": False, "ambiguous": True,
                "friendly": "Threads did not answer, so this reply may or may not be live. "
                            "Insights will check rather than post it again."}
    if published.status_code >= 500:
        return {"success": False, "ambiguous": True,
                "friendly": "Threads had a problem, so this reply may or may not be live. "
                            "Insights will check rather than post it again."}
    if published.status_code != 200 or not _payload(published).get("id"):
        state, friendly = classify_error("threads", published.status_code, _payload(published))
        return {"success": False, "friendly": friendly, "state": state}
    reply_id = _payload(published)["id"]
    permalink = None
    try:
        info = requests.get(f"{THREADS_API}/{reply_id}", params={
            "fields": "permalink", "access_token": token}, timeout=TIMEOUT)
        permalink = _payload(info).get("permalink")
    except requests.exceptions.RequestException:
        pass
    return {"success": True, "id": reply_id, "permalink": permalink}


# ── Instagram ───────────────────────────────────────────────────────────────

def instagram_own_posts(token: str, limit: int = 10) -> dict:
    result = _get_paged("instagram", f"{INSTAGRAM_API}/me/media", {
        "fields": "id,caption,permalink,timestamp,comments_count,username", "limit": limit,
        "access_token": token})
    if result["ok"]:
        result["items"] = [_item(id=p.get("id"), text=p.get("caption") or "", author=p.get("username"),
                                 created=_utc(p.get("timestamp")), permalink=p.get("permalink"),
                                 root_id=p.get("id"), own=True,
                                 comment_count=int(p.get("comments_count") or 0))
                           for p in result["items"][:limit]]
    return result


def instagram_comments(token: str, media_id: str, own_username: str = "",
                       permalink: str | None = None) -> dict:
    """Comments on one of the owner's posts and the replies under them, flattened."""
    fields = ("id,text,username,timestamp,from{id,username},"
              "replies{id,text,username,timestamp,from{id,username}}")
    result = _get_paged("instagram", f"{INSTAGRAM_API}/{media_id}/comments", {
        "fields": fields, "access_token": token})
    if result["ok"]:
        items = []
        own = (own_username or "").lower()

        def one(c, parent):
            author = c.get("username") or (c.get("from") or {}).get("username") or ""
            return _item(id=c.get("id"), text=c.get("text") or "", author=author,
                         author_id=(c.get("from") or {}).get("id") or "",
                         created=_utc(c.get("timestamp")), permalink=permalink,
                         parent_id=parent, root_id=media_id,
                         own=bool(own) and author.lower() == own)

        for c in result["items"]:
            items.append(one(c, media_id))
            for r in (c.get("replies") or {}).get("data") or []:
                items.append(one(r, c.get("id")))
        result["items"] = items
    return result


# ── Facebook Page ───────────────────────────────────────────────────────────

def facebook_own_posts(page_token: str, page_id: str, limit: int = 10) -> dict:
    result = _get_paged("facebook", f"{FACEBOOK_API}/{page_id}/posts", {
        "fields": "id,message,permalink_url,created_time,comments.limit(0).summary(true)",
        "limit": limit, "access_token": page_token})
    if result["ok"]:
        result["items"] = [_item(id=p.get("id"), text=p.get("message") or "",
                                 created=_utc(p.get("created_time")), permalink=p.get("permalink_url"),
                                 root_id=p.get("id"), own=True,
                                 comment_count=int(((p.get("comments") or {}).get("summary") or {})
                                                   .get("total_count") or 0))
                           for p in result["items"][:limit]]
    return result


def facebook_comments(page_token: str, post_id: str, page_id: str) -> dict:
    result = _get_paged("facebook", f"{FACEBOOK_API}/{post_id}/comments", {
        "fields": "id,message,from{id,name},created_time,parent{id},permalink_url",
        "filter": "stream", "order": "chronological", "limit": 100,
        "access_token": page_token})
    if result["ok"]:
        result["items"] = [_item(id=c.get("id"), text=c.get("message") or "",
                                 author=(c.get("from") or {}).get("name") or "",
                                 author_id=(c.get("from") or {}).get("id") or "",
                                 author_name=(c.get("from") or {}).get("name") or "",
                                 created=_utc(c.get("created_time")), permalink=c.get("permalink_url"),
                                 parent_id=(c.get("parent") or {}).get("id") or post_id,
                                 root_id=post_id,
                                 own=(c.get("from") or {}).get("id") == page_id)
                           for c in result["items"]]
    return result


# ── X ───────────────────────────────────────────────────────────────────────

def x_mentions(token: str, user_id: str, since_id: str | None = None) -> dict:
    """Posts that mention the owner. Reads are billed by X, so the caller keeps this off by default."""
    params = {"tweet.fields": "author_id,created_at,conversation_id,in_reply_to_user_id",
              "expansions": "author_id", "user.fields": "username,name", "max_results": 50}
    if since_id:
        params["since_id"] = since_id
    try:
        response = requests.get(f"{X_API}/users/{user_id}/mentions", params=params,
                                headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return _unreachable("twitter", exc)
    if response.status_code != 200:
        return _failed("twitter", response)
    payload = _payload(response)
    users = {u.get("id"): u for u in (payload.get("includes") or {}).get("users") or []}
    items = []
    for t in payload.get("data") or []:
        user = users.get(t.get("author_id")) or {}
        username = user.get("username") or ""
        items.append(_item(id=t.get("id"), text=t.get("text") or "", author=username,
                           author_id=t.get("author_id") or "", author_name=user.get("name") or "",
                           created=_utc(t.get("created_at")),
                           permalink=f"https://x.com/{username or 'i/web'}/status/{t.get('id')}",
                           parent_id=t.get("id"), root_id=t.get("conversation_id") or t.get("id"),
                           own=t.get("author_id") == user_id))
    newest = (payload.get("meta") or {}).get("newest_id")
    return {"ok": True, "items": items, "state": "ok", "friendly": "", "newest_id": newest}


def x_intent_link(tweet_id: str, text: str) -> str:
    """X's own compose box, opened as a reply with the words filled in."""
    return f"https://x.com/intent/post?in_reply_to={quote(str(tweet_id))}&text={quote(text or '')}"


# ── Replying ────────────────────────────────────────────────────────────────

def _simple_reply(platform: str, url: str, params: dict) -> dict:
    try:
        response = requests.post(url, data=params, timeout=TIMEOUT)
    except requests.exceptions.ConnectTimeout as exc:
        return {"success": False, "friendly": f"Could not reach {PLATFORM_NAMES[platform]} "
                                              f"({exc.__class__.__name__}). Nothing was posted."}
    except requests.exceptions.RequestException:
        return {"success": False, "ambiguous": True,
                "friendly": f"{PLATFORM_NAMES[platform]} did not answer, so this reply may or may "
                            "not be live. Insights will check rather than post it again."}
    if response.status_code >= 500:
        return {"success": False, "ambiguous": True,
                "friendly": f"{PLATFORM_NAMES[platform]} had a problem, so this reply may or may "
                            "not be live. Insights will check rather than post it again."}
    payload = _payload(response)
    if response.status_code != 200 or not payload.get("id"):
        state, friendly = classify_error(platform, response.status_code, payload)
        return {"success": False, "friendly": friendly, "state": state}
    return {"success": True, "id": payload["id"], "permalink": None}


def _x_reply(token: str, tweet_id: str, text: str) -> dict:
    try:
        response = requests.post(f"{X_API}/tweets", json={
            "text": text, "reply": {"in_reply_to_tweet_id": str(tweet_id)}},
            headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT)
    except requests.exceptions.ConnectTimeout:
        return {"success": False, "friendly": "Could not reach X. Nothing was posted."}
    except requests.exceptions.RequestException:
        return {"success": False, "ambiguous": True,
                "friendly": "X did not answer, so this reply may or may not be live. "
                            "Insights will not post it again on its own."}
    payload = _payload(response)
    if response.status_code in (200, 201) and (payload.get("data") or {}).get("id"):
        reply_id = payload["data"]["id"]
        return {"success": True, "id": reply_id, "permalink": f"https://x.com/i/web/status/{reply_id}"}
    if response.status_code >= 500:
        return {"success": False, "ambiguous": True,
                "friendly": "X had a problem, so this reply may or may not be live. "
                            "Insights will not post it again on its own."}
    if response.status_code == 403:
        # Since February 2026 X only lets an app reply when the person mentioned
        # or quoted the account. The owner can still reply by hand.
        return {"success": False, "refused": True, "open_url": x_intent_link(tweet_id, text),
                "friendly": "X does not let apps send this reply. Open it on X: the words are "
                            "filled in, so it is one tap to post."}
    state, friendly = classify_error("twitter", response.status_code, payload)
    return {"success": False, "friendly": friendly, "state": state}


def reply(platform: str, account_id, in_reply_to: str, text: str, db_path: str | None = None) -> dict:
    """Post ``text`` as a reply to ``in_reply_to`` from one of the owner's accounts.

    Returns ``{success, id, permalink, friendly, ambiguous, refused, open_url,
    state}``. A reply with no target is refused before any request: on Threads
    a missing ``reply_to_id`` would publish the words as a new top-level post.
    """
    if platform not in SOCIAL_CHANNELS:
        return {"success": False, "friendly": f"Insights cannot reply on {platform}."}
    if not str(in_reply_to or "").strip():
        return {"success": False, "friendly": "This draft does not say which comment it answers, "
                                              "so it was not sent."}
    if not (text or "").strip():
        return {"success": False, "friendly": "The reply is empty."}
    client, token, account, failure = social_publisher.session(platform, account_id, db_path=db_path)
    if failure:
        return {"success": False, "friendly": failure.get("error") or "That account is not usable",
                "state": "needs_reconnect" if failure.get("needs_reconnect") else "error"}
    text = text[:REPLY_LIMITS[platform]]
    if platform == "threads":
        return _threads_reply(token["access_token"], in_reply_to, text)
    if platform == "instagram":
        return _simple_reply("instagram", f"{INSTAGRAM_API}/{in_reply_to}/replies",
                             {"message": text, "access_token": token["access_token"]})
    if platform == "facebook":
        return _simple_reply("facebook", f"{FACEBOOK_API}/{in_reply_to}/comments",
                             {"message": text, "access_token": token["page_access_token"]})
    return _x_reply(token["access_token"], in_reply_to, text)

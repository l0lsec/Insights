"""One publish path for every platform and every connected account.

Publishing used to be written out once per platform per entry point: the
compose "post now" buttons, the schedule page's "post now", and the background
scheduler each carried their own copy of "fetch token, refresh if expired, call
the client, read the result". Five platforms times three call sites is fifteen
places for the same decision, and fanning one piece of copy out to several
accounts multiplies that again.

This module holds that decision once. A caller names a target, which is a
platform plus one of its connected accounts, and gets back a result in the same
shape whatever the target was:

    {"success": bool, "platform": str, "account_id": int | None,
     "account_label": str, "permalink": str | None, "error": str | None,
     "needs_reconnect": bool}

``publish_targets`` runs a list of those and never lets one target's failure
stop the rest, which is what makes "post this to all five" honest: the caller
finds out exactly which targets landed and which did not.

Instagram's format handling (carousel, reel, story, people tags) needs helpers
that live in the web app, so the app registers a publisher for it with
``set_instagram_publisher``. Left unregistered, Instagram falls back to a plain
feed photo, so this module stays importable and testable on its own.
"""

from __future__ import annotations

import database
from linkedin_client import (
    get_linkedin_client,
    calculate_token_expiry as linkedin_expiry,
    is_token_expired as linkedin_expired,
)
from threads_client import (
    get_threads_client,
    calculate_token_expiry as threads_expiry,
    is_token_expired as threads_expired,
)
from facebook_client import (
    get_facebook_client,
    is_token_expired as facebook_expired,
)
from twitter_client import (
    get_twitter_client,
    calculate_token_expiry as twitter_expiry,
    is_token_expired as twitter_expired,
)
from instagram_client import (
    get_instagram_client,
    calculate_token_expiry as instagram_expiry,
    is_token_expired as instagram_expired,
)

PLATFORMS = ("linkedin", "threads", "twitter", "facebook", "instagram")

# What each platform will accept in one post. Copy is trimmed rather than
# refused: the same text goes to every target, and a card that fits LinkedIn
# should still reach X rather than failing the whole fan-out.
CHAR_LIMITS = {
    "linkedin": 3000,
    "threads": 500,
    "twitter": 280,
    "facebook": 5000,
    "instagram": 2200,
}

PLATFORM_NAMES = {
    "linkedin": "LinkedIn",
    "threads": "Threads",
    "twitter": "X/Twitter",
    "facebook": "Facebook",
    "instagram": "Instagram",
}

# Set by the web app; see the module docstring.
_instagram_publisher = None


def set_instagram_publisher(fn) -> None:
    """Register the format-aware Instagram publisher the web app provides."""
    global _instagram_publisher
    _instagram_publisher = fn


def platform_name(platform: str) -> str:
    """The platform's name as a person writes it."""
    return PLATFORM_NAMES.get(platform, (platform or "").capitalize())


def account_label(account, platform: str | None = None) -> str:
    """What to call an account on screen.

    A user-set label wins, then whatever the platform told us about the login,
    and finally the platform's own name so a single-account setup reads the way
    it did before accounts existed.
    """
    if account is None:
        return platform_name(platform or "")
    keys = account.keys() if hasattr(account, "keys") else account
    for field in ("label", "display_name", "handle"):
        if field in keys and account[field]:
            return str(account[field])
    return platform_name(account["platform"] if "platform" in keys else (platform or ""))


def account_summary(account) -> dict:
    """An account as the browser sees it."""
    return {
        "id": account["id"],
        "platform": account["platform"],
        "label": account_label(account),
        "custom_label": account["label"],
        "display_name": account["display_name"],
        "handle": account["handle"],
        "avatar_url": account["avatar_url"],
        "is_default": bool(account["is_default"]),
        "status": account["status"],
        "external_id": account["external_id"],
    }


def _result(platform, account_id, account, success, permalink=None, error=None,
            needs_reconnect=False, permanent=False) -> dict:
    """One publish outcome, in the shape every caller here reads.

    ``permanent`` marks a failure that retrying will not fix: nothing is
    connected, the account is not configured, the media is unusable. The
    scheduler retries everything else, so getting this flag wrong means either
    a post that never retries or one that retries forever.
    """
    return {
        "success": bool(success),
        "platform": platform,
        "platform_name": platform_name(platform),
        "account_id": account_id,
        "account_label": account_label(account, platform),
        "permalink": permalink,
        "error": error,
        "needs_reconnect": needs_reconnect,
        "permanent": bool(permanent or needs_reconnect),
    }


def resolve_target(platform: str, account_id=None, db_path: str | None = None):
    """Turn a (platform, account) pair into the account row that will publish.

    Returns ``(account_row, error)``. An account row of None with no error means
    the install predates accounts and still has a bare token row, which the
    token accessors resolve on their own.
    """
    db_path = db_path or database.DB_PATH
    platform = (platform or "").strip().lower()
    if platform not in PLATFORMS:
        return None, f"{platform or 'That platform'} cannot be published to"

    if account_id:
        account = database.get_social_account(int(account_id), db_path=db_path)
        if not account:
            return None, "That account is no longer connected"
        if account["platform"] != platform:
            return None, "That account belongs to a different platform"
        return account, None

    account = database.get_default_social_account(platform, db_path=db_path)
    return account, None


def publish(
    platform: str,
    account_id=None,
    *,
    content: str,
    image_url: str | None = None,
    standalone_post_id: int | None = None,
    social_post_id: int | None = None,
    article_title: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Publish one piece of copy to one account, in one normalized result.

    Every step that can fail (no token, an expired token that will not refresh,
    the platform rejecting the post) comes back as ``success: False`` with a
    readable reason rather than an exception, because a fan-out has to keep
    going after one target fails.
    """
    db_path = db_path or database.DB_PATH
    platform = (platform or "").strip().lower()
    account, error = resolve_target(platform, account_id, db_path=db_path)
    if error:
        return _result(platform, account_id, None, False, error=error, permanent=True)

    resolved_id = account["id"] if account is not None else None
    text = content or ""
    if not text.strip() and platform != "instagram":
        return _result(platform, resolved_id, account, False,
                       error="This post has no content", permanent=True)

    try:
        handler = _HANDLERS[platform]
    except KeyError:
        return _result(platform, resolved_id, account, False,
                       error=f"{platform_name(platform)} cannot be published to",
                       permanent=True)

    try:
        return handler(
            resolved_id, account, text, image_url,
            standalone_post_id, social_post_id, article_title, db_path,
        )
    except Exception as exc:  # noqa: BLE001 - a target's failure must not end the fan-out
        return _result(platform, resolved_id, account, False, error=str(exc))


def publish_targets(targets, *, content, image_url=None, db_path=None) -> list[dict]:
    """Publish the same copy to several targets, reporting each one.

    ``targets`` are dicts of ``platform`` plus optionally ``account_id``,
    ``standalone_post_id`` and ``social_post_id``. Nothing short-circuits: the
    caller gets one result per target, in order, so a partial success reads as a
    partial success instead of a single yes or no.
    """
    results = []
    for target in targets:
        results.append(publish(
            target.get("platform"),
            target.get("account_id"),
            content=target.get("content", content),
            image_url=target.get("image_url", image_url),
            standalone_post_id=target.get("standalone_post_id"),
            social_post_id=target.get("social_post_id"),
            article_title=target.get("article_title"),
            db_path=db_path,
        ))
    return results


# ---------------------------------------------------------------------------
# Per-platform publishing
#
# Each handler owns exactly one platform's token refresh and client call. They
# all take and return the same things, so publish() does not branch beyond
# picking one.
# ---------------------------------------------------------------------------


def _linkedin(account_id, account, text, image_url, standalone_id, social_id,
              article_title, db_path):
    token = database.get_linkedin_token(account_id, db_path=db_path)
    if not token:
        return _result("linkedin", account_id, account, False,
                       error="LinkedIn is not connected", needs_reconnect=True)
    if not token["user_urn"]:
        return _result("linkedin", account_id, account, False,
                       error="This LinkedIn account still needs its Member ID configured",
                       permanent=True)

    client = get_linkedin_client()
    if linkedin_expired(token["expires_at"]):
        if not token["refresh_token"]:
            return _result("linkedin", account_id, account, False,
                           error="LinkedIn token expired. Please reconnect.",
                           needs_reconnect=True)
        try:
            fresh = client.refresh_access_token(token["refresh_token"])
            database.update_linkedin_token(
                access_token=fresh["access_token"],
                expires_at=linkedin_expiry(fresh.get("expires_in", 5184000)),
                refresh_token=fresh.get("refresh_token"),
                account_id=account_id,
                db_path=db_path,
            )
            token = database.get_linkedin_token(account_id, db_path=db_path)
        except Exception as exc:  # noqa: BLE001
            return _result("linkedin", account_id, account, False,
                           error=f"LinkedIn token expired: {exc}", needs_reconnect=True)

    body = text[:CHAR_LIMITS["linkedin"]]
    if image_url and not client.extract_first_url(body):
        result = client.create_image_post(
            access_token=token["access_token"],
            author_urn=token["user_urn"],
            text=body,
            image_url=image_url,
        )
    else:
        result = client.create_smart_post(
            access_token=token["access_token"],
            author_urn=token["user_urn"],
            text=body,
            article_title=article_title,
        )
    return _from_client("linkedin", account_id, account, result, urn_key="post_urn")


def _threads(account_id, account, text, image_url, standalone_id, social_id,
             article_title, db_path):
    token = database.get_threads_token(account_id, db_path=db_path)
    if not token:
        return _result("threads", account_id, account, False,
                       error="Threads is not connected", needs_reconnect=True)

    client = get_threads_client()
    if threads_expired(token["expires_at"]):
        try:
            fresh = client.refresh_access_token(token["access_token"])
            database.update_threads_token(
                access_token=fresh["access_token"],
                expires_at=threads_expiry(fresh.get("expires_in", 5184000)),
                account_id=account_id,
                db_path=db_path,
            )
            token = database.get_threads_token(account_id, db_path=db_path)
        except Exception as exc:  # noqa: BLE001
            return _result("threads", account_id, account, False,
                           error=f"Threads token expired: {exc}", needs_reconnect=True)

    body = text[:CHAR_LIMITS["threads"]]
    if image_url:
        result = client.publish_image_post(token["access_token"], body, image_url)
    else:
        result = client.publish_text_post(token["access_token"], body)
    return _from_client("threads", account_id, account, result)


def _twitter(account_id, account, text, image_url, standalone_id, social_id,
             article_title, db_path):
    token = database.get_twitter_token(account_id, db_path=db_path)
    if not token:
        return _result("twitter", account_id, account, False,
                       error="X/Twitter is not connected", needs_reconnect=True)

    client = get_twitter_client()
    if twitter_expired(token["expires_at"]):
        if not token["refresh_token"]:
            return _result("twitter", account_id, account, False,
                           error="X/Twitter token expired. Please reconnect.",
                           needs_reconnect=True)
        try:
            fresh = client.refresh_access_token(token["refresh_token"])
            database.update_twitter_token(
                access_token=fresh["access_token"],
                expires_at=twitter_expiry(fresh.get("expires_in", 7200)),
                refresh_token=fresh.get("refresh_token"),
                account_id=account_id,
                db_path=db_path,
            )
            token = database.get_twitter_token(account_id, db_path=db_path)
        except Exception as exc:  # noqa: BLE001
            return _result("twitter", account_id, account, False,
                           error=f"X/Twitter token expired: {exc}", needs_reconnect=True)

    body = text[:CHAR_LIMITS["twitter"]]
    if image_url:
        result = client.create_image_post(
            access_token=token["access_token"], text=body, image_url=image_url,
        )
    else:
        result = client.create_post(access_token=token["access_token"], text=body)
    return _from_client("twitter", account_id, account, result)


def _facebook(account_id, account, text, image_url, standalone_id, social_id,
              article_title, db_path):
    token = database.get_facebook_token(account_id, db_path=db_path)
    if not token:
        return _result("facebook", account_id, account, False,
                       error="Facebook is not connected", needs_reconnect=True)
    if not token["page_id"] or not token["page_access_token"]:
        return _result("facebook", account_id, account, False,
                       error="This Facebook account still needs a Page selected",
                       permanent=True)
    if facebook_expired(token["expires_at"]) and account_id:
        # Page tokens outlive the user token they came from, so an expired user
        # token is worth flagging on the accounts screen but not worth refusing
        # the post over.
        database.set_social_account_status(account_id, "expired", db_path=db_path)

    client = get_facebook_client()
    result = client.publish_smart_post(
        page_access_token=token["page_access_token"],
        page_id=token["page_id"],
        text=text[:CHAR_LIMITS["facebook"]],
        image_url=image_url,
    )
    return _from_client("facebook", account_id, account, result)


def _instagram(account_id, account, text, image_url, standalone_id, social_id,
               article_title, db_path):
    token = database.get_instagram_token(account_id, db_path=db_path)
    if not token:
        return _result("instagram", account_id, account, False,
                       error="Instagram is not connected", needs_reconnect=True)

    client = get_instagram_client()
    if instagram_expired(token["expires_at"]):
        try:
            fresh = client.refresh_access_token(token["access_token"])
            database.update_instagram_token(
                access_token=fresh["access_token"],
                expires_at=instagram_expiry(fresh.get("expires_in", 5184000)),
                account_id=account_id,
                db_path=db_path,
            )
            token = database.get_instagram_token(account_id, db_path=db_path)
        except Exception as exc:  # noqa: BLE001
            return _result("instagram", account_id, account, False,
                           error=f"Instagram token expired: {exc}", needs_reconnect=True)

    caption = text[:CHAR_LIMITS["instagram"]]
    if _instagram_publisher is not None:
        result = _instagram_publisher(
            token["access_token"],
            content=caption,
            image_url=image_url,
            standalone_post_id=standalone_id,
            social_post_id=social_id,
        )
    elif image_url:
        result = client.publish_image_post(token["access_token"], caption, image_url)
    else:
        return _result("instagram", account_id, account, False,
                       error="Instagram posts need an image or video", permanent=True)
    return _from_client("instagram", account_id, account, result)


_HANDLERS = {
    "linkedin": _linkedin,
    "threads": _threads,
    "twitter": _twitter,
    "facebook": _facebook,
    "instagram": _instagram,
}


def _from_client(platform, account_id, account, result, urn_key="permalink") -> dict:
    """Read a client's reply into the shape every caller here expects.

    The five clients disagree on what they return, so this is the one place that
    knows a LinkedIn success carries a post URN while the rest carry a permalink,
    and that Instagram attaches a friendlier message than its raw error.
    """
    if not result:
        return _result(platform, account_id, account, False,
                       error=f"{platform_name(platform)} gave no response")
    if result.get("success"):
        permalink = result.get("permalink") or result.get(urn_key) or result.get("post_urn")
        return _result(platform, account_id, account, True, permalink=permalink)

    error = result.get("friendly")
    if not error:
        raw = result.get("error")
        if isinstance(raw, dict):
            error = raw.get("message") or str(raw)
        else:
            error = raw or "Unknown error"
    # Instagram's media guard rejects a post the platform was never going to
    # accept, so retrying it just burns slots.
    return _result(platform, account_id, account, False, error=str(error),
                   permanent=bool(result.get("guard_error")))

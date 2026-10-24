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

import logging
from datetime import datetime

import database
from linkedin_client import (
    COMMENT_MAX_CHARS as LINKEDIN_COMMENT_MAX_CHARS,
    get_linkedin_client,
    calculate_token_expiry as linkedin_expiry,
    is_token_expired as linkedin_expired,
    split_link_for_comment,
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

logger = logging.getLogger(__name__)

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
        "link_in_comment": bool(
            account["link_in_comment"] if "link_in_comment" in account.keys() else 0),
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
    video_url: str | None = None,
    link_in_comment: bool | None = None,
) -> dict:
    """Publish one piece of copy to one account, in one normalized result.

    A ``video_url`` makes it a video post on every platform (Instagram sends it
    as a Reel) and the image, if any, is ignored. A video that cannot be
    delivered fails the post; it is never quietly swapped for a text or image
    post, because the copy was written to go with the video.

    ``link_in_comment`` is LinkedIn's alone, because LinkedIn holds back posts
    that link out. True takes the post's links out of the copy and posts them as
    its first comment, False keeps them in the copy, and None (what every caller
    passes) lets the saved post's own choice and then the account's decide. A
    comment LinkedIn refuses does not fail the post: it reports ``first_comment``
    on the result and keeps what a retry needs (see ``retry_first_comment``).

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

    # Only LinkedIn takes it, and only when a caller names one, so every other
    # platform (and a LinkedIn post nobody overrides) is called exactly as before.
    extra = {"link_in_comment": link_in_comment} \
        if platform == "linkedin" and link_in_comment is not None else {}

    try:
        return handler(
            resolved_id, account, text, image_url,
            standalone_post_id, social_post_id, article_title, db_path,
            video_url, **extra,
        )
    except Exception as exc:  # noqa: BLE001 - a target's failure must not end the fan-out
        return _result(platform, resolved_id, account, False, error=str(exc))


def publish_targets(targets, *, content, image_url=None, db_path=None, video_url=None) -> list[dict]:
    """Publish the same copy to several targets, reporting each one.

    ``targets`` are dicts of ``platform`` plus optionally ``account_id``,
    ``standalone_post_id``, ``social_post_id`` and a per-target ``video_url``. Nothing short-circuits: the
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
            video_url=target.get("video_url", video_url),
            standalone_post_id=target.get("standalone_post_id"),
            social_post_id=target.get("social_post_id"),
            article_title=target.get("article_title"),
            db_path=db_path,
            link_in_comment=target.get("link_in_comment"),
        ))
    return results


# ---------------------------------------------------------------------------
# Per-platform publishing
#
# Each handler owns exactly one platform's token refresh and client call. They
# all take and return the same things, so publish() does not branch beyond
# picking one.
# ---------------------------------------------------------------------------


def _linkedin_session(account_id, account, db_path):
    """The LinkedIn client and a usable token for an account.

    Returns ``(client, token, failure)``; ``failure`` is a ready-made result when
    there is no connection, no Member ID, or an expired token that will not
    refresh. Posting and retrying a comment need exactly the same thing, so they
    share this rather than carrying two copies of the token refresh.
    """
    token = database.get_linkedin_token(account_id, db_path=db_path)
    if not token:
        return None, None, _result("linkedin", account_id, account, False,
                                   error="LinkedIn is not connected", needs_reconnect=True)
    if not token["user_urn"]:
        return None, None, _result(
            "linkedin", account_id, account, False,
            error="This LinkedIn account still needs its Member ID configured",
            permanent=True)

    client = get_linkedin_client()
    if linkedin_expired(token["expires_at"]):
        if not token["refresh_token"]:
            return None, None, _result("linkedin", account_id, account, False,
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
            return None, None, _result("linkedin", account_id, account, False,
                                       error=f"LinkedIn token expired: {exc}",
                                       needs_reconnect=True)
    return client, token, None


def _link_comment_wanted(override, account, standalone_id, db_path) -> bool:
    """Whether this LinkedIn post sends its link as the first comment.

    A caller's explicit answer wins; otherwise the saved post's own choice, then
    its account's, then no (see ``database.link_in_comment_enabled``).
    """
    if override is not None:
        return bool(override)
    row = database.get_standalone_post(standalone_id, db_path=db_path) if standalone_id else None
    return database.link_in_comment_enabled(row, account)


def _linkedin(account_id, account, text, image_url, standalone_id, social_id,
              article_title, db_path, video_url=None, link_in_comment=None):
    # A post whose link is going to the comment is checked before anything is
    # sent: a refusal after the post is live would leave it without its link.
    comment_text = None
    if _link_comment_wanted(link_in_comment, account, standalone_id, db_path):
        text, links = split_link_for_comment(text)
        if links:
            comment_text = "\n".join(links)
            if len(comment_text) > LINKEDIN_COMMENT_MAX_CHARS:
                return _result(
                    "linkedin", account_id, account, False, permanent=True,
                    error=(f"The link comment would be {len(comment_text)} characters and "
                           f"LinkedIn allows {LINKEDIN_COMMENT_MAX_CHARS}. Remove some of the links."))
            if not text.strip():
                # LinkedIn posts here always carry words (publish() refuses an
                # empty one), and a link-only post would be left with none.
                return _result(
                    "linkedin", account_id, account, False, permanent=True,
                    error=("Nothing is left to post once the link moves to the comment. "
                           "Add some words to the post."))

    client, token, failure = _linkedin_session(account_id, account, db_path)
    if failure:
        return failure

    body = text[:CHAR_LIMITS["linkedin"]]
    if video_url:
        result = client.create_video_post(
            access_token=token["access_token"],
            author_urn=token["user_urn"],
            text=body,
            video_url=video_url,
        )
    elif image_url and not client.extract_first_url(body):
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
    outcome = _from_client("linkedin", account_id, account, result, urn_key="post_urn")

    if comment_text and outcome["success"]:
        # The post is live. Whatever happens to the comment is reported beside
        # that fact and never instead of it: an exception here would make
        # publish() call the post failed, and the scheduler would post it again.
        outcome["first_comment"] = _post_first_comment(
            client, token, comment_text, result.get("post_urn"), standalone_id, db_path)
    return outcome


def _post_first_comment(client, token, comment_text, post_urn, standalone_id, db_path) -> dict:
    """Comment on a live post and keep the record. Never raises.

    Returns the note ``{"status": "posted" | "failed", "text", "error",
    "comment_id", "post_urn", "at"}``. It is also saved on the post's row, which
    is what lets a failed comment be retried later, on the same post, with the
    same words, without trusting the copy to be unchanged since.
    """
    note = {"status": "failed", "text": comment_text, "error": None,
            "comment_id": None, "post_urn": post_urn,
            "at": datetime.utcnow().isoformat(timespec="seconds")}
    if not post_urn:
        note["error"] = ("LinkedIn did not say which post this is, so the link could "
                         "not be added as a comment. Add it by hand.")
    else:
        try:
            reply = client.create_comment(
                access_token=token["access_token"],
                actor_urn=token["user_urn"],
                post_urn=post_urn,
                text=comment_text,
            )
        except Exception as exc:  # noqa: BLE001
            reply = {"success": False, "friendly": f"The link comment failed: {exc}"}
        if reply and reply.get("success"):
            note["status"] = "posted"
            note["comment_id"] = reply.get("comment_id")
        else:
            raw = (reply or {}).get("error")
            note["error"] = str(
                (reply or {}).get("friendly")
                or (raw.get("message") if isinstance(raw, dict) else raw)
                or "LinkedIn gave no reason"
            )
    if note["status"] == "failed":
        logger.warning("LinkedIn link comment failed for %s: %s", post_urn, note["error"])

    if standalone_id:
        try:
            database.set_standalone_post_link_comment_state(
                standalone_id, note, db_path=db_path)
        except Exception:  # noqa: BLE001 - the record is a convenience, the post is live
            logger.exception("Could not save the link comment record for post %s", standalone_id)
    return note


def retry_first_comment(standalone_id, *, db_path: str | None = None) -> dict:
    """Post again a link comment LinkedIn refused, on the post that is live.

    Only the comment is sent; the post is never touched. It uses the words and
    the post id saved when the comment failed and the token of the account the
    post was published as. The reply is a normal publish result (``success`` is
    whether the comment landed) with ``first_comment`` carrying the new note, and
    a post with nothing to retry says so instead of commenting.
    """
    db_path = db_path or database.DB_PATH
    row = database.get_standalone_post(standalone_id, db_path=db_path)
    state = database.link_comment_state(row)
    if not state or state.get("status") != "failed":
        return _result("linkedin", None, None, False, permanent=True,
                       error="There is no failed link comment to retry for this post")
    if not state.get("post_urn") or not state.get("text"):
        return _result("linkedin", None, None, False, permanent=True,
                       error="The post's id was not recorded, so the comment cannot be "
                             "retried. Add the link to the post by hand.")

    account, error = resolve_target("linkedin", row["account_id"], db_path=db_path)
    if error:
        return _result("linkedin", row["account_id"], None, False, error=error, permanent=True)
    account_id = account["id"] if account is not None else None

    client, token, failure = _linkedin_session(account_id, account, db_path)
    if failure:
        return failure

    note = _post_first_comment(
        client, token, state["text"], state["post_urn"], standalone_id, db_path)
    outcome = _result("linkedin", account_id, account, note["status"] == "posted",
                      error=note["error"])
    outcome["first_comment"] = note
    return outcome


def _threads_session(account_id, account, db_path):
    """The Threads client and a usable token for an account; see ``_linkedin_session``."""
    token = database.get_threads_token(account_id, db_path=db_path)
    if not token:
        return None, None, _result("threads", account_id, account, False,
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
            return None, None, _result("threads", account_id, account, False,
                                       error=f"Threads token expired: {exc}",
                                       needs_reconnect=True)
    return client, token, None


def _threads(account_id, account, text, image_url, standalone_id, social_id,
             article_title, db_path, video_url=None):
    client, token, failure = _threads_session(account_id, account, db_path)
    if failure:
        return failure

    body = text[:CHAR_LIMITS["threads"]]
    if video_url:
        result = client.publish_video_post(token["access_token"], body, video_url)
    elif image_url:
        result = client.publish_image_post(token["access_token"], body, image_url)
    else:
        result = client.publish_text_post(token["access_token"], body)
    return _from_client("threads", account_id, account, result)


def _twitter_session(account_id, account, db_path):
    """The X client and a usable token for an account; see ``_linkedin_session``."""
    token = database.get_twitter_token(account_id, db_path=db_path)
    if not token:
        return None, None, _result("twitter", account_id, account, False,
                                   error="X/Twitter is not connected", needs_reconnect=True)

    client = get_twitter_client()
    if twitter_expired(token["expires_at"]):
        if not token["refresh_token"]:
            return None, None, _result("twitter", account_id, account, False,
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
            return None, None, _result("twitter", account_id, account, False,
                                       error=f"X/Twitter token expired: {exc}",
                                       needs_reconnect=True)
    return client, token, None


def _twitter(account_id, account, text, image_url, standalone_id, social_id,
             article_title, db_path, video_url=None):
    client, token, failure = _twitter_session(account_id, account, db_path)
    if failure:
        return failure

    body = text[:CHAR_LIMITS["twitter"]]
    if video_url:
        result = client.create_video_post(
            access_token=token["access_token"], text=body, video_url=video_url,
        )
    elif image_url:
        result = client.create_image_post(
            access_token=token["access_token"], text=body, image_url=image_url,
        )
    else:
        result = client.create_post(access_token=token["access_token"], text=body)
    return _from_client("twitter", account_id, account, result)


def _facebook_session(account_id, account, db_path):
    """The Facebook client and the Page token row for an account; see ``_linkedin_session``."""
    token = database.get_facebook_token(account_id, db_path=db_path)
    if not token:
        return None, None, _result("facebook", account_id, account, False,
                                   error="Facebook is not connected", needs_reconnect=True)
    if not token["page_id"] or not token["page_access_token"]:
        return None, None, _result("facebook", account_id, account, False,
                                   error="This Facebook account still needs a Page selected",
                                   permanent=True)
    if facebook_expired(token["expires_at"]) and account_id:
        # Page tokens outlive the user token they came from, so an expired user
        # token is worth flagging on the accounts screen but not worth refusing
        # the post over.
        database.set_social_account_status(account_id, "expired", db_path=db_path)
    return get_facebook_client(), token, None


def _facebook(account_id, account, text, image_url, standalone_id, social_id,
              article_title, db_path, video_url=None):
    client, token, failure = _facebook_session(account_id, account, db_path)
    if failure:
        return failure

    extra = {"video_url": video_url} if video_url else {}
    result = client.publish_smart_post(
        page_access_token=token["page_access_token"],
        page_id=token["page_id"],
        text=text[:CHAR_LIMITS["facebook"]],
        image_url=image_url,
        **extra,
    )
    return _from_client("facebook", account_id, account, result)


def _instagram_session(account_id, account, db_path):
    """The Instagram client and a usable token for an account; see ``_linkedin_session``."""
    token = database.get_instagram_token(account_id, db_path=db_path)
    if not token:
        return None, None, _result("instagram", account_id, account, False,
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
            return None, None, _result("instagram", account_id, account, False,
                                       error=f"Instagram token expired: {exc}",
                                       needs_reconnect=True)
    return client, token, None


def _instagram(account_id, account, text, image_url, standalone_id, social_id,
               article_title, db_path, video_url=None):
    client, token, failure = _instagram_session(account_id, account, db_path)
    if failure:
        return failure

    caption = text[:CHAR_LIMITS["instagram"]]
    if _instagram_publisher is not None:
        extra = {"video_url": video_url} if video_url else {}
        result = _instagram_publisher(
            token["access_token"],
            content=caption,
            image_url=image_url,
            standalone_post_id=standalone_id,
            social_post_id=social_id,
            **extra,
        )
    elif video_url:
        result = client.publish_reel_post(token["access_token"], caption, video_url)
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


_SESSIONS = {
    "linkedin": _linkedin_session,
    "threads": _threads_session,
    "twitter": _twitter_session,
    "facebook": _facebook_session,
    "instagram": _instagram_session,
}


def session(platform: str, account_id=None, *, db_path: str | None = None):
    """A platform client and a usable token for one account, for any caller.

    Publishing and the outreach listener both need "this account's token,
    refreshed if it has expired", so they share these sessions instead of each
    carrying a copy of the refresh. Returns ``(client, token, account, failure)``;
    ``failure`` is a publish-shaped result (``needs_reconnect`` set when only
    reconnecting will fix it) and the rest are None when it is set.
    """
    db_path = db_path or database.DB_PATH
    platform = (platform or "").strip().lower()
    account, error = resolve_target(platform, account_id, db_path=db_path)
    if error:
        return None, None, None, _result(platform, account_id, None, False, error=error,
                                         permanent=True)
    resolved_id = account["id"] if account is not None else None
    client, token, failure = _SESSIONS[platform](resolved_id, account, db_path)
    if failure:
        return None, None, account, failure
    return client, token, account, None


def _from_client(platform, account_id, account, result, urn_key="permalink") -> dict:
    """Read a client's reply into the shape every caller here expects.

    The five clients disagree on what they return, so this is the one place that
    knows a LinkedIn success carries a post URN while the rest carry a permalink,
    that Instagram attaches a friendlier message than its raw error, and that a
    client can flag a failure as one only reconnecting the account will fix.
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
    # accept, so retrying it just burns slots. A video the platform refused, or
    # one X refused for a missing grant, is the same kind of failure.
    return _result(platform, account_id, account, False, error=str(error),
                   needs_reconnect=bool(result.get("needs_reconnect")),
                   permanent=bool(result.get("guard_error")))

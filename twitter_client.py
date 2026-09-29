"""X/Twitter API client for OAuth 2.0 PKCE and posting functionality."""

from __future__ import annotations

import os
import hashlib
import base64
import logging
import secrets
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Optional

import requests

import video_media

logger = logging.getLogger(__name__)

# Seam for tests: a gate replaces this so status polling does not really wait.
_sleep = time.sleep

# X API v2 endpoints
TWITTER_AUTH_URL = "https://x.com/i/oauth2/authorize"
TWITTER_TOKEN_URL = "https://api.x.com/2/oauth2/token"
TWITTER_USERS_ME_URL = "https://api.x.com/2/users/me"
TWITTER_TWEETS_URL = "https://api.x.com/2/tweets"
TWITTER_MEDIA_UPLOAD_URL = "https://api.x.com/2/media/upload"

# media.write is what X's v2 media upload endpoint requires; without it an
# uploaded image or video is refused, so it has to be part of the grant.
TWITTER_SCOPES = os.environ.get(
    "TWITTER_SCOPES",
    "tweet.read tweet.write users.read media.write offline.access"
)

# X takes a video in segments of at most 5 MB; stay a little under.
VIDEO_SEGMENT_BYTES = 4 * 1024 * 1024
# 2s * 180 = six minutes of waiting for X to finish processing a video.
VIDEO_POLL_ATTEMPTS = 180
VIDEO_POLL_MAX_WAIT = 15


class MediaScopeError(video_media.VideoError):
    """X refused the media upload because the connected login lacks media.write."""

    def __init__(self, message: str):
        super().__init__(message, permanent=True)


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge pair.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    code_verifier = secrets.token_urlsafe(96)[:128]
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


class TwitterClient:
    """Client for interacting with the X/Twitter API v2."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ):
        self.client_id = client_id or os.environ.get("TWITTER_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("TWITTER_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.environ.get(
            "TWITTER_REDIRECT_URI", "http://localhost:5001/twitter/callback"
        )

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def get_authorization_url(self, state: str | None = None) -> tuple[str, str, str]:
        """Generate the OAuth 2.0 PKCE authorization URL.

        Returns:
            Tuple of (authorization_url, state_token, code_verifier)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        code_verifier, code_challenge = _generate_pkce_pair()

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": TWITTER_SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        url = f"{TWITTER_AUTH_URL}?{urlencode(params)}"
        return url, state, code_verifier

    def exchange_code_for_token(self, code: str, code_verifier: str) -> dict:
        """Exchange authorization code for access token using PKCE.

        Args:
            code: The authorization code from OAuth callback
            code_verifier: The PKCE code verifier stored during auth initiation

        Returns:
            Dict containing access_token, refresh_token, expires_in, etc.
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }

        response = requests.post(
            TWITTER_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Refresh an expired access token.

        Args:
            refresh_token: The refresh token

        Returns:
            Dict containing new access_token, refresh_token, expires_in, etc.
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

        response = requests.post(
            TWITTER_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_user_info(self, access_token: str) -> dict | None:
        """Get the authenticated user's profile info.

        Args:
            access_token: Valid X/Twitter access token

        Returns:
            Dict containing id, username, name, or None on failure
        """
        try:
            response = requests.get(
                TWITTER_USERS_ME_URL,
                params={"user.fields": "id,username,name,profile_image_url"},
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                return data
            else:
                logger.error(
                    "Failed to get Twitter user info: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None
        except Exception as e:
            logger.error("Error getting Twitter user info: %s", e)
            return None

    def create_post(
        self,
        access_token: str,
        text: str,
        media_ids: list[str] | None = None,
    ) -> dict:
        """Create a post (tweet) on X/Twitter.

        Args:
            access_token: Valid X/Twitter access token
            text: The post content (max 280 characters)
            media_ids: Optional list of media IDs to attach

        Returns:
            Dict with success status and post details
        """
        if len(text) > 280:
            text = text[:277] + "..."
            logger.warning("Twitter post truncated to 280 characters")

        payload: dict = {"text": text}
        if media_ids:
            payload["media"] = {"media_ids": media_ids}

        try:
            response = requests.post(
                TWITTER_TWEETS_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )

            if response.status_code in (200, 201):
                data = response.json().get("data", {})
                tweet_id = data.get("id", "")
                return {
                    "success": True,
                    "tweet_id": tweet_id,
                    "text": data.get("text", ""),
                    "permalink": f"https://x.com/i/web/status/{tweet_id}" if tweet_id else None,
                    "status_code": response.status_code,
                }
            else:
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"raw": response.text}

                logger.error(
                    "Twitter post failed: %s - %s",
                    response.status_code,
                    error_data,
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_data,
                }

        except requests.RequestException as e:
            logger.error("Twitter API request failed: %s", e)
            return {
                "success": False,
                "error": {"message": str(e)},
            }

    def upload_media(self, access_token: str, image_url: str) -> str | None:
        """Upload an image to X/Twitter via chunked media upload.

        Downloads the image from the URL, then uses the 3-step chunked upload:
        1. Initialize upload
        2. Append binary data
        3. Finalize upload

        Args:
            access_token: Valid X/Twitter access token
            image_url: URL of the image to upload

        Returns:
            The media_id string if successful, None otherwise
        """
        auth_headers = {"Authorization": f"Bearer {access_token}"}

        try:
            # Download the image
            logger.info("Downloading image from: %s", image_url)
            dl_headers = {
                "User-Agent": "Mozilla/5.0 (compatible; Insights/1.0)"
            }
            img_response = requests.get(
                image_url, headers=dl_headers, timeout=30, allow_redirects=True
            )
            img_response.raise_for_status()
            image_data = img_response.content
            content_type = img_response.headers.get("Content-Type", "image/jpeg")

            total_bytes = len(image_data)
            if total_bytes > 5 * 1024 * 1024:
                logger.warning("Image too large for Twitter (%d bytes), skipping upload", total_bytes)
                return None

            # Step 1: Initialize
            logger.info("Initializing Twitter media upload (%d bytes)", total_bytes)
            init_payload = {
                "media_type": content_type,
                "total_bytes": total_bytes,
                "media_category": "tweet_image",
            }
            init_response = requests.post(
                f"{TWITTER_MEDIA_UPLOAD_URL}/initialize",
                json=init_payload,
                headers={**auth_headers, "Content-Type": "application/json"},
                timeout=30,
            )

            if init_response.status_code not in (200, 201, 202):
                logger.error(
                    "Twitter media init failed: %s - %s",
                    init_response.status_code,
                    init_response.text,
                )
                return None

            media_id = _media_id(init_response.json())
            init_data = init_response.json()
            if not media_id:
                logger.error("No media_id in init response: %s", init_data)
                return None

            media_id = str(media_id)

            # Step 2: Append (single chunk for images <= 5MB)
            logger.info("Appending image data to media %s", media_id)
            append_response = requests.post(
                f"{TWITTER_MEDIA_UPLOAD_URL}/{media_id}/append",
                files={"media": ("image", image_data, content_type)},
                data={"segment_index": "0"},
                headers=auth_headers,
                timeout=60,
            )

            if append_response.status_code not in (200, 201, 202, 204):
                logger.error(
                    "Twitter media append failed: %s - %s",
                    append_response.status_code,
                    append_response.text,
                )
                return None

            # Step 3: Finalize
            logger.info("Finalizing media upload %s", media_id)
            finalize_response = requests.post(
                f"{TWITTER_MEDIA_UPLOAD_URL}/{media_id}/finalize",
                headers={**auth_headers, "Content-Type": "application/json"},
                timeout=30,
            )

            if finalize_response.status_code not in (200, 201, 202):
                logger.error(
                    "Twitter media finalize failed: %s - %s",
                    finalize_response.status_code,
                    finalize_response.text,
                )
                return None

            logger.info("Twitter media upload complete: %s", media_id)
            return media_id

        except requests.RequestException as e:
            logger.error("Error uploading media to Twitter: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error uploading media to Twitter: %s", e)
            return None

    def create_image_post(
        self,
        access_token: str,
        text: str,
        image_url: str,
    ) -> dict:
        """Create a post with an image on X/Twitter.

        Uploads the image first, then creates the tweet with the media attached.
        Falls back to text-only if the image upload fails.

        Args:
            access_token: Valid X/Twitter access token
            text: The post content (max 280 characters)
            image_url: URL of the image to attach

        Returns:
            Dict with success status and post details
        """
        logger.info("Uploading image for Twitter post: %s", image_url)
        media_id = self.upload_media(access_token, image_url)

        if not media_id:
            logger.warning("Failed to upload image, falling back to text-only post")
            return self.create_post(access_token=access_token, text=text)

        return self.create_post(
            access_token=access_token,
            text=text,
            media_ids=[media_id],
        )


    def upload_video(self, access_token: str, video_url: str) -> str:
        """Upload a video through X's chunked media upload and return its media id.

        Streams the file from disk in segments of at most 5 MB (initialize,
        append each segment, finalize), then waits while X processes it. The
        image upload above reads the whole file into memory, which is fine for
        5 MB and not for a video.

        Raises ``video_media.VideoError`` with a readable reason on any failure
        (``MediaScopeError`` when the login lacks the media.write grant).
        """
        auth = {"Authorization": f"Bearer {access_token}"}
        limits = video_media.PLATFORM_LIMITS["twitter"]
        with video_media.downloaded(video_url, max_bytes=limits["max_bytes"]) as (path, size):
            init = requests.post(
                f"{TWITTER_MEDIA_UPLOAD_URL}/initialize",
                json={
                    "media_type": _video_mime(video_url),
                    "total_bytes": size,
                    "media_category": "tweet_video",
                },
                headers={**auth, "Content-Type": "application/json"},
                timeout=30,
            )
            if init.status_code in (401, 403):
                raise MediaScopeError(
                    f"X refused the video upload ({init.status_code}). Reconnect X so it "
                    "can grant the media.write permission, then try again."
                )
            if init.status_code not in (200, 201, 202):
                raise video_media.VideoError(
                    f"X would not start the video upload ({init.status_code}): "
                    f"{_error_text(init)}",
                    permanent=init.status_code in (400, 401),
                )
            media_id = _media_id(init.json())
            if not media_id:
                raise video_media.VideoError(
                    "X's upload reply had no media id", permanent=False,
                )

            with open(path, "rb") as source:
                index = 0
                while True:
                    chunk = source.read(VIDEO_SEGMENT_BYTES)
                    if not chunk:
                        break
                    append = requests.post(
                        f"{TWITTER_MEDIA_UPLOAD_URL}/{media_id}/append",
                        files={"media": ("video", chunk, "application/octet-stream")},
                        data={"segment_index": str(index)},
                        headers=auth,
                        timeout=120,
                    )
                    if append.status_code not in (200, 201, 202, 204):
                        raise video_media.VideoError(
                            f"X rejected segment {index + 1} of the video "
                            f"({append.status_code}): {_error_text(append)}",
                            permanent=False,
                        )
                    index += 1

            final = requests.post(
                f"{TWITTER_MEDIA_UPLOAD_URL}/{media_id}/finalize",
                headers={**auth, "Content-Type": "application/json"},
                timeout=60,
            )
            if final.status_code not in (200, 201, 202):
                raise video_media.VideoError(
                    f"X could not finalize the video ({final.status_code}): "
                    f"{_error_text(final)}",
                    permanent=False,
                )
            info = (final.json().get("data") or {}).get("processing_info")

        self._wait_for_media(access_token, media_id, info)
        return media_id

    def _wait_for_media(self, access_token: str, media_id: str, info: dict | None) -> None:
        """Poll X's media STATUS until processing succeeds, or raise why not.

        ``info`` is the ``processing_info`` finalize returned; None means X had
        nothing left to do and the media is already usable.
        """
        state = (info or {}).get("state")
        for _attempt in range(VIDEO_POLL_ATTEMPTS):
            if not info or state == "succeeded":
                return
            if state == "failed":
                detail = (info.get("error") or {}).get("message") or "unsupported file"
                raise video_media.VideoError(
                    f"X could not process the video: {detail}", permanent=True,
                )
            _sleep(min(max(int(info.get("check_after_secs") or 1), 1), VIDEO_POLL_MAX_WAIT))
            response = requests.get(
                TWITTER_MEDIA_UPLOAD_URL,
                params={"command": "STATUS", "media_id": media_id},
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=30,
            )
            if response.status_code == 200:
                info = (response.json().get("data") or {}).get("processing_info")
                state = (info or {}).get("state")
        raise video_media.VideoError(
            "X was still processing the video after waiting; try again shortly",
            permanent=False,
        )

    def create_video_post(
        self,
        access_token: str,
        text: str,
        video_url: str,
    ) -> dict:
        """Create a post with a video on X/Twitter.

        Unlike ``create_image_post`` this never falls back to a text post when
        the upload fails: the copy was written to go with a video, so the
        failure is returned and nothing is posted.
        """
        try:
            media_id = self.upload_video(access_token, video_url)
        except MediaScopeError as exc:
            logger.error("X video upload lacks media.write: %s", exc)
            result = video_media.failure(str(exc), permanent=True)
            result["needs_reconnect"] = True
            return result
        except video_media.VideoError as exc:
            logger.error("X video upload failed: %s", exc)
            return video_media.failure(str(exc), permanent=exc.permanent)
        except requests.RequestException as exc:
            logger.error("X video upload request failed: %s", exc)
            return video_media.failure(f"X video upload failed: {exc}")

        return self.create_post(access_token=access_token, text=text, media_ids=[media_id])


def _media_id(data: dict) -> str | None:
    """The media id from an upload reply, whichever shape carries it.

    The v2 endpoint answers ``{"data": {"id": ...}}``; older replies put
    ``media_id`` at the top or under ``data``.
    """
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    found = inner.get("id") or inner.get("media_id") or data.get("media_id") or data.get("id")
    return str(found) if found else None


def _video_mime(video_url: str) -> str:
    path = video_url.split("?", 1)[0].lower()
    return "video/quicktime" if path.endswith(".mov") else "video/mp4"


def _error_text(response: requests.Response) -> str:
    """A platform error as one short line, for a message a person will read."""
    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}
    return str(data.get("detail") or data.get("title") or data.get("raw") or data)[:200]


def get_twitter_client() -> TwitterClient:
    """Factory function to get a configured Twitter client."""
    return TwitterClient()


def calculate_token_expiry(expires_in: int) -> str:
    """Calculate the token expiry datetime as ISO string.

    Args:
        expires_in: Seconds until token expires

    Returns:
        ISO format datetime string
    """
    expiry = datetime.utcnow() + timedelta(seconds=expires_in)
    return expiry.isoformat(timespec="seconds")


def is_token_expired(expires_at: str | None, buffer_minutes: int = 5) -> bool:
    """Check if a token is expired or about to expire.

    Args:
        expires_at: ISO format datetime string of expiry
        buffer_minutes: Consider expired if within this many minutes

    Returns:
        True if token is expired or will expire soon
    """
    if not expires_at:
        return True

    try:
        expiry = datetime.fromisoformat(expires_at)
        buffer = timedelta(minutes=buffer_minutes)
        return datetime.utcnow() >= (expiry - buffer)
    except (ValueError, TypeError):
        return True

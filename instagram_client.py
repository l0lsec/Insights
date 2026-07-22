"""Instagram API client for OAuth and posting functionality.

Uses the "Instagram API with Instagram Login" flavor (no Facebook Page link
required). Requires an Instagram professional (Business or Creator) account.
Feed posts REQUIRE an image — Instagram has no text-only posts.
"""

from __future__ import annotations

import json
import os
import logging
import secrets
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Instagram API endpoints
INSTAGRAM_AUTH_HOST = "https://www.instagram.com"   # authorize UI
INSTAGRAM_TOKEN_HOST = "https://api.instagram.com"  # code -> short-lived token
INSTAGRAM_API_HOST = "https://graph.instagram.com"  # everything else

# OAuth scopes needed for posting
# instagram_business_basic: Read profile info
# instagram_business_content_publish: Post content
INSTAGRAM_SCOPES = os.environ.get(
    "INSTAGRAM_SCOPES",
    "instagram_business_basic,instagram_business_content_publish"
)

# Instagram caption limit (feed posts)
INSTAGRAM_CAPTION_LIMIT = 2200

# Container poll retry counts (2s interval). Video (reels / video stories /
# video carousel items) processes much slower than images.
IMAGE_MAX_RETRIES = 45      # ~90s
VIDEO_MAX_RETRIES = 120     # ~4 min

# Long-lived tokens last ~60 days
DEFAULT_TOKEN_LIFETIME_SECONDS = 5184000

# Known error subcodes from the content publishing API
_ERROR_SUBCODE_MESSAGES = {
    2207003: (
        "Instagram could not download the image (URL unreachable or too slow). "
        "The image must be a publicly accessible HTTPS URL."
    ),
    2207009: (
        "Image aspect ratio rejected. Instagram requires between 4:5 "
        "(portrait) and 1.91:1 (landscape)."
    ),
    2207026: "Unsupported image format. Instagram feed photos must be JPEG.",
    2207042: "Instagram publishing rate limit reached (max API posts per 24 hours).",
}
_RATE_LIMIT_CODES = {4, 9, 17}


def _friendly_error(error_data: dict) -> str:
    """Map an Instagram API error payload to a human-readable message."""
    if not isinstance(error_data, dict):
        return str(error_data)
    err = error_data.get("error", error_data)
    if not isinstance(err, dict):
        return str(err)
    subcode = err.get("error_subcode")
    if subcode in _ERROR_SUBCODE_MESSAGES:
        return _ERROR_SUBCODE_MESSAGES[subcode]
    if err.get("code") in _RATE_LIMIT_CODES:
        return "Instagram publishing rate limit reached (max API posts per 24 hours)."
    return err.get("error_user_msg") or err.get("message") or str(error_data)


class InstagramClient:
    """Client for interacting with the Instagram API (Instagram Login flavor)."""

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        redirect_uri: str | None = None,
    ):
        """Initialize the Instagram client with credentials from env or params."""
        self.app_id = app_id or os.environ.get("INSTAGRAM_APP_ID")
        self.app_secret = app_secret or os.environ.get("INSTAGRAM_APP_SECRET")
        self.redirect_uri = redirect_uri or os.environ.get(
            "INSTAGRAM_REDIRECT_URI", "https://localhost:5001/instagram/callback"
        )

    def is_configured(self) -> bool:
        """Check if Instagram credentials are configured."""
        return bool(self.app_id and self.app_secret)

    def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
        """Generate the OAuth authorization URL.

        Returns:
            Tuple of (authorization_url, state_token)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.app_id,
            "redirect_uri": self.redirect_uri,
            "scope": INSTAGRAM_SCOPES,
            "response_type": "code",
            "state": state,
        }
        url = f"{INSTAGRAM_AUTH_HOST}/oauth/authorize?{urlencode(params)}"
        return url, state

    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for short-lived access token.

        Args:
            code: The authorization code from OAuth callback

        Returns:
            Dict containing access_token and user_id
        """
        params = {
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        }

        response = requests.post(
            f"{INSTAGRAM_TOKEN_HOST}/oauth/access_token",
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        # Instagram Business Login may wrap the payload: {"data": [{...}]}
        if isinstance(data.get("data"), list) and data["data"]:
            return data["data"][0]
        return data

    def get_long_lived_token(self, short_lived_token: str) -> dict:
        """Exchange a short-lived token for a long-lived token (60 days).

        Args:
            short_lived_token: The short-lived access token

        Returns:
            Dict containing access_token, token_type, and expires_in
        """
        params = {
            "grant_type": "ig_exchange_token",
            "client_secret": self.app_secret,
            "access_token": short_lived_token,
        }

        response = requests.get(
            f"{INSTAGRAM_API_HOST}/access_token",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, access_token: str) -> dict:
        """Refresh an unexpired long-lived access token.

        Only works on tokens that are at least 24 hours old and not yet
        expired; if the token has already expired the user must reconnect.

        Args:
            access_token: The current long-lived access token

        Returns:
            Dict containing new access_token and expires_in
        """
        params = {
            "grant_type": "ig_refresh_token",
            "access_token": access_token,
        }

        response = requests.get(
            f"{INSTAGRAM_API_HOST}/refresh_access_token",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_user_profile(self, access_token: str) -> dict | None:
        """Get the authenticated user's profile info.

        Args:
            access_token: Valid Instagram access token

        Returns:
            Dict containing id, user_id (IG professional account ID), username,
            name, account_type (BUSINESS/MEDIA_CREATOR), profile_picture_url
        """
        params = {
            "fields": "user_id,username,name,account_type,profile_picture_url",
            "access_token": access_token,
        }

        try:
            response = requests.get(
                f"{INSTAGRAM_API_HOST}/me",
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    "Failed to get Instagram profile: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None
        except Exception as e:
            logger.error("Error getting Instagram profile: %s", e)
            return None

    @staticmethod
    def _truncate_caption(caption: str) -> str:
        """Truncate a caption to the Instagram limit, keeping an ellipsis."""
        if caption and len(caption) > INSTAGRAM_CAPTION_LIMIT:
            logger.warning("Instagram caption truncated to %d characters", INSTAGRAM_CAPTION_LIMIT)
            return caption[:INSTAGRAM_CAPTION_LIMIT - 3] + "..."
        return caption

    def _create_container(self, params: dict) -> tuple[Optional[str], Optional[dict]]:
        """POST /me/media to create a media container.

        Returns (container_id, None) on success, or (None, error_result) where
        error_result is the standard failure dict.
        """
        try:
            response = requests.post(
                f"{INSTAGRAM_API_HOST}/me/media",
                params=params,
                timeout=30,
            )
        except requests.RequestException as e:
            logger.error("Instagram container request failed: %s", e)
            return None, {
                "success": False,
                "error": {"message": str(e)},
                "friendly": f"Instagram API request failed: {e}",
            }

        if response.status_code != 200:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"raw": response.text}
            logger.error(
                "Instagram container creation failed: %s - %s",
                response.status_code,
                error_data,
            )
            return None, {
                "success": False,
                "status_code": response.status_code,
                "error": error_data,
                "friendly": _friendly_error(error_data),
            }

        container_id = response.json().get("id")
        if not container_id:
            return None, {
                "success": False,
                "error": {"message": "No container ID returned"},
                "friendly": "Instagram did not return a media container ID.",
            }
        return container_id, None

    def _poll_and_publish(
        self,
        container_id: str,
        access_token: str,
        *,
        max_retries: int = IMAGE_MAX_RETRIES,
        poll_interval: float = 2.0,
    ) -> dict:
        """Poll a container until FINISHED, then publish it.

        Shared by every media type (feed image, carousel, reel, story). Returns
        the standard result dict: success ->
        {success, post_id, shortcode, permalink, status_code}; failure ->
        {success: False, status_code?, error, friendly}.
        """
        try:
            # Step 2: Poll container status until FINISHED. Instagram fetches and
            # processes the remote media, which can take a while (longer for video).
            for attempt in range(max_retries):
                status_response = requests.get(
                    f"{INSTAGRAM_API_HOST}/{container_id}",
                    params={"fields": "status_code,status", "access_token": access_token},
                    timeout=10,
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    container_status = status_data.get("status_code")

                    if container_status == "FINISHED":
                        logger.debug("Instagram container %s is ready (attempt %d)", container_id, attempt + 1)
                        break
                    elif container_status == "ERROR":
                        # The human-readable detail lives in the status field
                        error_msg = status_data.get("status", "Media container processing failed")
                        logger.error("Instagram container %s failed: %s", container_id, error_msg)
                        return {
                            "success": False,
                            "error": {"message": error_msg},
                            "friendly": (
                                "Instagram rejected the media: %s. Check the format, aspect "
                                "ratio, and (for video) duration/codec." % error_msg
                            ),
                        }
                    elif container_status == "EXPIRED":
                        logger.error("Instagram container %s expired", container_id)
                        return {
                            "success": False,
                            "error": {"message": "Container expired before publishing"},
                            "friendly": "The Instagram media container expired before publishing.",
                        }
                    elif container_status == "PUBLISHED":
                        logger.warning("Instagram container %s already published", container_id)
                        return {
                            "success": False,
                            "error": {"message": "Container already published"},
                            "friendly": "This media container was already published.",
                        }
                    else:
                        # IN_PROGRESS or other status, wait and retry
                        logger.debug("Instagram container %s status: %s, waiting...", container_id, container_status)
                        time.sleep(poll_interval)
                else:
                    logger.warning("Failed to check Instagram container status: %s", status_response.status_code)
                    time.sleep(poll_interval)
            else:
                # Exhausted retries — do NOT attempt a blind publish
                logger.error("Instagram container %s not ready after %d attempts", container_id, max_retries)
                return {
                    "success": False,
                    "error": {"message": "Media processing timed out"},
                    "friendly": "Instagram media processing timed out. Try again or use a smaller/shorter file.",
                }

            # Step 3: Publish the container
            publish_response = requests.post(
                f"{INSTAGRAM_API_HOST}/me/media_publish",
                params={"creation_id": container_id, "access_token": access_token},
                timeout=30,
            )

            if publish_response.status_code == 200:
                post_id = publish_response.json().get("id")

                # Fetch post details to get permalink and shortcode
                permalink = None
                shortcode = None
                if post_id:
                    try:
                        details_response = requests.get(
                            f"{INSTAGRAM_API_HOST}/{post_id}",
                            params={"fields": "permalink,shortcode", "access_token": access_token},
                            timeout=10,
                        )
                        if details_response.status_code == 200:
                            details_data = details_response.json()
                            permalink = details_data.get("permalink")
                            shortcode = details_data.get("shortcode")
                    except Exception as e:
                        logger.warning("Failed to fetch Instagram post details: %s", e)

                return {
                    "success": True,
                    "post_id": post_id,
                    "shortcode": shortcode,
                    "permalink": permalink,
                    "status_code": publish_response.status_code,
                }
            else:
                error_data = {}
                try:
                    error_data = publish_response.json()
                except Exception:
                    error_data = {"raw": publish_response.text}
                logger.error(
                    "Instagram publish failed: %s - %s",
                    publish_response.status_code,
                    error_data,
                )
                return {
                    "success": False,
                    "status_code": publish_response.status_code,
                    "error": error_data,
                    "friendly": _friendly_error(error_data),
                }

        except requests.RequestException as e:
            logger.error("Instagram API request failed: %s", e)
            return {
                "success": False,
                "error": {"message": str(e)},
                "friendly": f"Instagram API request failed: {e}",
            }

    def publish_image_post(
        self,
        access_token: str,
        caption: str,
        image_url: str,
        user_tags: Optional[list] = None,
    ) -> dict:
        """Publish a single-image feed post to Instagram.

        Instagram has no text-only posts, so image_url is required. The image
        must be a JPEG on a publicly accessible URL with aspect ratio between
        4:5 and 1.91:1.

        Args:
            access_token: Valid Instagram access token
            caption: The post caption (max 2200 characters)
            image_url: Public URL of the image to post
            user_tags: Optional list of {"username": str, "x": float, "y": float}
                people-tags (x/y are 0..1 positions on the photo). Feed photos
                only — the API ignores/rejects tags elsewhere.

        Returns:
            Dict with success status and post details
        """
        if not image_url:
            return {
                "success": False,
                "error": {"message": "Instagram posts require an image"},
                "friendly": "Instagram posts require an image.",
            }

        caption = self._truncate_caption(caption)
        logger.info("Creating Instagram image container with image: %s", image_url)
        params = {
            "caption": caption,
            "image_url": image_url,
            "access_token": access_token,
        }
        if user_tags:
            clean_tags = []
            for tag in user_tags:
                username = (tag or {}).get("username", "").strip().lstrip("@")
                if not username:
                    continue
                clean_tags.append({
                    "username": username,
                    "x": min(max(float(tag.get("x", 0.5)), 0.0), 1.0),
                    "y": min(max(float(tag.get("y", 0.5)), 0.0), 1.0),
                })
            if clean_tags:
                params["user_tags"] = json.dumps(clean_tags)
        container_id, error = self._create_container(params)
        if error:
            return error
        return self._poll_and_publish(container_id, access_token)

    def publish_carousel_post(
        self,
        access_token: str,
        caption: str,
        media_items: list,
    ) -> dict:
        """Publish a carousel (2-10 images and/or videos) feed post.

        Args:
            access_token: Valid Instagram access token
            caption: The post caption (max 2200 characters)
            media_items: list of {"url": str, "kind": "image"|"video"}

        Returns:
            Dict with success status and post details
        """
        items = media_items or []
        if not (2 <= len(items) <= 10):
            return {
                "success": False,
                "error": {"message": "carousel needs 2-10 items"},
                "friendly": "Instagram carousels need between 2 and 10 items.",
            }

        caption = self._truncate_caption(caption)

        # Step 1a: create a child container per item
        child_ids = []
        has_video = False
        for item in items:
            url = (item or {}).get("url")
            kind = (item or {}).get("kind", "image")
            if not url:
                return {
                    "success": False,
                    "error": {"message": "carousel item missing url"},
                    "friendly": "A carousel item is missing its media URL.",
                }
            if kind == "video":
                has_video = True
                params = {
                    "media_type": "VIDEO",
                    "video_url": url,
                    "is_carousel_item": "true",
                    "access_token": access_token,
                }
            else:
                params = {
                    "image_url": url,
                    "is_carousel_item": "true",
                    "access_token": access_token,
                }
            logger.info("Creating Instagram carousel child container (%s)", kind)
            container_id, error = self._create_container(params)
            if error:
                return error
            child_ids.append(container_id)

        # Step 1b: create the parent carousel container
        parent_id, error = self._create_container({
            "media_type": "CAROUSEL",
            "children": ",".join(child_ids),
            "caption": caption,
            "access_token": access_token,
        })
        if error:
            return error

        # Polling the parent covers child (video) processing — the parent will not
        # reach FINISHED until all children are ready.
        max_retries = VIDEO_MAX_RETRIES if has_video else IMAGE_MAX_RETRIES
        return self._poll_and_publish(parent_id, access_token, max_retries=max_retries)

    def publish_reel_post(
        self,
        access_token: str,
        caption: str,
        video_url: str,
        share_to_feed: bool = True,
    ) -> dict:
        """Publish a Reel (single video) to Instagram.

        Args:
            access_token: Valid Instagram access token
            caption: The post caption (max 2200 characters)
            video_url: Public URL of the video (MP4/MOV, H.264/AAC, ~90s)
            share_to_feed: Also show the reel on the main feed grid

        Returns:
            Dict with success status and post details
        """
        if not video_url:
            return {
                "success": False,
                "error": {"message": "reel requires a video"},
                "friendly": "Instagram Reels require a video.",
            }

        caption = self._truncate_caption(caption)
        logger.info("Creating Instagram reel container with video: %s", video_url)
        container_id, error = self._create_container({
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true" if share_to_feed else "false",
            "access_token": access_token,
        })
        if error:
            return error
        return self._poll_and_publish(container_id, access_token, max_retries=VIDEO_MAX_RETRIES)

    def publish_story_post(
        self,
        access_token: str,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
    ) -> dict:
        """Publish a Story (one image or one video) to Instagram.

        Stories have no caption via the Content Publishing API.

        Args:
            access_token: Valid Instagram access token
            image_url: Public URL of the image (mutually exclusive with video_url)
            video_url: Public URL of the video (mutually exclusive with image_url)

        Returns:
            Dict with success status and post details
        """
        if bool(image_url) == bool(video_url):
            return {
                "success": False,
                "error": {"message": "story requires exactly one of image_url/video_url"},
                "friendly": "A story needs exactly one image or one video.",
            }

        params = {"media_type": "STORIES", "access_token": access_token}
        if image_url:
            params["image_url"] = image_url
            max_retries = IMAGE_MAX_RETRIES
        else:
            params["video_url"] = video_url
            max_retries = VIDEO_MAX_RETRIES

        logger.info("Creating Instagram story container")
        container_id, error = self._create_container(params)
        if error:
            return error
        return self._poll_and_publish(container_id, access_token, max_retries=max_retries)

    def get_publishing_limit(self, access_token: str) -> dict | None:
        """Check the user's content publishing rate limit.

        Args:
            access_token: Valid Instagram access token

        Returns:
            Dict with quota_usage and config, or None on error
        """
        params = {
            "fields": "quota_usage,config",
            "access_token": access_token,
        }

        try:
            response = requests.get(
                f"{INSTAGRAM_API_HOST}/me/content_publishing_limit",
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                # Response is wrapped: {"data": [{"quota_usage": ..., "config": {...}}]}
                if isinstance(data.get("data"), list) and data["data"]:
                    return data["data"][0]
                return data
            else:
                logger.error(
                    "Failed to get Instagram publishing limit: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None
        except Exception as e:
            logger.error("Error getting Instagram publishing limit: %s", e)
            return None


def get_instagram_client() -> InstagramClient:
    """Factory function to get a configured Instagram client."""
    return InstagramClient()


def calculate_token_expiry(expires_in: int) -> str:
    """Calculate the token expiry datetime as ISO string.

    Args:
        expires_in: Seconds until token expires

    Returns:
        ISO format datetime string
    """
    expiry = datetime.utcnow() + timedelta(seconds=expires_in)
    return expiry.isoformat(timespec="seconds")


def is_token_expired(expires_at: str | None, buffer_minutes: int = 60) -> bool:
    """Check if a token is expired or about to expire.

    Args:
        expires_at: ISO format datetime string of expiry
        buffer_minutes: Consider expired if within this many minutes (default 1 hour)

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

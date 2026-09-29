"""Threads API client for OAuth and posting functionality."""

from __future__ import annotations

import os
import logging
import secrets
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Seam for tests: a gate replaces this so container polling does not really wait.
_sleep = time.sleep

# Threads processes a video for a while after the container is created (its own
# guidance is "wait about 30 seconds"), so a video is polled far longer than the
# 30 one-second checks an image gets: 2s * 150 = five minutes.
VIDEO_POLL_INTERVAL = 2.0
VIDEO_POLL_ATTEMPTS = 150

# Threads API endpoints
THREADS_AUTH_HOST = "https://www.threads.net"
THREADS_API_HOST = "https://graph.threads.net"

# OAuth scopes needed for posting
# threads_basic: Read profile info
# threads_content_publish: Post content
THREADS_SCOPES = os.environ.get(
    "THREADS_SCOPES",
    "threads_basic,threads_content_publish"
)


class ThreadsClient:
    """Client for interacting with Threads API."""

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        redirect_uri: str | None = None,
    ):
        """Initialize the Threads client with credentials from env or params."""
        self.app_id = app_id or os.environ.get("THREADS_APP_ID")
        self.app_secret = app_secret or os.environ.get("THREADS_APP_SECRET")
        self.redirect_uri = redirect_uri or os.environ.get(
            "THREADS_REDIRECT_URI", "https://localhost:5001/threads/callback"
        )

    def is_configured(self) -> bool:
        """Check if Threads credentials are configured."""
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
            "scope": THREADS_SCOPES,
            "response_type": "code",
            "state": state,
        }
        url = f"{THREADS_AUTH_HOST}/oauth/authorize?{urlencode(params)}"
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
            f"{THREADS_API_HOST}/oauth/access_token",
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_long_lived_token(self, short_lived_token: str) -> dict:
        """Exchange a short-lived token for a long-lived token (60 days).

        Args:
            short_lived_token: The short-lived access token

        Returns:
            Dict containing access_token, token_type, and expires_in
        """
        params = {
            "grant_type": "th_exchange_token",
            "client_secret": self.app_secret,
            "access_token": short_lived_token,
        }

        response = requests.get(
            f"{THREADS_API_HOST}/access_token",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, access_token: str) -> dict:
        """Refresh an unexpired long-lived access token.

        Args:
            access_token: The current long-lived access token

        Returns:
            Dict containing new access_token and expires_in
        """
        params = {
            "grant_type": "th_refresh_token",
            "access_token": access_token,
        }

        response = requests.get(
            f"{THREADS_API_HOST}/refresh_access_token",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_user_profile(self, access_token: str) -> dict | None:
        """Get the authenticated user's profile info.

        Args:
            access_token: Valid Threads access token

        Returns:
            Dict containing id, username, name, profile picture, bio
        """
        params = {
            "fields": "id,username,name,threads_profile_picture_url,threads_biography",
            "access_token": access_token,
        }

        try:
            response = requests.get(
                f"{THREADS_API_HOST}/me",
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    "Failed to get Threads profile: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None
        except Exception as e:
            logger.error("Error getting Threads profile: %s", e)
            return None

    def publish_text_post(
        self,
        access_token: str,
        text: str,
        reply_control: str = "everyone",
    ) -> dict:
        """Publish a text post to Threads.

        Uses auto_publish_text=true for single-request publishing.

        Args:
            access_token: Valid Threads access token
            text: The post content (max 500 characters)
            reply_control: Who can reply - "everyone", "accounts_you_follow", "mentioned_only"

        Returns:
            Dict with success status and post details
        """
        # Truncate text if needed (Threads has 500 char limit)
        if len(text) > 500:
            text = text[:497] + "..."
            logger.warning("Threads post truncated to 500 characters")

        params = {
            "text": text,
            "media_type": "TEXT",
            "access_token": access_token,
        }

        # Note: auto_publish_text is no longer supported as of recent API versions
        # We need to use the two-step process: create container, then publish

        try:
            # Step 1: Create media container
            response = requests.post(
                f"{THREADS_API_HOST}/me/threads",
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"raw": response.text}
                logger.error(
                    "Threads container creation failed: %s - %s",
                    response.status_code,
                    error_data,
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_data,
                }

            container_data = response.json()
            container_id = container_data.get("id")

            if not container_id:
                return {
                    "success": False,
                    "error": {"message": "No container ID returned"},
                }

            # Step 2: Poll container status until FINISHED
            # The Threads API needs time to process the container before publishing
            import time
            max_retries = 10
            poll_interval = 0.5  # seconds
            
            for attempt in range(max_retries):
                status_params = {
                    "fields": "status,error_message",
                    "access_token": access_token,
                }
                status_response = requests.get(
                    f"{THREADS_API_HOST}/{container_id}",
                    params=status_params,
                    timeout=10,
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    container_status = status_data.get("status")
                    
                    if container_status == "FINISHED":
                        logger.debug("Container %s is ready (attempt %d)", container_id, attempt + 1)
                        break
                    elif container_status == "ERROR":
                        error_msg = status_data.get("error_message", "Container processing failed")
                        logger.error("Container %s failed: %s", container_id, error_msg)
                        return {
                            "success": False,
                            "error": {"message": error_msg},
                        }
                    elif container_status == "EXPIRED":
                        logger.error("Container %s expired", container_id)
                        return {
                            "success": False,
                            "error": {"message": "Container expired before publishing"},
                        }
                    elif container_status == "PUBLISHED":
                        logger.warning("Container %s already published", container_id)
                        return {
                            "success": False,
                            "error": {"message": "Container already published"},
                        }
                    else:
                        # IN_PROGRESS or other status, wait and retry
                        logger.debug("Container %s status: %s, waiting...", container_id, container_status)
                        time.sleep(poll_interval)
                else:
                    logger.warning("Failed to check container status: %s", status_response.status_code)
                    time.sleep(poll_interval)
            else:
                # Exhausted retries
                logger.warning("Container %s not ready after %d attempts, attempting publish anyway", 
                             container_id, max_retries)

            # Step 3: Publish the container
            publish_params = {
                "creation_id": container_id,
                "access_token": access_token,
            }

            publish_response = requests.post(
                f"{THREADS_API_HOST}/me/threads_publish",
                params=publish_params,
                timeout=30,
            )

            if publish_response.status_code == 200:
                publish_data = publish_response.json()
                post_id = publish_data.get("id")
                
                # Fetch post details to get permalink and shortcode
                permalink = None
                shortcode = None
                if post_id:
                    try:
                        details_params = {
                            "fields": "permalink,shortcode",
                            "access_token": access_token,
                        }
                        details_response = requests.get(
                            f"{THREADS_API_HOST}/{post_id}",
                            params=details_params,
                            timeout=10,
                        )
                        if details_response.status_code == 200:
                            details_data = details_response.json()
                            permalink = details_data.get("permalink")
                            shortcode = details_data.get("shortcode")
                    except Exception as e:
                        logger.warning("Failed to fetch post details: %s", e)
                
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
                    "Threads publish failed: %s - %s",
                    publish_response.status_code,
                    error_data,
                )
                return {
                    "success": False,
                    "status_code": publish_response.status_code,
                    "error": error_data,
                }

        except requests.RequestException as e:
            logger.error("Threads API request failed: %s", e)
            return {
                "success": False,
                "error": {"message": str(e)},
            }

    def publish_text_post_with_link(
        self,
        access_token: str,
        text: str,
        link_url: str | None = None,
    ) -> dict:
        """Publish a text post with an optional link attachment.

        Args:
            access_token: Valid Threads access token
            text: The post content
            link_url: Optional URL to attach as link preview

        Returns:
            Dict with success status and post details
        """
        # For now, include the link in the text if provided
        # Threads will auto-detect and create a link preview
        if link_url and link_url not in text:
            full_text = f"{text}\n\n{link_url}"
        else:
            full_text = text

        return self.publish_text_post(access_token, full_text)

    def publish_image_post(
        self,
        access_token: str,
        text: str,
        image_url: str,
        reply_control: str = "everyone",
    ) -> dict:
        """Publish an image post to Threads.

        Args:
            access_token: Valid Threads access token
            text: The post content/caption (max 500 characters)
            image_url: Public URL of the image to post (must be accessible by Threads servers)
            reply_control: Who can reply - "everyone", "accounts_you_follow", "mentioned_only"

        Returns:
            Dict with success status and post details
        """
        # Truncate text if needed (Threads has 500 char limit)
        if len(text) > 500:
            text = text[:497] + "..."
            logger.warning("Threads post truncated to 500 characters")

        logger.info("Creating Threads image container with image: %s", image_url)
        return self._publish_media_container(
            access_token,
            {
                "text": text,
                "media_type": "IMAGE",
                "image_url": image_url,
                "access_token": access_token,
            },
            label="Image",
            # Image containers take longer to process than text
            max_retries=30,
            poll_interval=1.0,
        )

    def publish_video_post(
        self,
        access_token: str,
        text: str,
        video_url: str,
        reply_control: str = "everyone",
    ) -> dict:
        """Publish a video post to Threads.

        Threads fetches the video itself, so ``video_url`` must be publicly
        reachable. Videos must be MP4 or MOV, up to 5 minutes and 1 GB.

        Returns:
            Dict with success status and post details. A video that Threads
            cannot process is a failure; there is no text-only fallback.
        """
        if len(text) > 500:
            text = text[:497] + "..."
            logger.warning("Threads post truncated to 500 characters")

        logger.info("Creating Threads video container with video: %s", video_url)
        return self._publish_media_container(
            access_token,
            {
                "text": text,
                "media_type": "VIDEO",
                "video_url": video_url,
                "access_token": access_token,
            },
            label="Video",
            max_retries=VIDEO_POLL_ATTEMPTS,
            poll_interval=VIDEO_POLL_INTERVAL,
        )

    def _publish_media_container(
        self,
        access_token: str,
        params: dict,
        *,
        label: str,
        max_retries: int,
        poll_interval: float,
    ) -> dict:
        """Create a media container, wait for FINISHED, then publish it.

        The three-step Threads flow shared by image and video posts. ``label``
        only words the error messages; ``max_retries`` and ``poll_interval``
        are what differ, because a video takes far longer to process.
        """
        label_lower = label.lower()
        try:
            # Step 1: Create media container
            response = requests.post(
                f"{THREADS_API_HOST}/me/threads",
                params=params,
                timeout=30,
            )

            if response.status_code != 200:
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"raw": response.text}
                logger.error(
                    "Threads %s container creation failed: %s - %s",
                    label_lower,
                    response.status_code,
                    error_data,
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_data,
                }

            container_data = response.json()
            container_id = container_data.get("id")

            if not container_id:
                return {
                    "success": False,
                    "error": {"message": "No container ID returned"},
                }

            # Step 2: Poll container status until FINISHED
            for attempt in range(max_retries):
                status_params = {
                    "fields": "status,error_message",
                    "access_token": access_token,
                }
                status_response = requests.get(
                    f"{THREADS_API_HOST}/{container_id}",
                    params=status_params,
                    timeout=10,
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    container_status = status_data.get("status")

                    if container_status == "FINISHED":
                        logger.debug("%s container %s is ready (attempt %d)", label, container_id, attempt + 1)
                        break
                    elif container_status == "ERROR":
                        error_msg = status_data.get("error_message", f"{label} container processing failed")
                        logger.error("%s container %s failed: %s", label, container_id, error_msg)
                        return {
                            "success": False,
                            "error": {"message": error_msg},
                            # A file Threads has rejected will be rejected again.
                            "guard_error": label == "Video",
                        }
                    elif container_status == "EXPIRED":
                        logger.error("%s container %s expired", label, container_id)
                        return {
                            "success": False,
                            "error": {"message": "Container expired before publishing"},
                        }
                    elif container_status == "PUBLISHED":
                        logger.warning("%s container %s already published", label, container_id)
                        return {
                            "success": False,
                            "error": {"message": "Container already published"},
                        }
                    else:
                        # IN_PROGRESS or other status, wait and retry
                        logger.debug("%s container %s status: %s, waiting...", label, container_id, container_status)
                        _sleep(poll_interval)
                else:
                    logger.warning("Failed to check %s container status: %s", label_lower, status_response.status_code)
                    _sleep(poll_interval)
            else:
                # Exhausted retries
                logger.error("%s container %s not ready after %d attempts", label, container_id, max_retries)
                return {
                    "success": False,
                    "error": {"message": f"{label} processing timed out"},
                }

            # Step 3: Publish the container
            publish_params = {
                "creation_id": container_id,
                "access_token": access_token,
            }

            publish_response = requests.post(
                f"{THREADS_API_HOST}/me/threads_publish",
                params=publish_params,
                timeout=30,
            )

            if publish_response.status_code == 200:
                publish_data = publish_response.json()
                post_id = publish_data.get("id")

                # Fetch post details to get permalink and shortcode
                permalink = None
                shortcode = None
                if post_id:
                    try:
                        details_params = {
                            "fields": "permalink,shortcode",
                            "access_token": access_token,
                        }
                        details_response = requests.get(
                            f"{THREADS_API_HOST}/{post_id}",
                            params=details_params,
                            timeout=10,
                        )
                        if details_response.status_code == 200:
                            details_data = details_response.json()
                            permalink = details_data.get("permalink")
                            shortcode = details_data.get("shortcode")
                    except Exception as e:
                        logger.warning("Failed to fetch post details: %s", e)

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
                    "Threads %s publish failed: %s - %s",
                    label_lower,
                    publish_response.status_code,
                    error_data,
                )
                return {
                    "success": False,
                    "status_code": publish_response.status_code,
                    "error": error_data,
                }

        except requests.RequestException as e:
            logger.error("Threads %s API request failed: %s", label_lower, e)
            return {
                "success": False,
                "error": {"message": str(e)},
            }

    def get_publishing_limit(self, access_token: str) -> dict | None:
        """Check the user's publishing rate limit.

        Args:
            access_token: Valid Threads access token

        Returns:
            Dict with quota_usage and config, or None on error
        """
        params = {
            "fields": "quota_usage,config",
            "access_token": access_token,
        }

        try:
            response = requests.get(
                f"{THREADS_API_HOST}/me/threads_publishing_limit",
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    "Failed to get publishing limit: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None
        except Exception as e:
            logger.error("Error getting publishing limit: %s", e)
            return None


def get_threads_client() -> ThreadsClient:
    """Factory function to get a configured Threads client."""
    return ThreadsClient()


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

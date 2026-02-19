"""Facebook Graph API client for OAuth and posting functionality.

Supports posting to Facebook Pages and Groups via the Graph API v21.0.
Personal profile posting is not supported (deprecated by Facebook in 2018).
"""

from __future__ import annotations

import os
import logging
import secrets
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
FACEBOOK_OAUTH_URL = f"https://www.facebook.com/{GRAPH_API_VERSION}/dialog/oauth"

FACEBOOK_SCOPES = os.environ.get(
    "FACEBOOK_SCOPES",
    "pages_manage_posts,pages_read_engagement,pages_show_list,publish_to_groups",
)


class FacebookClient:
    """Client for interacting with the Facebook Graph API."""

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        redirect_uri: str | None = None,
    ):
        self.app_id = app_id or os.environ.get("FACEBOOK_APP_ID")
        self.app_secret = app_secret or os.environ.get("FACEBOOK_APP_SECRET")
        self.redirect_uri = redirect_uri or os.environ.get(
            "FACEBOOK_REDIRECT_URI", "http://localhost:5001/facebook/callback"
        )

    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    # ------------------------------------------------------------------
    # OAuth helpers
    # ------------------------------------------------------------------

    def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.app_id,
            "redirect_uri": self.redirect_uri,
            "scope": FACEBOOK_SCOPES,
            "response_type": "code",
            "state": state,
        }
        url = f"{FACEBOOK_OAUTH_URL}?{urlencode(params)}"
        return url, state

    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange an authorization code for a short-lived user access token."""
        params = {
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "redirect_uri": self.redirect_uri,
            "code": code,
        }
        response = requests.get(
            f"{GRAPH_API_BASE}/oauth/access_token",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_long_lived_token(self, short_lived_token: str) -> dict:
        """Exchange a short-lived token for one valid ~60 days."""
        params = {
            "grant_type": "fb_exchange_token",
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "fb_exchange_token": short_lived_token,
        }
        response = requests.get(
            f"{GRAPH_API_BASE}/oauth/access_token",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, access_token: str) -> dict:
        """Refresh a long-lived token (returns a new long-lived token).

        Facebook long-lived tokens can be refreshed by exchanging them again
        before they expire.
        """
        return self.get_long_lived_token(access_token)

    # ------------------------------------------------------------------
    # User / Page / Group discovery
    # ------------------------------------------------------------------

    def get_user_profile(self, access_token: str) -> dict | None:
        """Fetch the authenticated user's basic profile."""
        try:
            response = requests.get(
                f"{GRAPH_API_BASE}/me",
                params={"fields": "id,name,picture", "access_token": access_token},
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            logger.error("Facebook profile fetch failed: %s - %s", response.status_code, response.text)
            return None
        except Exception as e:
            logger.error("Error fetching Facebook profile: %s", e)
            return None

    def get_user_pages(self, access_token: str) -> list[dict]:
        """Return Pages the user manages, each with its own page access token.

        Each dict contains: id, name, access_token, category.
        """
        pages: list[dict] = []
        try:
            url = f"{GRAPH_API_BASE}/me/accounts"
            params = {
                "fields": "id,name,access_token,category",
                "access_token": access_token,
            }
            while url:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    logger.error("Failed to fetch pages: %s - %s", response.status_code, response.text)
                    break
                data = response.json()
                pages.extend(data.get("data", []))
                url = data.get("paging", {}).get("next")
                params = {}  # next URL already has params
        except Exception as e:
            logger.error("Error fetching Facebook pages: %s", e)
        return pages

    def get_user_groups(self, access_token: str) -> list[dict]:
        """Return Groups the user is an admin of.

        Each dict contains: id, name, privacy.
        """
        groups: list[dict] = []
        try:
            url = f"{GRAPH_API_BASE}/me/groups"
            params = {
                "fields": "id,name,privacy",
                "admin_only": "true",
                "access_token": access_token,
            }
            while url:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    logger.error("Failed to fetch groups: %s - %s", response.status_code, response.text)
                    break
                data = response.json()
                groups.extend(data.get("data", []))
                url = data.get("paging", {}).get("next")
                params = {}
        except Exception as e:
            logger.error("Error fetching Facebook groups: %s", e)
        return groups

    # ------------------------------------------------------------------
    # Publishing – Pages
    # ------------------------------------------------------------------

    def publish_text_post(
        self,
        page_access_token: str,
        page_id: str,
        text: str,
    ) -> dict:
        """Publish a text-only post to a Facebook Page."""
        try:
            response = requests.post(
                f"{GRAPH_API_BASE}/{page_id}/feed",
                data={"message": text, "access_token": page_access_token},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "post_id": data.get("id"),
                    "permalink": f"https://www.facebook.com/{data.get('id', '').replace('_', '/posts/')}",
                    "status_code": response.status_code,
                }
            error_data = _safe_json(response)
            logger.error("Facebook text post failed: %s - %s", response.status_code, error_data)
            return {"success": False, "status_code": response.status_code, "error": error_data}
        except requests.RequestException as e:
            logger.error("Facebook API request failed: %s", e)
            return {"success": False, "error": {"message": str(e)}}

    def publish_image_post(
        self,
        page_access_token: str,
        page_id: str,
        text: str,
        image_url: str,
    ) -> dict:
        """Publish a post with an image to a Facebook Page."""
        try:
            response = requests.post(
                f"{GRAPH_API_BASE}/{page_id}/photos",
                data={
                    "message": text,
                    "url": image_url,
                    "access_token": page_access_token,
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                post_id = data.get("post_id") or data.get("id", "")
                return {
                    "success": True,
                    "post_id": post_id,
                    "permalink": f"https://www.facebook.com/{post_id.replace('_', '/posts/')}" if post_id else None,
                    "status_code": response.status_code,
                }
            error_data = _safe_json(response)
            logger.error("Facebook image post failed: %s - %s", response.status_code, error_data)
            return {"success": False, "status_code": response.status_code, "error": error_data}
        except requests.RequestException as e:
            logger.error("Facebook image post request failed: %s", e)
            return {"success": False, "error": {"message": str(e)}}

    def publish_link_post(
        self,
        page_access_token: str,
        page_id: str,
        text: str,
        link: str,
    ) -> dict:
        """Publish a post with a link preview to a Facebook Page."""
        try:
            response = requests.post(
                f"{GRAPH_API_BASE}/{page_id}/feed",
                data={
                    "message": text,
                    "link": link,
                    "access_token": page_access_token,
                },
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "post_id": data.get("id"),
                    "permalink": f"https://www.facebook.com/{data.get('id', '').replace('_', '/posts/')}",
                    "status_code": response.status_code,
                }
            error_data = _safe_json(response)
            logger.error("Facebook link post failed: %s - %s", response.status_code, error_data)
            return {"success": False, "status_code": response.status_code, "error": error_data}
        except requests.RequestException as e:
            logger.error("Facebook link post request failed: %s", e)
            return {"success": False, "error": {"message": str(e)}}

    # ------------------------------------------------------------------
    # Publishing – Groups
    # ------------------------------------------------------------------

    def publish_group_post(
        self,
        user_access_token: str,
        group_id: str,
        text: str,
        link: str | None = None,
    ) -> dict:
        """Publish a post to a Facebook Group the user admins."""
        payload: dict = {
            "message": text,
            "access_token": user_access_token,
        }
        if link:
            payload["link"] = link

        try:
            response = requests.post(
                f"{GRAPH_API_BASE}/{group_id}/feed",
                data=payload,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "post_id": data.get("id"),
                    "permalink": f"https://www.facebook.com/groups/{group_id}/posts/{data.get('id', '').split('_')[-1]}" if data.get("id") else None,
                    "status_code": response.status_code,
                }
            error_data = _safe_json(response)
            logger.error("Facebook group post failed: %s - %s", response.status_code, error_data)
            return {"success": False, "status_code": response.status_code, "error": error_data}
        except requests.RequestException as e:
            logger.error("Facebook group post request failed: %s", e)
            return {"success": False, "error": {"message": str(e)}}

    # ------------------------------------------------------------------
    # Smart post (auto-detects URLs)
    # ------------------------------------------------------------------

    def publish_smart_post(
        self,
        page_access_token: str,
        page_id: str,
        text: str,
        image_url: str | None = None,
    ) -> dict:
        """Post to a Page, choosing text/image/link format automatically."""
        import re
        url_match = re.search(r'https?://[^\s<>"\')\]]+', text)

        if image_url:
            return self.publish_image_post(page_access_token, page_id, text, image_url)
        elif url_match:
            return self.publish_link_post(page_access_token, page_id, text, url_match.group(0).rstrip(".,;:!?"))
        else:
            return self.publish_text_post(page_access_token, page_id, text)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_json(response: requests.Response) -> dict:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def get_facebook_client() -> FacebookClient:
    """Factory function to get a configured Facebook client."""
    return FacebookClient()


def calculate_token_expiry(expires_in: int) -> str:
    expiry = datetime.utcnow() + timedelta(seconds=expires_in)
    return expiry.isoformat(timespec="seconds")


def is_token_expired(expires_at: str | None, buffer_minutes: int = 60) -> bool:
    if not expires_at:
        return True
    try:
        expiry = datetime.fromisoformat(expires_at)
        buffer = timedelta(minutes=buffer_minutes)
        return datetime.utcnow() >= (expiry - buffer)
    except (ValueError, TypeError):
        return True

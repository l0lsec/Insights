"""LinkedIn API client for OAuth and posting functionality."""

from __future__ import annotations

import os
import re
import logging
import secrets
import time
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from typing import Optional

import requests

import video_media

logger = logging.getLogger(__name__)

# Seam for tests: a gate replaces this so status polling does not really wait.
_sleep = time.sleep


def fetch_og_metadata(url: str, timeout: int = 10) -> dict:
    """Fetch Open Graph metadata from a URL.
    
    Args:
        url: The URL to fetch metadata from
        timeout: Request timeout in seconds
        
    Returns:
        Dict with 'title', 'description', 'image' keys (may be None if not found)
    """
    metadata = {"title": None, "description": None, "image": None}
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Insights/1.0; +https://insights.app)"
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        html = response.text
        
        # Extract og:title
        og_title_match = re.search(
            r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if not og_title_match:
            og_title_match = re.search(
                r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']',
                html, re.IGNORECASE
            )
        if og_title_match:
            metadata["title"] = og_title_match.group(1)
        else:
            # Fallback to <title> tag
            title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
        
        # Extract og:description
        og_desc_match = re.search(
            r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if not og_desc_match:
            og_desc_match = re.search(
                r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:description["\']',
                html, re.IGNORECASE
            )
        if og_desc_match:
            metadata["description"] = og_desc_match.group(1)
        
        # Extract og:image
        og_image_match = re.search(
            r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if not og_image_match:
            og_image_match = re.search(
                r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
                html, re.IGNORECASE
            )
        if og_image_match:
            metadata["image"] = og_image_match.group(1)
            
    except requests.RequestException as e:
        logger.warning("Failed to fetch OG metadata from %s: %s", url, e)
    except Exception as e:
        logger.warning("Error parsing OG metadata from %s: %s", url, e)
    
    return metadata

# LinkedIn API endpoints
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
LINKEDIN_POSTS_URL = "https://api.linkedin.com/rest/posts"
LINKEDIN_IMAGES_URL = "https://api.linkedin.com/rest/images"
LINKEDIN_VIDEOS_URL = "https://api.linkedin.com/rest/videos"

# LinkedIn finishes processing a video some seconds to a few minutes after the
# upload is finalized, and a post that points at a video that is not AVAILABLE
# yet is refused, so the client waits (2s * 150 = five minutes) before giving up.
VIDEO_POLL_INTERVAL = 2.0
VIDEO_POLL_ATTEMPTS = 150

# API version for LinkedIn REST API (format: YYYYMM)
# LinkedIn keeps versions active for ~1 year, use a recent stable version
# See: https://learn.microsoft.com/en-us/linkedin/marketing/versioning
LINKEDIN_API_VERSION = "202601"

# OAuth scopes needed for personal posting
# Configurable via environment variable
# 
# LinkedIn products that provide these scopes:
#   - "Share on LinkedIn" -> w_member_social (posting)
#   - "Sign In with LinkedIn using OpenID Connect" -> openid, profile, email (user info via /v2/userinfo)
#
# Note: openid alone is not sufficient - LinkedIn requires at least 'profile' with openid
LINKEDIN_SCOPES = os.environ.get(
    "LINKEDIN_SCOPES",
    "openid profile w_member_social"
)


class LinkedInClient:
    """Client for interacting with LinkedIn's API."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ):
        """Initialize the LinkedIn client with credentials from env or params."""
        self.client_id = client_id or os.environ.get("LINKEDIN_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("LINKEDIN_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.environ.get(
            "LINKEDIN_REDIRECT_URI", "http://localhost:5001/linkedin/callback"
        )

    def is_configured(self) -> bool:
        """Check if LinkedIn credentials are configured."""
        return bool(self.client_id and self.client_secret)

    def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
        """Generate the OAuth authorization URL.

        Returns:
            Tuple of (authorization_url, state_token)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": state,
            "scope": LINKEDIN_SCOPES,
        }
        url = f"{LINKEDIN_AUTH_URL}?{urlencode(params)}"
        return url, state

    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for access token.

        Args:
            code: The authorization code from OAuth callback

        Returns:
            Dict containing access_token, expires_in, and optionally refresh_token
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = requests.post(
            LINKEDIN_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Refresh an expired access token.

        Args:
            refresh_token: The refresh token

        Returns:
            Dict containing new access_token, expires_in, etc.
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = requests.post(
            LINKEDIN_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_user_info(self, access_token: str) -> dict | None:
        """Get the authenticated user's profile info.

        Tries multiple endpoints to get user info depending on available scopes.
        Returns None if no profile endpoints work (user has w_member_social only).

        Args:
            access_token: Valid LinkedIn access token

        Returns:
            Dict containing sub (member_id), name, email, etc. OR None if unavailable
        """
        # First try the OpenID userinfo endpoint (if openid scope was granted)
        try:
            response = requests.get(
                LINKEDIN_USERINFO_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                },
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                logger.info("Got user info from OpenID userinfo endpoint")
                return data
            else:
                logger.debug("userinfo endpoint returned %s", response.status_code)
        except Exception as e:
            logger.debug("userinfo endpoint failed: %s", e)

        # Fallback to /v2/me endpoint for basic profile
        try:
            response = requests.get(
                "https://api.linkedin.com/v2/me",
                headers={
                    "Authorization": f"Bearer {access_token}",
                },
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                # Convert to userinfo-like format
                member_id = data.get("id", "")
                first_name = data.get("localizedFirstName", "")
                last_name = data.get("localizedLastName", "")
                logger.info("Got user info from /v2/me endpoint")
                return {
                    "sub": member_id,
                    "name": f"{first_name} {last_name}".strip() or "LinkedIn User",
                    "email": "",  # Not available from /v2/me
                }
            else:
                logger.debug("/v2/me endpoint returned %s", response.status_code)
        except Exception as e:
            logger.debug("/v2/me endpoint failed: %s", e)

        # If profile endpoints don't work, return None
        # User will need to manually configure their member ID
        logger.warning(
            "Could not get user info from LinkedIn. "
            "User may need to manually configure their member ID, "
            "or add the 'Sign In with LinkedIn using OpenID Connect' product."
        )
        return None

    def _get_api_headers(self, access_token: str) -> dict:
        """Get standard headers for LinkedIn REST API calls."""
        return {
            "Authorization": f"Bearer {access_token}",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        }

    @staticmethod
    def extract_first_url(text: str) -> str | None:
        """Extract the first URL from text.
        
        Returns the first http/https URL found, or None if no URL is found.
        """
        import re
        # Match http/https URLs
        url_pattern = r'https?://[^\s<>"\')\]]+(?:\([^\s<>"\')\]]*\)[^\s<>"\')\]]*)*'
        match = re.search(url_pattern, text)
        if match:
            url = match.group(0)
            # Clean up trailing punctuation that might have been captured
            url = url.rstrip('.,;:!?')
            return url
        return None

    def upload_image_from_url(
        self,
        access_token: str,
        owner_urn: str,
        image_url: str,
    ) -> str | None:
        """Upload an image to LinkedIn from a URL and return the image URN.
        
        Args:
            access_token: Valid LinkedIn access token
            owner_urn: The owner URN (e.g., "urn:li:person:ABC123")
            image_url: URL of the image to upload
            
        Returns:
            The LinkedIn image URN if successful, None otherwise
        """
        try:
            # Step 1: Download the image from the URL
            logger.info("Downloading image from: %s", image_url)
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; Insights/1.0)"
            }
            img_response = requests.get(image_url, headers=headers, timeout=30, allow_redirects=True)
            img_response.raise_for_status()
            image_data = img_response.content
            content_type = img_response.headers.get("Content-Type", "image/jpeg")
            
            # Step 2: Initialize the upload with LinkedIn
            logger.info("Initializing LinkedIn image upload")
            init_payload = {
                "initializeUploadRequest": {
                    "owner": owner_urn
                }
            }
            
            init_response = requests.post(
                f"{LINKEDIN_IMAGES_URL}?action=initializeUpload",
                json=init_payload,
                headers=self._get_api_headers(access_token),
                timeout=30,
            )
            
            if init_response.status_code != 200:
                logger.error("Failed to initialize image upload: %s - %s", 
                           init_response.status_code, init_response.text)
                return None
            
            init_data = init_response.json()
            upload_url = init_data.get("value", {}).get("uploadUrl")
            image_urn = init_data.get("value", {}).get("image")
            
            if not upload_url or not image_urn:
                logger.error("Missing upload URL or image URN in response: %s", init_data)
                return None
            
            # Step 3: Upload the image binary to the provided URL
            logger.info("Uploading image to LinkedIn: %s", image_urn)
            upload_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": content_type,
            }
            
            upload_response = requests.put(
                upload_url,
                data=image_data,
                headers=upload_headers,
                timeout=60,
            )
            
            if upload_response.status_code in (200, 201):
                logger.info("Image uploaded successfully: %s", image_urn)
                return image_urn
            else:
                logger.error("Failed to upload image: %s - %s",
                           upload_response.status_code, upload_response.text)
                return None
                
        except requests.RequestException as e:
            logger.error("Error uploading image: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error uploading image: %s", e)
            return None

    def create_text_post(
        self,
        access_token: str,
        author_urn: str,
        text: str,
        visibility: str = "PUBLIC",
    ) -> dict:
        """Create a text-only post on LinkedIn.

        Args:
            access_token: Valid LinkedIn access token
            author_urn: The author URN (e.g., "urn:li:person:ABC123")
            text: The post content/commentary
            visibility: Post visibility - "PUBLIC" or "CONNECTIONS"

        Returns:
            Dict with post details including the post URN in x-restli-id header
        """
        payload = {
            "author": author_urn,
            "commentary": text,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }

        response = requests.post(
            LINKEDIN_POSTS_URL,
            json=payload,
            headers=self._get_api_headers(access_token),
            timeout=30,
        )

        if response.status_code == 201:
            # Success - the post URN is in the x-restli-id header
            post_urn = response.headers.get("x-restli-id", "")
            return {
                "success": True,
                "post_urn": post_urn,
                "status_code": response.status_code,
            }
        else:
            # Error occurred
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"raw": response.text}

            logger.error(
                "LinkedIn post failed: %s - %s",
                response.status_code,
                error_data,
            )
            return {
                "success": False,
                "status_code": response.status_code,
                "error": error_data,
            }

    def create_image_post(
        self,
        access_token: str,
        author_urn: str,
        text: str,
        image_url: str,
        visibility: str = "PUBLIC",
    ) -> dict:
        """Create a post with an image on LinkedIn.

        Args:
            access_token: Valid LinkedIn access token
            author_urn: The author URN (e.g., "urn:li:person:ABC123")
            text: The post content/commentary
            image_url: URL of the image to attach
            visibility: Post visibility - "PUBLIC" or "CONNECTIONS"

        Returns:
            Dict with post details including the post URN
        """
        # First, upload the image to get an image URN
        logger.info("Uploading image for LinkedIn post: %s", image_url)
        image_urn = self.upload_image_from_url(
            access_token=access_token,
            owner_urn=author_urn,
            image_url=image_url,
        )
        
        if not image_urn:
            logger.warning("Failed to upload image, falling back to text-only post")
            return self.create_text_post(
                access_token=access_token,
                author_urn=author_urn,
                text=text,
                visibility=visibility,
            )
        
        logger.info("Image uploaded successfully, URN: %s", image_urn)
        
        # Create post with image content
        payload = {
            "author": author_urn,
            "commentary": text,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {
                "media": {
                    "id": image_urn,
                }
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }

        response = requests.post(
            LINKEDIN_POSTS_URL,
            json=payload,
            headers=self._get_api_headers(access_token),
            timeout=30,
        )

        if response.status_code == 201:
            post_urn = response.headers.get("x-restli-id", "")
            return {
                "success": True,
                "post_urn": post_urn,
                "status_code": response.status_code,
            }
        else:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"raw": response.text}

            logger.error(
                "LinkedIn image post failed: %s - %s",
                response.status_code,
                error_data,
            )
            return {
                "success": False,
                "status_code": response.status_code,
                "error": error_data,
            }

    def upload_video_from_url(
        self,
        access_token: str,
        owner_urn: str,
        video_url: str,
    ) -> str:
        """Upload a video through LinkedIn's Videos API and return its URN.

        The protocol has four steps: declare the upload (which returns one
        pre-signed URL per 4 MB part), PUT each part and keep the ETag it
        answers with, finalize with those ETags in order, then wait for
        LinkedIn to finish processing. The file is streamed from disk a part at
        a time, so a large video is never held in memory.

        Raises ``video_media.VideoError`` with a readable reason on any failure;
        ``permanent`` on it says whether a retry could help.
        """
        limits = video_media.PLATFORM_LIMITS["linkedin"]
        with video_media.downloaded(video_url, max_bytes=limits["max_bytes"]) as (path, size):
            refusal = video_media.preflight("linkedin", size)
            if refusal:
                raise video_media.VideoError(refusal)

            init = requests.post(
                f"{LINKEDIN_VIDEOS_URL}?action=initializeUpload",
                json={"initializeUploadRequest": {
                    "owner": owner_urn,
                    "fileSizeBytes": size,
                    "uploadCaptions": False,
                    "uploadThumbnail": False,
                }},
                headers=self._get_api_headers(access_token),
                timeout=30,
            )
            if init.status_code != 200:
                raise video_media.VideoError(
                    f"LinkedIn would not start the video upload ({init.status_code}): "
                    f"{_error_text(init)}",
                    permanent=init.status_code in (400, 401, 403),
                )
            value = init.json().get("value", {})
            video_urn = value.get("video")
            instructions = value.get("uploadInstructions") or []
            upload_token = value.get("uploadToken", "")
            if not video_urn or not instructions:
                raise video_media.VideoError(
                    "LinkedIn's upload reply had no video id or upload URL", permanent=False,
                )

            part_ids = []
            with open(path, "rb") as source:
                for part in instructions:
                    first, last = int(part["firstByte"]), int(part["lastByte"])
                    source.seek(first)
                    body = source.read(last - first + 1)
                    put = requests.put(
                        part["uploadUrl"],
                        data=body,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=120,
                    )
                    if put.status_code not in (200, 201):
                        raise video_media.VideoError(
                            f"LinkedIn rejected part {len(part_ids) + 1} of the video "
                            f"({put.status_code}): {_error_text(put)}",
                            permanent=False,
                        )
                    etag = (put.headers.get("etag") or put.headers.get("ETag") or "").strip('"')
                    if not etag:
                        raise video_media.VideoError(
                            f"LinkedIn gave no ETag for part {len(part_ids) + 1}, so the "
                            "upload cannot be finalized",
                            permanent=False,
                        )
                    part_ids.append(etag)

            fin = requests.post(
                f"{LINKEDIN_VIDEOS_URL}?action=finalizeUpload",
                json={"finalizeUploadRequest": {
                    "video": video_urn,
                    "uploadToken": upload_token,
                    "uploadedPartIds": part_ids,
                }},
                headers=self._get_api_headers(access_token),
                timeout=30,
            )
            if fin.status_code != 200:
                raise video_media.VideoError(
                    f"LinkedIn could not finalize the video ({fin.status_code}): "
                    f"{_error_text(fin)}",
                    permanent=False,
                )

        self._wait_for_video(access_token, video_urn)
        return video_urn

    def _wait_for_video(self, access_token: str, video_urn: str) -> None:
        """Block until LinkedIn reports the video AVAILABLE, or raise why not."""
        status = None
        for _attempt in range(VIDEO_POLL_ATTEMPTS):
            response = requests.get(
                f"{LINKEDIN_VIDEOS_URL}/{quote(video_urn, safe='')}",
                headers=self._get_api_headers(access_token),
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                if status == "AVAILABLE":
                    return
                if status == "PROCESSING_FAILED":
                    raise video_media.VideoError(
                        "LinkedIn could not process the video: "
                        f"{data.get('processingFailureReason') or 'unsupported file'}",
                        permanent=True,
                    )
            _sleep(VIDEO_POLL_INTERVAL)
        raise video_media.VideoError(
            f"LinkedIn was still processing the video (last status {status or 'unknown'}) "
            "after waiting; try again shortly",
            permanent=False,
        )

    def create_video_post(
        self,
        access_token: str,
        author_urn: str,
        text: str,
        video_url: str,
        visibility: str = "PUBLIC",
    ) -> dict:
        """Create a post with a video on LinkedIn.

        Unlike ``create_image_post`` this never falls back to a text post when
        the upload fails: someone who attached a video does not want the words
        going out without it, so the failure is returned and nothing is posted.
        """
        try:
            video_urn = self.upload_video_from_url(access_token, author_urn, video_url)
        except video_media.VideoError as exc:
            logger.error("LinkedIn video upload failed: %s", exc)
            return video_media.failure(str(exc), permanent=exc.permanent)
        except requests.RequestException as exc:
            logger.error("LinkedIn video upload request failed: %s", exc)
            return video_media.failure(f"LinkedIn video upload failed: {exc}")

        payload = {
            "author": author_urn,
            "commentary": text,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {"media": {"id": video_urn}},
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }
        try:
            response = requests.post(
                LINKEDIN_POSTS_URL,
                json=payload,
                headers=self._get_api_headers(access_token),
                timeout=30,
            )
        except requests.RequestException as exc:
            return video_media.failure(f"LinkedIn video post failed: {exc}")

        if response.status_code == 201:
            return {
                "success": True,
                "post_urn": response.headers.get("x-restli-id", ""),
                "status_code": response.status_code,
            }
        error_data = _error_json(response)
        logger.error("LinkedIn video post failed: %s - %s", response.status_code, error_data)
        return {"success": False, "status_code": response.status_code, "error": error_data}

    def create_article_post(
        self,
        access_token: str,
        author_urn: str,
        commentary: str,
        article_url: str,
        article_title: str | None = None,
        article_description: str | None = None,
        article_thumbnail: str | None = None,
        visibility: str = "PUBLIC",
    ) -> dict:
        """Create a post with an article link on LinkedIn.

        Args:
            access_token: Valid LinkedIn access token
            author_urn: The author URN (e.g., "urn:li:person:ABC123")
            commentary: The text commentary for the post
            article_url: URL of the article to share
            article_title: Optional title override for the article
            article_description: Optional description override
            article_thumbnail: Optional thumbnail image URN (urn:li:image:...) for the preview card
            visibility: Post visibility - "PUBLIC" or "CONNECTIONS"

        Returns:
            Dict with post details including the post URN
        """
        article_content = {"source": article_url}
        if article_title:
            article_content["title"] = article_title
        if article_description:
            article_content["description"] = article_description[:200]  # LinkedIn limit
        if article_thumbnail:
            article_content["thumbnail"] = article_thumbnail

        payload = {
            "author": author_urn,
            "commentary": commentary,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {
                "article": article_content,
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }

        response = requests.post(
            LINKEDIN_POSTS_URL,
            json=payload,
            headers=self._get_api_headers(access_token),
            timeout=30,
        )

        if response.status_code == 201:
            post_urn = response.headers.get("x-restli-id", "")
            return {
                "success": True,
                "post_urn": post_urn,
                "status_code": response.status_code,
            }
        else:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"raw": response.text}

            logger.error(
                "LinkedIn article post failed: %s - %s",
                response.status_code,
                error_data,
            )
            return {
                "success": False,
                "status_code": response.status_code,
                "error": error_data,
            }

    def create_smart_post(
        self,
        access_token: str,
        author_urn: str,
        text: str,
        article_title: str | None = None,
        visibility: str = "PUBLIC",
    ) -> dict:
        """Create a post on LinkedIn, automatically detecting URLs for link previews.
        
        If the text contains a URL, this will create an article post with a link
        preview card. The title and thumbnail are automatically fetched from the
        URL's Open Graph metadata for a proper preview card.
        
        Args:
            access_token: Valid LinkedIn access token
            author_urn: The author URN (e.g., "urn:li:person:ABC123")
            text: The post content/commentary
            article_title: Optional fallback title (OG metadata takes priority)
            visibility: Post visibility - "PUBLIC" or "CONNECTIONS"
            
        Returns:
            Dict with post details including the post URN
        """
        url = self.extract_first_url(text)
        
        if url:
            # Fetch Open Graph metadata from the URL for the title and image
            logger.info("Fetching OG metadata from: %s", url)
            og_metadata = fetch_og_metadata(url)
            
            # Use OG metadata for title and description
            final_title = og_metadata.get("title")
            final_description = og_metadata.get("description")
            og_image_url = og_metadata.get("image")
            
            # Fallback title if OG metadata didn't provide one
            if not final_title:
                if article_title:
                    final_title = article_title
                else:
                    # Extract from post text as last resort
                    text_before_url = text.split(url)[0].strip()
                    if text_before_url:
                        sentences = re.split(r'[.!?]', text_before_url)
                        final_title = sentences[0].strip()[:100]
                    if not final_title:
                        # Final fallback: domain name
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        final_title = f"Link from {parsed.netloc}"
            
            # Upload the og:image to LinkedIn to get a URN for the thumbnail
            thumbnail_urn = None
            if og_image_url:
                logger.info("Uploading og:image to LinkedIn: %s", og_image_url)
                thumbnail_urn = self.upload_image_from_url(
                    access_token=access_token,
                    owner_urn=author_urn,
                    image_url=og_image_url,
                )
                if thumbnail_urn:
                    logger.info("Image uploaded, URN: %s", thumbnail_urn)
                else:
                    logger.warning("Failed to upload og:image, posting without thumbnail")
            
            logger.info("Using title: %s", final_title)
            
            # Use article post format for link preview
            return self.create_article_post(
                access_token=access_token,
                author_urn=author_urn,
                commentary=text,
                article_url=url,
                article_title=final_title,
                article_description=final_description,
                article_thumbnail=thumbnail_urn,
            )
        else:
            # No URL found, use text-only post
            return self.create_text_post(
                access_token=access_token,
                author_urn=author_urn,
                text=text,
                visibility=visibility,
            )

    def get_post(self, access_token: str, post_urn: str) -> dict:
        """Retrieve a post by its URN.

        Args:
            access_token: Valid LinkedIn access token
            post_urn: The post URN (URL encoded)

        Returns:
            Dict with post details
        """
        from urllib.parse import quote

        encoded_urn = quote(post_urn, safe="")
        url = f"{LINKEDIN_POSTS_URL}/{encoded_urn}"

        response = requests.get(
            url,
            headers=self._get_api_headers(access_token),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def delete_post(self, access_token: str, post_urn: str) -> bool:
        """Delete a post by its URN.

        Args:
            access_token: Valid LinkedIn access token
            post_urn: The post URN (URL encoded)

        Returns:
            True if deleted successfully
        """
        from urllib.parse import quote

        encoded_urn = quote(post_urn, safe="")
        url = f"{LINKEDIN_POSTS_URL}/{encoded_urn}"

        response = requests.delete(
            url,
            headers=self._get_api_headers(access_token),
            timeout=30,
        )
        return response.status_code == 204


def _error_json(response: requests.Response) -> dict:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _error_text(response: requests.Response) -> str:
    """A platform error as one short line, for a message a person will read."""
    data = _error_json(response)
    return str(data.get("message") or data.get("raw") or data)[:200]


def get_linkedin_client() -> LinkedInClient:
    """Factory function to get a configured LinkedIn client."""
    return LinkedInClient()


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

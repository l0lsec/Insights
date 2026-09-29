"""Video handling shared by every platform client and the Compose routes.

Threads, Facebook and Instagram fetch a video themselves from a public URL, so
for them a video is only a URL. LinkedIn and X do not: they want the bytes
pushed through their own upload protocol, which means the server has to pull the
file down first. This module is that one download, plus the two other things
every platform needs to know about a video before it is worth sending:

* what each platform will accept (``PLATFORM_LIMITS``, ``check_compat``), so the
  composer can say "too long for X" when the video is attached rather than when
  the post fails at 3am, and
* how big and how long a given file actually is (``probe_file``).

Videos are big, so the download streams to a temp file instead of holding the
file in memory the way the image path can afford to, and it is capped: a pasted
URL is user input, and an unbounded ``requests.get`` of it is a disk-fill.

The app registers its SSRF guard with ``set_url_guard`` (see insights_web), the
same way it registers the Instagram publisher with the shared publisher. Left
unregistered the guard only insists on http(s), which keeps this module usable
and testable on its own.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

MB = 1024 * 1024
GB = 1024 * MB

# Nothing bigger than this is downloaded for a push-style upload, whatever the
# platform allows: Cloudinary caps the composer's own uploads at 100 MB, so this
# is only ever reached by a pasted URL.
MAX_DOWNLOAD_BYTES = 1 * GB
DOWNLOAD_CHUNK_BYTES = 1 * MB
MAX_REDIRECTS = 5
USER_AGENT = "InsightsBot/1.0 (+https://insights.local)"

# What each platform accepts for a video post through its API. ``max_bytes`` and
# ``min_seconds`` are the limits the platform documents as hard, and a file over
# them is an ``error`` (it will be refused). ``max_seconds`` is reported as a
# ``warn``: some are entitlement-dependent (X allows longer for Premium) or have
# moved between API versions, so the platform, not this table, has the last word.
PLATFORM_LIMITS = {
    "linkedin": {"max_bytes": 500 * MB, "min_seconds": 3, "max_seconds": 30 * 60},
    "twitter": {"max_bytes": MAX_DOWNLOAD_BYTES, "min_seconds": None, "max_seconds": 140},
    "threads": {"max_bytes": 1 * GB, "min_seconds": None, "max_seconds": 5 * 60},
    "facebook": {"max_bytes": 1 * GB, "min_seconds": None, "max_seconds": 20 * 60},
    "instagram": {"max_bytes": 300 * MB, "min_seconds": 3, "max_seconds": 15 * 60},
}

_PLATFORM_LABELS = {
    "linkedin": "LinkedIn", "twitter": "X", "threads": "Threads",
    "facebook": "Facebook", "instagram": "Instagram",
}


class VideoError(Exception):
    """A video could not be fetched or is not something a platform will take.

    ``permanent`` says whether trying again can help. A file that is too big, a
    URL that is not a video or a 404 will fail the same way tomorrow, so the
    scheduler should give up on them; a dropped connection or a 503 is worth
    another go.
    """

    def __init__(self, message: str, permanent: bool = True):
        super().__init__(message)
        self.permanent = permanent


def failure(message: str, permanent: bool = False) -> dict:
    """A failed publish in the shape every client returns.

    ``guard_error`` is the flag the shared publisher reads to stop the scheduler
    retrying a post the platform was never going to accept.
    """
    return {"success": False, "error": {"message": message}, "guard_error": bool(permanent)}


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------


def _default_guard(url: str) -> None:
    parsed = urlparse((url or "").strip())
    if (parsed.scheme or "").lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("Video URL must be an http(s) link")


_url_guard = _default_guard


def set_url_guard(fn) -> None:
    """Register the app's SSRF guard. It raises ValueError for an unsafe URL."""
    global _url_guard
    _url_guard = fn or _default_guard


def check_url(url: str) -> str:
    """The trimmed URL if it may be fetched; ``VideoError`` with the reason if not."""
    url = (url or "").strip()
    try:
        _url_guard(url)
    except ValueError as exc:
        raise VideoError(str(exc)) from exc
    return url


def _open(url: str, method: str, timeout: float):
    """One request that follows redirects by hand, re-checking each hop.

    ``requests`` would follow them silently, and a public URL that redirects to
    an internal address is the usual way past a guard that only saw the first.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    current = check_url(url)
    for _hop in range(MAX_REDIRECTS + 1):
        response = session.request(
            method, current, stream=True, timeout=timeout, allow_redirects=False,
        )
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise VideoError(f"Redirect without a Location from {current}")
            current = check_url(urljoin(current, location))
            continue
        return response
    raise VideoError("Too many redirects fetching the video")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def downloaded(url: str, *, max_bytes: int = MAX_DOWNLOAD_BYTES, timeout: float = 60):
    """Stream ``url`` to a temp file and yield ``(path, size_bytes)``.

    The file is removed when the block exits, on success or on error, so a
    failed LinkedIn upload does not leave a gigabyte behind. Raises
    ``VideoError`` for anything that makes the file unusable: an unsafe URL, an
    HTTP error, a page that is not a video, or more bytes than ``max_bytes``.
    """
    path = None
    response = None
    try:
        try:
            response = _open(url, "GET", timeout)
        except requests.RequestException as exc:
            raise VideoError(f"Could not fetch the video: {exc}", permanent=False) from exc

        if not response.ok:
            raise VideoError(
                f"The video URL answered HTTP {response.status_code}",
                permanent=response.status_code < 500 and response.status_code != 429,
            )

        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type.startswith(("text/", "application/json")):
            raise VideoError(
                f"The video URL returned {content_type or 'a web page'}, not a video file"
            )

        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise VideoError(
                f"The video is {int(declared) / MB:.0f} MB, over the {max_bytes // MB} MB limit"
            )

        handle, path = tempfile.mkstemp(prefix="insights_video_", suffix=".mp4")
        size = 0
        with os.fdopen(handle, "wb") as out:
            try:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise VideoError(
                            f"The video is over the {max_bytes // MB} MB limit"
                        )
                    out.write(chunk)
            except requests.RequestException as exc:
                raise VideoError(
                    f"The video download was interrupted: {exc}", permanent=False,
                ) from exc

        if size == 0:
            raise VideoError("The video URL returned an empty file")
        yield path, size
    finally:
        if response is not None:
            response.close()
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Could not remove temp video %s", path)


def head_size(url: str, timeout: float = 15) -> int | None:
    """The video's size in bytes from a HEAD request, or None if it isn't told.

    Never raises: this feeds an early warning, and a host that refuses HEAD
    should not stop a video from being attached.
    """
    try:
        response = _open(url, "HEAD", timeout)
    except (VideoError, requests.RequestException):
        return None
    try:
        length = response.headers.get("Content-Length")
        return int(length) if length and length.isdigit() else None
    finally:
        response.close()


# ---------------------------------------------------------------------------
# What a file is, and what each platform will take
# ---------------------------------------------------------------------------


def probe_file(path: str) -> dict:
    """Duration, dimensions and codec of a local file; ``{}`` if ffprobe can't say.

    An unreadable file (or no ffprobe) gives ``{}`` rather than a dict of
    defaults, so "could not tell" is never mistaken for "no duration".
    """
    try:
        import media_probe
        found = media_probe.probe_video(path) or {}
    except Exception as exc:  # noqa: BLE001 - probing is best-effort
        logger.warning("Could not probe %s: %s", path, exc)
        return {}
    if found.get("duration") is None and found.get("width") is None:
        return {}
    return {key: found[key] for key in ("duration", "width", "height", "codec", "has_audio")
            if key in found}


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def check_compat(size_bytes: int | None = None, duration_seconds: float | None = None,
                 platforms=None) -> list[dict]:
    """What each platform will make of a video this size and length.

    Returns ``[{"platform", "level", "message"}]``, empty when every platform is
    fine. ``level`` is ``error`` when the platform documents a hard limit the
    video breaks and ``warn`` when it may be refused. A value that is None
    (size unknown, ffprobe missing) is simply not judged.
    """
    issues = []
    for platform in platforms or PLATFORM_LIMITS:
        limits = PLATFORM_LIMITS.get(platform)
        if not limits:
            continue
        name = _PLATFORM_LABELS.get(platform, platform)

        if size_bytes is not None and size_bytes > limits["max_bytes"]:
            issues.append({
                "platform": platform, "level": "error",
                "message": f"{name} takes videos up to {limits['max_bytes'] // MB} MB; "
                           f"this one is {size_bytes / MB:.0f} MB",
            })
        if duration_seconds is not None:
            minimum = limits["min_seconds"]
            if minimum and duration_seconds < minimum:
                issues.append({
                    "platform": platform, "level": "error",
                    "message": f"{name} needs at least {minimum} seconds of video",
                })
            if limits["max_seconds"] and duration_seconds > limits["max_seconds"]:
                issues.append({
                    "platform": platform, "level": "warn",
                    "message": f"{name} may reject videos longer than "
                               f"{_clock(limits['max_seconds'])}; this one runs "
                               f"{_clock(duration_seconds)}",
                })
    return issues


def preflight(platform: str, size_bytes: int | None) -> str | None:
    """A reason this platform will certainly refuse a video of this size, else None."""
    for issue in check_compat(size_bytes=size_bytes, platforms=[platform]):
        if issue["level"] == "error":
            return issue["message"]
    return None

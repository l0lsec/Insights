"""Filesystem and media inspection helpers for the Content Library.

This module is the only place that knows how to pull bytes off disk, and it
draws a hard line between two classes of operation:

* **Cold** helpers (:func:`stat_file`, :func:`is_materialized`, :func:`free_bytes`)
  read nothing but inode metadata. They are safe to run across an entire
  archive -- a 32k-file walk costs well under two seconds.
* **Warm** helpers (:func:`thumbnail_image`, :func:`probe_video`,
  :func:`extract_frames`, :func:`extract_audio`) open the file and therefore
  *materialize* it.

That distinction matters because the archives this feature targets live in
cloud-backed providers (OneDrive / iCloud Drive "Files On-Demand"). There, a
directory entry with ``st_size`` of 4 MB may occupy ``st_blocks == 0`` on the
local disk: the bytes are still in the cloud and touching the file silently
triggers a download. An archive can easily be an order of magnitude larger
than the free space on the volume, so any code path that reads bytes has to be
budgeted rather than applied to every file. See :class:`HydrationBudget`.

macOS's File Provider daemon offers no supported eviction call, so downloads
are effectively one-way: the only way to reclaim the space is Finder's
"Remove Download". The budget is therefore a hard ceiling, not a soft hint.

External tools are used in preference to Python imaging libraries because they
handle the formats that dominate a modern phone archive without extra
dependencies: ``sips`` (built into macOS) reads and transcodes HEIC, which
Pillow cannot do without ``pillow-heif``; ``ffprobe``/``ffmpeg`` handle
QuickTime metadata and frame extraction. All three are probed once at import
and their absence degrades individual features rather than breaking the
module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ── Tool discovery ──────────────────────────────────────────────────────────

SIPS = shutil.which("sips")
FFPROBE = shutil.which("ffprobe")
FFMPEG = shutil.which("ffmpeg")

# On a cloud-backed path a subprocess blocks while the provider downloads the
# file before it can read a byte, so these allow far more than local I/O would
# need. They are still bounded: a cancel request is only noticed between files,
# so an over-long timeout makes stopping a run feel unresponsive. At observed
# provider speeds these cover a sample several times larger than the default
# size cap.
PROBE_TIMEOUT = int(os.environ.get("LIBRARY_PROBE_TIMEOUT", "60"))
CONVERT_TIMEOUT = int(os.environ.get("LIBRARY_CONVERT_TIMEOUT", "90"))


def tool_status() -> dict:
    """Report which external helpers are usable, for the UI's setup panel."""
    return {
        "sips": bool(SIPS),
        "ffprobe": bool(FFPROBE),
        "ffmpeg": bool(FFMPEG),
    }


# ── File kinds ──────────────────────────────────────────────────────────────

IMAGE_EXTS = {"jpg", "jpeg", "png", "heic", "heif", "gif", "webp", "tif", "tiff", "bmp", "dng", "raw", "cr2", "nef"}
VIDEO_EXTS = {"mov", "mp4", "m4v", "avi", "mkv", "3gp", "mpg", "mpeg", "wmv", "flv", "webm"}
AUDIO_EXTS = {"m4a", "mp3", "wav", "aiff", "aac", "flac", "ogg", "caf"}
DOC_EXTS = {"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "csv", "txt", "md", "rtf", "pages", "key", "numbers", "html", "htm"}

# Files that are never content: OS bookkeeping, camera sidecars, archives of
# other files. ``lrf``/``thm`` are DJI low-res proxies and thumbnails that sit
# beside the real footage; ``scr`` likewise. Keeping them out of the catalog
# avoids inflating counts and wasting AI passes on duplicates of real clips.
SIDECAR_EXTS = {"lrf", "thm", "scr", "aae", "xmp", "ds_store", "localized", "part", "tmp", "download"}
IGNORE_NAMES = {".DS_Store", ".localized", "Thumbs.db", "desktop.ini", "Icon\r"}


def file_kind(path: str) -> str:
    """Classify a path into a coarse kind from its extension alone (cold)."""
    name = os.path.basename(path)
    if name in IGNORE_NAMES or name.startswith("._"):
        return "ignore"
    ext = os.path.splitext(name)[1].lstrip(".").lower()
    if not ext:
        return "other"
    if ext in SIDECAR_EXTS:
        return "sidecar"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in DOC_EXTS:
        return "document"
    return "other"


# ── Cold metadata ───────────────────────────────────────────────────────────

@dataclass
class FileStat:
    """Inode-level facts about a file. Costs no download."""

    path: str
    size: int
    mtime: float
    blocks: int
    kind: str
    ext: str

    @property
    def materialized(self) -> bool:
        """True when the bytes are actually on the local disk.

        Cloud placeholders report their true ``st_size`` but allocate no
        blocks. A tiny fully-resident file also reports few blocks, so treat
        anything with at least one allocated block as present -- the failure
        mode (attempting a download that turns out to be a no-op) is harmless.
        """
        return self.blocks > 0


def stat_file(path: str) -> Optional[FileStat]:
    """Stat a single path without opening it. Returns None if unreadable."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return FileStat(
        path=path,
        size=st.st_size,
        mtime=st.st_mtime,
        blocks=getattr(st, "st_blocks", 1),
        kind=file_kind(path),
        ext=ext,
    )


def is_materialized(path: str) -> bool:
    """Whether reading ``path`` would avoid a cloud download."""
    st = stat_file(path)
    return bool(st and st.materialized)


def walk_tree(root: str, skip_hidden: bool = True) -> Iterator[FileStat]:
    """Yield a :class:`FileStat` for every file under ``root`` (cold).

    Hidden directories are skipped by default; provider metadata folders and
    version-control internals otherwise dominate the file count.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if skip_hidden and name.startswith("."):
                continue
            st = stat_file(os.path.join(dirpath, name))
            if st is not None:
                yield st


def free_bytes(path: str) -> int:
    """Free space on the volume holding ``path``, in bytes."""
    try:
        vfs = os.statvfs(path)
        return vfs.f_bavail * vfs.f_frsize
    except OSError:
        return 0


# ── Hydration budget ────────────────────────────────────────────────────────

class BudgetExhausted(Exception):
    """Raised when a download would exceed the caller's byte ceiling."""


@dataclass
class HydrationBudget:
    """Hard ceiling on bytes downloaded during one classification pass.

    Because eviction is unavailable, every byte pulled is a byte that stays on
    the volume. The budget tracks both an explicit ceiling and the volume's
    real free space, refusing to start a download that would breach either.

    ``reserve_bytes`` keeps a floor of free space so a pass can never fill the
    boot volume; the default leaves 5 GB of headroom.
    """

    limit_bytes: int
    volume: str = "/"
    reserve_bytes: int = 5 * 1024 ** 3
    spent_bytes: int = 0
    downloaded_files: int = 0
    skipped_files: int = 0

    def remaining(self) -> int:
        return max(0, self.limit_bytes - self.spent_bytes)

    def can_afford(self, size: int) -> bool:
        if size > self.remaining():
            return False
        return free_bytes(self.volume) - size > self.reserve_bytes

    def charge(self, size: int) -> None:
        """Record a download, or raise if it does not fit."""
        if not self.can_afford(size):
            self.skipped_files += 1
            raise BudgetExhausted(
                f"downloading {size / 1e6:.1f} MB would exceed the "
                f"{self.limit_bytes / 1e9:.1f} GB budget or leave under "
                f"{self.reserve_bytes / 1e9:.1f} GB free"
            )
        self.spent_bytes += size
        self.downloaded_files += 1

    def as_dict(self) -> dict:
        return {
            "limit_bytes": self.limit_bytes,
            "spent_bytes": self.spent_bytes,
            "remaining_bytes": self.remaining(),
            "downloaded_files": self.downloaded_files,
            "skipped_files": self.skipped_files,
            "free_bytes": free_bytes(self.volume),
        }


# ── Capture timestamps ──────────────────────────────────────────────────────

# ``YYYY/MM/DD`` (or ``YYYY/MM``) embedded in a directory path. Date-foldered
# archives are common and this is by far the cheapest reliable date source.
_PATH_DATE_RE = re.compile(r"(?:^|/)(19[7-9]\d|20[0-4]\d)/(0[1-9]|1[0-2])(?:/(0[1-9]|[12]\d|3[01]))?(?=/|$)")

# Dates embedded in filenames, e.g. IMG_20180704_101500, 2018-07-04, 20180704.
_NAME_DATE_RES = [
    re.compile(r"(?<!\d)(19[7-9]\d|20[0-4]\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"),
]


def date_from_path(path: str) -> Optional[datetime]:
    """Recover a capture date from directory structure (cold).

    Matches the *last* occurrence so a root like ``/Archive/2019/`` cannot
    shadow the per-file ``2022/03/04`` further down the path.
    """
    matches = list(_PATH_DATE_RE.finditer(path.replace(os.sep, "/")))
    if not matches:
        return None
    y, m, d = matches[-1].groups()
    try:
        return datetime(int(y), int(m), int(d or 1))
    except ValueError:
        return None


def date_from_name(path: str) -> Optional[datetime]:
    """Recover a capture date from the filename (cold)."""
    name = os.path.basename(path)
    for rx in _NAME_DATE_RES:
        m = rx.search(name)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _parse_ts(value: str) -> Optional[datetime]:
    """Parse the timestamp spellings sips and ffprobe emit."""
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    return None


def cold_capture_date(st: FileStat) -> tuple[Optional[datetime], str]:
    """Best capture date obtainable without downloading. Returns (dt, source).

    Order matters. Path and filename dates reflect when the content was made;
    ``st_mtime`` reflects when the *file* was last written, which a cloud sync
    or a bulk copy will have rewritten to the migration date. On a 15-year
    archive that difference is the whole ballgame, so mtime is the last resort
    and is labelled as such so the UI can flag it as low-confidence.
    """
    dt = date_from_path(st.path)
    if dt:
        return dt, "path"
    dt = date_from_name(st.path)
    if dt:
        return dt, "filename"
    try:
        return datetime.fromtimestamp(st.mtime), "mtime"
    except (OSError, OverflowError, ValueError):
        return None, "unknown"


# ── Warm probes (these download the file) ───────────────────────────────────

def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe_image(path: str) -> dict:
    """Read image dimensions and EXIF-ish fields via ``sips`` (warm).

    ``sips -g all`` emits ``key: value`` lines; unknown keys are passed through
    so callers can look for fields this code does not name explicitly.
    """
    if not SIPS:
        return {}
    try:
        res = _run([SIPS, "-g", "all", path], PROBE_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("sips probe failed for %s: %s", path, exc)
        return {}
    out: dict = {}
    for line in res.stdout.splitlines()[1:]:
        if ":" not in line:
            continue
        key, _, val = line.strip().partition(":")
        out[key.strip()] = val.strip()
    meta = {
        "width": _int_or_none(out.get("pixelWidth")),
        "height": _int_or_none(out.get("pixelHeight")),
        "make": out.get("make"),
        "model": out.get("model"),
        "software": out.get("software"),
        "format": out.get("format"),
    }
    dt = _parse_ts(out.get("creation", ""))
    if dt:
        meta["captured_at"] = dt.isoformat(timespec="seconds")
    return {k: v for k, v in meta.items() if v not in (None, "")}


def _int_or_none(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def probe_video(path: str) -> dict:
    """Read container metadata via ``ffprobe`` (warm).

    Beyond duration and creation time this detects two things that matter for
    classification:

    * **Live Photo sidecars.** iPhones write a 2-3 second ``.MOV`` next to
      every still shot. They are a large fraction of a phone archive's video
      count but carry no independent content, so they are flagged and folded
      into their still rather than classified separately.
    * **GPS.** QuickTime stores an ISO-6709 location string, which gives a
      strong location signal for travel-type categories at no AI cost.
    """
    if not FFPROBE:
        return {}
    try:
        res = _run([FFPROBE, "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", path], PROBE_TIMEOUT)
        data = json.loads(res.stdout or "{}")
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return {}

    fmt = data.get("format", {}) or {}
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    streams = data.get("streams", []) or []

    duration = None
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        pass

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video = next((s for s in streams if s.get("codec_type") == "video"), {})

    is_live_photo = any("live-photo" in k for k in tags)

    meta = {
        "duration": duration,
        "has_audio": has_audio,
        "width": _int_or_none(video.get("width")),
        "height": _int_or_none(video.get("height")),
        "codec": video.get("codec_name"),
        "is_live_photo": is_live_photo,
        "make": tags.get("com.apple.quicktime.make") or tags.get("make"),
        "model": tags.get("com.apple.quicktime.model") or tags.get("model"),
    }

    dt = _parse_ts(tags.get("creation_time", ""))
    if dt:
        meta["captured_at"] = dt.isoformat(timespec="seconds")

    loc = tags.get("com.apple.quicktime.location.iso6709") or tags.get("location")
    coords = _parse_iso6709(loc) if loc else None
    if coords:
        meta["lat"], meta["lon"] = coords

    return {k: v for k, v in meta.items() if v not in (None, "")}


def _parse_iso6709(value: str) -> Optional[tuple[float, float]]:
    """Parse an ISO-6709 location string like ``+25.7617-080.1918+003.000/``."""
    m = re.match(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)", (value or "").strip())
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def thumbnail_image(path: str, dest: str, max_px: int = 640) -> bool:
    """Write a downscaled JPEG of an image to ``dest`` (warm).

    ``sips`` is used rather than Pillow because HEIC is the single most common
    format in a modern phone archive and Pillow cannot read it without an
    optional native dependency. Downscaling here also keeps the vision model's
    input small, which is most of its per-image latency.
    """
    if not SIPS:
        return False
    try:
        res = _run([SIPS, "-s", "format", "jpeg", "-Z", str(max_px),
                    path, "--out", dest], CONVERT_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("sips thumbnail failed for %s: %s", path, exc)
        return False
    return res.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def frame_is_blank(path: str, stdev_threshold: float = 6.0) -> bool:
    """Whether a rendered frame carries no visual information.

    Archives accumulate frames that are solid black or solid white: failed
    captures, video poster frames, placeholder assets. They look like valid
    JPEGs and a vision model will describe them at full cost and full
    confidence ("a solid black background"), which then classifies the whole
    event on the strength of nothing.

    Standard deviation of luminance is the discriminator rather than mean --
    a solid white frame has a high mean but, like a solid black one, no
    variation. Real photographs sit far above this threshold.
    """
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as im:
            stat = ImageStat.Stat(im.convert("L"))
            return stat.stddev[0] < stdev_threshold
    except Exception:
        # If the frame cannot be inspected, let the caller try to use it.
        return False


def extract_frames(path: str, dest_dir: str, count: int = 3,
                   max_px: int = 640, duration: Optional[float] = None) -> list[str]:
    """Extract up to ``count`` evenly-spaced JPEG frames from a video (warm).

    Frames are taken at interior offsets rather than from the start: the first
    moments of a clip are frequently a lens cap, a blurred pan, or a black
    fade, none of which describe the clip's subject.
    """
    if not FFMPEG:
        return []
    os.makedirs(dest_dir, exist_ok=True)
    if duration is None:
        duration = (probe_video(path) or {}).get("duration")
    frames: list[str] = []

    if not duration or duration <= 0:
        offsets = [0.0]
    else:
        # e.g. count=3 -> 25%, 50%, 75% of the way through.
        offsets = [duration * (i + 1) / (count + 1) for i in range(count)]

    for idx, offset in enumerate(offsets):
        out = os.path.join(dest_dir, f"frame_{idx:02d}.jpg")
        cmd = [FFMPEG, "-v", "quiet", "-y", "-ss", f"{offset:.2f}", "-i", path,
               "-frames:v", "1", "-vf", f"scale='min({max_px},iw)':-2", out]
        try:
            _run(cmd, CONVERT_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("frame extraction failed for %s: %s", path, exc)
            continue
        if os.path.exists(out) and os.path.getsize(out) > 0:
            frames.append(out)
    return frames


def extract_audio(path: str, dest: str, max_seconds: int = 600) -> bool:
    """Extract mono 16 kHz WAV audio for speech-to-text (warm).

    16 kHz mono is what Whisper resamples to internally, so producing it here
    avoids a second conversion and keeps the temp file small. Long recordings
    are truncated: the opening minutes are enough to categorise a clip, and a
    full transcode of an hour-long file would dominate the pass.
    """
    if not FFMPEG:
        return False
    cmd = [FFMPEG, "-v", "quiet", "-y", "-i", path, "-vn",
           "-ac", "1", "-ar", "16000", "-t", str(max_seconds), dest]
    try:
        res = _run(cmd, CONVERT_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("audio extraction failed for %s: %s", path, exc)
        return False
    return res.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


class TempWorkspace:
    """Self-cleaning scratch directory for thumbnails and extracted frames."""

    def __init__(self, prefix: str = "insights_lib_"):
        self.prefix = prefix
        self.path: Optional[str] = None

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self.prefix)
        return self.path

    def __exit__(self, *exc) -> None:
        if self.path:
            shutil.rmtree(self.path, ignore_errors=True)
        self.path = None

"""Content Library: catalogue a large media archive and sort it by year and category.

The problem this solves is a many-year backlog of photos and video spread over
a cloud-backed folder, where the owner has hand-sorted a small slice into named
categories and wants the rest to follow. Three properties of that situation
drive the whole design:

1. **The bytes are mostly not on the local disk.** Cloud providers keep files
   as placeholders and download on first read. An archive is routinely far
   larger than the volume's free space, so reading every file is not merely
   slow, it is impossible. Everything that can be derived from paths and inode
   metadata is therefore derived that way first -- see :func:`scan_root`, which
   handles tens of thousands of files in seconds and downloads nothing.

2. **The owner has already defined the taxonomy.** A hand-triaged folder's
   subdirectory names *are* the category list, expressed in the owner's own
   vocabulary. :func:`learn_taxonomy` reads them rather than asking a model to
   invent categories, so results land in the labels the owner already uses.

3. **Media arrives in events, not as independent files.** Forty photos from one
   afternoon share a subject. Classifying the *event* and propagating the label
   (:func:`group_events`) cuts AI work by more than an order of magnitude and
   produces better labels, because an event has a coherent subject where an
   individual frame often does not.

The classification ladder runs cheapest-first and stops as soon as a tier is
confident:

===== ============================== ============ ==================
Tier   Signal                         Downloads?   Cost
===== ============================== ============ ==================
0      path date, extension, size     no           free
1      path/filename keyword rules    no           free
2      vision caption -> category     yes          free (local Ollama)
3      speech transcript -> category  yes          free (local Whisper)
===== ============================== ============ ==================

Tiers 2 and 3 are governed by a :class:`~media_probe.HydrationBudget`, and only
run against a small sample of representative files per event.

This module deliberately imports neither ``insights_web`` nor ``database``: it
is a pure engine driven by callbacks, which keeps it testable against a scratch
directory and lets the web layer own persistence and progress reporting.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Iterable, Optional, Sequence

from concurrent.futures import ThreadPoolExecutor

import media_probe
from media_probe import (
    BudgetExhausted,
    FileStat,
    HydrationBudget,
    TempWorkspace,
    cold_capture_date,
)

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.2")

# Label used when no tier reaches the confidence floor. Kept as a real category
# so unresolved files stay visible and reviewable instead of vanishing.
UNSORTED = "Unsorted"

# A tier's result is accepted only at or above this confidence.
MIN_CONFIDENCE = 0.45

# Minimum gap between the best and second-best keyword match. Below this the
# match is treated as unresolved rather than decided on a near-tie.
AMBIGUITY_MARGIN = 0.15

# How many separate events must land on a proposed category before it is
# treated as established. A hand-sorted folder reflects what its owner had time
# to file, not the true shape of the archive, so the classifier is allowed to
# name subjects the taxonomy misses. But a model asked for a new name will
# happily invent one per event, so a proposal has to recur before it is
# promoted; single-event proposals are kept as inactive suggestions instead of
# fragmenting the taxonomy into hundreds of near-synonyms.
NEW_CATEGORY_MIN_EVENTS = 3

# Ceiling on discovered categories offered back to the model, so the prompt
# cannot grow without bound on a long run.
MAX_DISCOVERED_CATEGORIES = 40


# ── Taxonomy ────────────────────────────────────────────────────────────────

# Tokens too generic to identify a category. Matching on these produces
# confident-looking nonsense ("Misc" matching every path containing "misc").
_STOPWORDS = {
    "the", "and", "of", "a", "an", "to", "in", "on", "at", "for", "with",
    "misc", "other", "new", "content", "video", "videos", "photo", "photos",
    "pics", "pic", "image", "images", "media", "file", "files", "footage",
    "export", "exports", "raw", "final", "edit", "edits", "clip", "clips",
    # Calendar and qualifier words. These occur constantly in filenames while
    # carrying no category signal, and a category name that happens to contain
    # one will otherwise capture unrelated files: a clip called
    # "Portraits - Over the Years" matching a "New Years Fireworks" category on
    # the strength of the word "year" alone.
    "year", "day", "time", "night", "morning", "evening", "week", "month",
    "over", "out", "now", "old", "best", "top", "first", "last", "part",
}


@dataclass
class Category:
    """One class in the owner's taxonomy, learned from a triaged folder."""

    name: str
    subcategories: list[str] = field(default_factory=list)
    example_count: int = 0
    keywords: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _stem(token: str) -> str:
    """Strip a plural ``s`` so ``cars`` and ``car`` compare equal.

    Deliberately minimal rather than a real stemmer: category names and
    captions differ almost entirely in number ("Cars" vs "a sports car"), and
    aggressive stemming would collapse distinct words. Short tokens are left
    alone so acronyms survive intact.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Split a folder or file name into lowercase, singularised word tokens."""
    return [_stem(t) for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def learn_taxonomy(triaged_root: str, min_token_len: int = 3) -> list[Category]:
    """Read a hand-sorted folder and return its categories (cold).

    Each immediate subdirectory of ``triaged_root`` becomes a category; its own
    subdirectories become subcategories. Keywords are drawn from both, so a
    category like ``Travel - Caribbean`` with a ``Yacht`` subfolder matches
    paths mentioning either.

    Tokens shorter than ``min_token_len`` are kept only when the category name
    is itself that short (``MMA``, ``FPOV``), which preserves meaningful
    acronyms without admitting noise.
    """
    if not triaged_root or not os.path.isdir(triaged_root):
        return []

    categories: list[Category] = []
    for entry in sorted(os.listdir(triaged_root)):
        path = os.path.join(triaged_root, entry)
        if entry.startswith(".") or not os.path.isdir(path):
            continue

        subs: list[str] = []
        count = 0
        for dirpath, dirnames, filenames in os.walk(path):
            if dirpath == path:
                subs = [d for d in sorted(dirnames) if not d.startswith(".")]
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            count += sum(
                1 for f in filenames
                if media_probe.file_kind(os.path.join(dirpath, f)) in ("image", "video", "audio")
            )

        tokens: list[str] = []
        for source in [entry] + subs:
            for tok in _tokenize(source):
                if tok in _STOPWORDS or tok.isdigit():
                    continue
                if len(tok) < min_token_len and len(_tokenize(entry)) > 1:
                    continue
                tokens.append(tok)

        categories.append(Category(
            name=entry,
            subcategories=subs,
            example_count=count,
            keywords=sorted(set(tokens)),
        ))
    return categories


_NAME_CLEAN = re.compile(r"[^\w &'-]+")


def canonical_category_name(name: str) -> str:
    """Normalise a model-proposed category name into a filesystem-safe label.

    Proposals arrive with quotes, trailing punctuation, and inconsistent case.
    Normalising here means "beach trips", "Beach Trips." and '"Beach Trips"'
    converge on one category instead of three, and the result is directly
    usable as a folder name.
    """
    cleaned = _NAME_CLEAN.sub(" ", (name or "").strip()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    # Title case, but leave existing acronyms (MMA, FPOV) alone.
    words = [w if w.isupper() and len(w) <= 5 else w.capitalize()
             for w in cleaned.split(" ")]
    return " ".join(words)[:60]


def merge_category(name: str, taxonomy: Sequence[Category]) -> Optional[str]:
    """Return an existing category equivalent to ``name``, if there is one.

    Catches the case where a proposal restates a category that already exists
    under a slightly different wording ("Martial Arts" against "MMA" will not
    match, but "Beach Trip" against "Beach Trips" will), so discovery extends
    the taxonomy rather than shadowing it.
    """
    if not name:
        return None
    target = {t for t in _tokenize(name)} - _STOPWORDS
    if not target:
        return None
    for cat in taxonomy:
        existing = {t for t in _tokenize(cat.name)} - _STOPWORDS
        if existing and existing == target:
            return cat.name
    return None


# Names that describe a junk drawer rather than a subject.
_CATCH_ALL = {"misc", "miscellaneous", "other", "others", "random", "stuff",
              "general", "unsorted", "various", "assorted"}


def is_catch_all(name: str) -> bool:
    """Whether a category is a junk drawer rather than a real subject."""
    tokens = set(_tokenize(name))
    return bool(tokens) and tokens <= _CATCH_ALL


def taxonomy_prompt_block(taxonomy: Sequence[Category],
                          exclude_catch_all: bool = False) -> str:
    """Render the label space for an LLM prompt, one category per line.

    ``exclude_catch_all`` drops junk-drawer categories such as "Misc". They
    have to go during discovery, because they are strictly easier to choose
    than naming a subject and the model takes that option every time: observed
    on a real archive, memes, quote graphics and screenshots each recurred
    across several events -- exactly the subjects discovery exists to surface --
    and every one of them was filed under "Misc" with the reason "fits no
    specific recurring subject". Removing the escape hatch costs nothing, since
    "Unsorted" still covers genuinely unreadable content.
    """
    lines = []
    for cat in taxonomy:
        if exclude_catch_all and is_catch_all(cat.name):
            continue
        if cat.subcategories:
            hint = ", ".join(cat.subcategories[:6])
            lines.append(f"- {cat.name} (includes: {hint})")
        else:
            lines.append(f"- {cat.name}")
    lines.append(f"- {UNSORTED} (use when nothing above clearly fits)")
    return "\n".join(lines)


# ── Scanning ────────────────────────────────────────────────────────────────

@dataclass
class ScannedFile:
    """A catalogued file. Populated entirely from cold metadata at scan time."""

    path: str
    rel_path: str
    name: str
    ext: str
    kind: str
    size: int
    mtime: float
    materialized: bool
    captured_at: Optional[str]      # ISO 8601
    date_source: str                # path | filename | mtime | exif | container
    year: Optional[int]
    month: Optional[int]
    seq: Optional[int]              # trailing number in the filename, if any
    event_key: str = ""
    dup_group: int = 1              # how many archive files share this identity
    category: str = ""
    subcategory: str = ""
    confidence: float = 0.0
    classified_by: str = ""         # rule | vision | speech | propagated | manual
    caption: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


_SEQ_RE = re.compile(r"(\d{2,})(?!.*\d)")


def _sequence_number(name: str) -> Optional[int]:
    """Trailing number in a filename stem, e.g. ``IMG_5146`` -> 5146."""
    stem = os.path.splitext(name)[0]
    m = _SEQ_RE.search(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def scan_root(
    root: str,
    include_kinds: Sequence[str] = ("image", "video", "audio", "document"),
    progress: Optional[Callable[[int], None]] = None,
) -> list[ScannedFile]:
    """Walk ``root`` and build the catalogue from metadata alone (cold).

    Downloads nothing and opens nothing. Sidecars (DJI proxies, ``.THM``
    thumbnails, ``.AAE`` edit lists) and OS bookkeeping files are dropped here
    so they never reach the classifier or inflate the counts shown to the user.
    """
    root = os.path.abspath(os.path.expanduser(root))
    out: list[ScannedFile] = []
    seen = 0

    for st in media_probe.walk_tree(root):
        seen += 1
        if progress and seen % 2000 == 0:
            progress(seen)
        if st.kind not in include_kinds:
            continue

        dt, source = cold_capture_date(st)
        rel = os.path.relpath(st.path, root)
        name = os.path.basename(st.path)

        out.append(ScannedFile(
            path=st.path,
            rel_path=rel,
            name=name,
            ext=st.ext,
            kind=st.kind,
            size=st.size,
            mtime=st.mtime,
            materialized=st.materialized,
            captured_at=dt.isoformat(timespec="seconds") if dt else None,
            date_source=source,
            year=dt.year if dt else None,
            month=dt.month if dt else None,
            seq=_sequence_number(name),
        ))

    if progress:
        progress(seen)
    mark_duplicates(out)
    return out


def mark_duplicates(files: Sequence[ScannedFile]) -> dict[tuple, int]:
    """Stamp each file with the size of its duplicate group (cold).

    Identity is ``(filename, byte size)``. That is a proxy for content equality
    rather than a proof of it -- only hashing would prove it, and hashing means
    downloading the archive -- but for the case that actually matters it is
    exact: the same file copied or re-synced into many folders keeps both its
    name and its size. A true collision needs two *different* files sharing a
    name and a byte count exactly, which is vanishingly rare outside generated
    filenames.

    This catches two things worth surfacing on a long backlog: genuine
    duplicates the owner may want to reclaim space from, and replicated
    application assets -- a placeholder or poster frame that a messaging app
    has scattered across dozens of folders. The latter are actively harmful as
    classification samples, since they are small, often already local, and
    therefore exactly what a cost-ranked sampler reaches for first.
    """
    groups: dict[tuple, int] = Counter((f.name, f.size) for f in files)
    for f in files:
        f.dup_group = groups[(f.name, f.size)]
    return dict(groups)


def duplicate_report(files: Sequence[ScannedFile], min_group: int = 2) -> list[dict]:
    """Summarise duplicate groups, largest reclaimable space first."""
    by_key: dict[tuple, list[ScannedFile]] = defaultdict(list)
    for f in files:
        if f.dup_group >= min_group:
            by_key[(f.name, f.size)].append(f)

    report = []
    for (name, size), group in by_key.items():
        report.append({
            "name": name,
            "size": size,
            "count": len(group),
            # Keeping one copy is the point; the rest is what can be reclaimed.
            "reclaimable_bytes": size * (len(group) - 1),
            "paths": [f.rel_path for f in group[:20]],
        })
    report.sort(key=lambda r: r["reclaimable_bytes"], reverse=True)
    return report


# ── Event grouping ──────────────────────────────────────────────────────────

# Two files whose sequence numbers differ by more than this are treated as
# separate events. Phone cameras number monotonically, so a large gap inside
# one folder means unrelated shooting sessions.
SEQUENCE_GAP = 60

# Below this fraction of parseable sequence numbers, fall back to grouping the
# whole directory as one event -- UUID-named exports carry no ordering.
SEQUENCE_COVERAGE = 0.6

# Marks the motion half of a folder that holds both stills and video. Chosen to
# be a character that cannot occur in a path component, so a split key can never
# collide with a real directory.
MOTION_SUFFIX = "\x1fv"

# Sequence splitting only applies to folders with at least this many files.
# In a date-foldered archive the directory already *is* the event, and
# splitting a 6-photo day into three "events" triples the AI cost while making
# every label worse -- a single frame is far weaker evidence than a session.
SEQUENCE_SPLIT_MIN_FILES = 20


def group_events(files: Sequence[ScannedFile]) -> dict[str, list[ScannedFile]]:
    """Cluster files into shooting events and stamp ``event_key`` on each.

    Grouping is by containing directory, then split on sequence-number gaps.
    Directory is the primary key because date-foldered archives already encode
    the day, and because an export folder is itself an authored grouping.

    Runs on cold metadata only, so the full archive can be grouped before any
    decision about what to download.
    """
    by_dir: dict[str, list[ScannedFile]] = defaultdict(list)
    for f in files:
        by_dir[os.path.dirname(f.rel_path)].append(f)

    # Stills and motion in the same folder are separated before clustering.
    # Sampling prefers stills because they are far cheaper to fetch, so in a
    # mixed folder the label is always derived from a photo and then inherited
    # by the video beside it. That is right when both show the same occasion and
    # badly wrong when they do not -- a screenshot saved on the same day as a
    # drone flight described the flight as a motivational quote. Splitting means
    # each kind is sampled and labelled on its own evidence.
    #
    # The suffix is applied only to the motion side, and only when a folder
    # actually holds both. Keys are the catalogue's identity for an event, so
    # leaving the still side untouched keeps every existing label attached to
    # the event that earned it; only the newly separated motion events look
    # unclassified, which is exactly what should be reconsidered.
    split: dict[str, list[ScannedFile]] = {}
    for dirname, group in by_dir.items():
        stills = [f for f in group if f.kind == "image"]
        motion = [f for f in group if f.kind != "image"]
        if stills and motion:
            split[dirname] = stills
            split[dirname + MOTION_SUFFIX] = motion
        else:
            split[dirname] = group

    events: dict[str, list[ScannedFile]] = {}
    for dirname, group in split.items():
        with_seq = [f for f in group if f.seq is not None]
        coverage = len(with_seq) / len(group) if group else 0

        if coverage < SEQUENCE_COVERAGE or len(group) < SEQUENCE_SPLIT_MIN_FILES:
            key = f"{dirname}#0"
            for f in group:
                f.event_key = key
            events[key] = list(group)
            continue

        ordered = sorted(group, key=lambda f: (f.seq if f.seq is not None else 0, f.name))
        cluster_idx = 0
        current: list[ScannedFile] = []
        prev_seq: Optional[int] = None

        for f in ordered:
            if prev_seq is not None and f.seq is not None and f.seq - prev_seq > SEQUENCE_GAP:
                key = f"{dirname}#{cluster_idx}"
                for g in current:
                    g.event_key = key
                events[key] = current
                cluster_idx += 1
                current = []
            current.append(f)
            if f.seq is not None:
                prev_seq = f.seq

        if current:
            key = f"{dirname}#{cluster_idx}"
            for g in current:
                g.event_key = key
            events[key] = current

    return events


def event_summary(key: str, files: Sequence[ScannedFile]) -> dict:
    """Aggregate facts about one event, for display and for prompting."""
    years = sorted({f.year for f in files if f.year})
    kinds = Counter(f.kind for f in files)
    dates = sorted({f.captured_at[:10] for f in files if f.captured_at})
    return {
        "event_key": key,
        "directory": os.path.dirname(files[0].rel_path) if files else "",
        "file_count": len(files),
        "total_bytes": sum(f.size for f in files),
        "years": years,
        "year": years[0] if years else None,
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "kinds": dict(kinds),
        "materialized": sum(1 for f in files if f.materialized),
    }


# Largest single file worth downloading purely to identify an event.
#
# This ceiling is the difference between a pass that fits on the volume and one
# that cannot run at all. In a representative phone archive stills have a median
# size of ~1 MB while video runs to a median of ~13 MB and a 99th percentile
# above 1 GB, so a cap generous enough to admit "most" video admits the tail
# that dominates the total. 16 MB keeps roughly half of video-only events in
# reach for a few GB; the rest are deferred rather than silently downloaded.
MAX_SAMPLE_BYTES = 16 * 1024 ** 2

# A file whose (name, size) identity recurs at least this many times across the
# archive is treated as a shared asset and kept out of the sample pool.
DUPLICATE_SAMPLE_LIMIT = 3

# How far ahead of the classifier to warm downloads, and with how many threads.
#
# Concurrency is the single most effective tuning knob in this pipeline. Cloud
# providers serve these archives at well under a megabyte per second per file
# because per-request latency, not bandwidth, is the limit -- so parallel reads
# scale almost linearly. Measured on a real archive: one thread sustained
# 0.30 MB/s, four reached 3.11 MB/s, and eight only 3.56 MB/s. Six sits past the
# knee without piling on requests the provider will throttle.
#
# Depth stays small on purpose: the goal is to keep the network busy while the
# GPU captions, not to race ahead and materialize far more of the archive than
# the run will actually use.
PREFETCH_DEPTH = 6
PREFETCH_WORKERS = int(os.environ.get("LIBRARY_PREFETCH_WORKERS", "6"))


def choose_samples(files: Sequence[ScannedFile], max_samples: int = 2,
                   max_sample_bytes: int = MAX_SAMPLE_BYTES,
                   motion_samples: int = 1) -> list[ScannedFile]:
    """Pick the files that best represent an event, cheapest-to-read first.

    On a cloud-backed archive the dominant cost is bytes downloaded, and the
    cheapest evidence is usually just as good as the most expensive: a photo and
    a video shot minutes apart show the same subject, but the photo costs an
    order of magnitude less to fetch. Two rules follow, and together they are
    what makes a full pass feasible:

    * **If the event contains any still image, only stills are sampled.** Most
      events are mixed, so this converts the majority of the archive from a
      video-priced problem into an image-priced one at no cost in accuracy --
      the label applies to the whole event either way.
    * **Video-only events get a single sample, and only if it fits the cap.**
      Returning an empty list signals the caller to defer the event rather than
      spend hundreds of megabytes identifying a handful of files.

    Already-materialized files are always eligible whatever their size, since
    reading them costs nothing. Samples are otherwise spread across the event so
    a long session is not judged by its first frame alone.
    """
    usable = [f for f in files if f.kind in ("image", "video", "audio")]
    if not usable:
        return []

    # A file that recurs across the archive is an app asset, not this event's
    # content. Excluding it matters more than it sounds: such files are small
    # and frequently already local, so a cost-ranked sampler picks them first
    # and would describe many unrelated events with the same stock frame.
    distinctive = [f for f in usable if f.dup_group < DUPLICATE_SAMPLE_LIMIT]
    if distinctive:
        usable = distinctive

    images = [f for f in usable if f.kind == "image"]
    if images:
        pool, want = images, max_samples
    else:
        # Motion-only event. One clip is the cheap default, because each video
        # sample costs a whole file. But one clip also *is* the whole event's
        # evidence, so a dud -- a blank opening frame, a Live Photo, an
        # establishing shot of empty sky -- sends every file in the event to
        # Unsorted no matter how clear the others are. Measured on a real
        # archive, clips that failed as an event's lone sample classified
        # correctly when judged individually. ``motion_samples`` lets a
        # follow-up pass buy a second opinion where a first pass could not
        # afford one.
        pool, want = usable, max(1, motion_samples)

    # Files already on disk are free whatever their kind, but the cap still
    # applies: taking max_samples of them here would quietly ignore the motion
    # limit on an event whose clips happen to be local.
    resident = [f for f in usable if f.materialized]
    if len(resident) >= want:
        return _spread(resident, want)

    affordable = [f for f in pool if f.size <= max_sample_bytes or f.materialized]
    if not affordable:
        return []   # caller marks the event deferred; nothing is downloaded

    picked = _spread(affordable, max(1, want - len(resident)))
    return resident + [f for f in picked if f not in resident]


def sample_cost(files: Sequence[ScannedFile], max_samples: int = 2,
                max_sample_bytes: int = MAX_SAMPLE_BYTES,
                motion_samples: int = 1) -> int:
    """Bytes that classifying this event would download. 0 if already local."""
    return sum(
        s.size for s in choose_samples(files, max_samples, max_sample_bytes, motion_samples)
        if not s.materialized
    )


# Fallback hydration rate for a cloud-backed provider, in bytes per second,
# used only when calibration cannot run. Cold reads are network-bound, so on a
# large archive the download can rival or exceed inference as the dominant cost
# and an estimate that counts only model time understates the real wait.
# :func:`measure_hydration_rate` replaces this with a figure measured on the
# actual archive at the concurrency the run will use.
DEFAULT_HYDRATION_BPS = 3.0 * 1024 ** 2

# Measured warm latency of one local vision caption, in seconds.
VISION_SECONDS_PER_SAMPLE = 1.5


def measure_hydration_rate(files: Sequence["ScannedFile"],
                           probes: int = PREFETCH_WORKERS * 2) -> Optional[float]:
    """Time a few real cold reads to calibrate the download estimate (warm).

    Returns bytes per second, or None if nothing suitable was available.

    The measurement runs at the same concurrency as the real prefetcher, which
    matters more than any other detail here: these providers are limited by
    per-request latency rather than bandwidth, so a serial probe understates
    achievable throughput by roughly ten times and turns a three-hour estimate
    into a thirty-hour one. Timing a parallel batch reports what the run will
    actually sustain.

    The first read of a session carries a large one-off penalty while the
    provider wakes up, so it is done separately and excluded from the timing.
    """
    import time

    # The catalogue's materialized flag is a snapshot from scan time. Any file
    # a previous pass downloaded still reads as remote there, and timing one
    # would measure local disk -- yielding hundreds of MB/s and an estimate off
    # by orders of magnitude. Re-check each candidate against the live
    # filesystem so only genuinely cold files are timed.
    candidates: list[ScannedFile] = []
    for f in files:
        if f.materialized or f.kind != "image" or not (200_000 < f.size < 3_000_000):
            continue
        if media_probe.is_materialized(f.path):
            continue        # hydrated since the scan; would time a local read
        candidates.append(f)
        if len(candidates) > probes:
            break
    if len(candidates) < 3:
        return None

    def read_bytes(path: str) -> int:
        try:
            total = 0
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
            return total
        except OSError:
            return 0

    read_bytes(candidates[0].path)      # absorb the provider's wake-up cost
    batch = candidates[1:]
    if not batch:
        return None

    start = time.time()
    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        total = sum(pool.map(read_bytes, [f.path for f in batch]))
    elapsed = time.time() - start

    if elapsed <= 0 or total <= 0:
        return None
    return total / elapsed


def estimate_pass(
    events: dict[str, list["ScannedFile"]],
    taxonomy: Sequence[Category],
    max_samples: int = 2,
    max_sample_bytes: int = MAX_SAMPLE_BYTES,
    hydration_bps: Optional[float] = None,
    motion_samples: int = 1,
) -> dict:
    """Predict the cost of an AI pass before committing to it (cold).

    Runs the free rule tier and the sample chooser over every event without
    reading a byte, so the UI can state exactly how much will be downloaded,
    how long it will take, and how much of the archive will still be unlabelled
    afterwards. Every number here is a real prediction, not a rule of thumb.
    """
    total_files = sum(len(v) for v in events.values())
    by_rule = ai_events = deferred_events = 0
    ai_files = deferred_files = 0
    download = 0
    samples = 0

    for files in events.values():
        _, confidence, _ = rule_classify(files, taxonomy)
        if confidence >= MIN_CONFIDENCE:
            by_rule += 1
            continue
        picked = choose_samples(files, max_samples, max_sample_bytes, motion_samples)
        if not picked:
            deferred_events += 1
            deferred_files += len(files)
            continue
        ai_events += 1
        ai_files += len(files)
        samples += len(picked)
        download += sum(s.size for s in picked if not s.materialized)

    covered = total_files - deferred_files
    rate = hydration_bps or DEFAULT_HYDRATION_BPS
    download_seconds = int(download / rate) if rate else 0
    vision_seconds = int(samples * VISION_SECONDS_PER_SAMPLE)

    return {
        "events_total": len(events),
        "files_total": total_files,
        "resolved_by_rule": by_rule,
        "ai_events": ai_events,
        "ai_files": ai_files,
        "sample_files": samples,
        "download_bytes": download,
        "deferred_events": deferred_events,
        "deferred_files": deferred_files,
        "coverage": covered / total_files if total_files else 0.0,
        "download_seconds": download_seconds,
        "vision_seconds": vision_seconds,
        # Downloading and captioning overlap (see the prefetcher in
        # classify_events), so the wall clock is closer to the larger of the two
        # than to their sum -- but never better than the larger one.
        "est_seconds": max(download_seconds, vision_seconds) + min(download_seconds, vision_seconds) // 4,
        "hydration_bps": rate,
        "free_bytes": media_probe.free_bytes("/"),
    }


def _spread(files: Sequence[ScannedFile], count: int) -> list[ScannedFile]:
    """Take ``count`` files spaced evenly across a capture-ordered sequence."""
    if len(files) <= count:
        return sorted(files, key=lambda f: f.size)
    ordered = sorted(files, key=lambda f: (f.seq if f.seq is not None else 0, f.name))
    stride = len(ordered) / count
    out: list[ScannedFile] = []
    for i in range(count):
        f = ordered[min(int(i * stride), len(ordered) - 1)]
        if f not in out:
            out.append(f)
    return out


# ── Tier 1: rules ───────────────────────────────────────────────────────────

def _score_taxonomy(
    weighted: Sequence[tuple[Counter, float, float]],
    taxonomy: Sequence[Category],
    floor: float = 0.45,
    ceiling: float = 0.95,
) -> tuple[str, float, str]:
    """Score token counters against the taxonomy and return the best match.

    ``weighted`` is a list of ``(tokens, name_weight, keyword_weight)``: each
    token source contributes at its own strength, so a directory name can
    outrank a filename and a category's own name can outrank its subcategories'.
    """
    scored: list[tuple[str, float, str]] = []
    for cat in taxonomy:
        cat_tokens = {_stem(t) for t in _tokenize(cat.name)} - _STOPWORDS
        sub_tokens = {_stem(t) for t in cat.keywords} - cat_tokens

        score = 0.0
        hits: list[str] = []
        for tokens, name_w, kw_w in weighted:
            for tok in cat_tokens:
                if tok in tokens:
                    score += name_w
                    hits.append(tok)
            for tok in sub_tokens:
                if tok in tokens:
                    score += kw_w
                    hits.append(tok)

        if score <= 0:
            continue
        denom = max(1, len(cat_tokens))
        confidence = min(ceiling, floor + 0.5 * (score / denom))
        scored.append((cat.name, confidence, f"matched {sorted(set(hits))}"))

    if not scored:
        return "", 0.0, ""

    scored.sort(key=lambda s: s[1], reverse=True)
    best = scored[0]

    # Category names overlap in ordinary English -- a photo of a "sports car"
    # matches both "Sports Therapist" and "ATVs Bikes Cars". Where two
    # categories are close, the token overlap simply does not identify which,
    # and asserting the higher score would file real content under a category
    # it has nothing to do with. Declining is recoverable; a confident wrong
    # label is what makes a review queue untrustworthy.
    if len(scored) > 1 and best[1] - scored[1][1] < AMBIGUITY_MARGIN:
        return "", 0.0, (f"ambiguous between {best[0]!r} and {scored[1][0]!r}")

    return best


def keyword_classify_text(text: str, taxonomy: Sequence[Category]) -> tuple[str, float, str]:
    """Match free text -- a caption or transcript -- against the taxonomy.

    This exists because the mapping step is the pipeline's accuracy bottleneck.
    A small local text model reliably produces a good *caption* but will drop
    an obvious mapping: given "two people on all-terrain vehicles (ATVs)" and a
    category literally named ``ATVs Bikes Cars``, a 3B model often answers
    "Unsorted". Literal token overlap costs nothing, never hallucinates, and
    catches exactly the cases the model fumbles, so it runs as a safety net
    beneath the model rather than as a replacement for it.

    Confidence is capped below what the model can claim, so a genuine model
    answer always wins over a keyword coincidence.
    """
    if not text or not taxonomy:
        return "", 0.0, ""
    tokens = Counter(_tokenize(text))
    return _score_taxonomy([(tokens, 1.0, 0.6)], taxonomy, floor=0.40, ceiling=0.75)


def rule_classify(files: Sequence[ScannedFile], taxonomy: Sequence[Category]) -> tuple[str, float, str]:
    """Match an event's paths against taxonomy keywords (cold, free).

    Scoring favours specificity: a match on the category name itself outranks a
    match on one of its subcategory names, and a match on a directory component
    outranks one on a filename, because directories are deliberate groupings
    while filenames are mostly camera-assigned.

    Returns ``(category, confidence, reason)``; confidence is 0 when nothing
    matched, which sends the event down to the AI tiers.
    """
    if not files or not taxonomy:
        return "", 0.0, ""

    dir_tokens = Counter()
    name_tokens = Counter()
    for f in files:
        parts = f.rel_path.replace(os.sep, "/").split("/")
        for tok in _tokenize(" ".join(parts[:-1])):
            dir_tokens[tok] += 1
        for tok in _tokenize(parts[-1]):
            name_tokens[tok] += 1

    # Directories outrank filenames: a folder name is a deliberate grouping,
    # while a filename is usually whatever the camera assigned.
    category, confidence, hits = _score_taxonomy(
        [(dir_tokens, 1.0, 0.6), (name_tokens, 0.5, 0.3)], taxonomy)
    return (category, confidence, f"path {hits}" if hits else "")


# ── Tier 2/3: local model calls ─────────────────────────────────────────────

class OllamaUnavailable(Exception):
    """Raised when the local model server cannot be reached."""


def _ollama_generate(model: str, prompt: str, images: Optional[list[str]] = None,
                     num_predict: int = 90, timeout: int = 300) -> str:
    """Call Ollama's native generate endpoint and return the response text.

    The native endpoint is used rather than the OpenAI-compatible shim because
    it takes base64 images directly, avoiding a data-URI round trip for what is
    already a latency-sensitive inner loop.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.1},
    }
    if images:
        payload["images"] = images

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaUnavailable(f"Ollama call failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable(f"Ollama returned invalid JSON: {exc}") from exc
    return (body.get("response") or "").strip()


def ollama_status() -> dict:
    """Report reachability and whether the configured models are installed."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read())
    except Exception:
        return {"available": False, "models": [], "vision_ready": False, "text_ready": False,
                "vision_model": OLLAMA_VISION_MODEL, "text_model": OLLAMA_TEXT_MODEL}

    names = [m.get("name", "") for m in tags.get("models", [])]
    stems = {n.split(":")[0] for n in names}
    return {
        "available": True,
        "models": names,
        "vision_model": OLLAMA_VISION_MODEL,
        "text_model": OLLAMA_TEXT_MODEL,
        "vision_ready": OLLAMA_VISION_MODEL.split(":")[0] in stems,
        "text_ready": OLLAMA_TEXT_MODEL.split(":")[0] in stems,
    }


CAPTION_PROMPT = (
    "Describe this photo or video frame in one factual sentence. "
    "State the main subject, the setting, and what is happening. "
    "If it is a screenshot, a document, or a receipt, say so explicitly. "
    "Do not speculate about names, dates, or emotions."
)


def caption_file(f: ScannedFile, budget: HydrationBudget, workspace: str,
                 vision_model: str = "", video_frames: int = 2) -> tuple[str, dict]:
    """Download one file, render it to a JPEG, and caption it (warm).

    Returns ``(caption, metadata)``. ``metadata`` carries whatever the container
    probe recovered -- capture time, GPS, Live Photo status -- which is often
    more valuable than the caption itself and is kept even when captioning
    fails.

    Raises :class:`BudgetExhausted` before touching the file if its size does
    not fit the remaining download allowance.
    """
    model = vision_model or OLLAMA_VISION_MODEL
    meta: dict = {}

    if not f.materialized:
        budget.charge(f.size)   # raises BudgetExhausted; nothing read yet

    frames: list[str] = []
    if f.kind == "image":
        meta = media_probe.probe_image(f.path) or {}
        thumb = os.path.join(workspace, "thumb.jpg")
        if media_probe.thumbnail_image(f.path, thumb):
            frames = [thumb]
    elif f.kind == "video":
        meta = media_probe.probe_video(f.path) or {}
        # A Live Photo sidecar is the motion attached to a still, not its own
        # clip. Recording that fact is the useful outcome; captioning it would
        # spend a model call to redescribe a photo already in the catalogue.
        if meta.get("is_live_photo"):
            return "", meta
        # More frames is the cheapest available accuracy gain for video: the
        # file is already downloaded by this point, so each extra frame costs a
        # scrub and an inference and nothing on the network. One or two frames
        # of a moving subject is thin evidence -- a drone shot can open on empty
        # sky and land on a rooftop -- and thin evidence is what left so much
        # footage unresolved.
        frames = media_probe.extract_frames(
            f.path, os.path.join(workspace, "frames"),
            count=max(1, video_frames), duration=meta.get("duration"),
        )
    elif f.kind == "audio":
        return "", {}

    if not frames:
        return "", meta

    # Drop empty frames before paying for inference. A model will describe a
    # solid black frame confidently and at full price, and that description
    # then drives the label for every file in the event.
    usable_frames = [fr for fr in frames if not media_probe.frame_is_blank(fr)]
    if not usable_frames:
        meta["blank"] = True
        return "", meta
    frames = usable_frames

    captions: list[str] = []
    for frame in frames:
        try:
            with open(frame, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
        except OSError:
            continue
        try:
            text = _ollama_generate(model, CAPTION_PROMPT, images=[b64], num_predict=90)
        except OllamaUnavailable:
            raise
        if text:
            captions.append(text)

    return " ".join(captions).strip(), meta


def transcribe_file(f: ScannedFile, budget: HydrationBudget, workspace: str,
                    max_seconds: int = 300) -> str:
    """Extract speech from a video or audio file using local Whisper (warm).

    Transcription is the slowest tier by a wide margin, so it is reserved for
    events that vision could not resolve -- typically talking-head footage,
    podcasts, and interviews, where the words carry the topic and the picture
    does not.
    """
    if not f.materialized:
        budget.charge(f.size)

    if f.kind == "video":
        probe = media_probe.probe_video(f.path) or {}
        if not probe.get("has_audio") or (probe.get("duration") or 0) < 3:
            return ""

    wav = os.path.join(workspace, "audio.wav")
    if not media_probe.extract_audio(f.path, wav, max_seconds=max_seconds):
        return ""

    try:
        import insights
        return (insights.transcribe_audio(wav) or "").strip()
    except Exception as exc:
        logger.warning("transcription failed for %s: %s", f.path, exc)
        return ""


CLASSIFY_PROMPT = """You are sorting a personal media archive into the owner's own categories.

Categories:
{categories}

Evidence about one group of {count} files taken {when}:
Folder: {folder}
{evidence}

Reply with ONLY a JSON object, no other text:
{{"category": "<exact category name from the list>", "confidence": <0.0-1.0>, "reason": "<under 12 words>"}}

Rules:
- "category" must be copied exactly from the list above.
- Use "{unsorted}" if no category clearly fits. Do not guess.
- Base the answer only on the evidence given."""


DISCOVERY_PROMPT = """You are sorting a personal media archive into categories.

Categories used so far:
{categories}

Evidence about one group of {count} files taken {when}:
Folder: {folder}
{evidence}

Reply with ONLY a JSON object, no other text:
{{"category": "<name>", "is_new": <true or false>, "confidence": <0.0-1.0>, "reason": "<under 12 words>"}}

Rules:
- If one of the categories above genuinely fits, copy its name exactly and set
  "is_new": false. The list is a starting point, not a constraint -- it may be
  incomplete or wrong, so do not force content into a category it does not fit.
- If the content clearly belongs to some other subject, name that subject and
  set "is_new": true.
- A new name must be short (1-3 words) and describe a recurring subject, not
  this one moment: "Beach Trips", not "Sunset On A Tuesday". Prefer a name that
  other similar content could also use.
- Never answer with a catch-all like "Misc", "Other", "Random" or "General".
  Those are not subjects. Content of a recognisable kind -- a meme, a saved
  quote graphic, a screenshot of a conversation, a receipt -- has a nameable
  subject, so name it.
- Use "{unsorted}" only when the evidence is too weak to say anything at all.
- Base the answer only on the evidence given."""


def _cloud_map(prompt: str, model: str = "") -> str:
    """Run the mapping prompt against the configured cloud LLM.

    Only the *mapping* step is ever sent off-device, never the images. It is a
    few hundred tokens of text per event, so covering an entire archive costs
    well under a dollar -- a worthwhile option when the local 3B model's
    mapping accuracy is the limiting factor, and cheap enough that it does not
    undermine the local-first design.
    """
    import insights
    import usage_meter

    client, resolved, provider = insights._get_llm_client(use_local=False, model=model or None)
    response = client.chat.completions.create(
        model=resolved,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=160,
    )
    try:
        usage_meter.record_chat(response, category="content_library",
                                provider=provider, model=resolved)
    except Exception:
        logger.debug("usage metering failed for content library mapping", exc_info=True)
    return (response.choices[0].message.content or "").strip()


def classify_from_evidence(
    summary: dict,
    evidence: str,
    taxonomy: Sequence[Category],
    text_model: str = "",
    use_cloud: bool = False,
    allow_new: bool = False,
) -> tuple[str, float, str, bool]:
    """Map captions/transcripts onto the taxonomy with the local text model.

    Kept separate from captioning on purpose. Captions are expensive and depend
    only on the pixels, so they are cached on the file; category assignment is
    cheap and depends on the taxonomy, which the owner may revise. Splitting
    the two means re-running classification after a taxonomy edit costs seconds
    instead of re-reading the archive.
    """
    if not evidence.strip():
        return "", 0.0, ""

    names = {c.name for c in taxonomy}
    when = summary.get("date_start") or "an unknown date"
    prompt = (DISCOVERY_PROMPT if allow_new else CLASSIFY_PROMPT).format(
        categories=taxonomy_prompt_block(taxonomy, exclude_catch_all=allow_new),
        count=summary.get("file_count", 0),
        when=when,
        folder=summary.get("directory", ""),
        evidence=evidence[:4000],
        unsorted=UNSORTED,
    )

    if use_cloud:
        try:
            raw = _cloud_map(prompt, text_model)
        except Exception as exc:
            logger.warning("cloud mapping failed, falling back to local: %s", exc)
            raw = _ollama_generate(text_model or OLLAMA_TEXT_MODEL, prompt, num_predict=160)
    else:
        raw = _ollama_generate(text_model or OLLAMA_TEXT_MODEL, prompt, num_predict=160)

    # Literal token overlap, computed regardless of what the model says, so a
    # dropped-but-obvious mapping can still be recovered below.
    kw_category, kw_confidence, kw_reason = keyword_classify_text(evidence, taxonomy)

    parsed = _extract_json(raw)
    if not parsed:
        if kw_category:
            return kw_category, kw_confidence, kw_reason, False
        return "", 0.0, "", False

    category = str(parsed.get("category", "")).strip()
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(parsed.get("reason", "")).strip()[:120]
    claims_new = bool(parsed.get("is_new")) and allow_new

    # The model declining to choose is the single most common failure mode, and
    # it is frequently wrong in an easily checkable way. Defer to literal
    # evidence when the caption plainly names a category the model passed over.
    if category == UNSORTED or not category:
        if kw_category:
            return kw_category, kw_confidence, f"caption {kw_reason}", False
        return UNSORTED, confidence, reason, False

    def best_of(cat: str, conf: float, why: str, is_new: bool):
        """Prefer literal evidence when the model's own answer is too weak.

        Without this, a model reply that names the right thing in its reasoning
        but reports low confidence is discarded outright: an "Ace MMA & Fitness
        open mat" advert came back at confidence 0.00 and was filed Unsorted,
        even though the caption says MMA and an MMA category exists. The
        keyword match is only allowed to win when the model has effectively
        abstained, so a confident model answer is never second-guessed.
        """
        if conf < MIN_CONFIDENCE and kw_category and kw_confidence >= MIN_CONFIDENCE:
            return kw_category, kw_confidence, f"caption {kw_reason}", False
        return cat, max(0.0, min(1.0, conf)), why, is_new

    if category in names:
        return best_of(category, confidence, reason, False)

    # An unrecognised label means one of two things, and which one depends on
    # whether discovery is enabled. With it off, it is a near-miss or an
    # invention to be snapped back or discarded. With it on, it may be a
    # genuine gap in the taxonomy, so it is passed up as a proposal for the
    # caller to weigh against how often it recurs.
    if allow_new and claims_new:
        proposed = canonical_category_name(category)
        # A model told not to use a junk drawer will still occasionally propose
        # one as "new". Treat that as an abstention rather than creating it.
        if not proposed or is_catch_all(proposed):
            return best_of("", 0.0, "", False)
        merged = merge_category(proposed, taxonomy)
        if merged:
            return best_of(merged, confidence, reason, False)
        return best_of(proposed, confidence, reason, True)

    match = _closest_category(category, names)
    if not match:
        if kw_category:
            return kw_category, kw_confidence, f"caption {kw_reason}", False
        return "", 0.0, "", False
    return best_of(match, min(confidence, 0.6), reason, False)


def _closest_category(guess: str, names: Iterable[str]) -> Optional[str]:
    """Resolve a model's near-miss label to a real category name."""
    g = set(_tokenize(guess))
    if not g:
        return None
    best, best_score = None, 0.0
    for name in names:
        n = set(_tokenize(name))
        if not n:
            continue
        overlap = len(g & n) / len(g | n)
        if overlap > best_score:
            best, best_score = name, overlap
    return best if best_score >= 0.5 else None


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def build_evidence(captions: Sequence[str], transcript: str, sample_names: Sequence[str]) -> str:
    """Assemble the evidence block handed to :func:`classify_from_evidence`."""
    parts: list[str] = []
    if sample_names:
        parts.append("Filenames: " + ", ".join(sample_names[:6]))
    for i, cap in enumerate(c for c in captions if c):
        parts.append(f"Image {i + 1}: {cap}")
    if transcript:
        parts.append(f"Speech transcript: {transcript[:1500]}")
    return "\n".join(parts)


# ── Copy plan ───────────────────────────────────────────────────────────────

# Characters that are unsafe or annoying in a destination path.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_component(text: str, fallback: str = "Unsorted") -> str:
    """Sanitise one path component."""
    cleaned = _UNSAFE.sub("_", (text or "").strip()).strip(". ")
    return cleaned or fallback


def plan_destination(f: ScannedFile, layout: str = "category_year") -> str:
    """Destination path for one file, relative to the plan's output root.

    ``category_year`` groups by subject first, matching how a hand-triaged
    folder is usually organised; ``year_category`` inverts it for
    chronological browsing.
    """
    category = safe_component(f.category or UNSORTED)
    year = str(f.year) if f.year else "Undated"
    sub = safe_component(f.subcategory) if f.subcategory else ""

    if layout == "year_category":
        parts = [year, category]
    else:
        parts = [category, year]
    if sub:
        parts.append(sub)
    parts.append(f.name)
    return os.path.join(*parts)


@dataclass
class PlanItem:
    """One proposed copy, pending the owner's approval."""

    src: str
    dest_rel: str
    size: int
    category: str
    year: Optional[int]
    confidence: float
    classified_by: str
    collision: bool = False


def build_plan(files: Sequence[ScannedFile], layout: str = "category_year") -> list[PlanItem]:
    """Turn classified files into a reviewable list of copy operations.

    Destination collisions are detected and flagged rather than silently
    resolved: two different files landing on one name almost always means the
    archive holds duplicates the owner should see, and quietly renaming one
    would hide that.
    """
    items: list[PlanItem] = []
    taken: dict[str, int] = defaultdict(int)

    for f in files:
        dest = plan_destination(f, layout)
        taken[dest] += 1
        items.append(PlanItem(
            src=f.path,
            dest_rel=dest,
            size=f.size,
            category=f.category or UNSORTED,
            year=f.year,
            confidence=f.confidence,
            classified_by=f.classified_by,
            collision=taken[dest] > 1,
        ))
    return items


def apply_plan(
    items: Sequence[PlanItem],
    dest_root: str,
    manifest_path: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    mode: str = "copy",
) -> dict:
    """Materialize approved items under ``dest_root``, writing an undo manifest.

    Never moves, so a mistake in the plan costs disk space rather than
    originals. Every item is appended to a JSONL manifest as it completes,
    which makes the operation resumable and reversible even if the process dies
    partway.

    Two modes, and on a cloud-backed archive the choice is not cosmetic:

    ``copy``
        Duplicates the bytes. Copying a placeholder downloads it first, so the
        plan's total size is a real download and a real second copy on disk.
        Only viable when that total fits the volume.
    ``link``
        Writes a symlink pointing at the original. Costs no bytes and downloads
        nothing, because a symlink refers to the file without reading it. The
        result is a browsable Category/Year tree over an archive far larger
        than the disk, where opening any entry fetches just that file. The
        tradeoff is that it is a view, not a duplicate: it breaks if the
        originals move, and it is not a backup.
    """
    dest_root = os.path.abspath(os.path.expanduser(dest_root))
    os.makedirs(dest_root, exist_ok=True)

    copied = skipped = failed = 0
    errors: list[str] = []
    manifest = open(manifest_path, "a", encoding="utf-8") if manifest_path else None

    try:
        for idx, item in enumerate(items):
            target = os.path.join(dest_root, item.dest_rel)
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)

                # Idempotent: an entry already at the target means an earlier
                # run placed it, so resuming is safe.
                #
                # A link is compared by what it resolves to, not by the string
                # it stores -- those differ whenever a path reaches here in a
                # different but equivalent form, and a mismatch would silently
                # write a second link beside the first on every re-run.
                if os.path.islink(target):
                    same = False
                    try:
                        same = os.path.realpath(target) == os.path.realpath(item.src)
                    except OSError:
                        same = False
                    if same:
                        skipped += 1
                        continue
                elif os.path.exists(target) and os.path.getsize(target) == item.size:
                    skipped += 1
                    continue
                if os.path.exists(target) or os.path.islink(target):
                    target = _dedupe_name(target)

                if mode == "link":
                    os.symlink(item.src, target)
                else:
                    shutil.copy2(item.src, target)
                copied += 1
                if manifest:
                    manifest.write(json.dumps({
                        "src": item.src,
                        "dest": target,
                        "size": item.size,
                        "category": item.category,
                        "mode": mode,
                        "copied_at": datetime.now().isoformat(timespec="seconds"),
                    }) + "\n")
                    manifest.flush()
            except (OSError, shutil.Error) as exc:
                failed += 1
                if len(errors) < 25:
                    errors.append(f"{os.path.basename(item.src)}: {exc}")
            if progress:
                progress(idx + 1, len(items))
    finally:
        if manifest:
            manifest.close()

    return {"copied": copied, "skipped": skipped, "failed": failed, "errors": errors}


def _dedupe_name(path: str) -> str:
    """Append ``-1``, ``-2``, ... until the path is free."""
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}-{n}{ext}"):
        n += 1
    return f"{stem}-{n}{ext}"


def undo_plan(manifest_path: str) -> dict:
    """Delete the copies recorded in a manifest, leaving originals untouched.

    Only paths written by :func:`apply_plan` are removed, and only when they
    still match the recorded size, so an edited or replaced file is left alone.
    """
    removed = missing = failed = 0
    if not os.path.exists(manifest_path):
        return {"removed": 0, "missing": 0, "failed": 0}

    seen: set[str] = set()
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A destination can appear more than once if an apply was re-run,
            # and counting it twice would report removing more than exists.
            if rec.get("dest") in seen:
                continue
            seen.add(rec.get("dest"))
            dest = rec.get("dest")
            if not dest or not (os.path.exists(dest) or os.path.islink(dest)):
                missing += 1
                continue
            try:
                # A link is removed on identity, not size: os.path.getsize
                # follows it to the original, which must never be deleted.
                if os.path.islink(dest):
                    if os.path.realpath(dest) == os.path.realpath(rec.get("src") or ""):
                        os.remove(dest)
                        removed += 1
                    else:
                        missing += 1
                elif os.path.getsize(dest) == rec.get("size"):
                    os.remove(dest)
                    removed += 1
                else:
                    missing += 1
            except OSError:
                failed += 1
    return {"removed": removed, "missing": missing, "failed": failed}


# ── Orchestration ───────────────────────────────────────────────────────────

def classify_events(
    events: dict[str, list[ScannedFile]],
    taxonomy: Sequence[Category],
    budget: HydrationBudget,
    use_ai: bool = True,
    use_speech: bool = False,
    max_samples: int = 2,
    max_sample_bytes: int = MAX_SAMPLE_BYTES,
    on_event: Optional[Callable[[str, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    vision_model: str = "",
    text_model: str = "",
    use_cloud_mapping: bool = False,
    allow_new_categories: bool = False,
    video_frames: int = 2,
    motion_samples: int = 1,
    on_category: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """Run the full ladder over every event and label its files in place.

    Events are processed cheapest-first: rules resolve what they can for free,
    and the surviving events are ordered so that those with already-downloaded
    files come first. That ordering means a budget-limited run spends its
    allowance on the events it can inspect most cheaply, and a run that is
    stopped early still produces useful coverage.

    ``on_event`` is called after each event with its summary so the caller can
    persist incrementally; ``should_stop`` is polled between events so a run
    can be cancelled from the UI without losing completed work.
    """
    stats = Counter()
    stats["events_total"] = len(events)

    # Discovery state. ``live_taxonomy`` grows during the run so a subject named
    # once is offered back to the model for later events -- without that, the
    # same subject gets a slightly different name every time and never
    # accumulates the evidence needed to be promoted.
    live_taxonomy: list[Category] = list(taxonomy)
    original_names: set[str] = {c.name for c in taxonomy}
    proposals: Counter = Counter()
    promoted: set[str] = set()

    ai_queue: list[tuple[str, list[ScannedFile]]] = []

    # Pass 1 -- free rules over everything.
    for key, files in events.items():
        category, confidence, reason = rule_classify(files, taxonomy)
        if confidence >= MIN_CONFIDENCE:
            for f in files:
                f.category, f.confidence, f.classified_by = category, confidence, "rule"
                f.notes = reason
            stats["by_rule"] += 1
            if on_event:
                on_event(key, {**event_summary(key, files), "category": category,
                               "confidence": confidence, "classified_by": "rule",
                               "reason": reason})
        else:
            ai_queue.append((key, files))

    if not use_ai or not ai_queue:
        stats["unresolved"] = len(ai_queue)
        for key, files in ai_queue:
            for f in files:
                f.category, f.classified_by = UNSORTED, "unresolved"
        return dict(stats)

    status = ollama_status()
    if not status["available"] or not status["vision_ready"]:
        logger.warning("local vision model unavailable (%s); leaving %d events unresolved",
                       status, len(ai_queue))
        stats["unresolved"] = len(ai_queue)
        stats["ollama_unavailable"] = 1
        for key, files in ai_queue:
            for f in files:
                f.category, f.classified_by = UNSORTED, "unresolved"
        return dict(stats)

    # Downloading is network-bound and captioning is GPU-bound, so they can run
    # at the same time. Warming the next few events' samples while the current
    # one is being captioned turns a strictly serial download-then-caption loop
    # into an overlapped one, which on a cloud archive is the difference between
    # the two costs adding up and only the larger of them mattering.
    def _warm(path: str) -> None:
        try:
            with open(path, "rb") as fh:
                while fh.read(1 << 20):
                    pass
        except OSError:
            pass    # a failed prefetch just means the caption path downloads it

    # Cheapest-to-inspect first: events whose samples are already on disk.
    def event_cost(pair) -> tuple:
        _, files = pair
        samples = choose_samples(files, max_samples, max_sample_bytes, motion_samples)
        if not samples:
            return (2, 0)   # deferred: sort last, resolved without downloading
        return (
            0 if all(s.materialized for s in samples) else 1,
            sum(s.size for s in samples if not s.materialized),
        )

    ai_queue.sort(key=event_cost)

    prefetch = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS,
                                  thread_name_prefix="library-prefetch")
    prefetched = 0

    def pump(cursor: int) -> None:
        """Queue downloads for the events just ahead of the cursor."""
        nonlocal prefetched
        while prefetched < min(cursor + PREFETCH_DEPTH, len(ai_queue)):
            _, upcoming = ai_queue[prefetched]
            prefetched += 1
            for sample in choose_samples(upcoming, max_samples, max_sample_bytes, motion_samples):
                # Check affordability without charging; caption_file does the
                # accounting so a file is never counted twice.
                if not sample.materialized and budget.can_afford(sample.size):
                    prefetch.submit(_warm, sample.path)

    # Pass 2 -- vision, then speech for what vision could not resolve.
    try:
        for index, (key, files) in enumerate(ai_queue):
            pump(index)
            if should_stop and should_stop():
                stats["stopped"] = 1
                break

            summary = event_summary(key, files)
            samples = choose_samples(files, max_samples, max_sample_bytes, motion_samples)
            if not samples:
                # Every candidate exceeds the per-file download ceiling. Leave
                # the event labelled and visible so the owner can opt into the
                # cost for this one specifically, rather than quietly spending
                # the budget.
                stats["deferred_large"] += 1
                for f in files:
                    f.category, f.classified_by = UNSORTED, "deferred_large"
                    f.notes = "smallest file exceeds the download ceiling"
                continue

            captions: list[str] = []
            transcript = ""
            exhausted = False

            with TempWorkspace() as workspace:
                for sample in samples:
                    try:
                        caption, meta = caption_file(sample, budget, workspace,
                                                     vision_model, video_frames)
                    except BudgetExhausted:
                        exhausted = True
                        break
                    except OllamaUnavailable:
                        stats["ollama_unavailable"] = 1
                        exhausted = True
                        break
                    if caption:
                        captions.append(caption)
                        sample.caption = caption
                    if meta.get("is_live_photo"):
                        sample.notes = "live photo sidecar"
                        stats["live_photos"] += 1
                    if meta.get("blank"):
                        stats["blank_frames"] += 1
                    # A capture time read from the container beats a path date.
                    if meta.get("captured_at"):
                        sample.captured_at = meta["captured_at"]
                        sample.date_source = "exif"

                if not exhausted and use_speech and not captions:
                    speech_sample = next(
                        (s for s in files if s.kind in ("video", "audio")), None
                    )
                    if speech_sample is not None:
                        try:
                            transcript = transcribe_file(speech_sample, budget, workspace)
                            if transcript:
                                stats["transcribed"] += 1
                        except BudgetExhausted:
                            exhausted = True

            if exhausted and not captions and not transcript:
                stats["budget_skipped"] += 1
                for f in files:
                    if not f.category:
                        f.category, f.classified_by = UNSORTED, "budget_exhausted"
                continue

            evidence = build_evidence(captions, transcript, [s.name for s in samples])
            try:
                category, confidence, reason, is_new = classify_from_evidence(
                    summary, evidence, live_taxonomy, text_model,
                    use_cloud=use_cloud_mapping, allow_new=allow_new_categories)
            except OllamaUnavailable:
                stats["ollama_unavailable"] = 1
                break

            if is_new and category:
                stats["proposed"] += 1
                # Offer the name back to the model straight away so related
                # events converge on it instead of each coining a variant.
                if not any(c.name == category for c in live_taxonomy):
                    if len(live_taxonomy) - len(taxonomy) < MAX_DISCOVERED_CATEGORIES:
                        live_taxonomy.append(Category(name=category, keywords=[
                            t for t in _tokenize(category) if t not in _STOPWORDS]))

            method = "speech" if transcript and not captions else "vision"
            if not category or confidence < MIN_CONFIDENCE:
                category, method = UNSORTED, "low_confidence"
                stats["low_confidence"] += 1
            elif category == UNSORTED:
                # The model was confident that nothing fits. That is a
                # legitimate answer, but it is not a classification -- counting
                # it as one would overstate how much of the archive got sorted.
                method = "no_match"
                stats["no_match"] += 1
            else:
                stats[f"by_{method}"] += 1

            # Count every event that lands on a category the owner did not
            # define, not just the one that first proposed it. Counting
            # proposals alone silently breaks the promotion rule: a name is fed
            # back into the label space as soon as it is coined, so from the
            # second event onward the model is *selecting* it rather than
            # proposing it, and the counter stays at one forever. Observed on a
            # real run, "Memes" collected 18 events and was still never adopted.
            if category and category != UNSORTED and category not in original_names:
                proposals[category] += 1
                if (proposals[category] >= NEW_CATEGORY_MIN_EVENTS
                        and category not in promoted):
                    promoted.add(category)
                    stats["categories_discovered"] += 1
                    if on_category:
                        on_category(category, proposals[category])

            for f in files:
                f.category = category
                f.confidence = confidence
                f.classified_by = method if f in samples else "propagated"
                f.notes = reason

            if on_event:
                on_event(key, {**summary, "category": category, "confidence": confidence,
                               "classified_by": method, "reason": reason,
                               "captions": captions, "transcript": transcript[:500]})
    finally:
        # Abandon queued downloads immediately on stop; in-flight reads finish
        # on their own rather than being interrupted mid-file.
        prefetch.shutdown(wait=False, cancel_futures=True)

    stats["budget"] = budget.as_dict()
    # Report every proposal with its event count, flagged by whether it cleared
    # the promotion threshold. Below-threshold names are still worth showing --
    # they are the classifier's read on what the taxonomy is missing, and the
    # owner may recognise one as real even from a single event.
    stats["discovered"] = [
        {"name": name, "events": count, "promoted": name in promoted}
        for name, count in proposals.most_common()
    ]
    return dict(stats)


# Values of ``classified_by`` that mean an event was genuinely processed, and
# so should not be paid for twice when resuming.
#
# "low_confidence" and "no_match" belong here: the samples were downloaded and
# captioned, and the honest conclusion was that nothing fit. Re-running would
# spend the same money to reach the same answer.
#
# Deliberately absent are the states that represent work *not* done:
# "budget_exhausted" (the allowance ran out), "unresolved" (the model server was
# unreachable), and "deferred_large" (every candidate exceeded the size cap).
# Leaving those pending is what lets a resumed run with a bigger budget or a
# raised cap pick up exactly the events the previous run had to abandon.
CLASSIFIED_MARKERS = frozenset({
    "rule", "vision", "speech", "manual", "propagated", "low_confidence", "no_match",
})

# A hand-set label. Unlike the markers above it is never revisited, not even by
# a run that explicitly retries the unresolved: the owner has already answered.
PINNED_MARKER = "manual"


def event_is_classified(files: Sequence[ScannedFile]) -> bool:
    """Whether an event already carries the result of a completed pass."""
    return any((f.classified_by or "") in CLASSIFIED_MARKERS for f in files)


def pending_events(
    events: dict[str, list[ScannedFile]],
    retry_unresolved: bool = False,
) -> dict[str, list[ScannedFile]]:
    """Drop events a previous pass already resolved, for resuming a run.

    A full pass over a large cloud-backed archive runs for hours, and an
    interruption partway through -- a dropped connection, a reboot, a stopped
    job -- would otherwise mean starting from zero and re-downloading
    everything. Results are persisted per event as the run proceeds, so the
    completed ones are simply skipped on the next attempt.

    ``retry_unresolved`` additionally reopens events that were examined but
    ended up unlabelled. Ordinary resume treats those as finished, because
    re-running an unchanged pipeline would spend the same money to reach the
    same answer. That reasoning stops holding the moment the taxonomy changes:
    an event the model could not place among a hundred overlapping categories
    may be obvious among fifty clean ones, and its samples are usually already
    on disk, so the retry is close to free.
    """
    out = {}
    for key, files in events.items():
        # A correction the owner made by hand is final. Reopening it would undo
        # review work and refill the queue with labels already fixed.
        if any((f.classified_by or "") == PINNED_MARKER for f in files):
            continue
        if not event_is_classified(files):
            out[key] = files
        elif retry_unresolved and _event_is_unlabelled(files):
            out[key] = files
    return out


def _event_is_unlabelled(files: Sequence[ScannedFile]) -> bool:
    """Whether an event was examined but came out with no usable category."""
    return all((f.category or UNSORTED) == UNSORTED for f in files)


def filter_events_by_year(
    events: dict[str, list[ScannedFile]],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> dict[str, list[ScannedFile]]:
    """Restrict events to a year range, for staged runs over a large archive.

    An event is kept when any of its files fall in range; date-foldered events
    do not straddle years in practice, and keeping a partially-matching event
    whole avoids splitting one shoot across two runs.
    """
    if year_from is None and year_to is None:
        return events
    lo = year_from if year_from is not None else -10_000
    hi = year_to if year_to is not None else 10_000
    return {
        key: files for key, files in events.items()
        if any(f.year is not None and lo <= f.year <= hi for f in files)
    }

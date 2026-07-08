"""AI usage metering: capture token usage and estimated cost for paid API calls.

Imported by ``insights.py`` right after each paid OpenAI / Ollama call. Each recorder
reads the response's ``usage`` block, estimates a dollar cost from the local price
table below, resolves whether the call happened *proactively* (automated background
work) or *reactively* (a user-triggered web request), and writes a row via
``database.log_usage``.

Design rules
------------
* Recording MUST NEVER raise into generation code. Every public recorder swallows its
  own exceptions and logs a warning instead. ``insights.py`` also calls these through a
  guarded ``_meter`` helper, so even an import failure here cannot break generation.
* Prices are ESTIMATES. OpenAI does not return a dollar amount, so cost is computed from
  ``PRICING`` / the module constants below, each overridable via a ``USAGE_PRICE_*`` (or
  ``USAGE_WHISPER_PER_MIN``) environment variable. Token counts, by contrast, are exact.
* Ollama / local generation is recorded at $0 (tokens still captured) so local vs cloud
  spend stays visible.
"""

from __future__ import annotations

import contextvars
import logging
import os

logger = logging.getLogger(__name__)


# ── Pricing (USD) ──────────────────────────────────────────────────────────
# Chat rates are dollars per 1,000,000 tokens (input / output). These are estimates;
# tune them to your plan via the matching USAGE_PRICE_* env vars.

def _envf(name: str, default: float) -> float:
    """Return env var ``name`` as a float, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return float(default)


# Keys are matched by longest-prefix against the model name, so e.g.
# "gpt-4o-2024-08-06" resolves to the "gpt-4o" entry and "gpt-4o-mini-..." to
# "gpt-4o-mini".
PRICING = {
    "gpt-4o-mini": {
        "in": _envf("USAGE_PRICE_GPT_4O_MINI_IN", 0.15),
        "out": _envf("USAGE_PRICE_GPT_4O_MINI_OUT", 0.60),
    },
    "gpt-4o": {
        "in": _envf("USAGE_PRICE_GPT_4O_IN", 2.50),
        "out": _envf("USAGE_PRICE_GPT_4O_OUT", 10.00),
    },
    "gpt-4.1-mini": {
        "in": _envf("USAGE_PRICE_GPT_41_MINI_IN", 0.40),
        "out": _envf("USAGE_PRICE_GPT_41_MINI_OUT", 1.60),
    },
    "gpt-4.1": {
        "in": _envf("USAGE_PRICE_GPT_41_IN", 2.00),
        "out": _envf("USAGE_PRICE_GPT_41_OUT", 8.00),
    },
}

# Used for any model name not matched above (assume gpt-4o-class pricing).
DEFAULT_PRICE = {
    "in": _envf("USAGE_PRICE_DEFAULT_IN", 2.50),
    "out": _envf("USAGE_PRICE_DEFAULT_OUT", 10.00),
}

# OpenAI Whisper transcription, USD per minute of audio.
WHISPER_PER_MIN = _envf("USAGE_WHISPER_PER_MIN", 0.006)

# gpt-image-1 generation, USD per image (estimate for a large, high-quality image).
IMAGE_PER_CALL = _envf("USAGE_PRICE_IMAGE", 0.19)


def _price_for(model: str) -> dict:
    """Return the {in, out} per-1M-token price for ``model`` (longest-prefix match)."""
    name = (model or "").lower()
    best = None
    for key in PRICING:
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return PRICING[best] if best else DEFAULT_PRICE


# ── Proactive vs reactive resolution ───────────────────────────────────────
# An explicit override (set by the background episode worker) wins; otherwise we
# infer from the Flask request context: inside a request => reactive, else proactive.

_MODE_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("usage_mode", default=None)


class usage_context:
    """Context manager forcing the recorded mode (e.g. ``"proactive"``) for a block.

    ContextVars are per-thread, so wrapping the background worker thread tags every
    generation it performs as proactive without affecting request threads.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self._token = None

    def __enter__(self):
        self._token = _MODE_OVERRIDE.set(self.mode)
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _MODE_OVERRIDE.reset(self._token)
        return False


def _resolve_mode() -> str:
    override = _MODE_OVERRIDE.get()
    if override:
        return override
    try:
        from flask import has_request_context

        return "reactive" if has_request_context() else "proactive"
    except Exception:
        return "proactive"


def _resolve_user():
    """Best-effort ``(user_id, username)`` for the current request, else ``(None, None)``."""
    try:
        from flask import has_request_context

        if not has_request_context():
            return None, None
        import insights_web  # late import: avoids an import cycle at module load

        user = insights_web.current_user()
        if user:
            return user["id"], user["username"]
    except Exception:
        pass
    return None, None


# ── Recorders ──────────────────────────────────────────────────────────────

def _record(
    *,
    category: str,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    audio_seconds: float = 0.0,
    images: int = 0,
    cost_usd: float = 0.0,
    details=None,
) -> None:
    try:
        import database

        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        database.log_usage(
            mode=_resolve_mode(),
            category=category,
            provider=provider,
            model=model or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            audio_seconds=float(audio_seconds or 0.0),
            images=int(images or 0),
            cost_usd=round(float(cost_usd or 0.0), 6),
            user_id=_resolve_user()[0],
            username=_resolve_user()[1],
            details=details,
        )
    except Exception:
        logger.warning("usage metering failed (non-fatal)", exc_info=True)


def record_chat(response, *, category: str, provider: str, model: str) -> None:
    """Record a ``chat.completions`` call from its response's ``usage`` block."""
    try:
        usage = getattr(response, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        if provider == "openai":
            price = _price_for(model)
            cost = pt / 1_000_000 * price["in"] + ct / 1_000_000 * price["out"]
        else:
            cost = 0.0
        _record(
            category=category,
            provider=provider,
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
        )
    except Exception:
        logger.warning("record_chat failed (non-fatal)", exc_info=True)


def _audio_duration_seconds(audio_path) -> float:
    """Return the audio file's duration in seconds via ffprobe, or 0 on failure."""
    if not audio_path:
        return 0.0
    try:
        import subprocess

        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def record_transcription(
    *,
    audio_path=None,
    transcript=None,
    provider: str = "openai",
    model: str = "whisper-1",
    category: str = "transcription",
) -> None:
    """Record an audio transcription. ``provider="openai"`` bills per minute; local is $0."""
    try:
        seconds = _audio_duration_seconds(audio_path)
        if seconds <= 0 and transcript:
            # Rough fallback when ffprobe is unavailable: ~150 spoken words/minute.
            seconds = len(transcript.split()) / 150.0 * 60.0
        cost = (seconds / 60.0) * WHISPER_PER_MIN if provider == "openai" else 0.0
        _record(
            category=category,
            provider=provider,
            model=model,
            audio_seconds=seconds,
            cost_usd=cost,
        )
    except Exception:
        logger.warning("record_transcription failed (non-fatal)", exc_info=True)


def record_image(
    response=None,
    *,
    model: str = "gpt-image-1",
    images: int = 1,
    provider: str = "openai",
    category: str = "thumbnail",
) -> None:
    """Record an image-generation call (per-image estimate; captures usage tokens if present)."""
    try:
        cost = IMAGE_PER_CALL * images if provider == "openai" else 0.0
        pt = ct = 0
        usage = getattr(response, "usage", None) if response is not None else None
        if usage is not None:
            pt = getattr(usage, "input_tokens", 0) or 0
            ct = getattr(usage, "output_tokens", 0) or 0
        _record(
            category=category,
            provider=provider,
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            images=images,
            cost_usd=cost,
        )
    except Exception:
        logger.warning("record_image failed (non-fatal)", exc_info=True)

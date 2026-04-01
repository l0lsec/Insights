"""Command line utilities for transcribing and analysing podcasts."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shlex
import shutil
from typing import List
from urllib.parse import urlparse, parse_qs

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.2")

logger = logging.getLogger(__name__)

# ── Shared constants used by all generation functions ──────────────────────

TONE_GUIDES = {
    "professional": "Professional and authoritative, suitable for business audiences",
    "casual": "Casual and conversational, friendly and approachable",
    "witty": "Witty and clever, with humor where appropriate",
    "educational": "Educational and informative, focuses on teaching",
    "promotional": "Promotional and persuasive, drives action",
}

PLATFORM_GUIDELINES = {
    "twitter": "280 characters max, punchy and engaging, 3-5 relevant hashtags",
    "linkedin": "Professional tone, 1-3 paragraphs, thought leadership angle, 3-5 professional hashtags",
    "facebook": "Conversational, can be longer, engaging question or hook, 2-3 hashtags",
    "threads": "Casual and authentic, similar to Twitter but can be slightly longer, 3-5 hashtags",
    "instagram": "Visual-focused caption, emojis welcome, 10-15 relevant hashtags at the end",
}

# ── Shared LLM helpers ────────────────────────────────────────────────────

def _get_llm_client(use_local: bool = False, vision: bool = False):
    """Return ``(client, model_name)`` for either Ollama or OpenAI."""
    from openai import OpenAI

    if use_local:
        model = OLLAMA_VISION_MODEL if vision else OLLAMA_TEXT_MODEL
        return OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama"), model
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return client, OPENAI_MODEL


def _get_llm_params(use_local: bool, num_platforms: int = 1, posts_per_call: int = 1) -> dict:
    """Return generation kwargs tuned for local vs cloud models."""
    if use_local:
        tokens = max(800, 400 * num_platforms * posts_per_call)
        return {
            "temperature": 0.3,
            "max_tokens": min(tokens, 4000),
            "extra_body": {"repeat_penalty": 1.3, "top_p": 0.9},
        }
    return {"temperature": 0.8, "max_tokens": 3000}


_LOCAL_BATCH_SIZE = 1


def _build_format_instruction(platforms: list[str], posts_per_platform: int) -> str:
    """Build the JSON format instruction appended to every LLM prompt."""
    platform_keys = ", ".join(f'"{p}"' for p in platforms)
    if posts_per_platform > 1:
        example_val = '["First post #hashtag", "Second post #hashtag"]'
    else:
        example_val = '"Your post text here #hashtag"'
    json_example = "{" + ", ".join(f'"{p}": {example_val}' for p in platforms) + "}"

    lines = (
        f"\nReply with ONLY a JSON object. No other text.\n"
        f"Use ONLY these keys: {platform_keys}\n"
    )
    if posts_per_platform > 1:
        lines += f"Each key maps to an array of {posts_per_platform} post strings.\n"
    else:
        lines += "Each key maps to a single post string.\n"
    lines += f"Example:\n{json_example}"
    return lines


def _batch_generate(client, model, messages_fn, platforms, posts_per_platform, use_local):
    """Generate posts, batching into smaller calls for local models.

    Tracks actual posts received per platform and keeps requesting until the
    target is met or a safety cap of ``max_attempts`` is reached.
    """
    batch_size = _LOCAL_BATCH_SIZE if use_local else posts_per_platform

    def _call(plats, n):
        params = _get_llm_params(use_local, num_platforms=len(plats), posts_per_call=n)
        msgs = messages_fn(plats, n)
        resp = client.chat.completions.create(model=model, messages=msgs, **params)
        return _extract_json_from_llm(resp.choices[0].message.content.strip())

    if posts_per_platform <= batch_size:
        return _call(platforms, posts_per_platform)

    merged: dict[str, list] = {p: [] for p in platforms}
    max_attempts = (posts_per_platform // batch_size) * 3
    attempts = 0
    while attempts < max_attempts:
        shortest = min(len(merged[p]) for p in platforms)
        if shortest >= posts_per_platform:
            break
        need = min(batch_size, posts_per_platform - shortest)
        result = _call(platforms, need)
        if result:
            for p in platforms:
                val = result.get(p, [])
                if isinstance(val, str):
                    val = [val]
                merged[p].extend(val)
        attempts += 1

    for p in platforms:
        merged[p] = merged[p][:posts_per_platform]
    return merged


def check_ollama_status() -> dict:
    """Check if Ollama is running and list available models."""
    import urllib.request
    import urllib.error

    _VISION_KW = ("vision", "llava", "moondream", "bakllava")

    result = {
        "available": False,
        "models": [],
        "text_models": [],
        "configured_model": OLLAMA_VISION_MODEL,
        "configured_text_model": OLLAMA_TEXT_MODEL,
    }
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        all_models = [m["name"] for m in data.get("models", [])]
        result["models"] = [
            m for m in all_models
            if any(kw in m.lower() for kw in _VISION_KW)
        ]
        result["text_models"] = [
            m for m in all_models
            if not any(kw in m.lower() for kw in _VISION_KW)
        ]
        result["available"] = True
    except Exception:
        logger.debug("Ollama not reachable at %s", OLLAMA_BASE_URL)
    return result


def configure_logging(verbose: bool = False) -> None:
    """Configure ``logging`` so debug output can be toggled via ``--verbose``."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    # Write a debug message so callers know what level we're using
    logger.debug("Logging configured. Level=%s", logging.getLevelName(level))


# ── YouTube helpers ──────────────────────────────────────────────────────

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def is_youtube_url(url: str) -> bool:
    """Return *True* if *url* points to YouTube (video, channel, or playlist)."""
    try:
        host = urlparse(url).hostname or ""
        return host.lower() in _YT_HOSTS
    except Exception:
        return False


def classify_youtube_url(url: str) -> str:
    """Classify a YouTube URL as ``'video'``, ``'channel'``, ``'playlist'``, or ``'unknown'``.

    Handles formats like:
    - ``https://www.youtube.com/watch?v=VIDEO_ID``
    - ``https://youtu.be/VIDEO_ID``
    - ``https://www.youtube.com/@handle``
    - ``https://www.youtube.com/channel/UC...``
    - ``https://www.youtube.com/c/ChannelName``
    - ``https://www.youtube.com/playlist?list=PL...``
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    qs = parse_qs(parsed.query)

    if host == "youtu.be":
        return "video"

    if "list" in qs and path in ("/playlist", ""):
        return "playlist"
    if "v" in qs or path.startswith("/shorts/"):
        return "video"
    if path.startswith("/@") or path.startswith("/channel/") or path.startswith("/c/"):
        return "channel"
    if "list" in qs:
        return "playlist"

    return "unknown"


def _resolve_channel_id(url: str) -> str | None:
    """Use yt-dlp to resolve a channel/handle URL to a channel ID."""
    try:
        import yt_dlp

        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlist_items": "1",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("channel_id") or info.get("id")
    except Exception as exc:
        logger.warning("Could not resolve channel ID for %s: %s", url, exc)
        return None


def youtube_url_to_rss(url: str) -> str | None:
    """Convert a YouTube channel or playlist URL to its Atom RSS feed URL.

    Returns *None* when the URL cannot be converted (e.g. a single video).
    """
    kind = classify_youtube_url(url)
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if kind == "playlist":
        playlist_id = qs.get("list", [None])[0]
        if playlist_id:
            return f"https://www.youtube.com/feeds/videos.xml?playlist_id={playlist_id}"
        return None

    if kind == "channel":
        path = parsed.path.rstrip("/")
        if path.startswith("/channel/"):
            channel_id = path.split("/channel/")[1].split("/")[0]
        else:
            channel_id = _resolve_channel_id(url)
        if channel_id:
            return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        return None

    return None


def get_youtube_video_id(url: str) -> str | None:
    """Extract the video ID from a YouTube video URL."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    qs = parse_qs(parsed.query)

    if host == "youtu.be":
        return parsed.path.lstrip("/").split("/")[0] or None

    vid = qs.get("v", [None])[0]
    if vid:
        return vid

    path = parsed.path
    if path.startswith("/shorts/"):
        return path.split("/shorts/")[1].split("/")[0] or None

    return None


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(_APP_DIR, "downloads")


def download_youtube_audio(video_url: str, output_dir: str | None = None) -> str:
    """Download audio from a YouTube video using yt-dlp.

    Files are saved into *output_dir* (defaults to ``downloads/`` inside the
    application directory).  Returns the path to the downloaded audio file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is required for YouTube support. Install with: pip install yt-dlp"
        )

    if output_dir is None:
        output_dir = DOWNLOADS_DIR
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    logger.info("Downloading audio from YouTube: %s", video_url)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    video_id = info.get("id", "audio")
    expected_path = os.path.join(output_dir, f"{video_id}.mp3")
    if os.path.exists(expected_path):
        logger.info("YouTube audio saved to %s", expected_path)
        return expected_path

    mp3_files = glob.glob(os.path.join(output_dir, "*.mp3"))
    if mp3_files:
        return mp3_files[0]

    audio_files = glob.glob(os.path.join(output_dir, f"{video_id}.*"))
    audio_files = [f for f in audio_files if not f.endswith((".json", ".txt"))]
    if audio_files:
        return audio_files[0]

    raise FileNotFoundError(
        f"yt-dlp did not produce an audio file for {video_url}"
    )


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using local mlx-whisper, faster-whisper, or OpenAI API.

    Parameters
    ----------
    audio_path: str
        Path to the audio file.

    Returns
    -------
    str
        The transcribed text.
    """
    logger.debug("Starting transcription of %s", audio_path)
    
    # Try mlx-whisper first (optimized for Apple Silicon, free & local)
    try:
        import mlx_whisper
        logger.info("Using mlx-whisper for transcription (Apple Silicon optimized)")
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-base-mlx",
        )
        transcript = result.get("text", "").strip()
        logger.debug("Transcription complete via mlx-whisper")
        return transcript
    except ImportError:
        logger.debug("mlx-whisper not available, trying alternatives")
    except Exception as exc:
        logger.warning("mlx-whisper failed: %s, trying alternatives", exc)
    
    # Try faster-whisper next (works on Linux/Windows/older Macs)
    try:
        from faster_whisper import WhisperModel
        logger.info("Using faster-whisper for transcription")
        model = WhisperModel("base", device="cpu")
        segments, _ = model.transcribe(audio_path)
        segments_list = list(segments)
        transcript = " ".join(segment.text.strip() for segment in segments_list)
        logger.debug("Transcription finished with %d segments", len(segments_list))
        return transcript
    except ImportError:
        logger.debug("faster-whisper not available, trying OpenAI API")
    except Exception as exc:
        logger.warning("faster-whisper failed: %s, trying OpenAI API", exc)
    
    # Fall back to OpenAI Whisper API (costs money but always works)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            logger.info("Using OpenAI Whisper API for transcription")
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            transcript = response.text.strip()
            logger.debug("Transcription complete via OpenAI API")
            return transcript
        except Exception as exc:
            logger.exception("OpenAI Whisper API failed: %s", exc)
    
    raise NotImplementedError(
        "Audio transcription requires one of:\n"
        "1. mlx-whisper (pip install mlx-whisper) - Best for Apple Silicon Macs\n"
        "2. faster-whisper (pip install faster-whisper) - For other systems\n"
        "3. OPENAI_API_KEY environment variable set (uses paid API)"
    )


def summarize_text(text: str) -> str:
    """Summarize ``text`` using OpenAI."""

    try:
        # Create a minimal OpenAI client on demand so the dependency is optional
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        logger.debug("Requesting summary from OpenAI")
        # Ask the language model for a short summary of the transcript
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": f"Summarize the following text:\n{text}"}],
            temperature=0.2,
        )
        summary = response.choices[0].message.content.strip()
        logger.debug("Summary received")
        return summary
    except Exception as exc:
        logger.exception("OpenAI summarization failed")
        raise RuntimeError("Failed to summarize text with OpenAI") from exc


def extract_action_items(text: str) -> List[str]:
    """Extract action items from ``text`` using OpenAI."""

    try:
        # Similar to ``summarize_text`` but asking for a bullet list of tasks
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        logger.debug("Requesting action items from OpenAI")
        # Ask the language model for a plain list of actions without extra text
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract a concise list of action items from the "
                        "following text. Respond with one item per line "
                        "and no additional commentary.\n" + text
                    ),
                }
            ],
            temperature=0.2,
        )
        # Normalise each returned line into a bare task string
        lines = response.choices[0].message.content.splitlines()
        actions = [ln.lstrip("- ").strip() for ln in lines if ln.strip()]
        logger.debug("Action items received: %d", len(actions))
        return actions
    except Exception as exc:
        logger.exception("OpenAI action item extraction failed")
        raise RuntimeError("Failed to extract action items with OpenAI") from exc


def generate_article(
    transcript: str,
    summary: str,
    topic: str,
    podcast_title: str,
    episode_title: str,
    style: str = "blog",
    extra_context: str | None = None,
    is_text_source: bool = False,
) -> str:
    """Generate an article about a specific topic based on podcast or article content.

    Parameters
    ----------
    transcript: str
        The full podcast transcript or article content.
    summary: str
        A summary of the episode or article.
    topic: str
        The specific topic or angle the user wants the article to focus on.
    podcast_title: str
        The name of the podcast or publication for attribution.
    episode_title: str
        The title of the specific episode or article.
    style: str
        The article style (blog, news, opinion, technical). Defaults to blog.
    extra_context: str | None
        Optional additional context or instructions from the user.
    is_text_source: bool
        True if the source is a text article, False if it's a podcast.

    Returns
    -------
    str
        The generated article in markdown format.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        style_guides = {
            "blog": "Write in an engaging, conversational blog style with a personal voice.",
            "news": "Write in a professional news article style, factual and objective.",
            "opinion": "Write as an opinion/editorial piece with clear perspective and analysis.",
            "technical": "Write as a technical deep-dive with detailed explanations for practitioners.",
        }
        style_instruction = style_guides.get(style, style_guides["blog"])

        # Build extra context section if provided
        extra_context_section = ""
        if extra_context:
            extra_context_section = (
                f"\nADDITIONAL CONTEXT FROM THE AUTHOR:\n{extra_context}\n\n"
                "Please incorporate the above context, insights, or instructions into the article.\n"
            )

        # Adapt prompts based on source type
        if is_text_source:
            source_type = "article"
            source_label = "SOURCE PUBLICATION"
            content_label = "ARTICLE"
            full_content_label = "FULL ARTICLE CONTENT"
            credit_instruction = (
                "6. IMPORTANT: At the end of the article, include a section titled "
                "'## Read the Original Article' that credits the source publication by name, "
                "mentions the specific article title, and encourages readers to check out "
                "the original piece and the publication for more great journalism. Make this feel "
                "genuine and appreciative, not like a generic disclaimer."
            )
        else:
            source_type = "podcast"
            source_label = "SOURCE PODCAST"
            content_label = "EPISODE"
            full_content_label = "FULL TRANSCRIPT"
            credit_instruction = (
                "6. IMPORTANT: At the end of the article, include a section titled "
                "'## Listen to the Full Episode' that credits the source podcast by name, "
                "mentions the specific episode title, and encourages readers to check out "
                "the podcast for the full discussion and more great content. Make this feel "
                "genuine and enthusiastic, not like a generic disclaimer."
            )

        logger.debug("Generating article about: %s", topic)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert tech writer specializing in cybersecurity, privacy, "
                        "and technology topics. You write compelling, well-researched articles "
                        "that inform and engage readers. Use markdown formatting for the article "
                        "with proper headings, paragraphs, and emphasis where appropriate."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on the following {source_type} content, write an article focused on: {topic}\n\n"
                        f"Style: {style_instruction}\n\n"
                        f"{source_label}: {podcast_title}\n"
                        f"{content_label}: {episode_title}\n\n"
                        f"{extra_context_section}"
                        f"SUMMARY:\n{summary}\n\n"
                        f"{full_content_label}:\n{transcript[:15000]}\n\n"  # Limit to avoid token limits
                        "Write a compelling article (800-1500 words) that:\n"
                        "1. Has an attention-grabbing headline\n"
                        "2. Provides valuable insights on the topic\n"
                        f"3. References specific points from the {source_type}\n"
                        "4. Includes a strong conclusion with takeaways\n"
                        "5. Is suitable for a tech/security focused audience\n"
                        f"{credit_instruction}"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        article = response.choices[0].message.content.strip()
        logger.debug("Article generated successfully")
        return article
    except Exception as exc:
        logger.exception("Article generation failed")
        raise RuntimeError("Failed to generate article with OpenAI") from exc


def generate_social_copy(
    article_content: str,
    article_topic: str,
    platforms: List[str] | None = None,
    posts_per_platform: int = 1,
    extra_context: str | None = None,
) -> dict:
    """Generate social media promotional copy with hashtags for different platforms.

    Parameters
    ----------
    article_content: str
        The article content to promote.
    article_topic: str
        The main topic/title of the article.
    platforms: List[str] | None
        List of platforms to generate copy for. Defaults to all major platforms.
    posts_per_platform: int
        Number of unique posts to generate per platform. Defaults to 1.
    extra_context: str | None
        Optional additional context or instructions for generating posts.

    Returns
    -------
    dict
        Dictionary with platform names as keys. Values are lists of posts if 
        posts_per_platform > 1, otherwise single strings for backward compatibility.
    """
    if platforms is None:
        platforms = ["twitter", "linkedin", "facebook", "threads"]
    
    # Clamp posts_per_platform to reasonable range
    posts_per_platform = max(1, min(posts_per_platform, 21))

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        platform_guidelines = {
            "twitter": "280 characters max, punchy and engaging, 3-5 relevant hashtags",
            "linkedin": "Professional tone, 1-3 paragraphs, thought leadership angle, 3-5 professional hashtags",
            "facebook": "Conversational, can be longer, engaging question or hook, 2-3 hashtags",
            "threads": "Casual and authentic, similar to Twitter but can be slightly longer, 3-5 hashtags",
            "instagram": "Visual-focused caption, emojis welcome, 10-15 relevant hashtags at the end",
        }

        platform_list = "\n".join([
            f"- {p.upper()}: {platform_guidelines.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])

        # Build the multi-post instruction
        if posts_per_platform > 1:
            multi_post_instruction = (
                f"\nIMPORTANT: Generate {posts_per_platform} UNIQUE and DIFFERENT posts for EACH platform. "
                "Each post should have a distinct angle, hook, or approach - suitable for posting on different days. "
                "Vary the tone, focus, and call-to-action between posts. "
                f"Return an array of {posts_per_platform} posts for each platform.\n\n"
                "Format your response as JSON with platform names as keys and ARRAYS of posts as values. Example:\n"
                '{"twitter": ["First tweet here #hashtag", "Second tweet here #tech"], '
                '"linkedin": ["First LinkedIn post...", "Second LinkedIn post..."]}'
            )
        else:
            multi_post_instruction = (
                "\n\nFormat your response as JSON with platform names as keys. Example:\n"
                '{"twitter": "Your tweet here #hashtag", "linkedin": "Your LinkedIn post here"}'
            )

        logger.debug("Generating %d social media post(s) per platform for: %s", posts_per_platform, article_topic)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media marketing expert specializing in tech and cybersecurity content. "
                        "You create engaging, platform-optimized promotional copy that drives engagement and clicks. "
                        "You understand each platform's unique culture and best practices. "
                        "When asked to create multiple posts, you ensure each one is genuinely unique with different "
                        "angles, hooks, questions, or perspectives - not just rewording the same message."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate promotional social media copy for the following article:\n\n"
                        f"TOPIC: {article_topic}\n\n"
                        f"ARTICLE EXCERPT:\n{article_content[:3000]}\n\n"
                        + (f"ADDITIONAL CONTEXT/INSTRUCTIONS:\n{extra_context}\n\n" if extra_context else "")
                        + f"Create platform-specific promotional posts for each of these platforms:\n{platform_list}\n\n"
                        "For each post:\n"
                        "1. Write copy optimized for that platform's audience and format\n"
                        "2. Include relevant hashtags (tech, cybersecurity, privacy focused)\n"
                        "3. Include a call-to-action or hook\n"
                        "4. Make it shareable and engaging\n"
                        f"{multi_post_instruction}"
                    ),
                },
            ],
            temperature=0.8,  # Slightly higher for more variety in multiple posts
            max_tokens=3000 if posts_per_platform > 1 else 2000,
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content.strip()
        # Handle markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        logger.debug("Social media copy generated for %d platforms", len(result))
        return result
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON response, returning raw content")
        return {"raw": response.choices[0].message.content.strip()}
    except Exception as exc:
        logger.exception("Social media copy generation failed")
        raise RuntimeError("Failed to generate social media copy with OpenAI") from exc


def refine_article(
    current_content: str,
    user_feedback: str,
    article_topic: str,
) -> str:
    """Refine an article based on user feedback using AI.

    Parameters
    ----------
    current_content: str
        The current article content in markdown.
    user_feedback: str
        User's instructions for how to modify the article.
    article_topic: str
        The article's topic for context.

    Returns
    -------
    str
        The refined article content in markdown.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        logger.debug("Refining article based on feedback: %s", user_feedback[:100])
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert editor specializing in tech and cybersecurity content. "
                        "You help refine and improve articles based on user feedback while maintaining "
                        "the article's voice, structure, and key points. Return the complete revised "
                        "article in markdown format."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Please revise the following article based on my feedback.\n\n"
                        f"ARTICLE TOPIC: {article_topic}\n\n"
                        f"CURRENT ARTICLE:\n{current_content}\n\n"
                        f"MY FEEDBACK/INSTRUCTIONS:\n{user_feedback}\n\n"
                        "Please apply my feedback and return the complete revised article in markdown format. "
                        "Maintain the overall structure unless I specifically asked to change it. "
                        "Keep the same tone and style unless instructed otherwise."
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        refined = response.choices[0].message.content.strip()
        logger.debug("Article refined successfully")
        return refined
    except Exception as exc:
        logger.exception("Article refinement failed")
        raise RuntimeError("Failed to refine article with OpenAI") from exc


def generate_posts_from_prompt(
    prompt: str,
    platforms: List[str] | None = None,
    tone: str = "professional",
    posts_per_platform: int = 1,
    extra_context: str | None = None,
    use_local: bool = False,
) -> dict:
    """Generate social media posts from a freeform prompt/topic."""
    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]

    posts_per_platform = max(1, min(posts_per_platform, 10))

    try:
        client, model = _get_llm_client(use_local)
        tone_instruction = TONE_GUIDES.get(tone, TONE_GUIDES["professional"])

        platform_list = "\n".join([
            f"- {p.upper()}: {PLATFORM_GUIDELINES.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])

        def _messages(plats, n):
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator and marketing expert. "
                        "You create engaging, platform-optimized posts that resonate with audiences. "
                        "You ALWAYS reply with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Create social media posts about the following topic/prompt:\n\n"
                        f"TOPIC/PROMPT: {prompt}\n\n"
                        f"TONE: {tone_instruction}\n\n"
                        + (f"ADDITIONAL CONTEXT:\n{extra_context}\n\n" if extra_context else "")
                        + f"Create posts for these platforms:\n{platform_list}\n\n"
                        "For each post:\n"
                        "1. Optimize for the platform's audience and format\n"
                        "2. Include relevant hashtags\n"
                        "3. Make it engaging and shareable\n"
                        "4. Stay on topic and provide value\n"
                        + _build_format_instruction(plats, n)
                    ),
                },
            ]

        logger.debug("Generating posts from prompt via %s: %s",
                      "Ollama" if use_local else "OpenAI", prompt[:100])
        result = _batch_generate(client, model, _messages, platforms,
                                 posts_per_platform, use_local)
        if result:
            logger.debug("Generated posts for %d platforms from prompt", len(result))
            return result

        logger.warning("Could not extract JSON from prompt generation")
        return {p: "" for p in platforms}
    except Exception as exc:
        logger.exception("Post generation from prompt failed")
        raise RuntimeError("Failed to generate posts from prompt") from exc


def generate_posts_from_url(
    url: str,
    platforms: List[str] | None = None,
    tone: str = "professional",
    posts_per_platform: int = 1,
    extra_context: str | None = None,
    use_local: bool = False,
) -> dict:
    """Generate social media posts based on content from a URL."""
    import re
    import trafilatura

    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]

    # ── Fetch the URL content ──────────────────────────────────────────
    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            raise RuntimeError(f"Failed to fetch content from URL: {url}")

        body_content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        ) or ""

        metadata = trafilatura.extract_metadata(downloaded)

        title = ""
        description = ""
        og_image = None

        if metadata:
            title = metadata.title or ""
            description = metadata.description or ""
            og_image = metadata.image

        if not title or not description:
            title_match = re.search(r'<title>([^<]+)</title>', downloaded, re.IGNORECASE)
            if not title and title_match:
                title = title_match.group(1).strip()

            og_title_match = re.search(
                r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                downloaded, re.IGNORECASE,
            )
            if og_title_match:
                title = og_title_match.group(1)

            if not description:
                og_desc_match = re.search(
                    r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE,
                )
                if og_desc_match:
                    description = og_desc_match.group(1)
                else:
                    meta_desc_match = re.search(
                        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
                        downloaded, re.IGNORECASE,
                    )
                    if meta_desc_match:
                        description = meta_desc_match.group(1)

            if not og_image:
                og_image_match = re.search(
                    r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE,
                )
                if og_image_match:
                    og_image = og_image_match.group(1)

        extracted_content = f"TITLE: {title}\n\nDESCRIPTION: {description}\n\nCONTENT: {body_content}"

        source_data = {
            "url": url,
            "title": title,
            "description": description,
            "content": body_content,
            "og_image": og_image,
        }

    except Exception as e:
        logger.error("Failed to fetch URL %s: %s", url, e)
        raise RuntimeError(f"Failed to fetch content from URL: {e}") from e

    # ── Generate posts ─────────────────────────────────────────────────
    posts_per_platform = max(1, min(posts_per_platform, 10))

    try:
        client, model = _get_llm_client(use_local)
        tone_instruction = TONE_GUIDES.get(tone, TONE_GUIDES["professional"])

        platform_list = "\n".join([
            f"- {p.upper()}: {PLATFORM_GUIDELINES.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])

        def _messages(plats, n):
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator specializing in sharing and promoting web content. "
                        "You create engaging posts that summarize, comment on, or promote articles and web pages. "
                        "Include the URL in posts where appropriate (especially for LinkedIn). "
                        "You ALWAYS reply with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Create social media posts to share this web content:\n\n"
                        f"URL: {url}\n\n"
                        f"{extracted_content}\n\n"
                        f"TONE: {tone_instruction}\n\n"
                        + (f"ADDITIONAL CONTEXT:\n{extra_context}\n\n" if extra_context else "")
                        + f"Create posts for these platforms:\n{platform_list}\n\n"
                        "For each post:\n"
                        "1. Summarize or comment on the key points\n"
                        "2. Include the URL where appropriate\n"
                        "3. Add relevant hashtags\n"
                        "4. Make it engaging and encourage clicks/engagement\n"
                        + _build_format_instruction(plats, n)
                    ),
                },
            ]

        logger.debug("Generating posts from URL via %s: %s",
                      "Ollama" if use_local else "OpenAI", url)
        result = _batch_generate(client, model, _messages, platforms,
                                 posts_per_platform, use_local)
        if result:
            logger.debug("Generated posts for %d platforms from URL", len(result))
            return {"posts": result, "source_data": source_data}

        logger.warning("Could not extract JSON from URL generation")
        return {"posts": {p: "" for p in platforms}, "source_data": source_data}
    except Exception as exc:
        logger.exception("Post generation from URL failed")
        raise RuntimeError("Failed to generate posts from URL") from exc


def generate_posts_from_text(
    text: str,
    platforms: List[str] | None = None,
    tone: str = "professional",
    topic: str | None = None,
    posts_per_platform: int = 1,
    extra_context: str | None = None,
    use_local: bool = False,
) -> dict:
    """Generate social media posts from user-provided text content."""
    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]

    posts_per_platform = max(1, min(posts_per_platform, 10))

    try:
        client, model = _get_llm_client(use_local)
        tone_instruction = TONE_GUIDES.get(tone, TONE_GUIDES["professional"])

        platform_list = "\n".join([
            f"- {p.upper()}: {PLATFORM_GUIDELINES.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])

        topic_section = f"TOPIC: {topic}\n\n" if topic else ""

        def _messages(plats, n):
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator. You transform text content into engaging "
                        "social media posts optimized for different platforms. "
                        "You ALWAYS reply with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Transform the following content into social media posts:\n\n"
                        f"{topic_section}"
                        f"CONTENT:\n{text[:5000]}\n\n"
                        f"TONE: {tone_instruction}\n\n"
                        + (f"ADDITIONAL CONTEXT:\n{extra_context}\n\n" if extra_context else "")
                        + f"Create posts for these platforms:\n{platform_list}\n\n"
                        "For each post:\n"
                        "1. Capture the key message or insight\n"
                        "2. Optimize for the platform's format\n"
                        "3. Include relevant hashtags\n"
                        "4. Make it engaging and shareable\n"
                        + _build_format_instruction(plats, n)
                    ),
                },
            ]

        logger.debug("Generating posts from text via %s (length: %d)",
                      "Ollama" if use_local else "OpenAI", len(text))
        result = _batch_generate(client, model, _messages, platforms,
                                 posts_per_platform, use_local)
        if result:
            logger.debug("Generated posts for %d platforms from text", len(result))
            return result

        logger.warning("Could not extract JSON from text generation")
        return {p: "" for p in platforms}
    except Exception as exc:
        logger.exception("Post generation from text failed")
        raise RuntimeError("Failed to generate posts from text") from exc


def _normalize_llm_posts(parsed: dict | list) -> dict | None:
    """Convert various LLM output shapes into ``{platform: content_or_list}``."""
    if isinstance(parsed, dict):
        # Already the expected shape, but values might be dicts with a "post" key
        result = {}
        for k, v in parsed.items():
            if isinstance(v, str):
                result[k] = v
            elif isinstance(v, list):
                result[k] = [
                    item.get("post", item.get("content", str(item)))
                    if isinstance(item, dict) else str(item)
                    for item in v
                ]
            elif isinstance(v, dict):
                result[k] = v.get("post", v.get("content", str(v)))
        return result if result else None

    if isinstance(parsed, list):
        # Array of {"platform": "...", "post": "..."} objects
        result: dict[str, list[str]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            plat = item.get("platform", "").lower().strip()
            text = item.get("post", item.get("content", item.get("text", "")))
            if plat and text and isinstance(text, str):
                result.setdefault(plat, []).append(text)
        # Unwrap single-element lists
        for k, v in result.items():
            if len(v) == 1:
                result[k] = v[0]
        return result if result else None

    return None


def _extract_json_from_llm(text: str) -> dict | None:
    """Best-effort extraction of a JSON dict from an LLM response.

    Handles: plain JSON, markdown fences, embedded JSON, and
    array-of-objects format that local models sometimes produce.
    """
    import re

    def _sanitize(s: str) -> str:
        return s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

    def _try_parse(s: str) -> dict | None:
        try:
            return _normalize_llm_posts(json.loads(s))
        except (json.JSONDecodeError, ValueError):
            pass
        sanitized = _sanitize(s)
        if sanitized != s:
            try:
                return _normalize_llm_posts(json.loads(sanitized))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    # 1. Direct parse
    result = _try_parse(text)
    if result:
        return result

    # 2. Markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        result = _try_parse(fence_match.group(1).strip())
        if result:
            return result

    # 3. First { … last }  (object)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        result = _try_parse(text[start:end + 1])
        if result:
            return result

    # 4. First [ … last ]  (array)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        result = _try_parse(text[start:end + 1])
        if result:
            return result

    return None


def generate_posts_from_images(
    images: list[dict],
    prompt: str | None = None,
    platforms: List[str] | None = None,
    tone: str = "professional",
    posts_per_platform: int = 1,
    extra_context: str | None = None,
    use_local: bool = False,
) -> dict:
    """Generate social media posts by analysing one or more images.

    Images uses vision=True so the vision model is selected for local.
    Batch generation is NOT used here because image payloads are large
    and re-sending them multiple times is wasteful -- a single call is made.
    """
    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]

    posts_per_platform = max(1, min(posts_per_platform, 10))

    try:
        client, model = _get_llm_client(use_local, vision=True)
        params = _get_llm_params(use_local, num_platforms=len(platforms),
                                 posts_per_call=posts_per_platform)
        tone_instruction = TONE_GUIDES.get(tone, TONE_GUIDES["professional"])

        platform_list = "\n".join([
            f"- {p.upper()}: {PLATFORM_GUIDELINES.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])

        prompt_section = f"FOCUS: {prompt}\n" if prompt else ""
        extra_section = f"CONTEXT: {extra_context}\n" if extra_context else ""

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"Look at the image and write social media posts about it.\n"
                    f"{prompt_section}"
                    f"{extra_section}"
                    f"Tone: {tone_instruction}\n"
                    f"Platforms: {platform_list}\n\n"
                    + _build_format_instruction(platforms, posts_per_platform)
                ),
            }
        ]

        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['mime_type']};base64,{img['base64']}",
                },
            })

        logger.debug(
            "Generating posts from %d image(s) via %s",
            len(images),
            "Ollama" if use_local else "OpenAI",
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator. "
                        "You write engaging posts about images. "
                        "You ALWAYS reply with valid JSON only."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            **params,
        )

        raw_text = response.choices[0].message.content.strip()
        result = _extract_json_from_llm(raw_text)
        if result:
            logger.debug("Generated posts for %d platforms from images", len(result))
            return result

        logger.warning("Could not extract JSON; distributing raw text across platforms")
        return {p: raw_text for p in platforms}
    except Exception as exc:
        logger.exception("Post generation from images failed")
        raise RuntimeError("Failed to generate posts from images") from exc


def write_results_json(transcript: str, summary: str, actions: List[str], output_path: str) -> None:
    """Write the analysis results to ``output_path`` as JSON."""

    # Gather everything we generated so it can be saved and reused
    data = {
        "transcript": transcript,
        "summary": summary,
        "action_items": actions,
    }
    try:
        # ``indent`` makes the JSON readable for humans inspecting the file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Results written to %s", output_path)
    except Exception as exc:
        logger.exception("Failed to write results JSON")
        raise RuntimeError("Could not write results JSON") from exc


def main(audio_path: str, json_path: str | None = None, verbose: bool = False) -> None:
    """Run the full pipeline and write results to disk."""
    configure_logging(verbose)

    try:
        # 1) Transcribe the audio file
        logger.info("Transcribing audio...")
        transcript = transcribe_audio(audio_path)
        logger.info("Transcription complete")

        # 2) Summarise the transcript
        logger.info("Generating summary...")
        summary = summarize_text(transcript)
        logger.info("Summary complete")

        # 3) Pull out any explicit action items
        logger.info("Extracting action items...")
        actions = extract_action_items(transcript)
        logger.info("Action item extraction complete")

        # Log the results for easy visibility in the console
        logger.info("Summary:\n%s", summary)
        logger.info("Action Items:")
        for item in actions:
            logger.info("- %s", item)

        # Default JSON path is alongside the audio file
        if json_path is None:
            json_path = os.path.splitext(audio_path)[0] + ".json"
        write_results_json(transcript, summary, actions, json_path)
        logger.info("Results written to %s", json_path)
    except Exception as exc:
        # Any failure along the way is logged then re-raised
        logger.exception("Processing failed")


# ── YouTube Thumbnail Generation ─────────────────────────────────────────


def fetch_youtube_metadata(url: str) -> dict:
    """Fetch metadata for a YouTube video using yt-dlp.

    Returns a dict with keys: title, description, channel, tags,
    categories, thumbnail, video_id, duration.
    """
    video_id = get_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is required. Install with: pip install yt-dlp")

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "video_id": video_id,
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:1500],
        "channel": info.get("channel", info.get("uploader", "")),
        "tags": info.get("tags") or [],
        "categories": info.get("categories") or [],
        "thumbnail": info.get("thumbnail")
            or f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "duration": info.get("duration", 0),
    }


def _build_thumbnail_prompt(metadata: dict, aspect: str, style: str = "bold") -> str:
    """Build a single-stage image-generation prompt for a YouTube thumbnail.

    The prompt instructs ``gpt-image-1`` to render the *complete* thumbnail
    including bold text, visual effects, and photorealistic elements — no
    post-processing or text overlay needed.

    Parameters
    ----------
    metadata : dict
        Output of :func:`fetch_youtube_metadata`.
    aspect : str
        ``"16:9"`` for landscape or ``"9:16"`` for portrait / Shorts.
    style : str
        Visual style – ``"bold"``, ``"minimal"``, or ``"cinematic"``.
    """
    title = metadata.get("title", "Video")
    channel = metadata.get("channel", "")
    description = (metadata.get("description") or "")[:400]
    tags = ", ".join(metadata.get("tags", [])[:8])

    # Extract a short hook phrase from description (first sentence or clause)
    hook = ""
    if description:
        for sep in [".", "!", "?", "\n", " - "]:
            if sep in description[:200]:
                hook = description[:200].split(sep)[0].strip()
                break
        if not hook:
            hook = description[:80].strip()
        # Keep it punchy — truncate long hooks
        if len(hook) > 60:
            hook = hook[:57].rstrip() + "..."

    # Short title for the main text element (truncate if very long)
    display_title = title.upper()
    if len(display_title) > 50:
        display_title = display_title[:47].rstrip() + "..."

    orientation = "landscape" if aspect == "16:9" else "portrait"
    thumb_type = "YouTube thumbnail" if aspect == "16:9" else "YouTube Shorts thumbnail"

    # --- Style-specific prompt templates ---
    if style == "minimal":
        prompt = (
            f"Create a highly attractive {thumb_type} image. "
            f"Clean, modern design with strong color blocking and bold sans-serif typography. "
            f"A high-quality photographic background related to the topic: {title}. "
            f"Large, bold white text reading '{display_title}' prominently placed with a "
            f"subtle drop shadow for readability. "
        )
        if hook:
            prompt += (
                f"Below the main text, smaller clean text reading '{hook}' in a "
                f"contrasting accent color. "
            )
        if channel:
            prompt += f"Small text '{channel}' in the corner. "
        prompt += (
            f"Minimalist but eye-catching. Strong contrast between text and background. "
            f"MKBHD / tech-review style thumbnail. Professional, sharp, high resolution. "
            f"{orientation.capitalize()} ({aspect} aspect ratio)."
        )

    elif style == "cinematic":
        prompt = (
            f"Create a highly attractive {thumb_type} image styled like a movie poster. "
            f"Dramatic cinematic lighting with lens flares and volumetric light. "
            f"A photorealistic epic scene related to the topic: {title}. "
            f"The title text '{display_title}' rendered in large, bold cinematic movie-poster "
            f"font with metallic or chrome effect, centered prominently. "
        )
        if hook:
            prompt += (
                f"A tagline below reading '{hook}' in elegant serif font with subtle glow. "
            )
        if channel:
            prompt += f"Small text '{channel}' at the bottom in clean white font. "
        prompt += (
            f"Rich, moody color grading with teal and orange tones. Atmospheric depth of field. "
            f"Epic scale and drama. High contrast. "
            f"{orientation.capitalize()} ({aspect} aspect ratio)."
        )

    else:  # "bold" (default) — MrBeast-style maximum impact
        prompt = (
            f"Create a highly attractive {thumb_type}. "
            f"Photorealistic style with graphic elements. "
            f"A confident, expressive person pointing directly at the viewer or reacting with "
            f"an amazed expression, related to the topic: {title}. "
            f"Large bold 3D text reading '{display_title}' at the top in bright yellow with "
            f"thick black outline and neon glow effect. "
        )
        if hook:
            prompt += (
                f"Below that, even larger impactful text reading '{hook.upper()}' in bold "
                f"with red/orange glow and 3D depth effect. "
            )
        if channel:
            prompt += (
                f"Small text '{channel}' at the bottom in white with blue glow. "
            )
        prompt += (
            f"A red 'WATCH NOW!' button with play arrow icon in the bottom right corner. "
            f"Extremely vibrant, high contrast, saturated colors. "
            f"Dynamic composition with arrows or graphic elements directing attention. "
            f"Professional YouTube thumbnail style — loud, attention-grabbing, clickable. "
            f"{orientation.capitalize()} ({aspect} aspect ratio)."
        )

    # Add topic context from tags if available
    if tags:
        prompt += f" The visual theme should relate to: {tags}."

    return prompt


def _resolve_thumbnail_cli_argv(raw: str) -> List[str]:
    """Turn ``ASI_GENERATE_IMAGE`` into argv for ``subprocess`` (executable on ``PATH`` or full path)."""
    raw = raw.strip()
    if not raw:
        raise RuntimeError(
            "ASI_GENERATE_IMAGE is set but empty. Remove it to use OpenAI, or set a valid command."
        )
    parts = shlex.split(raw, posix=os.name != "nt")
    if not parts:
        raise RuntimeError("ASI_GENERATE_IMAGE could not be parsed as a command.")
    if len(parts) == 1:
        exe = os.path.expanduser(parts[0])
        if os.sep in exe or (os.altsep and exe and os.altsep in exe):
            if not os.path.isfile(exe):
                raise RuntimeError(
                    f"ASI_GENERATE_IMAGE path does not exist or is not a file: {exe}"
                )
            return [exe]
        resolved = shutil.which(exe)
        if not resolved:
            raise RuntimeError(
                "ASI_GENERATE_IMAGE command not found on PATH. Use a full path, fix PATH, "
                "or remove ASI_GENERATE_IMAGE to generate thumbnails via OpenAI instead."
            )
        return [resolved]
    head = os.path.expanduser(parts[0])
    if os.sep not in head and (not os.altsep or not head or os.altsep not in head):
        resolved_head = shutil.which(head)
        if not resolved_head:
            raise RuntimeError(
                f"ASI_GENERATE_IMAGE command not found on PATH: {parts[0]!r}"
            )
        head = resolved_head
    parts[0] = head
    return parts


def suggested_thumbnail_prompt(metadata: dict, aspect: str, style: str = "bold") -> str:
    """Return the default image-generation prompt for a YouTube thumbnail.

    Public wrapper around ``_build_thumbnail_prompt`` for use by the web layer.
    """
    return _build_thumbnail_prompt(metadata, aspect, style)


MAX_CUSTOM_PROMPT_LEN = 16_000


def generate_youtube_thumbnail(
    url: str,
    aspect: str = "16:9",
    style: str = "bold",
    use_local: bool = False,
    custom_prompt: str | None = None,
) -> dict:
    """Generate a complete YouTube thumbnail in a single AI pass.

    By default calls OpenAI's Images API with model ``gpt-image-1`` (requires
    ``OPENAI_API_KEY``). If ``ASI_GENERATE_IMAGE`` is set to a non-empty command,
    that CLI is invoked instead with a JSON payload (legacy integration).

    Parameters
    ----------
    url : str
        YouTube video URL.
    aspect : str
        ``"16:9"`` or ``"9:16"``.
    style : str
        ``"bold"``, ``"minimal"``, or ``"cinematic"``.
    use_local : bool
        Unused (kept for API compatibility).
    custom_prompt : str or None
        If provided and non-empty, used instead of the auto-generated prompt.

    Returns
    -------
    dict
        ``{"image_base64": ..., "prompt": ..., "metadata": ...}``
    """
    import base64
    import json
    import subprocess
    import tempfile

    from openai import OpenAI

    metadata = fetch_youtube_metadata(url)

    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()[:MAX_CUSTOM_PROMPT_LEN]
    else:
        prompt = _build_thumbnail_prompt(metadata, aspect, style)

    logger.info("Generating thumbnail (%s, %s style) for: %s", aspect, style, url)

    # Map aspect ratio to gpt-image-1 size parameter
    size = "1536x1024" if aspect == "16:9" else "1024x1536"

    cli_env = os.environ.get("ASI_GENERATE_IMAGE", "").strip()
    if cli_env:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_payload = json.dumps({
                "prompt": prompt,
                "filename": "thumbnail",
                "aspect_ratio": aspect,
                "size": size,
                "model": "gpt_image_1",
                "quality": "high",
            })
            result = subprocess.run(
                _resolve_thumbnail_cli_argv(cli_env) + [cmd_payload],
                cwd=tmpdir,
                env=os.environ,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error("thumbnail CLI failed: %s", result.stderr)
                raise RuntimeError(
                    f"Image generation failed: {result.stderr.strip() or 'unknown error'}"
                )

            img_path = os.path.join(tmpdir, "thumbnail.png")
            if not os.path.isfile(img_path):
                raise RuntimeError("Image generation did not produce expected output file")

            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY for thumbnail generation, or set ASI_GENERATE_IMAGE "
                "to an external image CLI."
            )
        client = OpenAI(api_key=api_key, timeout=120.0)
        # gpt-image-1 does not accept response_format (always returns base64); DALL·E uses url/b64_json.
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            quality="high",
            n=1,
        )
        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("OpenAI image generation returned no image data")
        image_base64 = response.data[0].b64_json

    return {
        "image_base64": image_base64,
        "prompt": prompt,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse

    # Command line interface for standalone usage
    parser = argparse.ArgumentParser(description="Transcribe and analyze podcasts")
    parser.add_argument("audio", help="Path to the podcast audio file")
    parser.add_argument(
        "-j",
        "--json",
        help=(
            "Optional path to save results as JSON; defaults to <audio>.json"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()
    # Execute the pipeline with the provided options
    main(args.audio, args.json, args.verbose)

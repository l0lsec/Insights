"""Command line utilities for transcribing and analysing podcasts."""

from __future__ import annotations

import os
import json
import logging
from typing import List

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure ``logging`` so debug output can be toggled via ``--verbose``."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    # Write a debug message so callers know what level we're using
    logger.debug("Logging configured. Level=%s", logging.getLevelName(level))


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
        platforms = ["twitter", "linkedin", "facebook", "threads", "bluesky"]
    
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
            "bluesky": "Similar to Twitter, concise and engaging, 3-5 hashtags",
            "instagram": "Visual-focused caption, emojis welcome, 10-15 relevant hashtags at the end",
            "mastodon": "Thoughtful and community-focused, 3-5 hashtags, can use content warnings if needed",
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
) -> dict:
    """Generate social media posts from a freeform prompt/topic.
    
    Parameters
    ----------
    prompt: str
        The topic, idea, or prompt to generate posts about.
    platforms: List[str] | None
        List of platforms to generate posts for. Defaults to major platforms.
    tone: str
        The tone/style for the posts (professional, casual, witty, educational, promotional).
    posts_per_platform: int
        Number of unique posts to generate per platform. Defaults to 1.
    extra_context: str | None
        Optional additional context or instructions.
        
    Returns
    -------
    dict
        Dictionary with platform names as keys and post content as values.
    """
    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]
    
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        
        tone_guides = {
            "professional": "Professional and authoritative, suitable for business audiences",
            "casual": "Casual and conversational, friendly and approachable",
            "witty": "Witty and clever, with humor where appropriate",
            "educational": "Educational and informative, focuses on teaching",
            "promotional": "Promotional and persuasive, drives action",
        }
        tone_instruction = tone_guides.get(tone, tone_guides["professional"])
        
        platform_guidelines = {
            "twitter": "280 characters max, punchy and engaging, 3-5 relevant hashtags",
            "linkedin": "Professional tone, 1-3 paragraphs, thought leadership angle, 3-5 professional hashtags",
            "facebook": "Conversational, can be longer, engaging question or hook, 2-3 hashtags",
            "threads": "Casual and authentic, similar to Twitter but can be slightly longer, 3-5 hashtags",
            "bluesky": "Similar to Twitter, concise and engaging, 3-5 hashtags",
            "instagram": "Visual-focused caption, emojis welcome, 10-15 relevant hashtags at the end",
            "mastodon": "Thoughtful and community-focused, 3-5 hashtags",
        }
        
        platform_list = "\n".join([
            f"- {p.upper()}: {platform_guidelines.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])
        
        if posts_per_platform > 1:
            format_instruction = (
                f"\nGenerate {posts_per_platform} UNIQUE posts for EACH platform with different angles.\n"
                "Format as JSON with platform names as keys and ARRAYS of posts as values. Example:\n"
                '{"twitter": ["First tweet #hashtag", "Second tweet #tech"], "linkedin": ["Post 1...", "Post 2..."]}'
            )
        else:
            format_instruction = (
                "\nFormat as JSON with platform names as keys. Example:\n"
                '{"twitter": "Your tweet #hashtag", "linkedin": "Your LinkedIn post..."}'
            )
        
        logger.debug("Generating posts from prompt: %s", prompt[:100])
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator and marketing expert. "
                        "You create engaging, platform-optimized posts that resonate with audiences. "
                        "You understand each platform's unique culture, character limits, and best practices."
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
                        f"{format_instruction}"
                    ),
                },
            ],
            temperature=0.8,
            max_tokens=3000,
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        logger.debug("Generated posts for %d platforms from prompt", len(result))
        return result
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response, returning raw content")
        return {"raw": response.choices[0].message.content.strip()}
    except Exception as exc:
        logger.exception("Post generation from prompt failed")
        raise RuntimeError("Failed to generate posts from prompt") from exc


def generate_posts_from_url(
    url: str,
    platforms: List[str] | None = None,
    tone: str = "professional",
    posts_per_platform: int = 1,
    extra_context: str | None = None,
) -> dict:
    """Generate social media posts based on content from a URL.
    
    Parameters
    ----------
    url: str
        The URL to fetch content from.
    platforms: List[str] | None
        List of platforms to generate posts for.
    tone: str
        The tone/style for the posts.
    posts_per_platform: int
        Number of unique posts per platform.
    extra_context: str | None
        Optional additional context.
        
    Returns
    -------
    dict
        Dictionary with 'posts' (platform-keyed posts) and 'source_data' (extracted metadata).
    """
    import requests
    import re
    import trafilatura
    
    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]
    
    # Fetch the URL content
    try:
        # Use trafilatura for robust article extraction
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            raise RuntimeError(f"Failed to fetch content from URL: {url}")
        
        # Extract main article content using trafilatura
        body_content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        ) or ""
        
        # Extract metadata using trafilatura's metadata extraction
        metadata = trafilatura.extract_metadata(downloaded)
        
        title = ""
        description = ""
        og_image = None
        
        if metadata:
            title = metadata.title or ""
            description = metadata.description or ""
            og_image = metadata.image
        
        # Fallback: extract metadata from HTML if trafilatura didn't get it
        if not title or not description:
            # Extract title from HTML
            title_match = re.search(r'<title>([^<]+)</title>', downloaded, re.IGNORECASE)
            if not title and title_match:
                title = title_match.group(1).strip()
            
            # Extract og:title if available
            og_title_match = re.search(
                r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                downloaded, re.IGNORECASE
            )
            if og_title_match:
                title = og_title_match.group(1)
            
            # Extract og:description
            if not description:
                og_desc_match = re.search(
                    r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE
                )
                if og_desc_match:
                    description = og_desc_match.group(1)
                else:
                    # Fallback to meta description
                    meta_desc_match = re.search(
                        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
                        downloaded, re.IGNORECASE
                    )
                    if meta_desc_match:
                        description = meta_desc_match.group(1)
            
            # Extract og:image if not already found
            if not og_image:
                og_image_match = re.search(
                    r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE
                )
                if og_image_match:
                    og_image = og_image_match.group(1)
        
        extracted_content = f"TITLE: {title}\n\nDESCRIPTION: {description}\n\nCONTENT: {body_content}"
        
        # Store source data for saving
        source_data = {
            "url": url,
            "title": title,
            "description": description,
            "content": body_content,
            "og_image": og_image,
        }
        
    except (requests.RequestException, Exception) as e:
        logger.error("Failed to fetch URL %s: %s", url, e)
        raise RuntimeError(f"Failed to fetch content from URL: {e}") from e
    
    # Now generate posts using the extracted content
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        
        tone_guides = {
            "professional": "Professional and authoritative, suitable for business audiences",
            "casual": "Casual and conversational, friendly and approachable",
            "witty": "Witty and clever, with humor where appropriate",
            "educational": "Educational and informative, focuses on teaching",
            "promotional": "Promotional and persuasive, drives action",
        }
        tone_instruction = tone_guides.get(tone, tone_guides["professional"])
        
        platform_guidelines = {
            "twitter": "280 characters max, punchy and engaging, 3-5 relevant hashtags",
            "linkedin": "Professional tone, 1-3 paragraphs, thought leadership angle, 3-5 professional hashtags",
            "facebook": "Conversational, can be longer, engaging question or hook, 2-3 hashtags",
            "threads": "Casual and authentic, similar to Twitter but can be slightly longer, 3-5 hashtags",
            "bluesky": "Similar to Twitter, concise and engaging, 3-5 hashtags",
            "instagram": "Visual-focused caption, emojis welcome, 10-15 relevant hashtags at the end",
            "mastodon": "Thoughtful and community-focused, 3-5 hashtags",
        }
        
        platform_list = "\n".join([
            f"- {p.upper()}: {platform_guidelines.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])
        
        if posts_per_platform > 1:
            format_instruction = (
                f"\nGenerate {posts_per_platform} UNIQUE posts for EACH platform with different angles.\n"
                "Format as JSON with platform names as keys and ARRAYS of posts as values."
            )
        else:
            format_instruction = (
                "\nFormat as JSON with platform names as keys. Example:\n"
                '{"twitter": "Your tweet #hashtag", "linkedin": "Your LinkedIn post..."}'
            )
        
        logger.debug("Generating posts from URL: %s", url)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator specializing in sharing and promoting web content. "
                        "You create engaging posts that summarize, comment on, or promote articles and web pages. "
                        "Include the URL in posts where appropriate (especially for LinkedIn)."
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
                        f"{format_instruction}"
                    ),
                },
            ],
            temperature=0.8,
            max_tokens=3000,
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        logger.debug("Generated posts for %d platforms from URL", len(result))
        return {"posts": result, "source_data": source_data}
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response, returning raw content")
        return {
            "posts": {"raw": response.choices[0].message.content.strip()},
            "source_data": source_data,
        }
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
) -> dict:
    """Generate social media posts from user-provided text content.
    
    Parameters
    ----------
    text: str
        The text content to generate posts from.
    platforms: List[str] | None
        List of platforms to generate posts for.
    tone: str
        The tone/style for the posts.
    topic: str | None
        Optional topic/title for the content.
    posts_per_platform: int
        Number of unique posts per platform.
    extra_context: str | None
        Optional additional context.
        
    Returns
    -------
    dict
        Dictionary with platform names as keys and post content as values.
    """
    if platforms is None:
        platforms = ["linkedin", "threads", "twitter"]
    
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        
        tone_guides = {
            "professional": "Professional and authoritative, suitable for business audiences",
            "casual": "Casual and conversational, friendly and approachable",
            "witty": "Witty and clever, with humor where appropriate",
            "educational": "Educational and informative, focuses on teaching",
            "promotional": "Promotional and persuasive, drives action",
        }
        tone_instruction = tone_guides.get(tone, tone_guides["professional"])
        
        platform_guidelines = {
            "twitter": "280 characters max, punchy and engaging, 3-5 relevant hashtags",
            "linkedin": "Professional tone, 1-3 paragraphs, thought leadership angle, 3-5 professional hashtags",
            "facebook": "Conversational, can be longer, engaging question or hook, 2-3 hashtags",
            "threads": "Casual and authentic, similar to Twitter but can be slightly longer, 3-5 hashtags",
            "bluesky": "Similar to Twitter, concise and engaging, 3-5 hashtags",
            "instagram": "Visual-focused caption, emojis welcome, 10-15 relevant hashtags at the end",
            "mastodon": "Thoughtful and community-focused, 3-5 hashtags",
        }
        
        platform_list = "\n".join([
            f"- {p.upper()}: {platform_guidelines.get(p, 'Standard social media post with hashtags')}"
            for p in platforms
        ])
        
        if posts_per_platform > 1:
            format_instruction = (
                f"\nGenerate {posts_per_platform} UNIQUE posts for EACH platform with different angles.\n"
                "Format as JSON with platform names as keys and ARRAYS of posts as values."
            )
        else:
            format_instruction = (
                "\nFormat as JSON with platform names as keys. Example:\n"
                '{"twitter": "Your tweet #hashtag", "linkedin": "Your LinkedIn post..."}'
            )
        
        topic_section = f"TOPIC: {topic}\n\n" if topic else ""
        
        logger.debug("Generating posts from text content (length: %d)", len(text))
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media content creator. You transform text content into engaging "
                        "social media posts optimized for different platforms. You understand how to "
                        "distill key points and make them compelling for social audiences."
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
                        f"{format_instruction}"
                    ),
                },
            ],
            temperature=0.8,
            max_tokens=3000,
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        logger.debug("Generated posts for %d platforms from text", len(result))
        return result
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response, returning raw content")
        return {"raw": response.choices[0].message.content.strip()}
    except Exception as exc:
        logger.exception("Post generation from text failed")
        raise RuntimeError("Failed to generate posts from text") from exc


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

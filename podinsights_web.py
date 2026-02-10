"""Flask web interface for the PodInsights tool."""

from __future__ import annotations
import os
import io
import tempfile
import threading
import uuid
from queue import Queue
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory
from PIL import Image
import cloudinary
import cloudinary.uploader
import re
import feedparser
import requests
from datetime import datetime, timedelta
import time
from html import unescape
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from flasgger import Swagger
from database import (
    init_db,
    get_episode,
    get_episode_by_id,
    save_episode,
    queue_episode,
    update_episode_status,
    delete_episode_by_id,
    delete_episodes_bulk,
    reset_episode_for_reprocess,
    add_feed,
    list_feeds,
    get_feed_by_id,
    delete_feed,
    delete_feeds_bulk,
    add_ticket,
    list_tickets,
    delete_ticket,
    delete_tickets_bulk,
    list_all_episodes,
    add_article,
    get_article,
    list_articles,
    update_article,
    delete_article,
    update_feed_metadata,
    add_social_post,
    get_social_post,
    list_social_posts,
    delete_social_post,
    delete_social_posts_bulk,
    delete_social_posts_for_article,
    mark_social_post_used,
    update_social_post,
    # LinkedIn token functions
    save_linkedin_token,
    get_linkedin_token,
    delete_linkedin_token,
    update_linkedin_token,
    # Threads token functions
    save_threads_token,
    get_threads_token,
    delete_threads_token,
    update_threads_token,
    # Scheduled posts functions
    add_scheduled_post,
    get_scheduled_post,
    get_pending_schedules_for_social_posts,
    list_scheduled_posts,
    get_pending_scheduled_posts,
    update_scheduled_post_status,
    update_scheduled_post_time,
    cancel_scheduled_post,
    delete_scheduled_post,
    delete_scheduled_posts_bulk,
    clear_pending_scheduled_posts,
    get_scheduled_posts_for_article,
    # Time slot functions for queue-based scheduling
    add_time_slot,
    list_time_slots,
    get_enabled_time_slots,
    update_time_slot,
    delete_time_slot,
    get_next_available_slot,
    initialize_default_time_slots,
    # Daily posting limits
    get_daily_limit,
    set_daily_limit,
    get_all_daily_limits,
    # Queue redistribution
    redistribute_scheduled_posts,
    increment_retry_count,
    # Standalone posts functions (Command Center)
    add_standalone_post,
    list_standalone_posts,
    get_standalone_post,
    update_standalone_post,
    update_standalone_post_image,
    update_social_post_image,
    delete_standalone_post,
    delete_standalone_posts_bulk,
    mark_standalone_post_used,
    # URL sources functions
    add_url_source,
    list_url_sources,
    get_url_source,
    get_url_source_by_url,
    delete_url_source,
    update_url_source_last_used,
    update_url_source_content,
    # Standalone post scheduling
    get_pending_schedules_for_standalone_posts,
    get_posted_info_for_standalone_posts,
    # Uploaded images library
    add_uploaded_image,
    list_uploaded_images,
    delete_uploaded_image,
    # Recent prompts
    list_recent_prompts,
    clear_recent_prompts,
    delete_prompt_by_content,
    delete_prompts_bulk,
)
from podinsights import (
    transcribe_audio,
    summarize_text,
    extract_action_items,
    write_results_json,
    configure_logging,
    generate_article,
    generate_social_copy,
    refine_article,
    # Command Center functions
    generate_posts_from_prompt,
    generate_posts_from_url,
    generate_posts_from_text,
)
from linkedin_client import (
    LinkedInClient,
    get_linkedin_client,
    calculate_token_expiry,
    is_token_expired,
)
from threads_client import (
    ThreadsClient,
    get_threads_client,
    calculate_token_expiry as threads_calculate_token_expiry,
    is_token_expired as threads_is_token_expired,
)
from stock_images import (
    search_stock_images,
    get_image_for_post,
    get_images_for_post,
    extract_keywords_from_text,
    is_configured as stock_images_configured,
    get_configured_services as get_stock_image_services,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# Configure Swagger/OpenAPI documentation
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger_template = {
    "info": {
        "title": "PodInsights API",
        "description": "API for managing podcasts, articles, social media posts, and scheduling",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["http", "https"],
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Configure image uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Cloudinary (optional - for public image URLs that work with Threads)
CLOUDINARY_CONFIGURED = False
if os.environ.get('CLOUDINARY_CLOUD_NAME'):
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
        secure=True
    )
    CLOUDINARY_CONFIGURED = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_and_clean_image(file_storage):
    """
    Validate image by parsing with Pillow and re-encode to strip embedded data.
    Returns (cleaned_bytes, extension) or raises ValueError.
    """
    try:
        file_storage.seek(0)
        img = Image.open(file_storage)
        img.verify()  # Verify it's a valid image
        
        # Re-open after verify (verify() can only be called once)
        file_storage.seek(0)
        img = Image.open(file_storage)
        
        # Check format against allowlist
        detected_format = img.format.lower() if img.format else None
        if detected_format not in ['png', 'jpeg', 'gif', 'webp']:
            raise ValueError(f"Image format not allowed: {detected_format}")
        
        # Re-encode to strip any embedded data (anti-polyglot)
        output = io.BytesIO()
        
        # Handle format-specific saving
        save_format = 'JPEG' if detected_format == 'jpeg' else detected_format.upper()
        if save_format == 'JPEG':
            img = img.convert('RGB')  # JPEG doesn't support alpha
        
        img.save(output, format=save_format, optimize=True)
        output.seek(0)
        
        ext = 'jpg' if detected_format == 'jpeg' else detected_format
        return output.read(), ext
        
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def save_stock_image_to_library(image_url: str, direct_save: bool = False) -> str:
    """
    Save a stock image to the library.
    
    If direct_save=True or URL is from Unsplash/Pexels/Pixabay, saves the URL directly
    to the library without downloading (these services allow hotlinking).
    
    Otherwise downloads and re-uploads to Cloudinary or local storage.
    Returns the saved image URL.
    """
    import requests as req
    import re
    
    # Check if this image URL is already in the library
    existing = list_uploaded_images()
    for img in existing:
        # Check if original URL matches (stored in filename or url)
        if img.get('url') == image_url or image_url in str(img.get('filename', '')):
            return img['url']
    
    # Check if this is a stock image URL that allows hotlinking
    stock_domains = ['images.unsplash.com', 'unsplash.com', 'pexels.com', 'pixabay.com']
    is_stock_url = any(domain in image_url for domain in stock_domains)
    
    # For stock URLs, save directly to library without downloading
    if direct_save or is_stock_url:
        # Extract a meaningful filename from the URL
        photo_match = re.search(r'photo-([a-zA-Z0-9_-]+)', image_url)
        if photo_match:
            filename = f"unsplash_{photo_match.group(1)}"
        else:
            filename = f"stock_{uuid.uuid4().hex[:8]}"
        
        # Determine storage type based on URL
        if 'unsplash' in image_url:
            storage = 'unsplash'
        elif 'pexels' in image_url:
            storage = 'pexels'
        elif 'pixabay' in image_url:
            storage = 'pixabay'
        else:
            storage = 'external'
        
        add_uploaded_image(
            filename=filename,
            url=image_url,
            storage=storage,
            size=0
        )
        return image_url
    
    # For non-stock URLs, download and re-upload
    response = req.get(image_url, timeout=30, stream=True)
    response.raise_for_status()
    
    # Get content type to determine extension
    content_type = response.headers.get('content-type', 'image/jpeg')
    ext_map = {
        'image/jpeg': 'jpg',
        'image/png': 'png',
        'image/gif': 'gif',
        'image/webp': 'webp',
    }
    ext = ext_map.get(content_type.split(';')[0].strip(), 'jpg')
    
    # Read image data
    image_data = response.content
    
    # Validate and clean the image
    img_file = io.BytesIO(image_data)
    try:
        img = Image.open(img_file)
        img.verify()
        img_file.seek(0)
        img = Image.open(img_file)
        
        # Re-encode to strip metadata
        output = io.BytesIO()
        save_format = 'JPEG' if ext == 'jpg' else ext.upper()
        if save_format == 'JPEG':
            img = img.convert('RGB')
        img.save(output, format=save_format, optimize=True)
        output.seek(0)
        cleaned_bytes = output.read()
    except Exception as e:
        raise ValueError(f"Invalid image from stock API: {e}")
    
    # Upload to Cloudinary if configured
    if CLOUDINARY_CONFIGURED:
        result = cloudinary.uploader.upload(
            cleaned_bytes,
            folder="podinsights/stock",
            resource_type="image"
        )
        saved_url = result['secure_url']
        filename = f"stock_{result['public_id'].split('/')[-1]}"
        file_size = result.get('bytes', len(cleaned_bytes))
        
        add_uploaded_image(
            filename=filename,
            url=saved_url,
            storage='cloudinary',
            size=file_size
        )
        return saved_url
    
    # Local storage fallback
    unique_filename = f"stock_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    with open(filepath, 'wb') as f:
        f.write(cleaned_bytes)
    
    saved_url = f"/static/uploads/{unique_filename}"
    
    add_uploaded_image(
        filename=unique_filename,
        url=saved_url,
        storage='local',
        size=len(cleaned_bytes)
    )
    
    return saved_url


configure_logging()
init_db()

# Background processing queue used to process episodes without blocking the web request
task_queue: Queue = Queue()


def worker() -> None:
    """Background thread processing queued episodes."""
    while True:
        # Wait for an episode to appear in the queue
        item = task_queue.get()
        if item is None:
            break
        url = item["url"]
        title = item.get("title", "Episode")
        feed_id = item.get("feed_id")
        published = item.get("published")
        try:
            update_episode_status(url, "processing")
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = os.path.join(tmpdir, "episode.mp3")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(audio_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                transcript = transcribe_audio(audio_path)
                summary = summarize_text(transcript)
                actions = extract_action_items(transcript)
                out_path = os.path.join(tmpdir, "results.json")
                write_results_json(transcript, summary, actions, out_path)
                save_episode(url, title, transcript, summary, actions, feed_id, published)
        except Exception:
            app.logger.exception("Failed to process episode %s", url)
            update_episode_status(url, "error")
        finally:
            task_queue.task_done()


# Worker thread will be started in main block to avoid duplicates in debug mode
# (app.debug is False at import time, so we can't check it here)


def strip_html(text: str) -> str:
    """Return plain text with HTML tags removed."""
    if not text:
        return ""
    # replace common HTML tags with line breaks then strip everything else
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def make_short_description(text: str, limit: int = 200) -> str:
    """Return a short preview from the provided text."""
    if not text:
        return ""
    text = text.strip()
    # Use the first couple of sentences as a human friendly snippet
    sentences = re.split(r"(?<=[.!?])\s+", text)
    short = " ".join(sentences[:2])
    if len(short) > limit:
        short = short[:limit].rstrip() + "..."
    return short


def fetch_article_content(url: str, timeout: int = 15) -> str:
    """Fetch and extract the main content from an article URL.
    
    Uses trafilatura for robust article extraction, with BeautifulSoup as fallback.
    Returns extracted text or empty string on failure.
    """
    import trafilatura
    
    # Skip non-HTML URLs (audio, video, images, etc.)
    media_extensions = (
        '.mp3', '.mp4', '.m4a', '.wav', '.ogg', '.webm', '.avi', '.mov',
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.pdf', '.zip'
    )
    url_lower = url.lower().split('?')[0]  # Remove query params for extension check
    if url_lower.endswith(media_extensions):
        app.logger.info("Skipping media URL (not an article): %s", url)
        return ""
    
    try:
        # Try trafilatura first - it's specifically designed for article extraction
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if content and len(content) > 200:
                app.logger.info("Extracted %d chars using trafilatura from: %s", len(content), url)
                return content
        
        # Fallback to BeautifulSoup for non-article pages or if trafilatura fails
        app.logger.info("Trafilatura extraction insufficient, falling back to BeautifulSoup for: %s", url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        
        # Check content type - only process HTML
        content_type = resp.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            return ""
        
        soup = BeautifulSoup(resp.content, 'lxml')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                                       'aside', 'iframe', 'noscript', 'form',
                                       'button', 'input', 'select', 'textarea']):
            element.decompose()
        
        # Remove common ad/tracking elements by class or id patterns
        for element in soup.find_all(class_=re.compile(r'(ad|ads|advert|banner|sidebar|comment|share|social|related|recommend|newsletter|popup|modal|cookie)', re.I)):
            element.decompose()
        for element in soup.find_all(id=re.compile(r'(ad|ads|advert|banner|sidebar|comment|share|social|related|recommend|newsletter|popup|modal|cookie)', re.I)):
            element.decompose()
        
        # Try to find the main content area
        article_content = None
        
        # Priority 1: Look for article tag
        article = soup.find('article')
        if article:
            article_content = article
        
        # Priority 2: Look for main content divs
        if not article_content:
            for selector in ['[role="main"]', '.article-content', '.post-content', 
                            '.entry-content', '.content', '#content', '.story-body',
                            '.article-body', '.post-body', 'main']:
                found = soup.select_one(selector)
                if found:
                    article_content = found
                    break
        
        # Priority 3: Use body as fallback
        if not article_content:
            article_content = soup.find('body') or soup
        
        # Extract text from paragraphs for cleaner output
        paragraphs = article_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'pre', 'code'])
        
        if paragraphs:
            text_parts = []
            for p in paragraphs:
                text = p.get_text(separator=' ', strip=True)
                if text and len(text) > 20:  # Filter out short fragments
                    text_parts.append(text)
            content = '\n\n'.join(text_parts)
        else:
            # Fallback: get all text
            content = article_content.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = content.strip()
        
        return content
        
    except requests.exceptions.Timeout:
        app.logger.warning("Timeout fetching article: %s", url)
        return ""
    except requests.exceptions.RequestException as e:
        app.logger.warning("Failed to fetch article %s: %s", url, str(e))
        return ""
    except Exception as e:
        app.logger.exception("Error extracting content from %s: %s", url, str(e))
        return ""

def create_jira_issue(summary: str, description: str) -> dict:
    """Create a JIRA issue using credentials from environment variables."""
    base = os.environ.get("JIRA_BASE_URL")
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    project = os.environ.get("JIRA_PROJECT_KEY")
    if not all([base, email, token, project]):
        raise RuntimeError("JIRA configuration is missing")

    url = f"{base}/rest/api/3/issue"
    data = {
        "fields": {
            "project": {"key": project},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": description
                            }
                        ]
                    }
                ]
            },
            "issuetype": {"name": "Task"},
        }
    }
    # Use basic auth with an API token
    resp = requests.post(url, json=data, auth=(email, token))
    resp.raise_for_status()
    return resp.json()


def get_jira_issue_status(issue_key: str) -> str:
    """Return the status name for a JIRA issue."""
    base = os.environ.get("JIRA_BASE_URL")
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    if not all([base, email, token, issue_key]):
        return ""
    try:
        # Fetch the issue data from JIRA
        url = f"{base}/rest/api/3/issue/{issue_key}"
        resp = requests.get(url, auth=(email, token))
        resp.raise_for_status()
        data = resp.json()
        return data.get("fields", {}).get("status", {}).get("name", "")
    except Exception:  # pragma: no cover - external call
        app.logger.exception("Failed to fetch status for %s", issue_key)
        return ""


def get_jira_issue_transitions(issue_key: str) -> list[dict]:
    """Return available transitions for a JIRA issue."""
    base = os.environ.get("JIRA_BASE_URL")
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    if not all([base, email, token, issue_key]):
        return []
    try:
        # Fetch transitions that allow moving the issue between states
        url = f"{base}/rest/api/3/issue/{issue_key}/transitions"
        resp = requests.get(url, auth=(email, token))
        resp.raise_for_status()
        data = resp.json()
        return [
            {"id": t.get("id"), "name": t.get("name")}
            for t in data.get("transitions", [])
        ]
    except Exception:  # pragma: no cover - external call
        app.logger.exception("Failed to fetch transitions for %s", issue_key)
        return []


def transition_jira_issue(issue_key: str, transition_id: str) -> None:
    """Move a JIRA issue to a new status via transition id."""
    base = os.environ.get("JIRA_BASE_URL")
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    if not all([base, email, token, issue_key, transition_id]):
        return
    try:
        # Perform the transition request
        url = f"{base}/rest/api/3/issue/{issue_key}/transitions"
        data = {"transition": {"id": transition_id}}
        resp = requests.post(url, json=data, auth=(email, token))
        resp.raise_for_status()
    except Exception:  # pragma: no cover - external call
        app.logger.exception(
            "Failed to transition %s using id %s", issue_key, transition_id
        )

# Templates are stored in the ``templates`` directory

def refresh_feed_metadata(feed_id: int, feed_url: str) -> dict:
    """Fetch feed and update cached metadata. Returns the metadata dict."""
    try:
        feed_data = feedparser.parse(feed_url)
        if not feed_data.entries:
            return {'type': 'unknown', 'last_post': None, 'item_count': 0}
        
        # Determine feed type from first entry with content
        is_audio = False
        for entry in feed_data.entries[:5]:  # Check first 5 entries
            if entry.get('enclosures'):
                is_audio = True
                break
        
        feed_type = 'audio' if is_audio else 'text'
        
        # Get last post date from most recent entry
        last_post = None
        last_post_str = None
        for entry in feed_data.entries[:1]:
            if getattr(entry, 'published_parsed', None):
                last_post = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                last_post_str = last_post.isoformat()
            elif getattr(entry, 'updated_parsed', None):
                last_post = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                last_post_str = last_post.isoformat()
        
        item_count = len(feed_data.entries)
        
        # Save to database
        update_feed_metadata(feed_id, feed_type, last_post_str, item_count)
        
        return {
            'type': feed_type,
            'last_post': last_post,
            'item_count': item_count
        }
    except Exception:
        return {'type': 'unknown', 'last_post': None, 'item_count': 0}


@app.route('/', methods=['GET', 'POST'])
def index():
    """List stored podcast feeds and allow new ones to be added."""
    if request.method == 'POST':
        # User submitted a new feed URL
        feed_url = request.form['feed_url']
        feed = feedparser.parse(feed_url)
        title = feed.feed.get('title', feed_url)
        feed_id = add_feed(feed_url, title)
        # Refresh metadata for the new feed
        refresh_feed_metadata(feed_id, feed_url)
        return redirect(url_for('view_feed', feed_id=feed_id))
    
    # Get filter and sort parameters
    filter_type = request.args.get('type', '')
    sort_by = request.args.get('sort', 'title')
    sort_order = request.args.get('order', 'asc')
    search_query = request.args.get('q', '').lower()
    
    raw_feeds = list_feeds()
    feeds_with_meta = []
    
    for f in raw_feeds:
        # Use cached metadata from database
        last_post = None
        if f['last_post']:
            try:
                last_post = datetime.fromisoformat(f['last_post'])
            except (ValueError, TypeError):
                pass
        
        feed_dict = {
            'id': f['id'],
            'title': f['title'],
            'url': f['url'],
            'type': f['feed_type'] or 'unknown',
            'last_post': last_post,
            'item_count': f['item_count'] or 0,
            'last_checked': f['last_checked'],
        }
        
        # Apply type filter
        if filter_type and feed_dict['type'] != filter_type:
            continue
        
        # Apply search filter
        if search_query and search_query not in feed_dict['title'].lower():
            continue
        
        feeds_with_meta.append(feed_dict)
    
    # Sort feeds
    if sort_by == 'last_post':
        feeds_with_meta.sort(
            key=lambda x: x['last_post'] or datetime.min,
            reverse=(sort_order == 'desc')
        )
    elif sort_by == 'type':
        feeds_with_meta.sort(
            key=lambda x: x['type'] or '',
            reverse=(sort_order == 'desc')
        )
    elif sort_by == 'items':
        feeds_with_meta.sort(
            key=lambda x: x['item_count'] or 0,
            reverse=(sort_order == 'desc')
        )
    else:  # Default: title
        feeds_with_meta.sort(
            key=lambda x: x['title'].lower(),
            reverse=(sort_order == 'desc')
        )
    
    return render_template(
        'feeds.html',
        feeds=feeds_with_meta,
        filter_type=filter_type,
        sort_by=sort_by,
        sort_order=sort_order,
        search_query=search_query,
        now=datetime.now(),
    )


@app.route('/feed/<int:feed_id>/delete', methods=['POST'])
def remove_feed(feed_id: int):
    """Delete a feed and all its associated data."""
    delete_feed(feed_id)
    return redirect(url_for('index'))


@app.route('/feeds/bulk-delete', methods=['POST'])
def bulk_delete_feeds():
    """Delete multiple feeds at once."""
    feed_ids = request.form.getlist('feed_ids', type=int)
    if feed_ids:
        delete_feeds_bulk(feed_ids)
    return redirect(url_for('index'))


@app.route('/feed/<int:feed_id>/refresh')
def refresh_feed(feed_id: int):
    """Refresh metadata for a specific feed."""
    feed = get_feed_by_id(feed_id)
    if feed:
        refresh_feed_metadata(feed_id, feed['url'])
    return redirect(url_for('index'))


@app.route('/feeds/refresh-all')
def refresh_all_feeds():
    """Refresh metadata for all feeds (runs in background thread)."""
    def refresh_worker():
        for f in list_feeds():
            try:
                refresh_feed_metadata(f['id'], f['url'])
            except Exception:
                pass
    
    thread = threading.Thread(target=refresh_worker, daemon=True)
    thread.start()
    return redirect(url_for('index'))


@app.route('/feed/<int:feed_id>')
def view_feed(feed_id: int):
    """Display episodes/articles for a particular feed with pagination."""
    feed = get_feed_by_id(feed_id)
    if not feed:
        return redirect(url_for('index'))
    
    # Refresh metadata when viewing a feed (it's just one feed, so it's fast)
    refresh_feed_metadata(feed_id, feed['url'])
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    per_page = min(per_page, 50)  # Cap at 50 items per page
    
    feed_data = feedparser.parse(feed['url'])
    all_episodes = []
    is_text_feed = True  # Assume text feed, switch to audio if we find enclosures
    
    # Audio file extensions to detect
    audio_extensions = ('.mp3', '.m4a', '.wav', '.ogg', '.aac', '.flac')
    
    for entry in feed_data.entries:
        # Determine if this is an audio podcast or text feed
        has_audio = bool(entry.get('enclosures'))
        if has_audio:
            is_text_feed = False
            url = entry.enclosures[0].href
            item_type = 'audio'
        else:
            # Text feed - use link as unique identifier
            url = entry.get('link', entry.get('id', ''))
            item_type = 'text'
            if not url:
                continue
            # Check if the link itself is an audio file (some feeds use link instead of enclosure)
            url_check = url.lower().split('?')[0]  # Remove query params
            if url_check.endswith(audio_extensions):
                is_text_feed = False
                item_type = 'audio'
        ep_db = get_episode(url)
        status = {
            'transcribed': ep_db is not None and bool(ep_db['transcript']),
            'summarized': ep_db is not None and bool(ep_db['summary']),
            'actions': ep_db is not None and bool(ep_db['action_items']),
            'state': ep_db['status'] if ep_db else 'new',
        }
        # Get full content for text feeds, description for podcasts
        content = ''
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].get('value', '')
        if not content:
            content = entry.get('summary') or entry.get('description', '')
        # Prefer the summary element but fall back to description
        desc = entry.get('summary') or entry.get('description', '')
        clean_desc = strip_html(desc)
        # Try a few different locations for artwork
        img = None
        if hasattr(entry, 'image') and getattr(entry.image, 'href', None):
            img = entry.image.href
        elif entry.get('itunes_image'):
            img = entry.itunes_image.get('href') if isinstance(entry.itunes_image, dict) else entry.itunes_image
        elif entry.get('media_thumbnail'):
            img = entry.media_thumbnail[0].get('url')
        elif entry.get('media_content'):
            img = entry.media_content[0].get('url')
        published_ts = None
        if getattr(entry, 'published_parsed', None):
            published_ts = datetime.fromtimestamp(time.mktime(entry.published_parsed))
        elif getattr(entry, 'updated_parsed', None):
            published_ts = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
        published_iso = published_ts.isoformat() if published_ts else None
        # Get author if available
        author = entry.get('author', '')
        all_episodes.append({
            'title': entry.title,
            'description': desc,
            'content': content,
            'clean_description': clean_desc,
            'short_description': make_short_description(clean_desc),
            'image': img,
            'enclosure': url,
            'link': entry.get('link', ''),
            'author': author,
            'type': item_type,
            'status': status,
            'published': published_iso,
        })
    
    # Calculate pagination
    total_items = len(all_episodes)
    total_pages = (total_items + per_page - 1) // per_page  # Ceiling division
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    episodes = all_episodes[start_idx:end_idx]
    
    pagination = {
        'page': page,
        'per_page': per_page,
        'total_items': total_items,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < total_pages else None,
    }
    
    return render_template('feed.html', feed=feed, episodes=episodes, is_text_feed=is_text_feed, pagination=pagination)


@app.route('/enqueue')
def enqueue_episode():
    """Queue an episode for background processing."""
    audio_url = request.args.get('url')
    title = request.args.get('title', 'Episode')
    feed_id = request.args.get('feed_id', type=int)
    published = request.args.get('published')
    if not audio_url or feed_id is None:
        return redirect(url_for('index'))
    queue_episode(audio_url, title, feed_id, published)
    task_queue.put({'url': audio_url, 'title': title, 'feed_id': feed_id, 'published': published})
    return redirect(url_for('status_page'))


@app.route('/process_text')
def process_text_article():
    """Process a text article (no transcription needed)."""
    article_url = request.args.get('url')
    title = request.args.get('title', 'Article')
    feed_id = request.args.get('feed_id', type=int)
    published = request.args.get('published')
    description = ""
    content = ""

    if not article_url:
        return redirect(url_for('index'))

    app.logger.info("Processing text article: %s", article_url)

    # Fetch content from the feed first
    if feed_id:
        feed = get_feed_by_id(feed_id)
        if feed:
            feed_data = feedparser.parse(feed['url'])
            for entry in feed_data.entries:
                entry_url = entry.get('link', entry.get('id', ''))
                if entry_url == article_url:
                    desc = entry.get('summary') or entry.get('description', '')
                    description = strip_html(desc)
                    # Get full content from feed
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].get('value', '')
                    if not content:
                        content = desc
                    content = strip_html(content)
                    break
    
    # If RSS content is too short (likely just metadata/link), fetch the actual article
    MIN_CONTENT_LENGTH = 500  # Minimum chars to consider content "full"
    if len(content) < MIN_CONTENT_LENGTH:
        app.logger.info("RSS content too short (%d chars), fetching from URL: %s", len(content), article_url)
        fetched_content = fetch_article_content(article_url)
        if fetched_content and len(fetched_content) > len(content):
            app.logger.info("Fetched %d chars from article URL", len(fetched_content))
            content = fetched_content
        else:
            app.logger.warning("Could not fetch better content from URL")

    # Reuse previously processed data if available
    existing = get_episode(article_url)
    if existing:
        transcript = existing["transcript"]
        summary = existing["summary"]
        actions = existing["action_items"].splitlines()
        tickets = [dict(t) for t in list_tickets(existing["id"])]
        for t in tickets:
            t["status"] = get_jira_issue_status(t["ticket_key"])
            t["transitions"] = get_jira_issue_transitions(t["ticket_key"])
        articles = [dict(a) for a in list_articles(existing["id"])]
        return render_template(
            'result.html',
            title=existing["title"],
            transcript=transcript,
            summary=summary,
            actions=actions,
            description=description,
            feed_id=feed_id,
            url=article_url,
            tickets=tickets,
            articles=articles,
            current_url=request.full_path,
            is_text=True,
            original_link=article_url,
        )

    if not content:
        app.logger.error("No content found for article: %s", article_url)
        return redirect(url_for('view_feed', feed_id=feed_id))

    # For text articles, content IS the transcript (no transcription needed)
    transcript = content
    app.logger.info("Article content loaded (%d chars)", len(transcript))

    app.logger.info("Generating summary")
    summary = summarize_text(transcript)
    app.logger.info("Summary complete")

    app.logger.info("Extracting action items")
    actions = extract_action_items(transcript)
    app.logger.info("Action item extraction complete")

    # Persist results
    save_episode(article_url, title, transcript, summary, actions, feed_id, published)

    return render_template(
        'result.html',
        title=title,
        transcript=transcript,
        summary=summary,
        actions=actions,
        description=description,
        feed_id=feed_id,
        url=article_url,
        tickets=[],
        articles=[],
        current_url=request.full_path,
        is_text=True,
        original_link=article_url,
    )


@app.route('/process')
def process_episode():
    """Process an episode synchronously and show the results."""
    audio_url = request.args.get('url')
    title = request.args.get('title', 'Episode')
    feed_id = request.args.get('feed_id', type=int)
    published = request.args.get('published')
    description = ""
    if not audio_url:
        return redirect(url_for('index'))
    app.logger.info("Processing episode: %s", audio_url)
    if not description and feed_id:
        feed = get_feed_by_id(feed_id)
        if feed:
            feed_data = feedparser.parse(feed['url'])
            for entry in feed_data.entries:
                if entry.get('enclosures') and entry.enclosures[0].href == audio_url:
                    desc = entry.get('summary') or entry.get('description', '')
                    description = strip_html(desc)
                    break
    # Reuse previously processed data if available
    existing = get_episode(audio_url)
    if existing:
        # Already processed - read results from the DB
        transcript = existing["transcript"]
        summary = existing["summary"]
        actions = existing["action_items"].splitlines()
        tickets = [dict(t) for t in list_tickets(existing["id"])]
        for t in tickets:
            t["status"] = get_jira_issue_status(t["ticket_key"])
            t["transitions"] = get_jira_issue_transitions(t["ticket_key"])
        articles = [dict(a) for a in list_articles(existing["id"])]
        return render_template(
            'result.html',
            title=existing["title"],
            transcript=transcript,
            summary=summary,
            actions=actions,
            description=description,
            feed_id=feed_id,
            url=audio_url,
            tickets=tickets,
            articles=articles,
            current_url=request.full_path,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, 'episode.mp3')
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(audio_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Run the same processing pipeline as the CLI
        app.logger.info("Transcribing audio")
        transcript = transcribe_audio(audio_path)
        app.logger.info("Transcription complete")

        app.logger.info("Generating summary")
        summary = summarize_text(transcript)
        app.logger.info("Summary complete")

        app.logger.info("Extracting action items")
        actions = extract_action_items(transcript)
        app.logger.info("Action item extraction complete")
        out_path = os.path.join(tmpdir, 'results.json')
        write_results_json(transcript, summary, actions, out_path)
        # Persist results so they can be reused later
        save_episode(audio_url, title, transcript, summary, actions, feed_id, published)
    return render_template(
        'result.html',
        title=title,
        transcript=transcript,
        summary=summary,
        actions=actions,
        description=description,
        feed_id=feed_id,
        url=audio_url,
        tickets=[],
        articles=[],
        current_url=request.full_path,
    )


@app.route('/create_jira', methods=['POST'])
def create_jira():
    """Create JIRA tickets for the selected action items."""
    # Items are the action item texts selected by the user
    items = request.form.getlist('items')
    episode_url = request.form.get('episode_url')
    title = request.form.get('title', 'Episode')
    if not items:
        return redirect(request.referrer or url_for('index'))
    episode = get_episode(episode_url) if episode_url else None
    episode_id = episode['id'] if episode else None
    summary_text = episode['summary'] if episode else ''
    
    # Get the source (feed/podcast name)
    source_name = ''
    if episode and episode['feed_id']:
        feed = get_feed_by_id(episode['feed_id'])
        if feed:
            source_name = feed['title']
    
    created = []
    for item in items:
        try:
            # Build description with source if available
            source_line = f"Source: {source_name}\n" if source_name else ""
            description = (
                f"Action item: {item}\n\n"
                f"{source_line}"
                f"From episode: {title}\n\n"
                f"Episode summary:\n{summary_text}"
            )
            issue = create_jira_issue(item, description)
            key = issue.get('key', '')
            ticket_url = f"{os.environ.get('JIRA_BASE_URL')}/browse/{key}" if key else ''
            if episode_id is not None and key:
                add_ticket(episode_id, item, key, ticket_url)
            created.append({'key': key, 'url': ticket_url})
        except Exception as exc:  # pragma: no cover - external call
            created.append({'error': str(exc)})
    return render_template('jira_result.html', created=created)


@app.route('/update_ticket', methods=['POST'])
def update_ticket():
    """Update a JIRA ticket's status using a selected transition."""
    # Ticket key and selected transition id from the form
    ticket_key = request.form.get('ticket_key')
    transition_id = request.form.get('transition_id')
    ref = request.form.get('ref') or url_for('view_tickets')
    if ticket_key and transition_id:
        transition_jira_issue(ticket_key, transition_id)
    return redirect(ref)


@app.route('/tickets/<int:ticket_id>/delete', methods=['POST'])
def delete_ticket_route(ticket_id: int):
    """Delete a single JIRA ticket from the database."""
    success = delete_ticket(ticket_id)
    if success:
        return jsonify({"success": True, "message": "Ticket deleted"})
    return jsonify({"error": "Ticket not found"}), 404


@app.route('/tickets/delete-selected', methods=['POST'])
def delete_tickets_selected():
    """Delete multiple selected JIRA tickets."""
    data = request.get_json()
    ticket_ids = data.get('ticket_ids', []) if data else []
    
    if not ticket_ids:
        return jsonify({"error": "No tickets selected"}), 400
    
    try:
        ticket_ids = [int(tid) for tid in ticket_ids]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid ticket IDs"}), 400
    
    count = delete_tickets_bulk(ticket_ids)
    return jsonify({
        "success": True,
        "message": f"Deleted {count} ticket{'s' if count != 1 else ''}"
    })


@app.route('/api/routes')
def list_api_routes():
    """List all available API routes.
    ---
    tags:
      - System
    responses:
      200:
        description: List of all API routes with their methods and endpoints
        schema:
          type: object
          properties:
            routes:
              type: array
              items:
                type: object
                properties:
                  endpoint:
                    type: string
                  methods:
                    type: array
                    items:
                      type: string
                  path:
                    type: string
            count:
              type: integer
    """
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods - {'HEAD', 'OPTIONS'}),
                'path': str(rule)
            })
    routes.sort(key=lambda x: x['path'])
    return jsonify({'routes': routes, 'count': len(routes)})


@app.route('/status')
def status_page():
    """Display processing status for all episodes."""
    sort = request.args.get('sort', 'released')
    sort_order = request.args.get('order', 'desc')
    filter_status = request.args.get('status', '')
    filter_feed = request.args.get('feed', '', type=str)
    filter_type = request.args.get('type', '')
    search_query = request.args.get('q', '').lower()
    
    if sort == 'released':
        order_by = 'published'
    elif sort == 'processed':
        order_by = 'processed_at'
    else:
        order_by = 'id'
    
    all_episodes = list_all_episodes(order_by=order_by)
    feeds_list = list_feeds()
    feeds = {f["id"]: f["title"] for f in feeds_list}
    
    # Filter episodes
    filtered_episodes = []
    for ep in all_episodes:
        # Status filter
        if filter_status and ep['status'] != filter_status:
            continue
        
        # Feed filter
        if filter_feed and str(ep['feed_id']) != filter_feed:
            continue
        
        # Type filter (audio vs text)
        is_audio = ep['url'].lower().split('?')[0].endswith(('.mp3', '.m4a', '.wav', '.ogg', '.aac', '.flac'))
        ep_type = 'audio' if is_audio else 'text'
        if filter_type and ep_type != filter_type:
            continue
        
        # Search filter
        if search_query and search_query not in (ep['title'] or '').lower():
            continue
        
        filtered_episodes.append(ep)
    
    # Reverse order if ascending
    if sort_order == 'asc':
        filtered_episodes = list(reversed(filtered_episodes))
    
    return render_template(
        'status.html',
        episodes=filtered_episodes,
        feeds=feeds,
        feeds_list=feeds_list,
        sort=sort,
        sort_order=sort_order,
        filter_status=filter_status,
        filter_feed=filter_feed,
        filter_type=filter_type,
        search_query=search_query,
    )


@app.route('/episode/<int:episode_id>/reprocess')
def reprocess_episode(episode_id: int):
    """Reprocess an episode - clears existing data and requeues."""
    episode = get_episode_by_id(episode_id)
    if not episode:
        return redirect(url_for('status_page'))

    # Detect if this is audio or text
    audio_extensions = ('.mp3', '.m4a', '.wav', '.ogg', '.aac', '.flac')
    is_audio = episode['url'].lower().split('?')[0].endswith(audio_extensions)

    # Reset episode data for reprocessing
    reset_episode_for_reprocess(episode_id)

    if is_audio:
        # Queue for background processing
        task_queue.put({
            'url': episode['url'],
            'title': episode['title'],
            'feed_id': episode['feed_id'],
            'published': episode['published'],
        })
        return redirect(url_for('status_page'))
    else:
        # Process text article directly
        return redirect(url_for(
            'process_text_article',
            url=episode['url'],
            title=episode['title'],
            feed_id=episode['feed_id'],
            published=episode['published'],
        ))


@app.route('/episode/<int:episode_id>/delete')
def delete_episode(episode_id: int):
    """Delete an episode and all associated data."""
    delete_episode_by_id(episode_id)
    return redirect(url_for('status_page'))


@app.route('/episodes/bulk-delete', methods=['POST'])
def bulk_delete_episodes():
    """Delete multiple episodes at once."""
    episode_ids = request.form.getlist('episode_ids', type=int)
    if episode_ids:
        delete_episodes_bulk(episode_ids)
    return redirect(url_for('status_page'))


@app.route('/tickets')
def view_tickets():
    """Display all created JIRA tickets."""
    sort_by = request.args.get('sort', 'id')
    sort_order = request.args.get('order', 'desc')
    filter_status = request.args.get('status', '')
    search_query = request.args.get('q', '').lower()
    
    raw_tickets = [dict(t) for t in list_tickets()]
    
    # Fetch JIRA statuses and filter
    tickets = []
    all_statuses = set()
    
    for t in raw_tickets:
        t["status"] = get_jira_issue_status(t["ticket_key"])
        t["transitions"] = get_jira_issue_transitions(t["ticket_key"])
        all_statuses.add(t["status"] or "Unknown")
        
        # Filter by status
        if filter_status and t["status"] != filter_status:
            continue
        
        # Search filter
        if search_query:
            searchable = f"{t['episode_title']} {t['action_item']} {t['ticket_key']}".lower()
            if search_query not in searchable:
                continue
        
        tickets.append(t)
    
    # Sort tickets
    if sort_by == 'episode':
        tickets.sort(key=lambda x: (x['episode_title'] or '').lower(), reverse=(sort_order == 'desc'))
    elif sort_by == 'status':
        tickets.sort(key=lambda x: (x['status'] or '').lower(), reverse=(sort_order == 'desc'))
    elif sort_by == 'ticket':
        tickets.sort(key=lambda x: x['ticket_key'], reverse=(sort_order == 'desc'))
    else:
        tickets.sort(key=lambda x: x['id'], reverse=(sort_order == 'desc'))
    
    return render_template(
        'tickets.html',
        tickets=tickets,
        sort_by=sort_by,
        sort_order=sort_order,
        filter_status=filter_status,
        search_query=search_query,
        all_statuses=sorted(all_statuses),
    )


@app.route('/generate_article', methods=['POST'])
def create_article():
    """Generate an article based on podcast or text article content."""
    episode_url = request.form.get('episode_url')
    topic = request.form.get('topic', '').strip()
    style = request.form.get('style', 'blog')
    extra_context = request.form.get('extra_context', '').strip()

    if not episode_url or not topic:
        return redirect(request.referrer or url_for('index'))

    episode = get_episode(episode_url)
    if not episode:
        return redirect(url_for('index'))

    # Detect if this is a text article (not an audio file)
    audio_extensions = ('.mp3', '.m4a', '.wav', '.ogg', '.aac', '.flac')
    is_text_source = not episode_url.lower().split('?')[0].endswith(audio_extensions)

    # Get the podcast/publication title for attribution
    feed = get_feed_by_id(episode['feed_id']) if episode['feed_id'] else None
    podcast_title = feed['title'] if feed else ("the publication" if is_text_source else "the podcast")
    episode_title = episode['title'] or ("this article" if is_text_source else "this episode")

    try:
        # Generate article using OpenAI
        article_content = generate_article(
            transcript=episode['transcript'],
            summary=episode['summary'],
            topic=topic,
            podcast_title=podcast_title,
            episode_title=episode_title,
            style=style,
            extra_context=extra_context if extra_context else None,
            is_text_source=is_text_source,
        )
        # Save article to database
        article_id = add_article(
            episode_id=episode['id'],
            topic=topic,
            style=style,
            content=article_content,
        )
        return redirect(url_for('view_article', article_id=article_id))
    except Exception as exc:
        app.logger.exception("Failed to generate article")
        return render_template(
            'article_error.html',
            error=str(exc),
            episode_url=episode_url,
            feed_id=episode['feed_id'],
        )


@app.route('/article/<int:article_id>')
def view_article(article_id: int):
    """Display a generated article."""
    article = get_article(article_id)
    if not article:
        return redirect(url_for('index'))
    
    # Get saved social posts grouped by platform
    posts = list_social_posts(article_id)
    
    # Get pending schedules for all posts
    post_ids = [post['id'] for post in posts]
    pending_schedules = get_pending_schedules_for_social_posts(post_ids)
    
    social_posts = {}
    for post in posts:
        platform = post['platform']
        if platform not in social_posts:
            social_posts[platform] = []
        
        # Check if this post is scheduled for this platform
        schedules = pending_schedules.get(post['id'], [])
        scheduled_for_platform = {}
        for sched in schedules:
            sched_platform = sched['platform']
            if sched_platform not in scheduled_for_platform:
                scheduled_for_platform[sched_platform] = sched['scheduled_for']
        
        social_posts[platform].append({
            'id': post['id'],
            'content': post['content'],
            'created_at': post['created_at'],
            'used': bool(post['used']),
            'scheduled': scheduled_for_platform,  # Dict of platform -> scheduled_for
        })
    
    return render_template('article.html', article=dict(article), social_posts=social_posts)


@app.route('/article/<int:article_id>/edit', methods=['GET', 'POST'])
def edit_article(article_id: int):
    """Edit an existing article."""
    article = get_article(article_id)
    if not article:
        return redirect(url_for('view_articles'))
    
    if request.method == 'POST':
        topic = request.form.get('topic', '').strip()
        style = request.form.get('style', '').strip()
        content = request.form.get('content', '').strip()
        
        if topic and content:
            update_article(article_id, topic=topic, style=style, content=content)
            return redirect(url_for('view_article', article_id=article_id))
    
    return render_template('article_edit.html', article=dict(article))


@app.route('/article/<int:article_id>/delete', methods=['POST'])
def remove_article(article_id: int):
    """Delete an article."""
    delete_article(article_id)
    return redirect(url_for('view_articles'))


@app.route('/article/<int:article_id>/social', methods=['POST'])
def generate_article_social(article_id: int):
    """Generate social media promotional copy for an article and save to database."""
    article = get_article(article_id)
    if not article:
        return {"error": "Article not found"}, 404
    
    platforms = request.form.getlist('platforms')
    if not platforms:
        platforms = ["twitter", "linkedin", "facebook", "threads", "bluesky"]
    
    # Get number of posts per platform (default to 1, max 21)
    posts_per_platform = request.form.get('posts_per_platform', 1, type=int)
    posts_per_platform = max(1, min(posts_per_platform, 21))
    
    # Get optional extra context for the prompt
    extra_context = request.form.get('extra_context', '').strip() or None
    
    try:
        social_copy = generate_social_copy(
            article_content=article['content'],
            article_topic=article['topic'],
            platforms=platforms,
            posts_per_platform=posts_per_platform,
            extra_context=extra_context,
        )
        
        # Save generated posts to database
        saved_posts = {}
        for platform, copy_data in social_copy.items():
            posts = copy_data if isinstance(copy_data, list) else [copy_data]
            saved_posts[platform] = []
            for post_content in posts:
                post_id = add_social_post(
                    article_id=article_id,
                    platform=platform,
                    content=post_content,
                )
                saved_posts[platform].append({
                    'id': post_id,
                    'content': post_content,
                })
        
        return {
            "success": True,
            "social_copy": social_copy,
            "saved_posts": saved_posts,
            "posts_per_platform": posts_per_platform,
        }
    except Exception as exc:
        app.logger.exception("Failed to generate social media copy")
        return {"error": str(exc)}, 500


@app.route('/article/<int:article_id>/social/list', methods=['GET'])
def list_article_social_posts(article_id: int):
    """Get all saved social posts for an article."""
    article = get_article(article_id)
    if not article:
        return {"error": "Article not found"}, 404
    
    posts = list_social_posts(article_id)
    
    # Group posts by platform
    grouped = {}
    for post in posts:
        platform = post['platform']
        if platform not in grouped:
            grouped[platform] = []
        grouped[platform].append({
            'id': post['id'],
            'content': post['content'],
            'created_at': post['created_at'],
            'used': bool(post['used']),
        })
    
    return {"success": True, "posts": grouped}


@app.route('/social/<int:post_id>/delete', methods=['POST'])
def delete_social_post_route(post_id: int):
    """Delete a single social post."""
    delete_social_post(post_id)
    return {"success": True}


@app.route('/social/bulk-delete', methods=['POST'])
def bulk_delete_social_posts():
    """Delete multiple social posts at once."""
    post_ids = request.form.getlist('post_ids', type=int)
    if post_ids:
        count = delete_social_posts_bulk(post_ids)
        return {"success": True, "deleted": count}
    return {"success": True, "deleted": 0}


@app.route('/article/<int:article_id>/social/clear', methods=['POST'])
def clear_article_social_posts(article_id: int):
    """Delete all social posts for an article."""
    count = delete_social_posts_for_article(article_id)
    return {"success": True, "deleted": count}


@app.route('/social/<int:post_id>/toggle-used', methods=['POST'])
def toggle_social_post_used(post_id: int):
    """Toggle the used status of a social post."""
    post = get_social_post(post_id)
    if not post:
        return {"error": "Post not found"}, 404
    
    new_used = not bool(post['used'])
    mark_social_post_used(post_id, new_used)
    return {"success": True, "used": new_used}


@app.route('/social/<int:post_id>/edit', methods=['POST'])
def edit_social_post(post_id: int):
    """Update the content of a social post."""
    post = get_social_post(post_id)
    if not post:
        return {"error": "Post not found"}, 404
    
    content = request.form.get('content', '').strip()
    if not content:
        return {"error": "Content cannot be empty"}, 400
    
    update_social_post(post_id, content)
    return {"success": True, "content": content}


@app.route('/social/<int:post_id>/image', methods=['POST'])
def edit_social_post_image(post_id: int):
    """Update the image URL of a social post."""
    post = get_social_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    image_url = request.form.get('image_url', '').strip() or None
    update_social_post_image(post_id, image_url)
    return jsonify({"success": True, "image_url": image_url})


@app.route('/social/posts/bulk-image', methods=['POST'])
def social_bulk_update_images():
    """Bulk update images for multiple social posts."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    post_ids = data.get('post_ids', [])
    image_url = data.get('image_url')  # None to remove image
    
    if not post_ids:
        return jsonify({"error": "No post IDs provided"}), 400
    
    # Convert to integers
    try:
        post_ids = [int(pid) for pid in post_ids]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid post IDs"}), 400
    
    # Update each post
    updated_count = 0
    for post_id in post_ids:
        post = get_social_post(post_id)
        if post:
            update_social_post_image(post_id, image_url)
            updated_count += 1
    
    return jsonify({
        "success": True,
        "updated_count": updated_count,
        "message": f"Updated {updated_count} posts"
    })


@app.route('/posts/bulk-replace', methods=['POST'])
def bulk_replace_posts():
    """Replace text in all posts of a given type, optionally filtered by post IDs."""
    from database import bulk_replace_post_content
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    find_text = data.get('find', '')
    replace_text = data.get('replace', '')
    post_type = data.get('post_type', 'social')  # 'social' or 'standalone'
    case_sensitive = data.get('case_sensitive', False)
    whole_word = data.get('whole_word', False)
    post_ids = data.get('post_ids')  # Optional list of post IDs to filter
    excluded_matches = data.get('excluded_matches')  # Optional dict of excluded matches
    
    if not find_text:
        return jsonify({"error": "Find text is required"}), 400
    
    if post_type not in ('social', 'standalone'):
        return jsonify({"error": "post_type must be 'social' or 'standalone'"}), 400
    
    # Convert post_ids to integers if provided
    if post_ids:
        try:
            post_ids = [int(pid) for pid in post_ids]
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid post IDs"}), 400
    
    try:
        affected_count = bulk_replace_post_content(
            find_text=find_text,
            replace_text=replace_text,
            post_type=post_type,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            post_ids=post_ids,
            excluded_matches=excluded_matches,
        )
        return jsonify({
            "success": True,
            "affected_count": affected_count,
            "message": f"Replaced in {affected_count} post(s)"
        })
    except Exception as exc:
        app.logger.exception("Failed to bulk replace posts")
        return jsonify({"error": str(exc)}), 500


@app.route('/article/<int:article_id>/refine', methods=['POST'])
def refine_article_with_ai(article_id: int):
    """Refine an article using AI based on user feedback."""
    article = get_article(article_id)
    if not article:
        return {"error": "Article not found"}, 404
    
    feedback = request.form.get('feedback', '').strip()
    if not feedback:
        return {"error": "Please provide feedback for how to refine the article"}, 400
    
    try:
        refined_content = refine_article(
            current_content=article['content'],
            user_feedback=feedback,
            article_topic=article['topic'],
        )
        
        # Optionally auto-save the refined content
        auto_save = request.form.get('auto_save', 'false') == 'true'
        if auto_save:
            update_article(article_id, content=refined_content)
        
        return {"success": True, "refined_content": refined_content, "saved": auto_save}
    except Exception as exc:
        app.logger.exception("Failed to refine article")
        return {"error": str(exc)}, 500


@app.route('/articles')
def view_articles():
    """Display all generated articles."""
    sort_by = request.args.get('sort', 'date')
    sort_order = request.args.get('order', 'desc')
    filter_style = request.args.get('style', '')
    filter_podcast = request.args.get('podcast', '')
    search_query = request.args.get('q', '').lower()
    
    raw_articles = [dict(a) for a in list_articles()]
    
    # Collect unique styles and podcasts for filter dropdowns
    all_styles = set()
    all_podcasts = set()
    
    articles = []
    for a in raw_articles:
        all_styles.add(a['style'])
        if a.get('podcast_title'):
            all_podcasts.add(a['podcast_title'])
        
        # Filter by style
        if filter_style and a['style'] != filter_style:
            continue
        
        # Filter by podcast
        if filter_podcast and a.get('podcast_title') != filter_podcast:
            continue
        
        # Search filter
        if search_query:
            searchable = f"{a['topic']} {a.get('episode_title', '')} {a.get('podcast_title', '')}".lower()
            if search_query not in searchable:
                continue
        
        articles.append(a)
    
    # Sort articles
    if sort_by == 'topic':
        articles.sort(key=lambda x: x['topic'].lower(), reverse=(sort_order == 'desc'))
    elif sort_by == 'style':
        articles.sort(key=lambda x: x['style'].lower(), reverse=(sort_order == 'desc'))
    elif sort_by == 'podcast':
        articles.sort(key=lambda x: (x.get('podcast_title') or '').lower(), reverse=(sort_order == 'desc'))
    else:  # Default: date
        articles.sort(key=lambda x: x['created_at'] or '', reverse=(sort_order == 'desc'))
    
    return render_template(
        'articles.html',
        articles=articles,
        sort_by=sort_by,
        sort_order=sort_order,
        filter_style=filter_style,
        filter_podcast=filter_podcast,
        search_query=search_query,
        all_styles=sorted(all_styles),
        all_podcasts=sorted(all_podcasts),
    )


# ============================================================================
# LinkedIn Integration Routes
# ============================================================================


@app.route('/linkedin/status')
def linkedin_status():
    """Check LinkedIn connection status."""
    client = get_linkedin_client()
    token = get_linkedin_token()
    
    if not client.is_configured():
        return jsonify({
            "connected": False,
            "configured": False,
            "message": "LinkedIn credentials not configured. Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET.",
        })
    
    if not token:
        return jsonify({
            "connected": False,
            "configured": True,
            "message": "Not connected to LinkedIn",
        })
    
    # Check if token is expired
    if is_token_expired(token['expires_at']):
        # Try to refresh if we have a refresh token
        if token['refresh_token']:
            try:
                new_token = client.refresh_access_token(token['refresh_token'])
                expires_at = calculate_token_expiry(new_token.get('expires_in', 5184000))
                update_linkedin_token(
                    access_token=new_token['access_token'],
                    expires_at=expires_at,
                    refresh_token=new_token.get('refresh_token'),
                )
                return jsonify({
                    "connected": True,
                    "configured": True,
                    "display_name": token['display_name'],
                    "email": token['email'],
                    "message": "Connected (token refreshed)",
                })
            except Exception as e:
                app.logger.warning("Failed to refresh LinkedIn token: %s", e)
                return jsonify({
                    "connected": False,
                    "configured": True,
                    "message": "Token expired. Please reconnect.",
                })
        else:
            return jsonify({
                "connected": False,
                "configured": True,
                "message": "Token expired. Please reconnect.",
            })
    
    # Check if user_urn is configured (needed for posting)
    needs_configuration = not token['user_urn']
    
    return jsonify({
        "connected": True,
        "configured": True,
        "needs_configuration": needs_configuration,
        "display_name": token['display_name'],
        "email": token['email'],
        "user_urn": token['user_urn'],
        "configure_url": url_for('linkedin_configure') if needs_configuration else None,
    })


@app.route('/linkedin/auth')
def linkedin_auth():
    """Start LinkedIn OAuth flow."""
    client = get_linkedin_client()
    
    if not client.is_configured():
        return jsonify({"error": "LinkedIn not configured"}), 400
    
    auth_url, state = client.get_authorization_url()
    session['linkedin_oauth_state'] = state
    
    return redirect(auth_url)


@app.route('/linkedin/callback')
def linkedin_callback():
    """Handle LinkedIn OAuth callback."""
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        app.logger.error("LinkedIn OAuth error: %s - %s", error, error_desc)
        return render_template(
            'article_error.html',
            error=f"LinkedIn authorization failed: {error_desc}",
        )
    
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Verify state to prevent CSRF
    stored_state = session.pop('linkedin_oauth_state', None)
    if not stored_state or stored_state != state:
        app.logger.warning("LinkedIn OAuth state mismatch")
        return render_template(
            'article_error.html',
            error="Security verification failed. Please try again.",
        )
    
    if not code:
        return render_template(
            'article_error.html',
            error="No authorization code received from LinkedIn.",
        )
    
    client = get_linkedin_client()
    
    try:
        # Exchange code for token
        token_data = client.exchange_code_for_token(code)
        access_token = token_data['access_token']
        expires_in = token_data.get('expires_in', 5184000)  # Default 60 days
        refresh_token = token_data.get('refresh_token')
        
        # Calculate expiry
        expires_at = calculate_token_expiry(expires_in)
        
        # Try to get user info - may return None if only w_member_social scope
        user_info = client.get_user_info(access_token)
        
        if user_info:
            member_id = user_info.get('sub') or user_info.get('id', '')
            user_urn = f"urn:li:person:{member_id}" if member_id else ''
            display_name = user_info.get('name', '') or user_info.get('localizedFirstName', 'LinkedIn User')
            email = user_info.get('email', '')
        else:
            # Profile endpoints didn't work - user needs to manually configure
            member_id = ''
            user_urn = ''
            display_name = 'LinkedIn User (needs configuration)'
            email = ''
        
        # Save token (with or without profile info)
        save_linkedin_token(
            access_token=access_token,
            expires_at=expires_at,
            member_id=member_id,
            user_urn=user_urn,
            display_name=display_name,
            email=email,
            refresh_token=refresh_token,
        )
        
        app.logger.info("LinkedIn connected for user: %s", display_name)
        
        # If we couldn't get profile info, redirect to configuration page
        if not user_urn:
            return redirect(url_for('linkedin_configure') + '?new=1')
        
        # Redirect to articles page with success message
        return redirect(url_for('view_articles') + '?linkedin=connected')
        
    except Exception as e:
        app.logger.exception("LinkedIn OAuth exchange failed")
        return render_template(
            'article_error.html',
            error=f"Failed to connect to LinkedIn: {str(e)}",
        )


@app.route('/linkedin/disconnect', methods=['POST'])
def linkedin_disconnect():
    """Disconnect LinkedIn account."""
    delete_linkedin_token()
    return jsonify({"success": True, "message": "LinkedIn disconnected"})


@app.route('/linkedin/configure', methods=['GET', 'POST'])
def linkedin_configure():
    """Configure LinkedIn member ID manually.
    
    This is needed when the user only has 'Share on LinkedIn' product
    which doesn't provide profile access scopes.
    """
    from database import update_linkedin_member_urn
    
    token = get_linkedin_token()
    if not token:
        return redirect(url_for('view_schedule') + '?error=not_connected')
    
    if request.method == 'POST':
        member_id = request.form.get('member_id', '').strip()
        display_name = request.form.get('display_name', '').strip() or 'LinkedIn User'
        
        if not member_id:
            return render_template(
                'linkedin_configure.html',
                token=token,
                error="Member ID is required",
                is_new=request.args.get('new') == '1',
            )
        
        # Update the token with the manual member ID
        success = update_linkedin_member_urn(
            member_id=member_id,
            display_name=display_name,
        )
        
        if success:
            app.logger.info("LinkedIn member ID configured manually: %s", member_id)
            return redirect(url_for('view_schedule') + '?linkedin=configured')
        else:
            return render_template(
                'linkedin_configure.html',
                token=token,
                error="Failed to save configuration",
                is_new=request.args.get('new') == '1',
            )
    
    # GET request - show configuration form
    return render_template(
        'linkedin_configure.html',
        token=token,
        is_new=request.args.get('new') == '1',
    )


@app.route('/linkedin/post/<int:post_id>', methods=['POST'])
def linkedin_post_social(post_id: int):
    """Post a social media post to LinkedIn immediately."""
    post = get_social_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    token = get_linkedin_token()
    if not token:
        return jsonify({"error": "LinkedIn not connected"}), 401
    
    if is_token_expired(token['expires_at']):
        return jsonify({"error": "LinkedIn token expired. Please reconnect."}), 401
    
    if not token['user_urn']:
        return jsonify({
            "error": "LinkedIn needs configuration. Please configure your Member ID.",
            "configure_url": url_for('linkedin_configure')
        }), 400
    
    client = get_linkedin_client()
    
    try:
        # Get image URL if available
        image_url = post['image_url'] if 'image_url' in post.keys() else None
        
        # Use smart post to automatically detect URLs and show link previews
        # Pass the article topic as fallback title for link previews
        article_topic = post['article_topic'] if 'article_topic' in post.keys() else None
        
        # Use image post if image URL is available and no URL in content
        if image_url and not client.extract_first_url(post['content']):
            app.logger.info("Posting to LinkedIn with image: %s", image_url)
            result = client.create_image_post(
                access_token=token['access_token'],
                author_urn=token['user_urn'],
                text=post['content'],
                image_url=image_url,
            )
        else:
            result = client.create_smart_post(
                access_token=token['access_token'],
                author_urn=token['user_urn'],
                text=post['content'],
                article_title=article_topic,
            )
        
        if result['success']:
            # Mark the post as used
            mark_social_post_used(post_id, True)
            
            # Record in scheduled_posts for history tracking
            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=post_id,
                article_id=post['article_id'] if 'article_id' in post.keys() else None,
                post_type='social',
                platform='linkedin',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('post_urn'),
            )
            
            return jsonify({
                "success": True,
                "post_urn": result['post_urn'],
                "message": "Posted to LinkedIn successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400
            
    except Exception as e:
        app.logger.exception("Failed to post to LinkedIn")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Threads Integration Routes
# ============================================================================


@app.route('/threads/status')
def threads_status():
    """Check Threads connection status."""
    client = get_threads_client()
    token = get_threads_token()
    
    if not client.is_configured():
        return jsonify({
            "connected": False,
            "configured": False,
            "message": "Threads credentials not configured. Set THREADS_APP_ID and THREADS_APP_SECRET.",
        })
    
    if not token:
        return jsonify({
            "connected": False,
            "configured": True,
            "message": "Not connected to Threads",
        })
    
    # Check if token is expired
    if threads_is_token_expired(token['expires_at']):
        # Try to refresh the token
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = threads_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_threads_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            return jsonify({
                "connected": True,
                "configured": True,
                "username": token['username'],
                "display_name": token['display_name'],
                "profile_picture_url": token['profile_picture_url'],
                "message": "Connected (token refreshed)",
            })
        except Exception as e:
            app.logger.warning("Failed to refresh Threads token: %s", e)
            return jsonify({
                "connected": False,
                "configured": True,
                "message": "Token expired. Please reconnect.",
            })
    
    return jsonify({
        "connected": True,
        "configured": True,
        "username": token['username'],
        "display_name": token['display_name'],
        "profile_picture_url": token['profile_picture_url'],
        "user_id": token['user_id'],
    })


@app.route('/threads/auth')
def threads_auth():
    """Start Threads OAuth flow."""
    client = get_threads_client()
    
    if not client.is_configured():
        return jsonify({"error": "Threads not configured"}), 400
    
    auth_url, state = client.get_authorization_url()
    session['threads_oauth_state'] = state
    
    return redirect(auth_url)


@app.route('/threads/callback')
def threads_callback():
    """Handle Threads OAuth callback."""
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        app.logger.error("Threads OAuth error: %s - %s", error, error_desc)
        return render_template(
            'article_error.html',
            error=f"Threads authorization failed: {error_desc}",
        )
    
    code = request.args.get('code')
    state = request.args.get('state')
    
    # Verify state to prevent CSRF
    stored_state = session.pop('threads_oauth_state', None)
    if not stored_state or stored_state != state:
        app.logger.warning("Threads OAuth state mismatch")
        return render_template(
            'article_error.html',
            error="Security verification failed. Please try again.",
        )
    
    if not code:
        return render_template(
            'article_error.html',
            error="No authorization code received from Threads.",
        )
    
    client = get_threads_client()
    
    try:
        # Step 1: Exchange code for short-lived token
        token_data = client.exchange_code_for_token(code)
        short_lived_token = token_data['access_token']
        user_id = token_data.get('user_id', '')
        
        # Step 2: Exchange for long-lived token (60 days)
        long_lived_data = client.get_long_lived_token(short_lived_token)
        access_token = long_lived_data['access_token']
        expires_in = long_lived_data.get('expires_in', 5184000)  # Default 60 days
        
        # Calculate expiry
        expires_at = threads_calculate_token_expiry(expires_in)
        
        # Get user profile
        user_info = client.get_user_profile(access_token)
        
        if user_info:
            user_id = user_info.get('id', user_id)
            username = user_info.get('username', '')
            display_name = user_info.get('name', '') or username
            profile_picture_url = user_info.get('threads_profile_picture_url', '')
        else:
            username = ''
            display_name = 'Threads User'
            profile_picture_url = ''
        
        # Save token
        save_threads_token(
            access_token=access_token,
            expires_at=expires_at,
            user_id=user_id,
            username=username,
            display_name=display_name,
            profile_picture_url=profile_picture_url,
        )
        
        app.logger.info("Threads connected for user: @%s", username)
        
        # Redirect to schedule page with success message
        return redirect(url_for('schedule_list') + '?threads=connected')
        
    except Exception as e:
        app.logger.exception("Threads OAuth exchange failed")
        return render_template(
            'article_error.html',
            error=f"Failed to connect to Threads: {str(e)}",
        )


@app.route('/threads/disconnect', methods=['POST'])
def threads_disconnect():
    """Disconnect Threads account."""
    delete_threads_token()
    return jsonify({"success": True, "message": "Threads disconnected"})


@app.route('/threads/post/<int:post_id>', methods=['POST'])
def threads_post_social(post_id: int):
    """Post a social media post to Threads immediately."""
    post = get_social_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    # Check if this is a Threads post
    if post['platform'] != 'threads':
        return jsonify({"error": "This post is not for Threads"}), 400
    
    # Get Threads token
    token = get_threads_token()
    if not token:
        return jsonify({"error": "Threads not connected. Please connect your account first."}), 401
    
    # Check if token is expired and try to refresh
    if threads_is_token_expired(token['expires_at']):
        client = get_threads_client()
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = threads_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_threads_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            token = get_threads_token()
        except Exception as e:
            app.logger.warning("Failed to refresh Threads token: %s", e)
            return jsonify({"error": "Threads token expired. Please reconnect."}), 401
    
    client = get_threads_client()
    
    try:
        # Get image URL if available
        image_url = post['image_url'] if 'image_url' in post.keys() else None
        
        # Use image post if image URL is available
        if image_url:
            app.logger.info("Posting to Threads with image: %s", image_url)
            result = client.publish_image_post(
                access_token=token['access_token'],
                text=post['content'],
                image_url=image_url,
            )
        else:
            result = client.publish_text_post(
                access_token=token['access_token'],
                text=post['content'],
            )
        
        if result['success']:
            # Mark the post as used
            mark_social_post_used(post_id, True)
            
            # Record in scheduled_posts for history tracking
            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=post_id,
                article_id=post['article_id'] if 'article_id' in post.keys() else None,
                post_type='social',
                platform='threads',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),  # Store permalink for view link
            )
            
            return jsonify({
                "success": True,
                "post_id": result.get('post_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to Threads successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400
            
    except Exception as e:
        app.logger.exception("Failed to post to Threads")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Scheduled Posts Routes
# ============================================================================


@app.route('/schedule/add', methods=['POST'])
def schedule_add():
    """Add a post to the schedule queue."""
    post_type = request.form.get('post_type')  # 'social' or 'article'
    scheduled_for = request.form.get('scheduled_for')  # ISO datetime (optional if using queue)
    social_post_id = request.form.get('social_post_id', type=int)
    article_id = request.form.get('article_id', type=int)
    use_queue = request.form.get('use_queue') == '1'  # Auto-schedule to next available slot
    platform = request.form.get('platform', 'linkedin')  # Platform: 'linkedin' or 'threads'
    
    # If using queue, get the next available time slot for this platform
    if use_queue or not scheduled_for:
        scheduled_for = get_next_available_slot(platform=platform)
        if not scheduled_for:
            return jsonify({
                "error": f"No time slots available for {platform}. Please add posting times in the Schedule settings."
            }), 400
    
    if post_type == 'social' and not social_post_id:
        return jsonify({"error": "Social post ID is required"}), 400
    
    if post_type == 'article' and not article_id:
        return jsonify({"error": "Article ID is required"}), 400
    
    # Verify the post/article exists
    if social_post_id:
        post = get_social_post(social_post_id)
        if not post:
            return jsonify({"error": "Social post not found"}), 404
    
    if article_id:
        article = get_article(article_id)
        if not article:
            return jsonify({"error": "Article not found"}), 404
    
    try:
        scheduled_id = add_scheduled_post(
            scheduled_for=scheduled_for,
            post_type=post_type,
            social_post_id=social_post_id,
            article_id=article_id,
            platform=platform,
        )
        
        # Format the display time
        scheduled_for_display = scheduled_for
        try:
            dt = datetime.fromisoformat(scheduled_for)
            scheduled_for_display = dt.strftime('%A, %B %d at %I:%M %p')
        except (ValueError, TypeError):
            pass
        
        return jsonify({
            "success": True,
            "scheduled_id": scheduled_id,
            "scheduled_for": scheduled_for,
            "scheduled_for_display": scheduled_for_display,
            "message": f"Post scheduled for {scheduled_for_display}",
        })
        
    except Exception as e:
        app.logger.exception("Failed to schedule post")
        return jsonify({"error": str(e)}), 500


@app.route('/schedule')
def schedule_list():
    """View all scheduled posts."""
    status_filter = request.args.get('status', '')
    platform_filter = request.args.get('platform', '')
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    sort_order = request.args.get('sort', 'asc')
    
    # Validate sort order
    if sort_order not in ('asc', 'desc'):
        sort_order = 'asc'
    
    # Initialize default time slots if none exist
    initialize_default_time_slots()
    
    posts = list_scheduled_posts(
        status=status_filter if status_filter else None,
        platform=platform_filter if platform_filter else None,
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
        sort_order=sort_order,
    )
    
    # Convert to list of dicts and format dates
    scheduled = []
    for p in posts:
        post_dict = dict(p)
        # Parse scheduled_for for display
        if post_dict.get('scheduled_for'):
            try:
                dt = datetime.fromisoformat(post_dict['scheduled_for'])
                post_dict['scheduled_for_display'] = dt.strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError):
                post_dict['scheduled_for_display'] = post_dict['scheduled_for']
        scheduled.append(post_dict)
    
    # Check LinkedIn connection status
    token = get_linkedin_token()
    linkedin_connected = token is not None and not is_token_expired(token['expires_at']) if token else False
    
    # Check Threads connection status
    threads_token = get_threads_token()
    threads_connected = threads_token is not None and not threads_is_token_expired(threads_token['expires_at']) if threads_token else False
    
    # Get configured time slots
    time_slots = list_time_slots()
    time_slots_list = []
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for slot in time_slots:
        slot_dict = dict(slot)
        if slot_dict['day_of_week'] == -1:
            slot_dict['day_display'] = 'Every day'
        else:
            slot_dict['day_display'] = day_names[slot_dict['day_of_week']]
        time_slots_list.append(slot_dict)
    
    # Get next available slot for each platform
    next_slots = {}
    for plat in ['linkedin', 'threads']:
        slot = get_next_available_slot(platform=plat)
        if slot:
            try:
                dt = datetime.fromisoformat(slot)
                next_slots[plat] = dt.strftime('%A, %B %d at %I:%M %p')
            except (ValueError, TypeError):
                next_slots[plat] = slot
        else:
            next_slots[plat] = None
    
    # For backwards compatibility
    next_slot_display = next_slots.get('linkedin')
    
    # Get daily posting limits
    daily_limits = get_all_daily_limits()
    
    # Get Threads username for constructing view URLs (kept for backward compatibility)
    threads_username = threads_token['username'] if threads_token and 'username' in threads_token.keys() else None
    
    # Count posts by platform
    linkedin_count = sum(1 for p in scheduled if p.get('platform') == 'linkedin')
    threads_count = sum(1 for p in scheduled if p.get('platform') == 'threads')
    
    return render_template(
        'schedule.html',
        scheduled_posts=scheduled,
        status_filter=status_filter,
        platform_filter=platform_filter,
        date_from=date_from,
        date_to=date_to,
        sort_order=sort_order,
        linkedin_connected=linkedin_connected,
        threads_connected=threads_connected,
        time_slots=time_slots_list,
        next_slot_display=next_slot_display,
        next_slots=next_slots,
        daily_limits=daily_limits,
        threads_username=threads_username,
        linkedin_count=linkedin_count,
        threads_count=threads_count,
    )


@app.route('/schedule/list-json')
def schedule_list_json():
    """Return scheduled posts as JSON for AJAX refresh."""
    status_filter = request.args.get('status', '')
    platform_filter = request.args.get('platform', '')
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    sort_order = request.args.get('sort', 'asc')
    
    # Validate sort order
    if sort_order not in ('asc', 'desc'):
        sort_order = 'asc'
    
    posts = list_scheduled_posts(
        status=status_filter if status_filter else None,
        platform=platform_filter if platform_filter else None,
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
        sort_order=sort_order,
    )
    
    # Convert to list of dicts and format dates
    scheduled = []
    for p in posts:
        post_dict = dict(p)
        # Parse scheduled_for for display
        if post_dict.get('scheduled_for'):
            try:
                dt = datetime.fromisoformat(post_dict['scheduled_for'])
                post_dict['scheduled_for_display'] = dt.strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError):
                post_dict['scheduled_for_display'] = post_dict['scheduled_for']
        
        # Add content preview (truncated)
        content = post_dict.get('social_content') or post_dict.get('standalone_content') or ''
        post_dict['content_preview'] = content[:100] + ('...' if len(content) > 100 else '')
        
        # Determine if it's draggable (only pending posts)
        post_dict['is_draggable'] = post_dict.get('status') == 'pending'
        
        scheduled.append(post_dict)
    
    # Count posts by platform
    linkedin_count = sum(1 for p in scheduled if p.get('platform') == 'linkedin')
    threads_count = sum(1 for p in scheduled if p.get('platform') == 'threads')
    
    return jsonify({
        "success": True,
        "posts": scheduled,
        "linkedin_count": linkedin_count,
        "threads_count": threads_count,
        "total_count": len(scheduled),
    })


@app.route('/schedule/<int:scheduled_id>/cancel', methods=['POST'])
def schedule_cancel(scheduled_id: int):
    """Cancel a scheduled post."""
    success = cancel_scheduled_post(scheduled_id)
    
    if success:
        return jsonify({"success": True, "message": "Post cancelled"})
    else:
        return jsonify({"error": "Could not cancel post (may already be posted or cancelled)"}), 400


@app.route('/schedule/remove-from-queue', methods=['POST'])
def schedule_remove_from_queue():
    """Remove a post from queue by its source post ID (social or standalone) and platform."""
    from database import cancel_scheduled_post_by_source
    
    post_type = request.form.get('post_type', '')  # 'social' or 'standalone'
    post_id = request.form.get('post_id', type=int)
    platform = request.form.get('platform', 'linkedin')
    
    if not post_type or not post_id:
        return jsonify({"error": "post_type and post_id are required"}), 400
    
    if post_type not in ['social', 'standalone']:
        return jsonify({"error": "Invalid post_type"}), 400
    
    success = cancel_scheduled_post_by_source(post_type, post_id, platform)
    
    if success:
        return jsonify({"success": True, "message": f"Removed from {platform} queue"})
    else:
        return jsonify({"error": "Post not found in queue"}), 400


@app.route('/schedule/<int:scheduled_id>/delete', methods=['POST'])
def schedule_delete(scheduled_id: int):
    """Delete a scheduled post."""
    delete_scheduled_post(scheduled_id)
    return jsonify({"success": True, "message": "Post deleted"})


@app.route('/schedule/<int:scheduled_id>/post-now', methods=['POST'])
def schedule_post_now(scheduled_id: int):
    """Immediately post a pending scheduled post."""
    post = get_scheduled_post(scheduled_id)
    
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    if post['status'] != 'pending':
        return jsonify({"error": "Only pending posts can be posted immediately"}), 400
    
    platform = post['platform'] if 'platform' in post.keys() else 'linkedin'
    
    # Get content and image URL
    image_url = None
    if post['post_type'] == 'social' and post['social_content']:
        content = post['social_content']
        image_url = post['social_image_url'] if 'social_image_url' in post.keys() else None
    elif post['post_type'] == 'standalone' and post['standalone_content']:
        content = post['standalone_content']
        image_url = post['standalone_image_url'] if 'standalone_image_url' in post.keys() else None
    elif post['post_type'] == 'article' and post['article_content']:
        content = f"{post['article_topic']}\n\n{post['article_content'][:2800]}"
    else:
        return jsonify({"error": "No content found"}), 400
    
    # Get article topic for title
    article_topic = post['article_topic'] if 'article_topic' in post.keys() else None
    
    try:
        if platform == 'threads':
            # Handle Threads posting
            threads_token = get_threads_token()
            if not threads_token:
                return jsonify({"error": "Threads not connected"}), 400
            
            threads_client = get_threads_client()
            
            # Use image post if image URL is available
            if image_url:
                app.logger.info("Posting to Threads with image: %s", image_url)
                result = threads_client.publish_image_post(
                    threads_token['access_token'],
                    content[:500],
                    image_url,
                )
            else:
                result = threads_client.publish_text_post(
                    threads_token['access_token'],
                    content[:500],
                )
            
            if result and result.get('success'):
                update_scheduled_post_status(
                    scheduled_id,
                    status='posted',
                    linkedin_post_urn=result.get('permalink'),
                )
                if post['social_post_id']:
                    mark_social_post_used(post['social_post_id'], True)
                if post['standalone_post_id']:
                    mark_standalone_post_used(post['standalone_post_id'], True)
                
                # If posted before scheduled time, redistribute remaining posts to fill the gap
                scheduled_time = datetime.fromisoformat(post['scheduled_for'])
                if datetime.now() < scheduled_time:
                    redistribute_scheduled_posts(platform)
                
                return jsonify({"success": True, "message": "Posted to Threads!"})
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                return jsonify({"error": f"Failed: {error_msg}"}), 400
        else:
            # Handle LinkedIn posting
            token = get_linkedin_token()
            if not token:
                return jsonify({"error": "LinkedIn not connected"}), 400
            
            client = get_linkedin_client()
            
            # Use image post if image URL is available and no URL in content
            if image_url and not client.extract_first_url(content):
                app.logger.info("Posting to LinkedIn with image: %s", image_url)
                result = client.create_image_post(
                    token['access_token'],
                    token['user_urn'],
                    content[:3000],
                    image_url,
                )
            else:
                result = client.create_smart_post(
                    token['access_token'],
                    token['user_urn'],
                    content[:3000],
                    article_title=article_topic,
                )
            
            if result and result.get('success'):
                update_scheduled_post_status(
                    scheduled_id,
                    status='posted',
                    linkedin_post_urn=result.get('post_urn'),
                )
                if post['social_post_id']:
                    mark_social_post_used(post['social_post_id'], True)
                if post['standalone_post_id']:
                    mark_standalone_post_used(post['standalone_post_id'], True)
                
                # If posted before scheduled time, redistribute remaining posts to fill the gap
                scheduled_time = datetime.fromisoformat(post['scheduled_for'])
                if datetime.now() < scheduled_time:
                    redistribute_scheduled_posts(platform)
                
                return jsonify({"success": True, "message": "Posted to LinkedIn!"})
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                return jsonify({"error": f"Failed: {error_msg}"}), 400
            
    except Exception as e:
        app.logger.exception("Error posting now for post %d", scheduled_id)
        return jsonify({"error": str(e)}), 500


@app.route('/schedule/<int:scheduled_id>/retry', methods=['POST'])
def schedule_retry(scheduled_id: int):
    """Retry a failed scheduled post."""
    post = get_scheduled_post(scheduled_id)
    
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    if post['status'] != 'failed':
        return jsonify({"error": "Only failed posts can be retried"}), 400
    
    platform = post['platform'] if 'platform' in post.keys() else 'linkedin'
    
    # Get content
    if post['post_type'] == 'social' and post['social_content']:
        content = post['social_content']
    elif post['post_type'] == 'article' and post['article_content']:
        content = f"{post['article_topic']}\n\n{post['article_content'][:2800]}"
    else:
        return jsonify({"error": "No content found"}), 400
    
    # Get article topic for title
    article_topic = post['article_topic'] if 'article_topic' in post.keys() else None
    
    try:
        if platform == 'threads':
            # Handle Threads posting
            threads_token = get_threads_token()
            if not threads_token:
                return jsonify({"error": "Threads not connected"}), 400
            
            from threads_client import ThreadsClient
            threads_client = ThreadsClient()
            result = threads_client.create_text_post(
                threads_token['access_token'],
                threads_token['user_id'],
                content,
            )
        else:
            # Handle LinkedIn posting
            token = get_linkedin_token()
            if not token:
                return jsonify({"error": "LinkedIn not connected"}), 400
            
            client = get_linkedin_client()
            result = client.create_smart_post(
                token['access_token'],
                token['user_urn'],
                content,
                article_title=article_topic,
            )
        
        if result and result.get('success'):
            update_scheduled_post_status(
                scheduled_id,
                status='posted',
                linkedin_post_urn=result.get('post_urn'),
            )
            return jsonify({"success": True, "message": "Post successful!"})
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No response'
            update_scheduled_post_status(
                scheduled_id,
                status='failed',
                error_message=f"Retry failed: {error_msg}",
            )
            return jsonify({"error": f"Failed: {error_msg}"}), 400
            
    except Exception as e:
        app.logger.exception("Error retrying post %d", scheduled_id)
        update_scheduled_post_status(
            scheduled_id,
            status='failed',
            error_message=f"Retry exception: {str(e)}",
        )
        return jsonify({"error": str(e)}), 500


@app.route('/schedule/clear-queue', methods=['POST'])
def schedule_clear_queue():
    """Clear all pending scheduled posts."""
    count = clear_pending_scheduled_posts()
    return jsonify({
        "success": True, 
        "message": f"Cleared {count} pending post{'s' if count != 1 else ''} from queue"
    })


@app.route('/schedule/delete-selected', methods=['POST'])
def schedule_delete_selected():
    """Delete multiple selected scheduled posts."""
    data = request.get_json()
    post_ids = data.get('post_ids', []) if data else []
    
    if not post_ids:
        return jsonify({"error": "No posts selected"}), 400
    
    # Ensure all IDs are integers
    try:
        post_ids = [int(pid) for pid in post_ids]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid post IDs"}), 400
    
    count = delete_scheduled_posts_bulk(post_ids)
    return jsonify({
        "success": True,
        "message": f"Deleted {count} post{'s' if count != 1 else ''}"
    })


@app.route('/schedule/reorder', methods=['POST'])
def schedule_reorder():
    """Reorder pending scheduled posts by swapping their scheduled times."""
    from database import reorder_scheduled_posts
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    post_ids = data.get('post_ids', [])
    
    if not post_ids or len(post_ids) < 2:
        return jsonify({"error": "At least 2 post IDs required"}), 400
    
    try:
        post_ids = [int(pid) for pid in post_ids]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid post IDs"}), 400
    
    success = reorder_scheduled_posts(post_ids)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"Reordered {len(post_ids)} posts"
        })
    else:
        return jsonify({"error": "Failed to reorder posts"}), 500


@app.route('/schedule/move-position', methods=['POST'])
def schedule_move_position():
    """Move selected pending posts to the top or bottom of the queue."""
    from database import move_posts_to_position
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    post_ids = data.get('post_ids', [])
    position = data.get('position', '')
    
    if not post_ids:
        return jsonify({"error": "No post IDs provided"}), 400
    
    if position not in ('top', 'bottom'):
        return jsonify({"error": "Position must be 'top' or 'bottom'"}), 400
    
    try:
        post_ids = [int(pid) for pid in post_ids]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid post IDs"}), 400
    
    success = move_posts_to_position(post_ids, position)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"Moved {len(post_ids)} post(s) to {position}"
        })
    else:
        return jsonify({"error": "Failed to move posts"}), 500


@app.route('/schedule/<int:scheduled_id>/edit', methods=['POST'])
def schedule_edit(scheduled_id: int):
    """Edit the scheduled time for a pending post."""
    scheduled_for = request.form.get('scheduled_for', '').strip()
    
    if not scheduled_for:
        return jsonify({"error": "Scheduled time is required"}), 400
    
    # Validate datetime format
    try:
        dt = datetime.fromisoformat(scheduled_for.replace('Z', '+00:00'))
        # Ensure it's in the future (use local time for comparison since form uses local time)
        # Allow a small buffer (1 minute) to avoid edge cases
        if dt <= datetime.now() - timedelta(minutes=1):
            return jsonify({"error": "Scheduled time must be in the future"}), 400
        # Normalize to ISO format
        scheduled_for = dt.isoformat(timespec="seconds")
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid datetime format"}), 400
    
    success = update_scheduled_post_time(scheduled_id, scheduled_for)
    
    if success:
        # Format the display time
        try:
            display = dt.strftime('%A, %B %d at %I:%M %p')
        except:
            display = scheduled_for
        
        return jsonify({
            "success": True,
            "scheduled_for": scheduled_for,
            "scheduled_for_display": display,
            "message": f"Post rescheduled for {display}",
        })
    else:
        return jsonify({
            "error": "Could not update post (may not be pending or not found)"
        }), 400


# ============================================================================
# Time Slot Management Routes
# ============================================================================


@app.route('/schedule/slots', methods=['GET'])
def schedule_slots():
    """Get all configured time slots."""
    slots = list_time_slots()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    slots_list = []
    for slot in slots:
        # list_time_slots() already returns dicts with 'platforms' key
        slot_dict = dict(slot)
        if slot_dict['day_of_week'] == -1:
            slot_dict['day_display'] = 'Every day'
        else:
            slot_dict['day_display'] = day_names[slot_dict['day_of_week']]
        slots_list.append(slot_dict)
    
    return jsonify({"slots": slots_list})


@app.route('/schedule/slots/add', methods=['POST'])
def schedule_slot_add():
    """Add a new time slot."""
    day_of_week = request.form.get('day_of_week', type=int, default=-1)
    time_slot = request.form.get('time_slot', '').strip()
    platforms = request.form.getlist('platforms')
    
    if not time_slot:
        return jsonify({"error": "Time is required"}), 400
    
    # Validate time format (HH:MM)
    try:
        hour, minute = map(int, time_slot.split(':'))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError()
        time_slot = f"{hour:02d}:{minute:02d}"
    except (ValueError, AttributeError):
        return jsonify({"error": "Invalid time format. Use HH:MM (24-hour)"}), 400
    
    # Validate day_of_week
    if day_of_week < -1 or day_of_week > 6:
        return jsonify({"error": "Invalid day of week"}), 400
    
    # Validate platforms (empty list = all platforms)
    valid_platforms = {'linkedin', 'threads'}
    platforms = [p for p in platforms if p in valid_platforms]
    
    slot_id = add_time_slot(
        day_of_week=day_of_week,
        time_slot=time_slot,
        enabled=True,
        platforms=platforms if platforms else None,
    )
    
    # Redistribute all pending posts to use the new optimal slots
    linkedin_redistributed = redistribute_scheduled_posts('linkedin')
    threads_redistributed = redistribute_scheduled_posts('threads')
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_display = 'Every day' if day_of_week == -1 else day_names[day_of_week]
    
    return jsonify({
        "success": True,
        "slot": {
            "id": slot_id,
            "day_of_week": day_of_week,
            "day_display": day_display,
            "time_slot": time_slot,
            "enabled": True,
            "platforms": platforms,
        },
        "redistributed": {
            "linkedin": linkedin_redistributed,
            "threads": threads_redistributed,
        }
    })


@app.route('/schedule/slots/<int:slot_id>/toggle', methods=['POST'])
def schedule_slot_toggle(slot_id: int):
    """Toggle a time slot's enabled state."""
    slots = list_time_slots()
    slot = next((s for s in slots if s['id'] == slot_id), None)
    
    if not slot:
        return jsonify({"error": "Slot not found"}), 404
    
    new_enabled = not bool(slot['enabled'])
    update_time_slot(slot_id, enabled=new_enabled)
    
    # Redistribute all pending posts to use the new optimal slots
    redistribute_scheduled_posts('linkedin')
    redistribute_scheduled_posts('threads')
    
    return jsonify({
        "success": True,
        "enabled": new_enabled,
    })


@app.route('/schedule/slots/<int:slot_id>/edit', methods=['POST'])
def schedule_slot_edit(slot_id: int):
    """Edit a time slot's day, time, and platform assignments."""
    day_of_week = request.form.get('day_of_week', type=int)
    time_slot = request.form.get('time_slot', '').strip()
    platforms = request.form.getlist('platforms')
    
    if day_of_week is None:
        return jsonify({"error": "Day of week is required"}), 400
    
    if not time_slot:
        return jsonify({"error": "Time is required"}), 400
    
    # Validate time format (HH:MM)
    try:
        hour, minute = map(int, time_slot.split(':'))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError()
        time_slot = f"{hour:02d}:{minute:02d}"
    except (ValueError, AttributeError):
        return jsonify({"error": "Invalid time format. Use HH:MM (24-hour)"}), 400
    
    # Validate day_of_week
    if day_of_week < -1 or day_of_week > 6:
        return jsonify({"error": "Invalid day of week"}), 400
    
    # Validate platforms (empty list = all platforms)
    valid_platforms = {'linkedin', 'threads'}
    platforms = [p for p in platforms if p in valid_platforms]
    
    # Update the slot (pass empty list to clear platform restrictions)
    update_time_slot(
        slot_id,
        day_of_week=day_of_week,
        time_slot=time_slot,
        platforms=platforms,
    )
    
    # Redistribute all pending posts to use the new optimal slots
    linkedin_redistributed = redistribute_scheduled_posts('linkedin')
    threads_redistributed = redistribute_scheduled_posts('threads')
    
    return jsonify({
        "success": True,
        "platforms": platforms,
        "redistributed": {
            "linkedin": linkedin_redistributed,
            "threads": threads_redistributed,
        }
    })


@app.route('/schedule/slots/<int:slot_id>/delete', methods=['POST'])
def schedule_slot_delete(slot_id: int):
    """Delete a time slot."""
    delete_time_slot(slot_id)
    
    # Redistribute all pending posts to use the remaining slots
    redistribute_scheduled_posts('linkedin')
    redistribute_scheduled_posts('threads')
    return jsonify({"success": True, "message": "Slot deleted"})


@app.route('/schedule/daily-limits', methods=['GET', 'POST'])
def schedule_daily_limits():
    """Get or update daily posting limits per platform."""
    if request.method == 'GET':
        limits = get_all_daily_limits()
        return jsonify({
            "linkedin": limits.get('linkedin', 0),
            "threads": limits.get('threads', 0),
        })
    
    # POST - update limits
    platform = request.form.get('platform', '').strip().lower()
    limit = request.form.get('limit', type=int, default=0)
    
    if platform not in ('linkedin', 'threads'):
        return jsonify({"error": "Invalid platform. Must be 'linkedin' or 'threads'"}), 400
    
    if limit < 0:
        return jsonify({"error": "Limit must be 0 or greater (0 = unlimited)"}), 400
    
    set_daily_limit(platform, limit)
    
    # Redistribute posts for this platform to respect the new limit
    redistributed = redistribute_scheduled_posts(platform)
    
    return jsonify({
        "success": True,
        "platform": platform,
        "limit": limit,
        "redistributed": redistributed,
        "message": f"{'Unlimited' if limit == 0 else limit} posts per day for {platform.capitalize()}"
    })


@app.route('/schedule/next-slot', methods=['GET'])
def schedule_next_slot():
    """Get the next available posting slot for a platform."""
    platform = request.args.get('platform', 'linkedin')
    next_slot = get_next_available_slot(platform=platform)
    
    if not next_slot:
        return jsonify({
            "available": False,
            "message": f"No time slots available for {platform}",
        })
    
    try:
        dt = datetime.fromisoformat(next_slot)
        display = dt.strftime('%A, %B %d at %I:%M %p')
    except (ValueError, TypeError):
        display = next_slot
    
    return jsonify({
        "available": True,
        "scheduled_for": next_slot,
        "display": display,
    })


@app.route('/schedule/debug', methods=['GET'])
def schedule_debug():
    """Debug endpoint to check time slots and scheduling state."""
    import sqlite3
    from database import get_enabled_time_slots, list_time_slots
    
    # Get all time slots (enabled and disabled)
    all_slots = list_time_slots()
    enabled_slots = get_enabled_time_slots()
    
    # Get current server time info
    now = datetime.now()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Get next available slots
    next_linkedin = get_next_available_slot('linkedin')
    next_threads = get_next_available_slot('threads')
    
    # Get pending scheduled posts
    with sqlite3.connect('pod_insights.db') as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id, scheduled_for, platform, status 
            FROM scheduled_posts 
            WHERE status = 'pending'
            ORDER BY scheduled_for ASC
            LIMIT 10
            """
        )
        pending_posts = [dict(row) for row in cur.fetchall()]
    
    return jsonify({
        "server_time": {
            "now": now.isoformat(),
            "display": now.strftime('%A, %B %d, %Y at %I:%M:%S %p'),
            "day_of_week": now.weekday(),
            "day_name": day_names[now.weekday()],
        },
        "time_slots": {
            "all_count": len(all_slots),
            "enabled_count": len(enabled_slots),
            "all": [{"id": s['id'], "day_of_week": s['day_of_week'], "time_slot": s['time_slot'], "enabled": s['enabled']} for s in all_slots],
            "enabled": [{"id": s['id'], "day_of_week": s['day_of_week'], "time_slot": s['time_slot']} for s in enabled_slots],
        },
        "next_available": {
            "linkedin": next_linkedin,
            "threads": next_threads,
        },
        "pending_posts": pending_posts,
    })


# ============================================================================
# Command Center Routes
# ============================================================================


@app.route('/compose')
def compose_page():
    """Command Center page for generating social media posts."""
    # Get existing standalone posts grouped by platform
    posts = list_standalone_posts()
    
    # Get scheduled info for all standalone posts
    post_ids = [post['id'] for post in posts]
    scheduled_info = get_pending_schedules_for_standalone_posts(post_ids) if post_ids else {}
    posted_info = get_posted_info_for_standalone_posts(post_ids) if post_ids else {}
    
    posts_by_platform = {}
    for post in posts:
        platform = post['platform']
        if platform not in posts_by_platform:
            posts_by_platform[platform] = []
        post_dict = dict(post)
        # Add scheduled info to each post
        post_dict['scheduled'] = scheduled_info.get(post['id'], {})
        # Add posted info (with URL) to each post
        post_dict['posted'] = posted_info.get(post['id'], {})
        posts_by_platform[platform].append(post_dict)
    
    # Get next available slots for display
    next_slots = {
        'linkedin': get_next_available_slot('linkedin'),
        'threads': get_next_available_slot('threads'),
    }
    
    # Get saved URL sources for the "From Saved Source" tab
    saved_sources = list_url_sources()
    
    # Check if a specific source is requested
    selected_source_id = request.args.get('source_id', type=int)
    selected_source = None
    if selected_source_id:
        selected_source = get_url_source(selected_source_id)
    
    # Check platform connections
    linkedin_connected = bool(get_linkedin_token())
    threads_connected = bool(get_threads_token())
    
    # Get recent freeform prompts for reuse
    recent_prompts = list_recent_prompts(limit=20)
    recent_prompts_list = [
        {
            'content': p['source_content'],
            'preview': p['source_content'][:80] + ('...' if len(p['source_content']) > 80 else ''),
            'created_at': p['created_at'],
        }
        for p in recent_prompts
    ]
    
    return render_template(
        'compose.html',
        posts_by_platform=posts_by_platform,
        next_slots=next_slots,
        saved_sources=saved_sources,
        selected_source=selected_source,
        linkedin_connected=linkedin_connected,
        threads_connected=threads_connected,
        recent_prompts=recent_prompts_list,
    )


@app.route('/compose/recent-prompts')
def compose_recent_prompts():
    """Return recent freeform prompts for reuse via AJAX."""
    prompts = list_recent_prompts(limit=20)
    prompts_list = [
        {
            'content': p['source_content'],
            'preview': p['source_content'][:80] + ('...' if len(p['source_content']) > 80 else ''),
            'created_at': p['created_at'],
        }
        for p in prompts
    ]
    return jsonify({
        "success": True,
        "prompts": prompts_list,
    })


@app.route('/compose/clear-prompts', methods=['POST'])
def compose_clear_prompts():
    """Clear the recent prompts history."""
    try:
        count = clear_recent_prompts()
        return jsonify({
            "success": True,
            "message": f"Cleared {count} prompt(s) from history",
            "count": count,
        })
    except Exception as e:
        app.logger.exception("Failed to clear prompts")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/delete-prompt', methods=['POST'])
def compose_delete_prompt():
    """Delete a single prompt from history."""
    prompt_content = request.form.get('prompt', '').strip()
    
    if not prompt_content:
        return jsonify({"error": "Prompt content is required"}), 400
    
    try:
        count = delete_prompt_by_content(prompt_content)
        return jsonify({
            "success": True,
            "message": f"Deleted prompt from history",
            "count": count,
        })
    except Exception as e:
        app.logger.exception("Failed to delete prompt")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/delete-prompts-bulk', methods=['POST'])
def compose_delete_prompts_bulk():
    """Delete multiple prompts from history."""
    data = request.get_json() or {}
    prompts = data.get('prompts', [])
    
    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400
    
    try:
        count = delete_prompts_bulk(prompts)
        return jsonify({
            "success": True,
            "message": f"Deleted {len(prompts)} prompt(s) from history",
            "count": count,
        })
    except Exception as e:
        app.logger.exception("Failed to delete prompts")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/generate', methods=['POST'])
def compose_generate():
    """Generate social media posts using LLM based on source type."""
    source_type = request.form.get('source_type', 'freeform')
    content = request.form.get('content', '').strip()
    platforms = request.form.getlist('platforms')
    tone = request.form.get('tone', 'professional')
    posts_per_platform = request.form.get('posts_per_platform', 1, type=int)
    extra_context = request.form.get('extra_context', '').strip() or None
    topic = request.form.get('topic', '').strip() or None
    image_url = request.form.get('image_url', '').strip() or None
    
    if not content:
        return jsonify({"error": "Content is required"}), 400
    
    if not platforms:
        platforms = ['linkedin', 'threads', 'twitter']
    
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        # Generate posts based on source type
        source_data = None
        if source_type == 'freeform':
            generated = generate_posts_from_prompt(
                prompt=content,
                platforms=platforms,
                tone=tone,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
            )
        elif source_type == 'url':
            result = generate_posts_from_url(
                url=content,
                platforms=platforms,
                tone=tone,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
            )
            # New structure: {"posts": {...}, "source_data": {...}}
            generated = result.get("posts", result)
            source_data = result.get("source_data")
            
            # Auto-save URL content to url_sources
            if source_data:
                source_id = add_url_source(
                    url=source_data.get("url", content),
                    title=source_data.get("title", ""),
                    description=source_data.get("description", ""),
                    content=source_data.get("content", ""),
                    og_image=source_data.get("og_image"),
                )
                source_data["source_id"] = source_id
        elif source_type == 'text':
            generated = generate_posts_from_text(
                text=content,
                platforms=platforms,
                tone=tone,
                topic=topic,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
            )
        else:
            return jsonify({"error": f"Unknown source type: {source_type}"}), 400
        
        # Save generated posts to database
        saved_posts = {}
        for platform, post_data in generated.items():
            if platform == 'raw':
                # Handle raw response (JSON parsing failed)
                continue
            
            posts_list = post_data if isinstance(post_data, list) else [post_data]
            saved_posts[platform] = []
            
            for post_content in posts_list:
                post_id = add_standalone_post(
                    source_type=source_type,
                    source_content=content[:1000],  # Truncate for storage
                    platform=platform,
                    content=post_content,
                    image_url=image_url,
                )
                saved_posts[platform].append({
                    'id': post_id,
                    'content': post_content,
                    'image_url': image_url,
                })
        
        response_data = {
            "success": True,
            "generated": generated,
            "saved_posts": saved_posts,
        }
        if source_data:
            response_data["source_data"] = source_data
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.exception("Failed to generate posts")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/post/<int:post_id>', methods=['GET'])
def compose_get_post(post_id: int):
    """Get a standalone post by ID."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    return jsonify(dict(post))


@app.route('/compose/post/create', methods=['POST'])
def compose_create_post():
    """Create a standalone post manually (no AI generation)."""
    platform = request.form.get('platform', '').strip().lower()
    content = request.form.get('content', '').strip()
    image_url = request.form.get('image_url', '').strip() or None

    if not platform:
        return jsonify({"error": "Platform is required"}), 400
    if not content:
        return jsonify({"error": "Content is required"}), 400

    valid_platforms = ['linkedin', 'threads', 'twitter', 'bluesky', 'facebook', 'mastodon']
    if platform not in valid_platforms:
        return jsonify({"error": f"Invalid platform. Must be one of: {', '.join(valid_platforms)}"}), 400

    post_id = add_standalone_post(
        source_type='manual',
        source_content='Manual post',
        platform=platform,
        content=content,
        image_url=image_url,
    )

    return jsonify({
        "success": True,
        "post": {
            "id": post_id,
            "platform": platform,
            "content": content,
            "image_url": image_url,
        }
    })


@app.route('/compose/post/<int:post_id>/edit', methods=['POST'])
def compose_edit_post(post_id: int):
    """Edit a standalone post's content."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    new_content = request.form.get('content', '').strip()
    if not new_content:
        return jsonify({"error": "Content is required"}), 400
    
    update_standalone_post(post_id, new_content)
    return jsonify({"success": True, "content": new_content})


@app.route('/compose/post/<int:post_id>/image', methods=['POST'])
def compose_update_post_image(post_id: int):
    """Update a standalone post's image URL."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    image_url = request.form.get('image_url', '').strip() or None
    
    update_standalone_post_image(post_id, image_url)
    return jsonify({"success": True, "image_url": image_url})


@app.route('/compose/post/<int:post_id>/stock-image', methods=['GET'])
def compose_get_stock_images(post_id: int):
    """Search for stock images based on post content.
    ---
    tags:
      - Compose
    parameters:
      - name: post_id
        in: path
        type: integer
        required: true
      - name: count
        in: query
        type: integer
        default: 5
        description: Number of images to return
    responses:
      200:
        description: List of stock images
      404:
        description: Post not found
      503:
        description: No stock image API configured
    """
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    if not stock_images_configured():
        return jsonify({
            "error": "No stock image API configured. Add UNSPLASH_ACCESS_KEY, PEXELS_API_KEY, or PIXABAY_API_KEY to your .env file.",
            "configured": False,
        }), 503
    
    count = request.args.get('count', 5, type=int)
    count = min(max(count, 1), 20)  # Clamp between 1 and 20
    
    content = post['content'] if post['content'] else ''
    keywords = extract_keywords_from_text(content)
    images = get_images_for_post(content, count=count)
    
    return jsonify({
        "success": True,
        "keywords": keywords,
        "images": images,
        "configured_services": get_stock_image_services(),
    })


@app.route('/compose/post/<int:post_id>/stock-image', methods=['POST'])
def compose_apply_stock_image(post_id: int):
    """Apply a stock image to a post, saving it to the library first.
    ---
    tags:
      - Compose
    parameters:
      - name: post_id
        in: path
        type: integer
        required: true
      - name: body
        in: body
        schema:
          type: object
          properties:
            image_url:
              type: string
            save_to_library:
              type: boolean
              default: true
    responses:
      200:
        description: Image applied successfully
      404:
        description: Post not found
    """
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    data = request.get_json() or {}
    image_url = data.get('image_url', '').strip()
    save_to_library = data.get('save_to_library', True)
    
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400
    
    # If save_to_library is true, download and save the stock image
    saved_url = image_url
    if save_to_library:
        try:
            saved_url = save_stock_image_to_library(image_url)
        except Exception as e:
            app.logger.warning(f"Failed to save stock image to library: {e}, using original URL")
            saved_url = image_url
    
    update_standalone_post_image(post_id, saved_url)
    return jsonify({"success": True, "image_url": saved_url, "saved_to_library": saved_url != image_url})


@app.route('/compose/stock-images/search', methods=['GET'])
def compose_search_stock_images():
    """Search for stock images by custom query.
    ---
    tags:
      - Compose
    parameters:
      - name: q
        in: query
        type: string
        required: true
        description: Search query
      - name: count
        in: query
        type: integer
        default: 5
    responses:
      200:
        description: List of stock images
      503:
        description: No stock image API configured
    """
    if not stock_images_configured():
        return jsonify({
            "error": "No stock image API configured",
            "configured": False,
        }), 503
    
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    count = request.args.get('count', 5, type=int)
    count = min(max(count, 1), 20)
    
    images = search_stock_images(query, per_page=count)
    
    return jsonify({
        "success": True,
        "query": query,
        "images": images,
        "configured_services": get_stock_image_services(),
    })


@app.route('/compose/stock-images/status', methods=['GET'])
def compose_stock_images_status():
    """Check if stock image APIs are configured.
    ---
    tags:
      - Compose
    responses:
      200:
        description: Configuration status
    """
    return jsonify({
        "configured": stock_images_configured(),
        "services": get_stock_image_services(),
    })


@app.route('/compose/upload-image', methods=['POST'])
def compose_upload_image():
    """Upload an image file and return its URL."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check extension first (fast rejection)
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    # Validate image content and re-encode to strip embedded data
    try:
        cleaned_bytes, ext = validate_and_clean_image(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    # Upload to Cloudinary if configured (for public URLs that work with Threads)
    if CLOUDINARY_CONFIGURED:
        try:
            result = cloudinary.uploader.upload(
                cleaned_bytes,
                folder="podinsights",
                resource_type="image"
            )
            image_url = result['secure_url']
            filename = result['public_id'].split('/')[-1]
            file_size = result.get('bytes', len(cleaned_bytes))
            
            # Save to database for image library
            add_uploaded_image(
                filename=filename,
                url=image_url,
                storage='cloudinary',
                size=file_size
            )
            
            return jsonify({
                "success": True,
                "image_url": image_url,
                "filename": filename,
                "storage": "cloudinary"
            })
        except Exception as e:
            app.logger.error("Cloudinary upload failed: %s", str(e))
            # Fall back to local storage
    
    # Local storage fallback
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    with open(filepath, 'wb') as f:
        f.write(cleaned_bytes)
    
    image_url = f"{request.host_url}static/uploads/{unique_filename}"
    
    # Save to database for image library
    add_uploaded_image(
        filename=unique_filename,
        url=f"/static/uploads/{unique_filename}",
        storage='local',
        size=len(cleaned_bytes)
    )
    
    return jsonify({
        "success": True,
        "image_url": image_url,
        "filename": unique_filename,
        "storage": "local",
        "warning": "Image stored locally - may not work with Threads/external platforms" if not CLOUDINARY_CONFIGURED else None
    })


@app.route('/compose/list-images', methods=['GET'])
def compose_list_images():
    """List all uploaded images from database and local folder."""
    from datetime import datetime as dt
    
    images = []
    seen_urls = set()
    
    # First, get images from the database (includes Cloudinary images)
    db_images = list_uploaded_images()
    for img in db_images:
        # Convert ISO datetime string to timestamp for consistent sorting
        created_at = img['created_at']
        try:
            if created_at:
                timestamp = dt.fromisoformat(created_at).timestamp()
            else:
                timestamp = 0
        except (ValueError, TypeError):
            timestamp = 0
        
        images.append({
            'id': img['id'],
            'filename': img['filename'],
            'url': img['url'],
            'size': img['size'] or 0,
            'storage': img['storage'],
            'created_at': created_at,
            'modified': timestamp,
        })
        seen_urls.add(img['url'])
    
    # Also scan local uploads folder for any images not in database (backward compatibility)
    upload_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            if allowed_file(filename):
                local_url = f"/static/uploads/{filename}"
                if local_url not in seen_urls:
                    filepath = os.path.join(upload_dir, filename)
                    stat = os.stat(filepath)
                    images.append({
                        'filename': filename,
                        'url': local_url,
                        'size': stat.st_size,
                        'storage': 'local',
                        'modified': stat.st_mtime
                    })
    
    # Sort by most recently modified/created first (all values are now floats)
    images.sort(key=lambda x: x.get('modified', 0), reverse=True)
    
    return jsonify({'success': True, 'images': images})


@app.route('/compose/post/<int:post_id>/delete', methods=['POST'])
def compose_delete_post(post_id: int):
    """Delete a standalone post."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    delete_standalone_post(post_id)
    return jsonify({"success": True})


@app.route('/compose/posts/delete-bulk', methods=['POST'])
def compose_delete_bulk():
    """Delete multiple standalone posts."""
    data = request.get_json()
    post_ids = data.get('post_ids', [])
    
    if not post_ids:
        return jsonify({"error": "No posts selected"}), 400
    
    # Convert to integers
    post_ids = [int(pid) for pid in post_ids]
    deleted = delete_standalone_posts_bulk(post_ids)
    
    return jsonify({
        "success": True,
        "deleted_count": deleted,
    })


@app.route('/compose/posts/bulk-image', methods=['POST'])
def compose_bulk_update_images():
    """Bulk update images for multiple standalone posts.
    ---
    tags:
      - Compose
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            post_ids:
              type: array
              items:
                type: integer
            image_url:
              type: string
              description: Image URL to set, or null/empty to remove
    responses:
      200:
        description: Images updated successfully
      400:
        description: No posts selected
    """
    data = request.get_json()
    post_ids = data.get('post_ids', [])
    image_url = data.get('image_url')  # None or empty string to remove
    
    if not post_ids:
        return jsonify({"error": "No posts selected"}), 400
    
    # Convert to integers
    post_ids = [int(pid) for pid in post_ids]
    
    # Update each post's image
    updated = 0
    for post_id in post_ids:
        post = get_standalone_post(post_id)
        if post:
            update_standalone_post_image(post_id, image_url if image_url else None)
            updated += 1
    
    return jsonify({
        "success": True,
        "updated_count": updated,
        "image_url": image_url
    })


@app.route('/compose/post/<int:post_id>/toggle-used', methods=['POST'])
def compose_toggle_used(post_id: int):
    """Toggle a standalone post's used status."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    new_status = not bool(post['used'])
    mark_standalone_post_used(post_id, new_status)
    return jsonify({"success": True, "used": new_status})


@app.route('/compose/post/<int:post_id>/linkedin', methods=['POST'])
def compose_post_to_linkedin(post_id: int):
    """Post a standalone post to LinkedIn immediately."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    # Get LinkedIn token
    token = get_linkedin_token()
    if not token:
        return jsonify({"error": "LinkedIn not connected. Please connect your account first."}), 401
    
    # Check if token has user_urn
    if not token['user_urn']:
        return jsonify({
            "error": "LinkedIn account needs configuration. Please configure your Member ID.",
            "needs_configuration": True,
        }), 401
    
    # Check if token is expired and try to refresh
    if is_token_expired(token['expires_at']):
        if token['refresh_token']:
            client = get_linkedin_client()
            try:
                new_token = client.refresh_access_token(token['refresh_token'])
                update_linkedin_token(
                    access_token=new_token['access_token'],
                    expires_at=calculate_token_expiry(new_token.get('expires_in', 3600)),
                    refresh_token=new_token.get('refresh_token'),
                )
                token = get_linkedin_token()
            except Exception as e:
                app.logger.warning("Failed to refresh LinkedIn token: %s", e)
                return jsonify({"error": "LinkedIn token expired. Please reconnect."}), 401
        else:
            return jsonify({"error": "LinkedIn token expired. Please reconnect."}), 401
    
    client = get_linkedin_client()
    
    try:
        # Get image URL if available
        image_url = post['image_url'] if 'image_url' in post.keys() else None
        
        # Use image post if image URL is available and no URL in content
        if image_url and not client.extract_first_url(post['content']):
            app.logger.info("Posting to LinkedIn with image: %s", image_url)
            result = client.create_image_post(
                access_token=token['access_token'],
                author_urn=token['user_urn'],
                text=post['content'],
                image_url=image_url,
            )
        else:
            result = client.create_smart_post(
                access_token=token['access_token'],
                author_urn=token['user_urn'],
                text=post['content'],
            )
        
        if result['success']:
            # Mark the post as used
            mark_standalone_post_used(post_id, True)
            
            # Record in scheduled_posts for history tracking
            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=None,  # Not a social_post from articles
                article_id=None,
                standalone_post_id=post_id,
                post_type='standalone',
                platform='linkedin',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('post_urn'),
            )
            
            return jsonify({
                "success": True,
                "post_urn": result['post_urn'],
                "message": "Posted to LinkedIn successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400
            
    except Exception as e:
        app.logger.exception("Failed to post to LinkedIn")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/post/<int:post_id>/threads', methods=['POST'])
def compose_post_to_threads(post_id: int):
    """Post a standalone post to Threads immediately."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    # Get Threads token
    token = get_threads_token()
    if not token:
        return jsonify({"error": "Threads not connected. Please connect your account first."}), 401
    
    # Check if token is expired and try to refresh
    if threads_is_token_expired(token['expires_at']):
        client = get_threads_client()
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = threads_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_threads_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            token = get_threads_token()
        except Exception as e:
            app.logger.warning("Failed to refresh Threads token: %s", e)
            return jsonify({"error": "Threads token expired. Please reconnect."}), 401
    
    client = get_threads_client()
    
    try:
        # Get image URL if available
        image_url = post['image_url'] if 'image_url' in post.keys() else None
        
        # Use image post if image URL is available
        if image_url:
            app.logger.info("Posting to Threads with image: %s", image_url)
            result = client.publish_image_post(
                access_token=token['access_token'],
                text=post['content'],
                image_url=image_url,
            )
        else:
            result = client.publish_text_post(
                access_token=token['access_token'],
                text=post['content'],
            )
        
        if result['success']:
            # Mark the post as used
            mark_standalone_post_used(post_id, True)
            
            # Record in scheduled_posts for history tracking
            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=None,
                article_id=None,
                standalone_post_id=post_id,
                post_type='standalone',
                platform='threads',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),  # Store permalink for view link
            )
            
            return jsonify({
                "success": True,
                "post_id": result.get('post_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to Threads successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400
            
    except Exception as e:
        app.logger.exception("Failed to post to Threads")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/post/<int:post_id>/queue', methods=['POST'])
def compose_add_to_queue(post_id: int):
    """Add a standalone post to the schedule queue."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    platform = request.form.get('platform', post['platform'])
    scheduled_for = request.form.get('scheduled_for', '').strip()
    
    # Validate platform
    if platform not in ['linkedin', 'threads']:
        return jsonify({"error": f"Platform {platform} does not support scheduling yet"}), 400
    
    # Use provided scheduled_for or get next available slot
    if scheduled_for:
        # Use custom datetime provided by user
        schedule_time = scheduled_for
    else:
        # Get next available slot from queue
        next_slot = get_next_available_slot(platform)
        if not next_slot:
            return jsonify({"error": f"No available time slots for {platform}. Please add time slots first."}), 400
        schedule_time = next_slot
    
    # Create scheduled post entry with standalone_post_id
    scheduled_id = add_scheduled_post(
        social_post_id=None,
        article_id=None,
        standalone_post_id=post_id,
        post_type='standalone',
        platform=platform,
        scheduled_for=schedule_time,
        status='pending',
    )
    
    # Format the display time
    try:
        dt = datetime.fromisoformat(schedule_time)
        display = dt.strftime("%A, %b %d at %I:%M %p")
    except:
        display = schedule_time
    
    return jsonify({
        "success": True,
        "scheduled_id": scheduled_id,
        "scheduled_for": schedule_time,
        "scheduled_for_display": display,
        "message": f"Scheduled for {display}",
    })


@app.route('/compose/post/<int:post_id>/unqueue', methods=['POST'])
def compose_remove_from_queue(post_id: int):
    """Remove a standalone post from the schedule queue."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    
    # Find and delete any scheduled posts for this standalone post
    scheduled_posts = list_scheduled_posts(status='pending')
    removed = 0
    for sp in scheduled_posts:
        if sp.get('standalone_post_id') == post_id:
            delete_scheduled_post(sp['id'])
            removed += 1
    
    if removed == 0:
        return jsonify({"error": "Post not found in queue"}), 404
    
    return jsonify({
        "success": True,
        "removed_count": removed,
        "message": f"Removed from queue",
    })


@app.route('/compose/clear-all', methods=['POST'])
def compose_clear_all():
    """Clear all standalone posts."""
    posts = list_standalone_posts()
    if not posts:
        return jsonify({"success": True, "message": "No posts to clear"})
    
    post_ids = [p['id'] for p in posts]
    deleted = delete_standalone_posts_bulk(post_ids)
    
    return jsonify({
        "success": True,
        "deleted_count": deleted,
        "message": f"Cleared {deleted} posts",
    })


# ============================================================================
# URL Sources Management
# ============================================================================


@app.route('/sources')
def sources_page():
    """Display saved URL sources."""
    sources = list_url_sources()
    return render_template('sources.html', sources=sources)


@app.route('/sources', methods=['POST'])
def add_source():
    """Add a new URL source by extracting content from a URL."""
    import trafilatura
    
    # Accept URL from JSON or form data
    if request.is_json:
        url = request.json.get('url', '').strip()
    else:
        url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Check if URL already exists
    existing = get_url_source_by_url(url)
    if existing:
        return jsonify({
            "error": "This URL has already been saved",
            "existing_id": existing['id']
        }), 409
    
    try:
        # Fetch the URL content using trafilatura
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return jsonify({"error": "Failed to fetch URL content. Please check the URL is accessible."}), 400
        
        # Extract main article content
        body_content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        ) or ""
        
        # Extract metadata
        metadata = trafilatura.extract_metadata(downloaded)
        
        title = ""
        description = ""
        og_image = None
        
        if metadata:
            title = metadata.title or ""
            description = metadata.description or ""
            og_image = metadata.image
        
        # Fallback metadata extraction from HTML if needed
        if not title or not description or not og_image:
            import re as re_module
            
            if not title:
                title_match = re_module.search(r'<title>([^<]+)</title>', downloaded, re_module.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                
                og_title_match = re_module.search(
                    r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re_module.IGNORECASE
                )
                if og_title_match:
                    title = og_title_match.group(1)
            
            if not description:
                # Try og:description
                og_desc_match = re_module.search(
                    r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re_module.IGNORECASE
                )
                if og_desc_match:
                    description = og_desc_match.group(1)
                else:
                    # Try meta description
                    meta_desc_match = re_module.search(
                        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
                        downloaded, re_module.IGNORECASE
                    )
                    if meta_desc_match:
                        description = meta_desc_match.group(1)
            
            if not og_image:
                og_image_match = re_module.search(
                    r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re_module.IGNORECASE
                )
                if og_image_match:
                    og_image = og_image_match.group(1)
        
        # Use URL as title fallback
        if not title:
            title = url
        
        # Save to database
        source_id = add_url_source(
            url=url,
            title=title,
            description=description,
            content=body_content,
            og_image=og_image,
        )
        
        return jsonify({
            "success": True,
            "source": {
                "id": source_id,
                "url": url,
                "title": title,
                "description": description,
                "content": body_content,
                "og_image": og_image,
            }
        })
        
    except Exception as e:
        app.logger.exception("Failed to add URL source %s: %s", url, str(e))
        return jsonify({"error": f"Failed to extract content: {str(e)}"}), 500


@app.route('/sources/<int:source_id>')
def get_source(source_id: int):
    """Get a single URL source by ID."""
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    return jsonify(dict(source))


@app.route('/sources/<int:source_id>', methods=['DELETE'])
def delete_source(source_id: int):
    """Delete a URL source."""
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    
    deleted = delete_url_source(source_id)
    return jsonify({"success": deleted})


@app.route('/sources/<int:source_id>/reextract', methods=['POST'])
def reextract_source(source_id: int):
    """Re-extract content from a URL source using improved extraction."""
    import trafilatura
    
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    
    url = source['url']
    
    try:
        # Use trafilatura for robust article extraction
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return jsonify({"error": "Failed to fetch URL content"}), 500
        
        # Extract main article content
        body_content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        ) or ""
        
        # Extract metadata
        metadata = trafilatura.extract_metadata(downloaded)
        
        title = ""
        description = ""
        og_image = None
        
        if metadata:
            title = metadata.title or ""
            description = metadata.description or ""
            og_image = metadata.image
        
        # Fallback metadata extraction from HTML if needed
        if not title or not description:
            import re
            
            if not title:
                title_match = re.search(r'<title>([^<]+)</title>', downloaded, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                
                og_title_match = re.search(
                    r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE
                )
                if og_title_match:
                    title = og_title_match.group(1)
            
            if not description:
                og_desc_match = re.search(
                    r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE
                )
                if og_desc_match:
                    description = og_desc_match.group(1)
            
            if not og_image:
                og_image_match = re.search(
                    r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
                    downloaded, re.IGNORECASE
                )
                if og_image_match:
                    og_image = og_image_match.group(1)
        
        # Update the source in the database
        updated = update_url_source_content(
            source_id=source_id,
            title=title,
            description=description,
            content=body_content,
            og_image=og_image,
        )
        
        if not updated:
            return jsonify({"error": "Failed to update source"}), 500
        
        # Return the updated source data
        return jsonify({
            "success": True,
            "source": {
                "id": source_id,
                "url": url,
                "title": title,
                "description": description,
                "content": body_content,
                "og_image": og_image,
            }
        })
        
    except Exception as e:
        app.logger.exception("Failed to re-extract source %d: %s", source_id, str(e))
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500


@app.route('/compose/generate-from-source', methods=['POST'])
def compose_generate_from_source():
    """Generate posts from a saved URL source."""
    source_id = request.form.get('source_id', type=int)
    platforms = request.form.getlist('platforms')
    tone = request.form.get('tone', 'professional')
    posts_per_platform = request.form.get('posts_per_platform', 1, type=int)
    extra_context = request.form.get('extra_context', '').strip() or None
    image_url = request.form.get('image_url', '').strip() or None
    
    if not source_id:
        return jsonify({"error": "Source ID is required"}), 400
    
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    
    if not platforms:
        platforms = ['linkedin', 'threads', 'twitter']
    
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        # Import here to avoid circular dependency
        from podinsights import generate_posts_from_text
        
        # Build context from the saved source
        source_text = f"TITLE: {source['title']}\n\n"
        if source['description']:
            source_text += f"DESCRIPTION: {source['description']}\n\n"
        source_text += f"CONTENT: {source['content']}\n\n"
        source_text += f"ORIGINAL URL: {source['url']}"
        
        # Generate posts using the saved content
        generated = generate_posts_from_text(
            text=source_text,
            platforms=platforms,
            tone=tone,
            topic=source['title'],
            posts_per_platform=posts_per_platform,
            extra_context=extra_context,
        )
        
        # Update last_used_at timestamp
        update_url_source_last_used(source_id)
        
        # Save generated posts to database
        saved_posts = {}
        for platform, post_data in generated.items():
            if platform == 'raw':
                continue
            
            posts_list = post_data if isinstance(post_data, list) else [post_data]
            saved_posts[platform] = []
            
            for post_content in posts_list:
                post_id = add_standalone_post(
                    source_type='saved_source',
                    source_content=source['url'][:1000],
                    platform=platform,
                    content=post_content,
                    image_url=image_url,
                )
                saved_posts[platform].append({
                    'id': post_id,
                    'content': post_content,
                    'image_url': image_url,
                })
        
        return jsonify({
            "success": True,
            "generated": generated,
            "saved_posts": saved_posts,
            "source_title": source['title'],
        })
        
    except Exception as e:
        app.logger.exception("Failed to generate posts from source")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Background Scheduler Worker
# ============================================================================


def scheduled_post_worker() -> None:
    """Background thread that processes scheduled posts."""
    import time as time_module
    from datetime import datetime as dt_class
    
    MISSED_WINDOW_MINUTES = 10  # Posts overdue by more than this are rescheduled
    MAX_RETRIES = 5             # Max retry attempts before permanent failure
    
    while True:
        try:
            # Check for pending posts every 60 seconds
            time_module.sleep(60)
            
            # Get pending posts that are due
            pending = get_pending_scheduled_posts()
            
            if not pending:
                continue
            
            # Cache tokens to avoid repeated DB queries
            linkedin_token = None
            threads_token = None
            redistributed_platforms = set()  # Track platforms we've redistributed
            
            for post in pending:
                try:
                    platform = post['platform'] if 'platform' in post.keys() else 'linkedin'
                    
                    # Skip if we already redistributed this platform's posts in this cycle
                    if platform in redistributed_platforms:
                        continue
                    
                    # --- Missed window detection ---
                    # If the post is overdue by more than MISSED_WINDOW_MINUTES,
                    # the app likely wasn't running. Reschedule instead of posting.
                    try:
                        scheduled_dt = dt_class.fromisoformat(post['scheduled_for'])
                        now = dt_class.now()
                        overdue_minutes = (now - scheduled_dt).total_seconds() / 60
                    except (ValueError, TypeError):
                        overdue_minutes = 0
                    
                    if overdue_minutes > MISSED_WINDOW_MINUTES:
                        retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                        if retry_count >= MAX_RETRIES:
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message=f'Missed window - max retries ({MAX_RETRIES}) exhausted',
                            )
                            app.logger.warning(
                                "Post %d missed window and exhausted retries (%d/%d), marked failed",
                                post['id'], retry_count, MAX_RETRIES,
                            )
                            continue
                        increment_retry_count(post['id'])
                        redistributed = redistribute_scheduled_posts(platform)
                        redistributed_platforms.add(platform)
                        app.logger.info(
                            "Post %d missed window by %.0f min, rescheduled (retry %d/%d, %d posts redistributed)",
                            post['id'], overdue_minutes, retry_count + 1, MAX_RETRIES, redistributed,
                        )
                        continue
                    
                    # Get article topic safely from sqlite3.Row
                    article_topic = post['article_topic'] if 'article_topic' in post.keys() else None
                    
                    # Determine content and image based on post type
                    image_url = None
                    if post['post_type'] == 'social' and post['social_content']:
                        content = post['social_content']
                        image_url = post['social_image_url'] if 'social_image_url' in post.keys() else None
                    elif post['post_type'] == 'article' and post['article_content']:
                        content = f"{post['article_topic']}\n\n{post['article_content'][:2800]}"
                    elif post['post_type'] == 'standalone' and post['standalone_content']:
                        content = post['standalone_content']
                        image_url = post['standalone_image_url'] if 'standalone_image_url' in post.keys() else None
                    else:
                        app.logger.warning("Scheduled post %d has no content", post['id'])
                        update_scheduled_post_status(
                            post['id'],
                            status='failed',
                            error_message='No content found',
                        )
                        continue
                    
                    result = None
                    
                    if platform == 'threads':
                        # Handle Threads posting
                        if threads_token is None:
                            threads_token = get_threads_token()
                        
                        if not threads_token:
                            app.logger.warning("Scheduled Threads post %d due but Threads not connected", post['id'])
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message='Threads not connected',
                            )
                            continue
                        
                        # Check token expiry and refresh if needed
                        if threads_is_token_expired(threads_token['expires_at']):
                            threads_client = get_threads_client()
                            try:
                                new_token = threads_client.refresh_access_token(threads_token['access_token'])
                                expires_at = threads_calculate_token_expiry(new_token.get('expires_in', 5184000))
                                update_threads_token(
                                    access_token=new_token['access_token'],
                                    expires_at=expires_at,
                                )
                                threads_token = get_threads_token()
                            except Exception as e:
                                app.logger.error("Failed to refresh Threads token: %s", e)
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message='Threads token expired',
                                )
                                continue
                        
                        threads_client = get_threads_client()
                        
                        # Use image post if image URL is available
                        if image_url:
                            app.logger.info("Posting Threads with image: %s", image_url)
                            result = threads_client.publish_image_post(
                                access_token=threads_token['access_token'],
                                text=content[:500],  # Threads has 500 char limit
                                image_url=image_url,
                            )
                        else:
                            result = threads_client.publish_text_post(
                                access_token=threads_token['access_token'],
                                text=content[:500],  # Threads has 500 char limit
                            )
                        
                        if result['success']:
                            update_scheduled_post_status(
                                post['id'],
                                status='posted',
                                linkedin_post_urn=result.get('permalink'),  # Store permalink for view link
                            )
                            if post['social_post_id']:
                                mark_social_post_used(post['social_post_id'], True)
                            if post['standalone_post_id']:
                                mark_standalone_post_used(post['standalone_post_id'], True)
                            app.logger.info("Scheduled Threads post %d published successfully", post['id'])
                        else:
                            error_msg = str(result.get('error', 'Unknown error'))[:500]
                            retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                            if retry_count < MAX_RETRIES:
                                increment_retry_count(post['id'])
                                redistribute_scheduled_posts(platform)
                                redistributed_platforms.add(platform)
                                app.logger.warning(
                                    "Threads post %d failed (%s), rescheduled to next slot (retry %d/%d)",
                                    post['id'], error_msg, retry_count + 1, MAX_RETRIES,
                                )
                                break  # Slots redistributed, restart on next cycle
                            else:
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message=f'{error_msg} (max retries exhausted)',
                                )
                                app.logger.error("Scheduled Threads post %d failed permanently: %s", post['id'], error_msg)
                    
                    else:
                        # Handle LinkedIn posting (default)
                        if linkedin_token is None:
                            linkedin_token = get_linkedin_token()
                        
                        if not linkedin_token:
                            app.logger.warning("Scheduled LinkedIn post %d due but LinkedIn not connected", post['id'])
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message='LinkedIn not connected',
                            )
                            continue
                        
                        # Check token expiry and refresh if needed
                        if is_token_expired(linkedin_token['expires_at']):
                            linkedin_client = get_linkedin_client()
                            if linkedin_token['refresh_token']:
                                try:
                                    new_token = linkedin_client.refresh_access_token(linkedin_token['refresh_token'])
                                    expires_at = calculate_token_expiry(new_token.get('expires_in', 5184000))
                                    update_linkedin_token(
                                        access_token=new_token['access_token'],
                                        expires_at=expires_at,
                                        refresh_token=new_token.get('refresh_token'),
                                    )
                                    linkedin_token = get_linkedin_token()
                                except Exception as e:
                                    app.logger.error("Failed to refresh LinkedIn token: %s", e)
                                    update_scheduled_post_status(
                                        post['id'],
                                        status='failed',
                                        error_message='LinkedIn token expired',
                                    )
                                    continue
                            else:
                                app.logger.warning("LinkedIn token expired for post %d", post['id'])
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message='LinkedIn token expired',
                                )
                                continue
                        
                        linkedin_client = get_linkedin_client()
                        
                        # Use image post if image URL is available and no URL in content
                        if image_url and not linkedin_client.extract_first_url(content):
                            app.logger.info("Posting LinkedIn with image: %s", image_url)
                            result = linkedin_client.create_image_post(
                                access_token=linkedin_token['access_token'],
                                author_urn=linkedin_token['user_urn'],
                                text=content[:3000],
                                image_url=image_url,
                            )
                        else:
                            result = linkedin_client.create_smart_post(
                                access_token=linkedin_token['access_token'],
                                author_urn=linkedin_token['user_urn'],
                                text=content[:3000],
                                article_title=article_topic,
                            )
                        
                        if result['success']:
                            update_scheduled_post_status(
                                post['id'],
                                status='posted',
                                linkedin_post_urn=result.get('post_urn'),
                            )
                            if post['social_post_id']:
                                mark_social_post_used(post['social_post_id'], True)
                            if post['standalone_post_id']:
                                mark_standalone_post_used(post['standalone_post_id'], True)
                            app.logger.info("Scheduled LinkedIn post %d published successfully", post['id'])
                        else:
                            error_msg = str(result.get('error', 'Unknown error'))[:500]
                            retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                            if retry_count < MAX_RETRIES:
                                increment_retry_count(post['id'])
                                redistribute_scheduled_posts(platform)
                                redistributed_platforms.add(platform)
                                app.logger.warning(
                                    "LinkedIn post %d failed (%s), rescheduled to next slot (retry %d/%d)",
                                    post['id'], error_msg, retry_count + 1, MAX_RETRIES,
                                )
                                break  # Slots redistributed, restart on next cycle
                            else:
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message=f'{error_msg} (max retries exhausted)',
                                )
                                app.logger.error("Scheduled LinkedIn post %d failed permanently: %s", post['id'], error_msg)
                        
                except Exception as e:
                    app.logger.exception("Error processing scheduled post %d", post['id'])
                    error_msg = str(e)[:500]
                    retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                    if retry_count < MAX_RETRIES:
                        increment_retry_count(post['id'])
                        redistribute_scheduled_posts(platform)
                        redistributed_platforms.add(platform)
                        app.logger.warning(
                            "Post %d exception (%s), rescheduled to next slot (retry %d/%d)",
                            post['id'], error_msg, retry_count + 1, MAX_RETRIES,
                        )
                        break  # Slots redistributed, restart on next cycle
                    else:
                        update_scheduled_post_status(
                            post['id'],
                            status='failed',
                            error_message=f'{error_msg} (max retries exhausted)',
                        )
                    
        except Exception as e:
            app.logger.exception("Error in scheduled post worker")


def start_workers():
    """Start all background worker threads."""
    # Episode processing worker
    episode_worker = threading.Thread(target=worker, daemon=True)
    episode_worker.start()
    app.logger.info("Episode processing worker started")
    
    # Scheduled post worker
    scheduled_worker = threading.Thread(target=scheduled_post_worker, daemon=True)
    scheduled_worker.start()
    app.logger.info("Scheduled post worker started")


if __name__ == '__main__':
    # In debug mode with reloader, Flask spawns two processes:
    # - Parent process (reloader): WERKZEUG_RUN_MAIN is NOT set
    # - Child process (actual server): WERKZEUG_RUN_MAIN='true'
    # We only want ONE set of workers to avoid duplicate posts
    #
    # WERKZEUG_RUN_MAIN will be:
    # - 'true' in the child process (actual server) when using reloader
    # - Not set in the parent process (reloader)
    # - Not set when running without reloader (production)
    
    use_reloader = True  # Set to False for production
    
    if use_reloader:
        # Only start workers in the child process (not the reloader parent)
        if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
            start_workers()
    else:
        # No reloader, just start the workers
        start_workers()
    
    # Run the Flask app
    app.run(debug=use_reloader, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), use_reloader=use_reloader)

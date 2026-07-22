"""Flask web interface for the Insights tool."""

from __future__ import annotations
import os

# Load variables from .env into os.environ before any os.environ.get(...) calls
# below (e.g. FLASK_SECRET_KEY, OPENAI_API_KEY, JIRA_*). override=False so a
# value already exported in the shell still wins.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import io
import csv
import json
import sqlite3
import tempfile
import threading
import uuid
from queue import Queue
from flask import (
    Flask,
    Response,
    request,
    render_template,
    redirect,
    url_for,
    session,
    jsonify,
    send_from_directory,
    abort,
    g,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
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
import usage_meter
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
    get_youtube_episodes_missing_channel,
    set_episode_channel,
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
    update_threads_user_info,
    # Facebook token functions
    save_facebook_token,
    get_facebook_token,
    delete_facebook_token,
    update_facebook_token,
    update_facebook_page_selection,
    update_facebook_group_ids,
    # Twitter token functions
    save_twitter_token,
    get_twitter_token,
    delete_twitter_token,
    update_twitter_token,
    # Instagram token functions
    save_instagram_token,
    get_instagram_token,
    delete_instagram_token,
    update_instagram_token,
    update_instagram_user_info,
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
    get_existing_standalone_content,
    list_standalone_posts,
    list_standalone_posts_by_source_url,
    count_standalone_posts_by_source_urls,
    get_standalone_post,
    update_standalone_post,
    update_standalone_post_image,
    update_social_post_image,
    set_standalone_post_media,
    set_standalone_post_user_tags,
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
    list_recent_image_prompts,
    clear_recent_prompts,
    delete_prompt_by_content,
    delete_prompts_bulk,
    # Prompt library (curated, named reusable prompts)
    add_library_prompt,
    list_library_prompts,
    get_library_prompt,
    update_library_prompt,
    delete_library_prompt,
    # Generated thumbnails
    add_generated_thumbnail,
    list_generated_thumbnails,
    get_generated_thumbnail,
    delete_generated_thumbnail,
    # Users (authentication)
    create_user,
    get_user_by_username,
    get_user_by_id,
    count_users,
    list_users,
    update_last_login,
    # Activity log
    log_activity,
    list_activity,
    count_activity,
    iter_activity_for_export,
    distinct_activity_actions,
    # AI usage metering
    usage_totals,
    usage_by_mode,
    usage_by_category,
    usage_by_model,
    usage_daily,
    list_usage,
    count_usage,
    iter_usage_for_export,
    distinct_usage_categories,
    # Content agent (briefs + runs)
    create_content_brief,
    update_content_brief,
    get_content_brief,
    list_content_briefs,
    delete_content_brief,
    set_content_brief_enabled,
    set_content_brief_schedule,
    get_due_content_briefs,
    create_brief_run,
    finalize_brief_run,
    get_brief_run,
    list_brief_runs,
    get_active_brief_run,
)
from insights import (
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
    generate_posts_from_images,
    condense_document_text,
    check_ollama_status,
    provider_model_options,
    # YouTube functions
    is_youtube_url,
    classify_youtube_url,
    youtube_url_to_rss,
    download_youtube_audio,
    get_youtube_video_id,
    DOWNLOADS_DIR,
    # Thumbnail generation
    fetch_youtube_metadata,
    generate_youtube_thumbnail,
    suggested_thumbnail_prompt,
)
# Content agent orchestrator. Safe to import at top: content_agent (and the
# research_engine/web_search it pulls in) only late-import insights_web inside
# functions, so there is no import cycle at module load.
import content_agent
import document_extractor
from starter_prompts import grouped_starter_prompts
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
from facebook_client import (
    FacebookClient,
    get_facebook_client,
    calculate_token_expiry as facebook_calculate_token_expiry,
    is_token_expired as facebook_is_token_expired,
)
from twitter_client import (
    TwitterClient,
    get_twitter_client,
    calculate_token_expiry as twitter_calculate_token_expiry,
    is_token_expired as twitter_is_token_expired,
)
from instagram_client import (
    InstagramClient,
    get_instagram_client,
    calculate_token_expiry as instagram_calculate_token_expiry,
    is_token_expired as instagram_is_token_expired,
)
from stock_images import (
    search_stock_images,
    get_image_for_post,
    get_images_for_post,
    extract_keywords_from_text,
    is_configured as stock_images_configured,
    get_configured_services as get_stock_image_services,
)
from github_client import (
    is_github_repo_url,
    parse_github_repo_url,
    fetch_github_repo,
)

app = Flask(__name__)

# Session secret key. A stable value is required so login sessions survive
# restarts; if FLASK_SECRET_KEY is missing we fall back to a per-process
# ephemeral key and log a loud warning (login cookies will be invalidated on
# every restart, which defeats the purpose).
_secret_key_env = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key_env:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "FLASK_SECRET_KEY is not set. Generating an ephemeral key — all "
        "sessions will be invalidated on restart. Set FLASK_SECRET_KEY in .env "
        "to a stable value (e.g. python -c \"import secrets; print(secrets.token_hex(32))\")."
    )
app.secret_key = _secret_key_env or os.urandom(24).hex()

# Harden the session cookie. SESSION_COOKIE_SECURE should be set to "true"
# only when serving over HTTPS, otherwise the browser will refuse the cookie
# and login will silently fail on http://localhost.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

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
        "title": "Insights API",
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
# Video uploads for Instagram Reels / video Stories / video carousel items.
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov'}
MAX_IMAGE_BYTES = 16 * 1024 * 1024   # 16MB — enforced manually on the image path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Global cap is raised for video; the image route re-checks MAX_IMAGE_BYTES so the
# effective image limit is unchanged.
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB (Cloudinary free-tier per-file)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Instagram target canvases (width, height) for the media "fit" endpoint.
# Story/Reel are full-screen 9:16; feed images must sit between 4:5 and 1.91:1.
IG_FIT_TARGETS = {
    'story':          (1080, 1920),
    'reel':           (1080, 1920),
    'feed_square':    (1080, 1080),
    'feed_portrait':  (1080, 1350),
    'feed_landscape': (1080, 566),
}

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


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


# ── Authentication & activity tracking ────────────────────────────────────
#
# Multi-user login: every request must carry session['user_id'] except for a
# small allow-list of endpoints (login/signup/logout pages, static assets, and
# the health check). Mutating (non-GET) requests are recorded in
# activity_log via after_request; named auth events use log_event().

PUBLIC_ENDPOINTS = {"login", "signup", "logout", "health"}

# Auto-tracking writes a row for every non-GET request, but we suppress
# rows for these endpoints because they're either covered by an explicit
# log_event() call or aren't interesting in the audit log.
SKIP_TRACK_ENDPOINTS = {"login", "logout", "signup", "health"}


def _is_static_endpoint(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    return endpoint == "static" or endpoint.endswith(".static")


def _client_ip() -> str | None:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr


def _short_ua() -> str:
    return (request.headers.get("User-Agent") or "")[:500]


def _signups_allowed() -> bool:
    """First user is always allowed (bootstrap). After that, gated by env."""
    try:
        if count_users() == 0:
            return True
    except sqlite3.OperationalError:
        return True
    return os.environ.get("ALLOW_SIGNUP", "").lower() == "true"


def current_user():
    """Return the logged-in user row for this request, cached on flask.g."""
    if "user_id" not in session:
        return None
    cached = getattr(g, "_current_user", None)
    if cached is not None:
        return cached
    user = get_user_by_id(session["user_id"])
    if user is None:
        # Stale session pointing at a deleted user
        session.pop("user_id", None)
        return None
    g._current_user = user
    return user


@app.context_processor
def _inject_current_user():
    return {"current_user": current_user()}


def _target_from_view_args(view_args: dict | None) -> str | None:
    """Best-effort: turn {'episode_id': 42} into 'episode:42' for the log."""
    if not view_args:
        return None
    for key, value in view_args.items():
        if not isinstance(key, str):
            continue
        if key.endswith("_id") and isinstance(value, (int, str)):
            entity = key[: -len("_id")] or "id"
            return f"{entity}:{value}"
    # Fall back to the first scalar value
    for key, value in view_args.items():
        if isinstance(value, (int, str)):
            return f"{key}:{value}"
    return None


def log_event(
    action: str,
    *,
    status_code: int = 200,
    details: dict | None = None,
    username: str | None = None,
) -> None:
    """Record a named event (login_success, login_failed, logout, signup, ...)."""
    user = current_user() if "user_id" in session else None
    try:
        log_activity(
            action=action,
            user_id=user["id"] if user else None,
            username=username or (user["username"] if user else None),
            method=request.method,
            path=request.path,
            endpoint=request.endpoint,
            status_code=status_code,
            ip=_client_ip(),
            user_agent=_short_ua(),
            details=json.dumps(details) if details else None,
        )
    except Exception:
        app.logger.exception("activity log_event failed for action=%s", action)


@app.before_request
def _auth_and_timer():
    # Start a per-request timer used by the activity logger.
    g._track_t0 = time.monotonic()

    endpoint = request.endpoint
    if _is_static_endpoint(endpoint):
        return None
    if endpoint in PUBLIC_ENDPOINTS or endpoint is None:
        return None

    if "user_id" not in session:
        if request.method == "GET":
            # Preserve the original target so we can bounce them back post-login.
            return redirect(url_for("login", next=request.full_path))
        # For JSON / form POSTs, return 401 instead of a redirect so the
        # caller (browser fetch, API client) gets a clear error.
        return jsonify({"error": "authentication required"}), 401

    # Resolve the user once and cache it; this also clears a stale session
    # pointing at a deleted account.
    if current_user() is None:
        if request.method == "GET":
            return redirect(url_for("login", next=request.full_path))
        return jsonify({"error": "authentication required"}), 401

    return None


@app.after_request
def _track_request(response):
    endpoint = request.endpoint
    # Only record mutations to keep the log signal-rich. Auth endpoints use
    # log_event() explicitly so we skip them here to avoid duplicate rows.
    if (
        request.method == "GET"
        or endpoint is None
        or endpoint in SKIP_TRACK_ENDPOINTS
        or _is_static_endpoint(endpoint)
    ):
        return response

    try:
        user = current_user()
        duration_ms = None
        t0 = getattr(g, "_track_t0", None)
        if t0 is not None:
            duration_ms = int((time.monotonic() - t0) * 1000)
        log_activity(
            action=endpoint,
            user_id=user["id"] if user else None,
            username=user["username"] if user else None,
            method=request.method,
            path=request.path,
            endpoint=endpoint,
            target=_target_from_view_args(request.view_args),
            status_code=response.status_code,
            ip=_client_ip(),
            user_agent=_short_ua(),
            duration_ms=duration_ms,
        )
    except Exception:
        app.logger.exception("activity after_request log failed")
    return response


def _safe_next_url(next_value: str | None) -> str | None:
    """Only accept relative paths for the post-login redirect (no open redirects)."""
    if not next_value:
        return None
    parsed = urlparse(next_value)
    if parsed.scheme or parsed.netloc:
        return None
    if not next_value.startswith("/"):
        return None
    return next_value


@app.route("/login", methods=["GET", "POST"])
def login():
    error: str | None = None
    next_url = request.values.get("next")

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = get_user_by_username(username) if username else None
        if user is None or not check_password_hash(user["password_hash"], password):
            log_event(
                "login_failed",
                status_code=401,
                username=username or None,
                details={"reason": "invalid_credentials"},
            )
            error = "Invalid username or password."
        else:
            session.clear()
            session["user_id"] = user["id"]
            session.permanent = True
            try:
                update_last_login(user["id"])
            except Exception:
                app.logger.exception("update_last_login failed")
            log_event("login_success", username=user["username"])
            target = _safe_next_url(next_url) or url_for("index")
            return redirect(target)

    return render_template(
        "login.html",
        error=error,
        next_url=next_url or "",
        signups_allowed=_signups_allowed(),
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if not _signups_allowed():
        abort(404)

    error: str | None = None
    username = ""

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        if len(username) < 3 or len(username) > 32 or not re.match(r"^[A-Za-z0-9_.-]+$", username):
            error = "Username must be 3–32 characters (letters, numbers, . _ -)."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif get_user_by_username(username) is not None:
            error = "That username is already taken."
        else:
            try:
                user_id = create_user(
                    username,
                    generate_password_hash(password, method="pbkdf2:sha256"),
                )
            except sqlite3.IntegrityError:
                error = "That username is already taken."
            else:
                session.clear()
                session["user_id"] = user_id
                session.permanent = True
                log_event("signup", username=username)
                return redirect(url_for("index"))

    return render_template("signup.html", error=error, username=username)


@app.route("/logout", methods=["POST"])
def logout():
    if "user_id" in session:
        log_event("logout")
    session.clear()
    return redirect(url_for("login"))


# ── Activity log viewer ───────────────────────────────────────────────────


def _parse_activity_filters():
    """Pull and normalize filter params from the querystring."""
    raw_user = request.args.get("user_id") or ""
    user_id: int | None = None
    if raw_user.isdigit():
        user_id = int(raw_user)

    action = (request.args.get("action") or "").strip() or None
    start = (request.args.get("start") or "").strip() or None
    end = (request.args.get("end") or "").strip() or None

    # Accept YYYY-MM-DD inputs (datetime-local without time also works);
    # broaden bare dates to full-day ranges.
    if start and len(start) == 10:
        start = f"{start} 00:00:00"
    if end and len(end) == 10:
        end = f"{end} 23:59:59"

    return {
        "user_id": user_id,
        "action": action,
        "start_ts": start,
        "end_ts": end,
    }


@app.route("/activity")
def activity_page():
    filters = _parse_activity_filters()

    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    per_page = 50
    offset = (page - 1) * per_page

    rows = list_activity(**filters, limit=per_page, offset=offset)
    total = count_activity(**filters)
    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template(
        "activity.html",
        rows=rows,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        users=list_users(),
        actions=distinct_activity_actions(),
        filter_user_id=filters["user_id"] or "",
        filter_action=request.args.get("action", ""),
        filter_start=request.args.get("start", ""),
        filter_end=request.args.get("end", ""),
    )


@app.route("/activity.csv")
def activity_csv():
    filters = _parse_activity_filters()

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "ts",
                "user_id",
                "username",
                "action",
                "method",
                "path",
                "endpoint",
                "target",
                "status_code",
                "ip",
                "user_agent",
                "duration_ms",
                "details",
            ]
        )
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for row in iter_activity_for_export(**filters):
            writer.writerow(
                [
                    row["id"],
                    row["ts"],
                    row["user_id"],
                    row["username"],
                    row["action"],
                    row["method"],
                    row["path"],
                    row["endpoint"],
                    row["target"],
                    row["status_code"],
                    row["ip"],
                    row["user_agent"],
                    row["duration_ms"],
                    row["details"],
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    headers = {
        "Content-Disposition": f'attachment; filename="activity-{stamp}.csv"'
    }
    return Response(generate(), mimetype="text/csv", headers=headers)


# ── AI usage / cost meters ─────────────────────────────────────────────────


def _parse_usage_filters():
    """Pull and normalize usage-page filter params from the querystring."""
    mode = (request.args.get("mode") or "").strip() or None
    if mode not in ("proactive", "reactive"):
        mode = None

    category = (request.args.get("category") or "").strip() or None
    start = (request.args.get("start") or "").strip() or None
    end = (request.args.get("end") or "").strip() or None

    # Accept bare YYYY-MM-DD inputs; broaden them to full-day ranges.
    if start and len(start) == 10:
        start = f"{start} 00:00:00"
    if end and len(end) == 10:
        end = f"{end} 23:59:59"

    return {"mode": mode, "category": category, "start_ts": start, "end_ts": end}


@app.route("/usage")
def usage_page():
    filters = _parse_usage_filters()
    # Headline tiles and breakdowns reflect the date range only, so the
    # proactive-vs-reactive split stays meaningful; mode/category filter the log.
    range_filters = {"start_ts": filters["start_ts"], "end_ts": filters["end_ts"]}

    totals = usage_totals(**range_filters)
    by_mode = usage_by_mode(**range_filters)
    by_category = usage_by_category(**range_filters)
    by_model = usage_by_model(**range_filters)
    daily = usage_daily(**range_filters)

    empty = {"cost_usd": 0.0, "total_tokens": 0, "events": 0}
    proactive = by_mode.get("proactive", empty)
    reactive = by_mode.get("reactive", empty)

    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    per_page = 100
    offset = (page - 1) * per_page

    rows = list_usage(**filters, limit=per_page, offset=offset)
    total_events = count_usage(**filters)
    total_pages = max(1, (total_events + per_page - 1) // per_page)

    return render_template(
        "usage.html",
        totals=totals,
        proactive=proactive,
        reactive=reactive,
        by_category=by_category,
        by_model=by_model,
        daily=daily,
        rows=rows,
        total_events=total_events,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        categories=distinct_usage_categories(),
        filter_mode=request.args.get("mode", ""),
        filter_category=request.args.get("category", ""),
        filter_start=request.args.get("start", ""),
        filter_end=request.args.get("end", ""),
    )


@app.route("/usage.csv")
def usage_csv():
    filters = _parse_usage_filters()

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id", "ts", "mode", "category", "provider", "model",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "audio_seconds", "images", "cost_usd",
                "user_id", "username", "details",
            ]
        )
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for row in iter_usage_for_export(**filters):
            writer.writerow(
                [
                    row["id"], row["ts"], row["mode"], row["category"],
                    row["provider"], row["model"], row["prompt_tokens"],
                    row["completion_tokens"], row["total_tokens"],
                    row["audio_seconds"], row["images"], row["cost_usd"],
                    row["user_id"], row["username"], row["details"],
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    headers = {
        "Content-Disposition": f'attachment; filename="usage-{stamp}.csv"'
    }
    return Response(generate(), mimetype="text/csv", headers=headers)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── SSRF-hardened outbound fetch utilities ─────────────────────────────────
#
# Any code path that fetches a URL whose value is influenced by user input
# (post bodies, recall image lists, og:image candidates, etc.) MUST go through
# `_fetch_safely` so we get protocol allowlisting, public-IP DNS validation,
# per-redirect re-checks, response-size caps, and content-type checks.

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

_ALLOWED_URL_SCHEMES = {"http", "https"}
_ALLOWED_URL_PORTS = {None, 80, 443}
_FETCH_USER_AGENT = "InsightsBot/1.0 (+https://insights.local)"
_MAX_FETCH_REDIRECTS = 5
_URL_REGEX = re.compile(r'https?://[^\s<>"\'\)\]\}]+', re.IGNORECASE)
_TRAILING_PUNCT = '.,;:!?>'

# Background pool for non-blocking og:image fetches triggered from save paths
_link_image_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="link-image")


class UnsafeURLError(ValueError):
    """Raised when a URL fails SSRF validation."""


def _ip_is_public(ip_obj) -> bool:
    """Return True only for globally routable, non-private IP addresses."""
    if (
        ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_private
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    ):
        return False
    # Block IPv4-mapped/IPv4-compatible IPv6 that wrap a private address
    if isinstance(ip_obj, ipaddress.IPv6Address):
        mapped = ip_obj.ipv4_mapped or getattr(ip_obj, "sixtofour", None)
        if mapped is not None and not _ip_is_public(mapped):
            return False
    return True


def _assert_safe_url(url: str):
    """Validate scheme, port, and resolved IPs for a URL. Returns the parsed URL."""
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL is empty")
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise UnsafeURLError(f"URL scheme not allowed: {scheme!r}")
    if not parsed.hostname:
        raise UnsafeURLError("URL has no hostname")
    if "@" in (parsed.netloc or ""):
        raise UnsafeURLError("URL must not contain userinfo")
    port = parsed.port
    if port not in _ALLOWED_URL_PORTS:
        raise UnsafeURLError(f"URL port not allowed: {port}")

    host = parsed.hostname
    # Reject bare IP literals that fall in private space without a DNS lookup
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not _ip_is_public(literal_ip):
        raise UnsafeURLError(f"URL host resolves to non-public address: {host}")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"DNS resolution failed for {host!r}: {exc}") from exc
    if not infos:
        raise UnsafeURLError(f"No addresses resolved for {host!r}")
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        # IPv6 scoped addresses arrive like "fe80::1%eth0" -- strip zone id
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise UnsafeURLError(f"Could not parse resolved address {ip_str!r}") from exc
        if not _ip_is_public(ip_obj):
            raise UnsafeURLError(
                f"URL host {host!r} resolves to non-public address {ip_str}"
            )
    return parsed


def _fetch_safely(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 10.0,
    allowed_content_types: tuple = (),
):
    """Fetch a URL with SSRF, redirect, size, and content-type protections.

    Returns (body_bytes, content_type, final_url). Raises UnsafeURLError on
    SSRF/validation failures, or RuntimeError on fetch problems.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    current = url
    session = requests.Session()
    session.headers.update({"User-Agent": _FETCH_USER_AGENT, "Accept": "*/*"})
    try:
        for hop in range(_MAX_FETCH_REDIRECTS + 1):
            _assert_safe_url(current)
            response = session.get(
                current,
                stream=True,
                timeout=timeout,
                allow_redirects=False,
            )
            try:
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise RuntimeError(
                            f"Redirect status {response.status_code} without Location"
                        )
                    next_url = urljoin(current, location)
                    if hop >= _MAX_FETCH_REDIRECTS:
                        raise RuntimeError("Too many redirects")
                    current = next_url
                    continue

                if not response.ok:
                    raise RuntimeError(
                        f"Upstream returned HTTP {response.status_code} for {current}"
                    )

                content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if allowed_content_types and not any(
                    content_type.startswith(prefix) for prefix in allowed_content_types
                ):
                    raise RuntimeError(
                        f"Disallowed content type {content_type!r} for {current}"
                    )

                content_length_header = response.headers.get("Content-Length")
                if content_length_header is not None:
                    try:
                        if int(content_length_header) > max_bytes:
                            raise RuntimeError(
                                f"Response too large ({content_length_header} bytes) for {current}"
                            )
                    except ValueError:
                        pass  # malformed header -- enforce via streaming below

                buffer = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    buffer.extend(chunk)
                    if len(buffer) > max_bytes:
                        raise RuntimeError(
                            f"Response exceeded max_bytes ({max_bytes}) for {current}"
                        )
                return bytes(buffer), content_type, current
            finally:
                response.close()
        raise RuntimeError("Too many redirects")
    finally:
        session.close()


def extract_urls_from_text(text: str | None, *, limit: int = 5) -> list:
    """Pull HTTP(S) URLs out of arbitrary text, deduped, trailing punct stripped."""
    if not text:
        return []
    seen = []
    for match in _URL_REGEX.findall(text):
        cleaned = match.rstrip(_TRAILING_PUNCT)
        # Balance trailing closing parens that the regex grabbed
        while cleaned.endswith(')') and cleaned.count('(') < cleaned.count(')'):
            cleaned = cleaned[:-1]
        if not cleaned:
            continue
        try:
            parsed = urlparse(cleaned)
        except ValueError:
            continue
        if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES or not parsed.hostname:
            continue
        if cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= limit:
            break
    return seen


def fetch_og_image_for_url(url: str):
    """Fetch a webpage safely and return an absolute, SSRF-validated og:image URL."""
    try:
        body, _ctype, final_url = _fetch_safely(
            url,
            max_bytes=2_000_000,
            timeout=8,
            allowed_content_types=("text/html", "application/xhtml"),
        )
    except (UnsafeURLError, RuntimeError, requests.RequestException) as exc:
        app.logger.info("og:image page fetch failed for %s: %s", url, exc)
        return None

    try:
        # Parse only enough to find meta/link tags
        soup = BeautifulSoup(body, "html.parser")
    except Exception as exc:
        app.logger.info("og:image HTML parse failed for %s: %s", url, exc)
        return None

    candidate = None
    for selector in (
        ("meta", {"property": re.compile(r"^og:image(?::url)?$", re.IGNORECASE)}),
        ("meta", {"name": re.compile(r"^og:image$", re.IGNORECASE)}),
        ("meta", {"name": re.compile(r"^twitter:image(?::src)?$", re.IGNORECASE)}),
        ("meta", {"property": re.compile(r"^twitter:image(?::src)?$", re.IGNORECASE)}),
        ("link", {"rel": re.compile(r"image_src", re.IGNORECASE)}),
    ):
        tag_name, attrs = selector
        tag = soup.find(tag_name, attrs=attrs)
        if not tag:
            continue
        value = tag.get("content") or tag.get("href")
        if value:
            candidate = value.strip()
            break

    if not candidate:
        return None

    absolute = urljoin(final_url, candidate)
    try:
        _assert_safe_url(absolute)
    except UnsafeURLError as exc:
        app.logger.info("og:image candidate rejected for %s: %s", url, exc)
        return None
    return absolute


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
    # Check if this image URL is already in the library.
    # `list_uploaded_images()` returns `sqlite3.Row` objects which use mapping
    # access (`row['col']`) but do not implement `dict.get()`.
    existing = list_uploaded_images()
    for img in existing:
        keys = img.keys() if hasattr(img, 'keys') else []
        existing_url = img['url'] if 'url' in keys else None
        existing_filename = img['filename'] if 'filename' in keys else ''
        if existing_url == image_url or (existing_filename and image_url in str(existing_filename)):
            return existing_url or image_url
    
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
    
    # For non-stock URLs, download and re-upload through the SSRF-hardened helper
    image_data, content_type, _final_url = _fetch_safely(
        image_url,
        max_bytes=16 * 1024 * 1024,
        timeout=15,
        allowed_content_types=("image/",),
    )

    ext_map = {
        'image/jpeg': 'jpg',
        'image/png': 'png',
        'image/gif': 'gif',
        'image/webp': 'webp',
    }
    ext = ext_map.get((content_type or '').split(';')[0].strip(), 'jpg')

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
            folder="insights/stock",
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


def _backfill_youtube_source(url: str, transcript: str, summary: str) -> None:
    """Update the matching URL source with the transcript after transcription."""
    source = get_url_source_by_url(url)
    if source:
        update_url_source_content(
            source['id'],
            title=source['title'],
            description=summary,
            content=transcript,
            og_image=source['og_image'],
        )


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
            if is_youtube_url(url):
                channel = None
                try:
                    meta = fetch_youtube_metadata(url)
                    channel = meta.get("channel") or None
                except Exception:
                    pass
                audio_path = download_youtube_audio(url)
                try:
                    transcript = transcribe_audio(audio_path)
                    summary = summarize_text(transcript)
                    actions = extract_action_items(transcript)
                    save_episode(url, title, transcript, summary, actions, feed_id, published, channel=channel)
                    _backfill_youtube_source(url, transcript, summary)
                finally:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
            else:
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


def _proactive_worker() -> None:
    """Run the episode ``worker`` with every generation tagged as proactive.

    The background queue is automated (non-interactive) work, so all AI usage it
    incurs — transcription, summaries, action items — is metered as ``proactive``.
    ContextVars are per-thread, so this override is scoped to the worker thread.
    """
    with usage_meter.usage_context("proactive"):
        worker()


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


def fetch_article_content_safe(url: str, timeout: int = 12, max_bytes: int = 3_000_000) -> tuple[str, str]:
    """SSRF-safe article fetch + extraction for agent/automated flows.

    Unlike ``fetch_article_content`` (which calls ``trafilatura.fetch_url``/``requests``
    directly and bypasses the SSRF guard), this fetches bytes through
    ``_fetch_safely`` (per-redirect public-IP validation, content-type + size caps)
    and extracts text from the in-memory HTML. Never performs a second, unguarded
    network fetch.

    Returns ``(text, final_url)``; ``("", url)`` on any failure.
    """
    try:
        body, ctype, final_url = _fetch_safely(
            url,
            max_bytes=max_bytes,
            timeout=timeout,
            allowed_content_types=("text/html", "application/xhtml"),
        )
    except (UnsafeURLError, RuntimeError, requests.RequestException) as exc:
        app.logger.info("Safe fetch rejected/failed for %s: %s", url, exc)
        return "", url
    except Exception:
        app.logger.exception("Unexpected error during safe fetch of %s", url)
        return "", url

    html = body.decode("utf-8", errors="replace")

    # Prefer trafilatura's precision extraction on the already-downloaded HTML.
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if text and len(text) > 200:
            return text.strip(), final_url
    except Exception:
        app.logger.debug("trafilatura.extract failed for %s", final_url, exc_info=True)

    # Fallback: compact BeautifulSoup reader (mirrors fetch_article_content).
    try:
        soup = BeautifulSoup(html, "lxml")
        for element in soup.find_all(["script", "style", "nav", "header", "footer",
                                       "aside", "iframe", "noscript", "form",
                                       "button", "input", "select", "textarea"]):
            element.decompose()
        container = soup.find("article")
        if not container:
            for selector in ['[role="main"]', ".article-content", ".post-content",
                             ".entry-content", ".content", "#content", ".story-body",
                             ".article-body", ".post-body", "main"]:
                found = soup.select_one(selector)
                if found:
                    container = found
                    break
        if not container:
            container = soup.find("body") or soup
        parts = []
        for p in container.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                                      "li", "blockquote", "pre", "code"]):
            t = p.get_text(separator=" ", strip=True)
            if t and len(t) > 20:
                parts.append(t)
        text = "\n\n".join(parts) if parts else container.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text, final_url
    except Exception:
        app.logger.exception("BeautifulSoup extraction failed for %s", final_url)
        return "", final_url


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
        
        # Determine feed type from URL and entries
        if 'youtube.com/feeds/videos.xml' in feed_url:
            feed_type = 'youtube'
        else:
            is_audio = False
            for entry in feed_data.entries[:5]:
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
        feed_url = request.form['feed_url']

        if is_youtube_url(feed_url):
            yt_kind = classify_youtube_url(feed_url)

            if yt_kind in ("channel", "playlist"):
                rss_url = youtube_url_to_rss(feed_url)
                if rss_url:
                    feed = feedparser.parse(rss_url)
                    title = feed.feed.get('title', feed_url)
                    feed_id = add_feed(rss_url, title)
                    update_feed_metadata(feed_id, 'youtube', None, len(feed.entries))
                    return redirect(url_for('view_feed', feed_id=feed_id))

            if yt_kind == "video":
                video_id = get_youtube_video_id(feed_url)
                title, description, og_image = feed_url, "", None
                try:
                    import yt_dlp
                    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                        info = ydl.extract_info(feed_url, download=False)
                        title = info.get("title", title)
                        description = info.get("description", "")
                        og_image = info.get("thumbnail")
                except Exception:
                    pass
                if not og_image and video_id:
                    og_image = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                add_url_source(
                    url=feed_url, title=title,
                    description=description, content="", og_image=og_image,
                )
                return redirect(url_for('sources_page'))

        feed = feedparser.parse(feed_url)
        title = feed.feed.get('title', feed_url)
        feed_id = add_feed(feed_url, title)
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

    is_youtube_feed = feed['feed_type'] == 'youtube'

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    per_page = min(per_page, 50)

    all_episodes = []
    is_text_feed = True

    refresh_feed_metadata(feed_id, feed['url'])
    feed_data = feedparser.parse(feed['url'])

    audio_extensions = ('.mp3', '.m4a', '.wav', '.ogg', '.aac', '.flac')

    for entry in feed_data.entries:
        has_audio = bool(entry.get('enclosures'))
        if has_audio:
            is_text_feed = False
            url = entry.enclosures[0].href
            item_type = 'audio'
        elif is_youtube_feed:
            url = entry.get('link', entry.get('id', ''))
            item_type = 'youtube'
            is_text_feed = False
            if not url:
                continue
        else:
            url = entry.get('link', entry.get('id', ''))
            item_type = 'text'
            if not url:
                continue
            url_check = url.lower().split('?')[0]
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
        content = ''
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].get('value', '')
        if not content:
            content = entry.get('summary') or entry.get('description', '')
        desc = entry.get('summary') or entry.get('description', '')
        clean_desc = strip_html(desc)
        img = None
        if is_youtube_feed:
            vid = get_youtube_video_id(url)
            if vid:
                img = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
        if not img:
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
        author = entry.get('author', '')
        ep_data = {
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
        }
        if item_type == 'youtube':
            ep_data['video_id'] = get_youtube_video_id(url)
        all_episodes.append(ep_data)

    total_items = len(all_episodes)
    total_pages = (total_items + per_page - 1) // per_page
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

    return render_template(
        'feed.html',
        feed=feed,
        episodes=episodes,
        is_text_feed=is_text_feed,
        is_youtube_feed=is_youtube_feed,
        pagination=pagination,
    )


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
                entry_url = entry.get('link', entry.get('id', ''))
                enclosure_url = entry.enclosures[0].href if entry.get('enclosures') else None
                if enclosure_url == audio_url or entry_url == audio_url:
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

    channel = None
    if is_youtube_url(audio_url):
        try:
            meta = fetch_youtube_metadata(audio_url)
            channel = meta.get("channel") or None
        except Exception:
            pass
        audio_path = download_youtube_audio(audio_url)
    else:
        tmpdir = tempfile.mkdtemp()
        audio_path = os.path.join(tmpdir, 'episode.mp3')
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(audio_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    try:
        app.logger.info("Transcribing audio")
        transcript = transcribe_audio(audio_path)
        app.logger.info("Transcription complete")

        app.logger.info("Generating summary")
        summary = summarize_text(transcript)
        app.logger.info("Summary complete")

        app.logger.info("Extracting action items")
        actions = extract_action_items(transcript)
        app.logger.info("Action item extraction complete")
        save_episode(audio_url, title, transcript, summary, actions, feed_id, published, channel=channel)
        if is_youtube_url(audio_url):
            _backfill_youtube_source(audio_url, transcript, summary)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
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

    ep_channels = {
        ep["id"]: ep["channel"]
        for ep in all_episodes
        if not ep["feed_id"] and ep["channel"]
    }

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
        ep_channels=ep_channels,
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

    audio_extensions = ('.mp3', '.m4a', '.wav', '.ogg', '.aac', '.flac')
    is_audio = (
        episode['url'].lower().split('?')[0].endswith(audio_extensions)
        or is_youtube_url(episode['url'])
    )

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


@app.route('/episodes/backfill-channels', methods=['POST'])
def backfill_youtube_channels():
    """Populate channel names for existing YouTube episodes missing them."""
    episodes = get_youtube_episodes_missing_channel()
    updated = 0
    for ep in episodes:
        try:
            meta = fetch_youtube_metadata(ep["url"])
            channel = meta.get("channel") or None
            if channel:
                set_episode_channel(ep["id"], channel)
                updated += 1
        except Exception:
            continue
    return jsonify({"success": True, "updated": updated, "total": len(episodes)})


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
        platforms = ["twitter", "linkedin", "facebook", "threads"]
    
    # Get number of posts per platform (default to 10, max 21)
    posts_per_platform = request.form.get('posts_per_platform', 10, type=int)
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

    # Auto-fetch og:image if the edit added a new URL and no image is set
    existing_image = post['image_url'] if 'image_url' in post.keys() else None
    if not existing_image:
        old_urls = set(extract_urls_from_text(post['content'] if 'content' in post.keys() else ''))
        new_urls = extract_urls_from_text(content)
        if any(u not in old_urls for u in new_urls):
            _maybe_attach_link_image(post_id, content, kind='social')

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


@app.route('/social/<int:post_id>/link-image', methods=['GET', 'POST'])
def social_apply_link_image(post_id: int):
    """Fetch the og:image of a URL referenced by a social post and attach it."""
    post = get_social_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    content = post['content'] if 'content' in post.keys() else ''
    detected = extract_urls_from_text(content)
    current_image = post['image_url'] if 'image_url' in post.keys() else None

    if request.method == 'GET':
        return jsonify({
            "success": True,
            "detected_urls": detected,
            "image_url": current_image,
        })

    data = request.get_json(silent=True) or {}
    override = (data.get('url') or '').strip()
    allow_external = bool(data.get('allow_external'))

    target = override or (detected[0] if detected else None)
    if not target:
        return jsonify({
            "error": "No URL found in post and none provided.",
            "detected_urls": detected,
        }), 400
    if override and override not in detected and not allow_external:
        return jsonify({
            "error": "Override URL is not present in the post body.",
            "detected_urls": detected,
        }), 400

    try:
        _assert_safe_url(target)
    except UnsafeURLError as exc:
        return jsonify({"error": f"URL rejected: {exc}", "detected_urls": detected}), 400

    og_image = fetch_og_image_for_url(target)
    if not og_image:
        return jsonify({
            "error": "Could not find an og:image at that URL.",
            "detected_urls": detected,
            "source_url": target,
        }), 422

    try:
        saved_url = save_stock_image_to_library(og_image)
    except (UnsafeURLError, RuntimeError, ValueError, requests.RequestException) as exc:
        app.logger.warning(
            "Failed to save og:image %s for social post %s: %s", og_image, post_id, exc
        )
        return jsonify({
            "error": f"Failed to download og:image: {exc}",
            "detected_urls": detected,
            "source_url": target,
            "og_image_url": og_image,
        }), 422

    update_social_post_image(post_id, saved_url)
    return jsonify({
        "success": True,
        "image_url": saved_url,
        "source_url": target,
        "og_image_url": og_image,
        "detected_urls": detected,
    })


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
                    "expires_at": expires_at,
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
        "expires_at": token['expires_at'],
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
                "expires_at": expires_at,
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
        "expires_at": token['expires_at'],
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
            remote_picture_url = user_info.get('threads_profile_picture_url', '')
        else:
            username = ''
            display_name = 'Threads User'
            remote_picture_url = ''
        
        # Download and store profile picture locally (CDN URLs expire)
        profile_picture_url = ''
        if remote_picture_url:
            try:
                img_resp = requests.get(remote_picture_url, timeout=15)
                if img_resp.status_code == 200:
                    pic_filename = f"threads_profile_{user_id}.jpg"
                    pic_path = os.path.join(app.static_folder, pic_filename)
                    with open(pic_path, 'wb') as f:
                        f.write(img_resp.content)
                    profile_picture_url = url_for('static', filename=pic_filename)
                    app.logger.info("Saved Threads profile picture to %s", pic_path)
                else:
                    app.logger.warning("Failed to download Threads profile picture: %s", img_resp.status_code)
            except Exception as pic_err:
                app.logger.warning("Could not save Threads profile picture: %s", pic_err)
        
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


@app.route('/threads/configure', methods=['GET', 'POST'])
def threads_configure():
    """Configure Threads user info manually or view setup instructions."""
    token = get_threads_token()

    if request.method == 'POST':
        if not token:
            return redirect(url_for('schedule_list') + '?error=threads_not_connected')

        user_id = request.form.get('user_id', '').strip()
        username = request.form.get('username', '').strip()
        display_name = request.form.get('display_name', '').strip() or 'Threads User'

        if not user_id:
            return render_template(
                'threads_configure.html',
                token=token,
                error="User ID is required",
            )

        success = update_threads_user_info(
            user_id=user_id,
            username=username or None,
            display_name=display_name,
        )

        if success:
            app.logger.info("Threads user info configured manually: %s (@%s)", user_id, username)
            return redirect(url_for('schedule_list') + '?threads=configured')
        else:
            return render_template(
                'threads_configure.html',
                token=token,
                error="Failed to save configuration. Make sure Threads is connected first.",
            )

    # GET request – show configuration / setup instructions
    return render_template(
        'threads_configure.html',
        token=token,
        is_new=request.args.get('new') == '1',
    )


@app.route('/instagram/status')
def instagram_status():
    """Check Instagram connection status."""
    client = get_instagram_client()
    token = get_instagram_token()

    if not client.is_configured():
        return jsonify({
            "connected": False,
            "configured": False,
            "message": "Instagram credentials not configured. Set INSTAGRAM_APP_ID and INSTAGRAM_APP_SECRET.",
        })

    if not token:
        return jsonify({
            "connected": False,
            "configured": True,
            "message": "Not connected to Instagram",
        })

    account_type = token['account_type'] if 'account_type' in token.keys() else None
    warning = None
    if account_type and account_type.upper() not in ('BUSINESS', 'MEDIA_CREATOR', 'CREATOR'):
        warning = (
            "Connected account is not a professional (Business/Creator) account. "
            "Instagram content publishing requires one."
        )

    # Check if token is expired
    if instagram_is_token_expired(token['expires_at']):
        # Try to refresh the token
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = instagram_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_instagram_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            return jsonify({
                "connected": True,
                "configured": True,
                "username": token['username'],
                "display_name": token['display_name'],
                "profile_picture_url": token['profile_picture_url'],
                "account_type": account_type,
                "warning": warning,
                "expires_at": expires_at,
                "message": "Connected (token refreshed)",
            })
        except Exception as e:
            app.logger.warning("Failed to refresh Instagram token: %s", e)
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
        "account_type": account_type,
        "warning": warning,
        "user_id": token['user_id'],
        "expires_at": token['expires_at'],
    })


@app.route('/instagram/auth')
def instagram_auth():
    """Start Instagram OAuth flow."""
    client = get_instagram_client()

    if not client.is_configured():
        return jsonify({"error": "Instagram not configured"}), 400

    auth_url, state = client.get_authorization_url()
    session['instagram_oauth_state'] = state

    return redirect(auth_url)


@app.route('/instagram/callback')
def instagram_callback():
    """Handle Instagram OAuth callback."""
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        app.logger.error("Instagram OAuth error: %s - %s", error, error_desc)
        return render_template(
            'article_error.html',
            error=f"Instagram authorization failed: {error_desc}",
        )

    code = request.args.get('code')
    state = request.args.get('state')

    # Verify state to prevent CSRF
    stored_state = session.pop('instagram_oauth_state', None)
    if not stored_state or stored_state != state:
        app.logger.warning("Instagram OAuth state mismatch")
        return render_template(
            'article_error.html',
            error="Security verification failed. Please try again.",
        )

    if not code:
        return render_template(
            'article_error.html',
            error="No authorization code received from Instagram.",
        )

    client = get_instagram_client()

    try:
        # Step 1: Exchange code for short-lived token
        token_data = client.exchange_code_for_token(code)
        short_lived_token = token_data['access_token']
        user_id = str(token_data.get('user_id', ''))

        # Step 2: Exchange for long-lived token (60 days)
        long_lived_data = client.get_long_lived_token(short_lived_token)
        access_token = long_lived_data['access_token']
        expires_in = long_lived_data.get('expires_in', 5184000)  # Default 60 days

        # Calculate expiry
        expires_at = instagram_calculate_token_expiry(expires_in)

        # Get user profile
        user_info = client.get_user_profile(access_token)

        if user_info:
            user_id = str(user_info.get('id', user_id))
            ig_user_id = str(user_info.get('user_id', '') or '')
            username = user_info.get('username', '')
            display_name = user_info.get('name', '') or username
            account_type = user_info.get('account_type', '')
            remote_picture_url = user_info.get('profile_picture_url', '')
        else:
            ig_user_id = ''
            username = ''
            display_name = 'Instagram User'
            account_type = ''
            remote_picture_url = ''

        # Download and store profile picture locally (CDN URLs expire)
        profile_picture_url = ''
        if remote_picture_url:
            try:
                img_resp = requests.get(remote_picture_url, timeout=15)
                if img_resp.status_code == 200:
                    pic_filename = f"instagram_profile_{user_id}.jpg"
                    pic_path = os.path.join(app.static_folder, pic_filename)
                    with open(pic_path, 'wb') as f:
                        f.write(img_resp.content)
                    profile_picture_url = url_for('static', filename=pic_filename)
                    app.logger.info("Saved Instagram profile picture to %s", pic_path)
                else:
                    app.logger.warning("Failed to download Instagram profile picture: %s", img_resp.status_code)
            except Exception as pic_err:
                app.logger.warning("Could not save Instagram profile picture: %s", pic_err)

        # Save token
        save_instagram_token(
            access_token=access_token,
            expires_at=expires_at,
            user_id=user_id,
            username=username,
            ig_user_id=ig_user_id or None,
            display_name=display_name,
            profile_picture_url=profile_picture_url,
            account_type=account_type or None,
        )

        app.logger.info("Instagram connected for user: @%s (%s)", username, account_type or 'unknown type')

        # Redirect to schedule page with success message
        return redirect(url_for('schedule_list') + '?instagram=connected')

    except Exception as e:
        app.logger.exception("Instagram OAuth exchange failed")
        return render_template(
            'article_error.html',
            error=f"Failed to connect to Instagram: {str(e)}",
        )


@app.route('/instagram/disconnect', methods=['POST'])
def instagram_disconnect():
    """Disconnect Instagram account."""
    delete_instagram_token()
    return jsonify({"success": True, "message": "Instagram disconnected"})


@app.route('/instagram/configure', methods=['GET', 'POST'])
def instagram_configure():
    """Configure Instagram user info manually or view setup instructions."""
    token = get_instagram_token()

    if request.method == 'POST':
        if not token:
            return redirect(url_for('schedule_list') + '?error=instagram_not_connected')

        user_id = request.form.get('user_id', '').strip()
        username = request.form.get('username', '').strip()
        display_name = request.form.get('display_name', '').strip() or 'Instagram User'
        ig_user_id = request.form.get('ig_user_id', '').strip()

        if not user_id:
            return render_template(
                'instagram_configure.html',
                token=token,
                error="User ID is required",
            )

        success = update_instagram_user_info(
            user_id=user_id,
            username=username or None,
            display_name=display_name,
            ig_user_id=ig_user_id or None,
        )

        if success:
            app.logger.info("Instagram user info configured manually: %s (@%s)", user_id, username)
            return redirect(url_for('schedule_list') + '?instagram=configured')
        else:
            return render_template(
                'instagram_configure.html',
                token=token,
                error="Failed to save configuration. Make sure Instagram is connected first.",
            )

    # GET request – show configuration / setup instructions
    return render_template(
        'instagram_configure.html',
        token=token,
        is_new=request.args.get('new') == '1',
    )


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
# Facebook Integration Routes
# ============================================================================


@app.route('/facebook/status')
def facebook_status():
    """Check Facebook connection status."""
    client = get_facebook_client()
    token = get_facebook_token()

    if not client.is_configured():
        return jsonify({
            "connected": False,
            "configured": False,
            "message": "Facebook credentials not configured. Set FACEBOOK_APP_ID and FACEBOOK_APP_SECRET.",
        })

    if not token:
        return jsonify({
            "connected": False,
            "configured": True,
            "message": "Not connected to Facebook",
        })

    if facebook_is_token_expired(token['expires_at']):
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = facebook_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_facebook_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            token = get_facebook_token()
        except Exception as e:
            app.logger.warning("Failed to refresh Facebook token: %s", e)
            return jsonify({
                "connected": False,
                "configured": True,
                "message": "Token expired. Please reconnect.",
            })

    return jsonify({
        "connected": True,
        "configured": True,
        "user_name": token['user_name'],
        "user_id": token['user_id'],
        "page_name": token['page_name'],
        "page_id": token['page_id'],
        "expires_at": token['expires_at'],
    })


@app.route('/facebook/auth')
def facebook_auth():
    """Start Facebook OAuth flow."""
    client = get_facebook_client()

    if not client.is_configured():
        return jsonify({"error": "Facebook not configured"}), 400

    auth_url, state = client.get_authorization_url()
    session['facebook_oauth_state'] = state

    return redirect(auth_url)


@app.route('/facebook/callback')
def facebook_callback():
    """Handle Facebook OAuth callback."""
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        app.logger.error("Facebook OAuth error: %s - %s", error, error_desc)
        return render_template(
            'article_error.html',
            error=f"Facebook authorization failed: {error_desc}",
        )

    code = request.args.get('code')
    state = request.args.get('state')

    stored_state = session.pop('facebook_oauth_state', None)
    if not stored_state or stored_state != state:
        app.logger.warning("Facebook OAuth state mismatch")
        return render_template(
            'article_error.html',
            error="Security verification failed. Please try again.",
        )

    if not code:
        return render_template(
            'article_error.html',
            error="No authorization code received from Facebook.",
        )

    client = get_facebook_client()

    try:
        token_data = client.exchange_code_for_token(code)
        short_lived_token = token_data['access_token']

        long_lived_data = client.get_long_lived_token(short_lived_token)
        access_token = long_lived_data['access_token']
        expires_in = long_lived_data.get('expires_in', 5184000)

        expires_at = facebook_calculate_token_expiry(expires_in)

        user_info = client.get_user_profile(access_token)
        user_id = user_info.get('id', '') if user_info else ''
        user_name = user_info.get('name', 'Facebook User') if user_info else 'Facebook User'

        pages = client.get_user_pages(access_token)
        page_id = pages[0]['id'] if pages else None
        page_name = pages[0]['name'] if pages else None
        page_access_token = pages[0]['access_token'] if pages else None

        groups = client.get_user_groups(access_token)
        group_ids = ','.join(g['id'] for g in groups) if groups else None

        save_facebook_token(
            access_token=access_token,
            expires_at=expires_at,
            user_id=user_id,
            user_name=user_name,
            page_id=page_id,
            page_name=page_name,
            page_access_token=page_access_token,
            group_ids=group_ids,
        )

        app.logger.info("Facebook connected for user: %s", user_name)

        if pages and len(pages) > 1:
            return redirect(url_for('facebook_configure') + '?new=1')

        return redirect(url_for('schedule_list') + '?facebook=connected')

    except Exception as e:
        app.logger.exception("Facebook OAuth exchange failed")
        return render_template(
            'article_error.html',
            error=f"Failed to connect to Facebook: {str(e)}",
        )


@app.route('/facebook/disconnect', methods=['POST'])
def facebook_disconnect():
    """Disconnect Facebook account."""
    delete_facebook_token()
    return jsonify({"success": True, "message": "Facebook disconnected"})


@app.route('/facebook/configure', methods=['GET', 'POST'])
def facebook_configure():
    """Configure which Facebook Page/Group to post to."""
    token = get_facebook_token()

    if request.method == 'POST':
        if not token:
            return redirect(url_for('schedule_list') + '?error=facebook_not_connected')

        page_id = request.form.get('page_id', '').strip()
        group_ids = request.form.get('group_ids', '').strip()

        if page_id:
            client = get_facebook_client()
            pages = client.get_user_pages(token['access_token'])
            selected = next((p for p in pages if p['id'] == page_id), None)
            if selected:
                update_facebook_page_selection(
                    page_id=selected['id'],
                    page_name=selected['name'],
                    page_access_token=selected['access_token'],
                )

        if group_ids is not None:
            update_facebook_group_ids(group_ids)

        app.logger.info("Facebook page/group selection updated")
        return redirect(url_for('schedule_list') + '?facebook=configured')

    pages = []
    groups = []
    if token:
        client = get_facebook_client()
        pages = client.get_user_pages(token['access_token'])
        groups = client.get_user_groups(token['access_token'])

    return render_template(
        'facebook_configure.html',
        token=token,
        pages=pages,
        groups=groups,
        is_new=request.args.get('new') == '1',
    )


@app.route('/facebook/post/<int:post_id>', methods=['POST'])
def facebook_post_social(post_id: int):
    """Post a social media post to Facebook immediately."""
    post = get_social_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    if post['platform'] != 'facebook':
        return jsonify({"error": "This post is not for Facebook"}), 400

    token = get_facebook_token()
    if not token:
        return jsonify({"error": "Facebook not connected. Please connect your account first."}), 401

    if not token['page_id'] or not token['page_access_token']:
        return jsonify({"error": "No Facebook Page selected. Please configure in Settings."}), 400

    if facebook_is_token_expired(token['expires_at']):
        client = get_facebook_client()
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = facebook_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_facebook_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            token = get_facebook_token()
        except Exception as e:
            app.logger.warning("Failed to refresh Facebook token: %s", e)
            return jsonify({"error": "Facebook token expired. Please reconnect."}), 401

    client = get_facebook_client()

    try:
        image_url = post['image_url'] if 'image_url' in post.keys() else None

        result = client.publish_smart_post(
            page_access_token=token['page_access_token'],
            page_id=token['page_id'],
            text=post['content'],
            image_url=image_url,
        )

        if result['success']:
            mark_social_post_used(post_id, True)

            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=post_id,
                article_id=post['article_id'] if 'article_id' in post.keys() else None,
                post_type='social',
                platform='facebook',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),
            )

            return jsonify({
                "success": True,
                "post_id": result.get('post_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to Facebook successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400

    except Exception as e:
        app.logger.exception("Failed to post to Facebook")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/post/<int:post_id>/facebook', methods=['POST'])
def compose_post_to_facebook(post_id: int):
    """Post a standalone post to Facebook immediately."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    token = get_facebook_token()
    if not token:
        return jsonify({"error": "Facebook not connected. Please connect your account first."}), 401

    if not token['page_id'] or not token['page_access_token']:
        return jsonify({"error": "No Facebook Page selected. Please configure in Settings."}), 400

    if facebook_is_token_expired(token['expires_at']):
        client = get_facebook_client()
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = facebook_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_facebook_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            token = get_facebook_token()
        except Exception as e:
            app.logger.warning("Failed to refresh Facebook token: %s", e)
            return jsonify({"error": "Facebook token expired. Please reconnect."}), 401

    client = get_facebook_client()

    try:
        image_url = post['image_url'] if 'image_url' in post.keys() else None

        result = client.publish_smart_post(
            page_access_token=token['page_access_token'],
            page_id=token['page_id'],
            text=post['content'],
            image_url=image_url,
        )

        if result['success']:
            mark_standalone_post_used(post_id, True)

            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=None,
                article_id=None,
                standalone_post_id=post_id,
                post_type='standalone',
                platform='facebook',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),
            )

            return jsonify({
                "success": True,
                "post_id": result.get('post_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to Facebook successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400

    except Exception as e:
        app.logger.exception("Failed to post to Facebook")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Twitter/X Integration Routes
# ============================================================================


@app.route('/twitter/status')
def twitter_status():
    """Check Twitter/X connection status."""
    client = get_twitter_client()
    token = get_twitter_token()

    if not client.is_configured():
        return jsonify({
            "connected": False,
            "configured": False,
            "message": "Twitter credentials not configured. Set TWITTER_CLIENT_ID and TWITTER_CLIENT_SECRET.",
        })

    if not token:
        return jsonify({
            "connected": False,
            "configured": True,
            "message": "Not connected to X/Twitter",
        })

    if twitter_is_token_expired(token['expires_at']):
        if token['refresh_token']:
            try:
                new_token = client.refresh_access_token(token['refresh_token'])
                expires_at = twitter_calculate_token_expiry(new_token.get('expires_in', 7200))
                update_twitter_token(
                    access_token=new_token['access_token'],
                    expires_at=expires_at,
                    refresh_token=new_token.get('refresh_token'),
                )
                return jsonify({
                    "connected": True,
                    "configured": True,
                    "username": token['username'],
                    "display_name": token['display_name'],
                    "user_id": token['user_id'],
                    "expires_at": expires_at,
                    "message": "Connected (token refreshed)",
                })
            except Exception as e:
                app.logger.warning("Failed to refresh Twitter token: %s", e)
                return jsonify({
                    "connected": False,
                    "configured": True,
                    "message": "Token expired. Please reconnect.",
                })
        else:
            return jsonify({
                "connected": False,
                "configured": True,
                "message": "Token expired and no refresh token. Please reconnect.",
            })

    return jsonify({
        "connected": True,
        "configured": True,
        "username": token['username'],
        "display_name": token['display_name'],
        "user_id": token['user_id'],
        "expires_at": token['expires_at'],
    })


@app.route('/twitter/auth')
def twitter_auth():
    """Start Twitter OAuth 2.0 PKCE flow."""
    client = get_twitter_client()

    if not client.is_configured():
        return jsonify({"error": "Twitter not configured"}), 400

    auth_url, state, code_verifier = client.get_authorization_url()
    session['twitter_oauth_state'] = state
    session['twitter_code_verifier'] = code_verifier

    return redirect(auth_url)


@app.route('/twitter/callback')
def twitter_callback():
    """Handle Twitter OAuth 2.0 callback."""
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        app.logger.error("Twitter OAuth error: %s - %s", error, error_desc)
        return render_template(
            'article_error.html',
            error=f"Twitter authorization failed: {error_desc}",
        )

    code = request.args.get('code')
    state = request.args.get('state')

    stored_state = session.pop('twitter_oauth_state', None)
    code_verifier = session.pop('twitter_code_verifier', None)

    if not stored_state or stored_state != state:
        app.logger.warning("Twitter OAuth state mismatch")
        return render_template(
            'article_error.html',
            error="Security verification failed. Please try again.",
        )

    if not code or not code_verifier:
        return render_template(
            'article_error.html',
            error="No authorization code or PKCE verifier. Please try again.",
        )

    client = get_twitter_client()

    try:
        token_data = client.exchange_code_for_token(code, code_verifier)
        access_token = token_data['access_token']
        refresh_token = token_data.get('refresh_token', '')
        expires_in = token_data.get('expires_in', 7200)

        expires_at = twitter_calculate_token_expiry(expires_in)

        user_info = client.get_user_info(access_token)

        if user_info:
            user_id = user_info.get('id', '')
            username = user_info.get('username', '')
            display_name = user_info.get('name', username or 'X User')
        else:
            user_id = ''
            username = ''
            display_name = 'X User'

        save_twitter_token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            user_id=user_id,
            username=username,
            display_name=display_name,
        )

        app.logger.info("Twitter connected for user: @%s", username)

        return redirect(url_for('view_articles') + '?twitter=connected')

    except Exception as e:
        app.logger.exception("Twitter OAuth exchange failed")
        return render_template(
            'article_error.html',
            error=f"Failed to connect to X/Twitter: {str(e)}",
        )


@app.route('/twitter/disconnect', methods=['POST'])
def twitter_disconnect():
    """Disconnect Twitter/X account."""
    delete_twitter_token()
    return jsonify({"success": True, "message": "Twitter disconnected"})


@app.route('/twitter/post/<int:post_id>', methods=['POST'])
def twitter_post_social(post_id: int):
    """Post a social media post to Twitter/X immediately."""
    post = get_social_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    token = get_twitter_token()
    if not token:
        return jsonify({"error": "Twitter not connected"}), 401

    if twitter_is_token_expired(token['expires_at']):
        if token['refresh_token']:
            client = get_twitter_client()
            try:
                new_token = client.refresh_access_token(token['refresh_token'])
                update_twitter_token(
                    access_token=new_token['access_token'],
                    expires_at=twitter_calculate_token_expiry(new_token.get('expires_in', 7200)),
                    refresh_token=new_token.get('refresh_token'),
                )
                token = get_twitter_token()
            except Exception as e:
                app.logger.warning("Failed to refresh Twitter token: %s", e)
                return jsonify({"error": "Twitter token expired. Please reconnect."}), 401
        else:
            return jsonify({"error": "Twitter token expired. Please reconnect."}), 401

    client = get_twitter_client()

    try:
        image_url = post['image_url'] if 'image_url' in post.keys() else None

        if image_url:
            app.logger.info("Posting to Twitter with image: %s", image_url)
            result = client.create_image_post(
                access_token=token['access_token'],
                text=post['content'],
                image_url=image_url,
            )
        else:
            result = client.create_post(
                access_token=token['access_token'],
                text=post['content'],
            )

        if result['success']:
            mark_social_post_used(post_id, True)

            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=post_id,
                article_id=post['article_id'] if 'article_id' in post.keys() else None,
                post_type='social',
                platform='twitter',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),
            )

            return jsonify({
                "success": True,
                "tweet_id": result.get('tweet_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to X/Twitter successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400

    except Exception as e:
        app.logger.exception("Failed to post to Twitter")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/post/<int:post_id>/twitter', methods=['POST'])
def compose_post_to_twitter(post_id: int):
    """Post a standalone post to Twitter/X immediately."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    token = get_twitter_token()
    if not token:
        return jsonify({"error": "Twitter not connected. Please connect your account first."}), 401

    if twitter_is_token_expired(token['expires_at']):
        if token['refresh_token']:
            client = get_twitter_client()
            try:
                new_token = client.refresh_access_token(token['refresh_token'])
                update_twitter_token(
                    access_token=new_token['access_token'],
                    expires_at=twitter_calculate_token_expiry(new_token.get('expires_in', 7200)),
                    refresh_token=new_token.get('refresh_token'),
                )
                token = get_twitter_token()
            except Exception as e:
                app.logger.warning("Failed to refresh Twitter token: %s", e)
                return jsonify({"error": "Twitter token expired. Please reconnect."}), 401
        else:
            return jsonify({"error": "Twitter token expired. Please reconnect."}), 401

    client = get_twitter_client()

    try:
        image_url = post['image_url'] if 'image_url' in post.keys() else None

        if image_url:
            app.logger.info("Posting to Twitter with image: %s", image_url)
            result = client.create_image_post(
                access_token=token['access_token'],
                text=post['content'],
                image_url=image_url,
            )
        else:
            result = client.create_post(
                access_token=token['access_token'],
                text=post['content'],
            )

        if result['success']:
            mark_standalone_post_used(post_id, True)

            now = datetime.now().isoformat(timespec='seconds')
            add_scheduled_post(
                social_post_id=None,
                article_id=None,
                standalone_post_id=post_id,
                post_type='standalone',
                platform='twitter',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),
            )

            return jsonify({
                "success": True,
                "tweet_id": result.get('tweet_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to X/Twitter successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error'),
            }), 400

    except Exception as e:
        app.logger.exception("Failed to post to Twitter")
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
    status_filter = request.args.get('status', 'pending')
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
    for plat in ['linkedin', 'threads', 'facebook', 'twitter', 'instagram']:
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
    facebook_count = sum(1 for p in scheduled if p.get('platform') == 'facebook')
    
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
        facebook_count=facebook_count,
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
    facebook_count = sum(1 for p in scheduled if p.get('platform') == 'facebook')
    
    return jsonify({
        "success": True,
        "posts": scheduled,
        "linkedin_count": linkedin_count,
        "threads_count": threads_count,
        "facebook_count": facebook_count,
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
        elif platform == 'twitter':
            twitter_token = get_twitter_token()
            if not twitter_token:
                return jsonify({"error": "Twitter/X not connected"}), 400
            
            if twitter_is_token_expired(twitter_token['expires_at']):
                tw_client = get_twitter_client()
                try:
                    new_token = tw_client.refresh_access_token(twitter_token['refresh_token'])
                    expires_at = twitter_calculate_token_expiry(new_token.get('expires_in', 7200))
                    update_twitter_token(
                        access_token=new_token['access_token'],
                        expires_at=expires_at,
                        refresh_token=new_token.get('refresh_token'),
                    )
                    twitter_token = get_twitter_token()
                except Exception as e:
                    return jsonify({"error": f"Twitter token expired: {e}"}), 400
            
            tw_client = get_twitter_client()
            
            if image_url:
                result = tw_client.create_image_post(
                    access_token=twitter_token['access_token'],
                    text=content[:280],
                    image_url=image_url,
                )
            else:
                result = tw_client.create_post(
                    access_token=twitter_token['access_token'],
                    text=content[:280],
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
                
                scheduled_time = datetime.fromisoformat(post['scheduled_for'])
                if datetime.now() < scheduled_time:
                    redistribute_scheduled_posts(platform)
                
                return jsonify({"success": True, "message": "Posted to X/Twitter!"})
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                return jsonify({"error": f"Failed: {error_msg}"}), 400

        elif platform == 'facebook':
            fb_token = get_facebook_token()
            if not fb_token:
                return jsonify({"error": "Facebook not connected"}), 400
            
            fb_client = get_facebook_client()
            
            result = fb_client.publish_smart_post(
                page_access_token=fb_token['page_access_token'],
                page_id=fb_token['page_id'],
                text=content[:5000],
                image_url=image_url,
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
                
                scheduled_time = datetime.fromisoformat(post['scheduled_for'])
                if datetime.now() < scheduled_time:
                    redistribute_scheduled_posts(platform)
                
                return jsonify({"success": True, "message": "Posted to Facebook!"})
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                return jsonify({"error": f"Failed: {error_msg}"}), 400

        elif platform == 'instagram':
            ig_token = get_instagram_token()
            if not ig_token:
                return jsonify({"error": "Instagram not connected"}), 400

            ig_client = get_instagram_client()

            if instagram_is_token_expired(ig_token['expires_at']):
                try:
                    new_token = ig_client.refresh_access_token(ig_token['access_token'])
                    expires_at = instagram_calculate_token_expiry(new_token.get('expires_in', 5184000))
                    update_instagram_token(
                        access_token=new_token['access_token'],
                        expires_at=expires_at,
                    )
                    ig_token = get_instagram_token()
                except Exception as e:
                    return jsonify({"error": f"Instagram token expired: {e}"}), 400

            # Publish honoring the post's Instagram format (feed/carousel/reel/story)
            result = _instagram_publish_for_post(
                ig_token['access_token'],
                content=content,
                image_url=image_url,
                standalone_post_id=post['standalone_post_id'],
                social_post_id=post['social_post_id'],
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

                scheduled_time = datetime.fromisoformat(post['scheduled_for'])
                if datetime.now() < scheduled_time:
                    redistribute_scheduled_posts(platform)

                return jsonify({"success": True, "message": "Posted to Instagram!"})
            else:
                error_msg = (result.get('friendly') or result.get('error', 'Unknown error')) if result else 'No response'
                return jsonify({"error": f"Failed: {error_msg}"}), 400

        else:
            # Handle LinkedIn posting (default)
            token = get_linkedin_token()
            if not token:
                return jsonify({"error": "LinkedIn not connected"}), 400
            
            client = get_linkedin_client()
            
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
    
    article_topic = post['article_topic'] if 'article_topic' in post.keys() else None
    
    try:
        result = None
        urn_key = 'post_urn'

        if platform == 'threads':
            threads_token = get_threads_token()
            if not threads_token:
                return jsonify({"error": "Threads not connected"}), 400
            
            threads_client = get_threads_client()
            if image_url:
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
            urn_key = 'permalink'

        elif platform == 'twitter':
            twitter_token = get_twitter_token()
            if not twitter_token:
                return jsonify({"error": "Twitter/X not connected"}), 400
            
            if twitter_is_token_expired(twitter_token['expires_at']):
                tw_client = get_twitter_client()
                try:
                    new_token = tw_client.refresh_access_token(twitter_token['refresh_token'])
                    expires_at = twitter_calculate_token_expiry(new_token.get('expires_in', 7200))
                    update_twitter_token(
                        access_token=new_token['access_token'],
                        expires_at=expires_at,
                        refresh_token=new_token.get('refresh_token'),
                    )
                    twitter_token = get_twitter_token()
                except Exception as e:
                    return jsonify({"error": f"Twitter token expired: {e}"}), 400
            
            tw_client = get_twitter_client()
            if image_url:
                result = tw_client.create_image_post(
                    access_token=twitter_token['access_token'],
                    text=content[:280],
                    image_url=image_url,
                )
            else:
                result = tw_client.create_post(
                    access_token=twitter_token['access_token'],
                    text=content[:280],
                )
            urn_key = 'permalink'

        elif platform == 'facebook':
            fb_token = get_facebook_token()
            if not fb_token:
                return jsonify({"error": "Facebook not connected"}), 400

            fb_client = get_facebook_client()
            result = fb_client.publish_smart_post(
                page_access_token=fb_token['page_access_token'],
                page_id=fb_token['page_id'],
                text=content[:5000],
                image_url=image_url,
            )
            urn_key = 'permalink'

        elif platform == 'instagram':
            ig_token = get_instagram_token()
            if not ig_token:
                return jsonify({"error": "Instagram not connected"}), 400

            ig_client = get_instagram_client()

            if instagram_is_token_expired(ig_token['expires_at']):
                try:
                    new_token = ig_client.refresh_access_token(ig_token['access_token'])
                    expires_at = instagram_calculate_token_expiry(new_token.get('expires_in', 5184000))
                    update_instagram_token(
                        access_token=new_token['access_token'],
                        expires_at=expires_at,
                    )
                    ig_token = get_instagram_token()
                except Exception as e:
                    return jsonify({"error": f"Instagram token expired: {e}"}), 400

            # Publish honoring the post's Instagram format (feed/carousel/reel/story)
            result = _instagram_publish_for_post(
                ig_token['access_token'],
                content=content,
                image_url=image_url,
                standalone_post_id=post['standalone_post_id'],
                social_post_id=post['social_post_id'],
            )
            if result and not result.get('success') and result.get('friendly'):
                result = dict(result, error=result['friendly'])
            urn_key = 'permalink'

        else:
            token = get_linkedin_token()
            if not token:
                return jsonify({"error": "LinkedIn not connected"}), 400

            client = get_linkedin_client()
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
                linkedin_post_urn=result.get(urn_key),
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
    valid_platforms = {'linkedin', 'threads', 'facebook', 'twitter', 'instagram'}
    platforms = [p for p in platforms if p in valid_platforms]
    
    slot_id = add_time_slot(
        day_of_week=day_of_week,
        time_slot=time_slot,
        enabled=True,
        platforms=platforms if platforms else None,
    )
    
    # Redistribute all pending posts to use the new optimal slots
    redistributed = {}
    for p in valid_platforms:
        redistributed[p] = redistribute_scheduled_posts(p)
    
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
        "redistributed": redistributed,
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
    valid_platforms = {'linkedin', 'threads', 'facebook', 'twitter', 'instagram'}
    platforms = [p for p in platforms if p in valid_platforms]
    
    # Update the slot (pass empty list to clear platform restrictions)
    update_time_slot(
        slot_id,
        day_of_week=day_of_week,
        time_slot=time_slot,
        platforms=platforms,
    )
    
    # Redistribute all pending posts to use the new optimal slots
    redistributed = {}
    for p in valid_platforms:
        redistributed[p] = redistribute_scheduled_posts(p)
    
    return jsonify({
        "success": True,
        "platforms": platforms,
        "redistributed": redistributed,
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
            "facebook": limits.get('facebook', 0),
            "twitter": limits.get('twitter', 0),
            "instagram": limits.get('instagram', 0),
        })

    # POST - update limits
    platform = request.form.get('platform', '').strip().lower()
    limit = request.form.get('limit', type=int, default=0)

    if platform not in ('linkedin', 'threads', 'facebook', 'twitter', 'instagram'):
        return jsonify({"error": "Invalid platform. Must be one of: linkedin, threads, facebook, twitter, instagram"}), 400
    
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
    from database import DB_PATH as _DB_PATH
    with sqlite3.connect(_DB_PATH) as conn:
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

# Saved posts are paginated per platform so the Compose page stays small even
# with thousands of imported/generated posts.
POSTS_PAGE_SIZE = 20


def _enrich_standalone_posts(rows, scheduled_info, posted_info, brief_names):
    """Turn standalone_posts rows into template dicts (scheduled/posted/media/brief)."""
    enriched = []
    for post in rows:
        post_dict = dict(post)
        post_dict['scheduled'] = scheduled_info.get(post['id'], {})
        post_dict['posted'] = posted_info.get(post['id'], {})
        # Instagram media format + parsed media list (feed by default)
        post_dict['ig_post_type'] = post_dict.get('ig_post_type') or 'feed'
        _raw_media = post_dict.get('media_items')
        try:
            post_dict['media_items'] = json.loads(_raw_media) if _raw_media else []
        except (ValueError, TypeError):
            post_dict['media_items'] = []
        _raw_tags = post_dict.get('ig_user_tags')
        try:
            post_dict['ig_user_tags'] = json.loads(_raw_tags) if _raw_tags else []
        except (ValueError, TypeError):
            post_dict['ig_user_tags'] = []
        # Attach the originating brief's name (agent-curated posts only)
        brief_id = post_dict.get('brief_id')
        post_dict['brief_name'] = brief_names.get(brief_id) if brief_id else None
        enriched.append(post_dict)
    return enriched


@app.route('/compose')
def compose_page():
    """Command Center page for generating social media posts."""
    # All saved posts (newest first) — cheap to load; we only render the first page.
    posts = list_standalone_posts()

    # Map brief ids -> names so agent-curated posts can show/filter by their brief
    brief_names = {brief['id']: brief['name'] for brief in list_content_briefs()}

    # Group raw rows by platform and count totals (for the "Load more" controls)
    raw_by_platform = {}
    for post in posts:
        raw_by_platform.setdefault(post['platform'], []).append(post)
    platform_totals = {p: len(rows) for p, rows in raw_by_platform.items()}

    # Only enrich + render the first page per platform (keeps the payload small).
    capped_by_platform = {p: rows[:POSTS_PAGE_SIZE] for p, rows in raw_by_platform.items()}
    capped_ids = [r['id'] for rows in capped_by_platform.values() for r in rows]
    scheduled_info = get_pending_schedules_for_standalone_posts(capped_ids) if capped_ids else {}
    posted_info = get_posted_info_for_standalone_posts(capped_ids) if capped_ids else {}

    posts_by_platform = {
        platform: _enrich_standalone_posts(rows, scheduled_info, posted_info, brief_names)
        for platform, rows in capped_by_platform.items()
    }

    # Distinct briefs across ALL saved posts (not just the first page), for the filter
    briefs_in_posts = {}
    for post in posts:
        bid = post['brief_id'] if 'brief_id' in post.keys() else None
        if bid and bid in brief_names:
            briefs_in_posts[bid] = brief_names[bid]
    brief_filter_options = [
        {'id': bid, 'name': name}
        for bid, name in sorted(briefs_in_posts.items(), key=lambda kv: kv[1].lower())
    ]
    
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
    
    # Get recent image prompt+image combos for reuse
    recent_img = list_recent_image_prompts(limit=10)
    recent_image_prompts_list = [
        {
            'prompt': r['source_content'] or '',
            'image_url': r['image_url'],
            'preview': (r['source_content'][:60] + '...' if r['source_content'] and len(r['source_content']) > 60
                        else r['source_content']) or '(no prompt)',
            'created_at': r['created_at'],
        }
        for r in recent_img
    ]

    # Curated, named prompts from the reusable library
    library_prompts_list = _serialize_library_prompts()

    return render_template(
        'compose.html',
        posts_by_platform=posts_by_platform,
        platform_totals=platform_totals,
        page_size=POSTS_PAGE_SIZE,
        brief_filter_options=brief_filter_options,
        next_slots=next_slots,
        saved_sources=saved_sources,
        selected_source=selected_source,
        linkedin_connected=linkedin_connected,
        threads_connected=threads_connected,
        recent_prompts=recent_prompts_list,
        recent_image_prompts=recent_image_prompts_list,
        library_prompts=library_prompts_list,
        starter_prompt_groups=grouped_starter_prompts(),
        model_options=provider_model_options(),
        file_formats=[f for f in document_extractor.supported_formats() if f['available']],
        file_accept=document_extractor.accept_attribute(),
    )


@app.route('/compose/posts/more')
def compose_posts_more():
    """Return the next page of saved posts for a platform as an HTML fragment."""
    platform = (request.args.get('platform') or '').strip()
    if not platform:
        return jsonify({"error": "platform is required"}), 400
    offset = request.args.get('offset', 0, type=int) or 0

    # Fetch one extra row to know whether there are more after this page
    rows = list_standalone_posts(platform=platform, limit=POSTS_PAGE_SIZE + 1, offset=offset)
    has_more = len(rows) > POSTS_PAGE_SIZE
    rows = rows[:POSTS_PAGE_SIZE]

    ids = [r['id'] for r in rows]
    scheduled_info = get_pending_schedules_for_standalone_posts(ids) if ids else {}
    posted_info = get_posted_info_for_standalone_posts(ids) if ids else {}
    brief_names = {brief['id']: brief['name'] for brief in list_content_briefs()}
    enriched = _enrich_standalone_posts(rows, scheduled_info, posted_info, brief_names)

    html = render_template(
        'partials/post_items.html',
        posts=enriched,
        platform=platform,
        start_index=offset,
    )
    return jsonify({
        "html": html,
        "has_more": has_more,
        "next_offset": offset + len(rows),
    })


def _valid_post_platform(raw):
    """Normalize an optional ?platform= filter, returning None when unset."""
    platform = (raw or '').strip()
    return platform or None


@app.route('/compose/posts/content')
def compose_posts_content():
    """Return id/platform/content for every saved post (optionally one platform).

    Text only — no images, schedules or brief metadata — so Find & Replace can
    search across ALL posts, not just the page currently rendered.
    """
    platform = _valid_post_platform(request.args.get('platform'))
    rows = list_standalone_posts(platform=platform)

    # Index mirrors the "Post N" label on the page: 1-based within each platform,
    # newest first (the same order the Compose page renders).
    per_platform_index = {}
    posts = []
    for row in rows:
        p = row['platform']
        per_platform_index[p] = per_platform_index.get(p, 0) + 1
        posts.append({
            'id': row['id'],
            'platform': p,
            'content': row['content'] or '',
            'index': per_platform_index[p],
        })

    return jsonify({
        "success": True,
        "posts": posts,
        "count": len(posts),
    })


@app.route('/compose/posts/ids')
def compose_posts_ids():
    """Return the id + platform of every saved post (optionally one platform).

    Powers the "select all across every page" bulk actions, which need the full
    matching id set rather than the checkboxes currently in the DOM.
    """
    platform = _valid_post_platform(request.args.get('platform'))
    rows = list_standalone_posts(platform=platform)
    posts = [{'id': row['id'], 'platform': row['platform']} for row in rows]

    return jsonify({
        "success": True,
        "posts": posts,
        "count": len(posts),
    })


@app.route('/compose/posts/queue-all-unscheduled', methods=['POST'])
def compose_queue_all_unscheduled():
    """Queue every unscheduled, unused saved post for a platform (server-side).

    Replaces the old DOM-scraping "Queue All" so it works across all posts, not
    just the ones currently loaded on the page.
    """
    platform = (request.form.get('platform') or (request.get_json(silent=True) or {}).get('platform') or '').strip()
    if platform not in ['linkedin', 'threads', 'facebook', 'twitter', 'instagram']:
        return jsonify({"error": f"Platform {platform} does not support scheduling"}), 400

    rows = list_standalone_posts(platform=platform)
    all_ids = [r['id'] for r in rows]
    already_scheduled = get_pending_schedules_for_standalone_posts(all_ids) if all_ids else {}

    queued = 0
    skipped = 0
    no_slots = False
    for row in rows:
        post = dict(row)
        pid = post['id']
        if post.get('used'):
            continue
        if pid in already_scheduled:
            continue

        if platform == 'instagram':
            ig_post_type = post.get('ig_post_type')
            raw_items = post.get('media_items')
            try:
                media_items = json.loads(raw_items) if raw_items else []
            except (ValueError, TypeError):
                media_items = []
            _, media_err = _ensure_instagram_media(
                post.get('content'),
                post.get('image_url'),
                ig_post_type,
                media_items,
                standalone_post_id=pid,
            )
            if media_err:
                skipped += 1
                continue

        slot = get_next_available_slot(platform)
        if not slot:
            no_slots = True
            break
        add_scheduled_post(
            social_post_id=None,
            article_id=None,
            standalone_post_id=pid,
            post_type='standalone',
            platform=platform,
            scheduled_for=slot,
            status='pending',
        )
        queued += 1

    return jsonify({
        "success": True,
        "queued": queued,
        "skipped": skipped,
        "no_slots": no_slots,
    })


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


# ---------------------------------------------------------------------------
# Prompt Library (curated, named reusable prompts)
# ---------------------------------------------------------------------------


def _serialize_library_prompts():
    """Return the saved library prompts as plain dicts for JSON / templates."""
    result = []
    for p in list_library_prompts():
        content = p['content'] or ''
        result.append({
            'id': p['id'],
            'title': p['title'],
            'content': content,
            'preview': content[:80] + ('...' if len(content) > 80 else ''),
            'updated_at': p['updated_at'],
        })
    return result


@app.route('/compose/library')
def compose_library_list():
    """Return all saved library prompts as JSON."""
    return jsonify({"success": True, "prompts": _serialize_library_prompts()})


@app.route('/compose/library/save', methods=['POST'])
def compose_library_save():
    """Create a new named prompt in the library."""
    data = request.get_json(silent=True) or request.form
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()

    if not title:
        return jsonify({"error": "A name for the prompt is required"}), 400
    if not content:
        return jsonify({"error": "Prompt text cannot be empty"}), 400

    try:
        prompt_id = add_library_prompt(title, content)
        return jsonify({
            "success": True,
            "message": "Prompt saved to library",
            "prompts": _serialize_library_prompts(),
            "id": prompt_id,
        })
    except Exception as e:
        app.logger.exception("Failed to save library prompt")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/library/update', methods=['POST'])
def compose_library_update():
    """Update an existing library prompt's name and/or text."""
    data = request.get_json(silent=True) or request.form
    try:
        prompt_id = int(data.get('id'))
    except (TypeError, ValueError):
        return jsonify({"error": "A valid prompt id is required"}), 400
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()

    if not title:
        return jsonify({"error": "A name for the prompt is required"}), 400
    if not content:
        return jsonify({"error": "Prompt text cannot be empty"}), 400

    if not get_library_prompt(prompt_id):
        return jsonify({"error": "Prompt not found"}), 404

    try:
        update_library_prompt(prompt_id, title, content)
        return jsonify({
            "success": True,
            "message": "Prompt updated",
            "prompts": _serialize_library_prompts(),
        })
    except Exception as e:
        app.logger.exception("Failed to update library prompt")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/library/delete', methods=['POST'])
def compose_library_delete():
    """Delete a library prompt by id."""
    data = request.get_json(silent=True) or request.form
    try:
        prompt_id = int(data.get('id'))
    except (TypeError, ValueError):
        return jsonify({"error": "A valid prompt id is required"}), 400

    try:
        count = delete_library_prompt(prompt_id)
        return jsonify({
            "success": True,
            "message": "Prompt deleted from library",
            "prompts": _serialize_library_prompts(),
            "count": count,
        })
    except Exception as e:
        app.logger.exception("Failed to delete library prompt")
        return jsonify({"error": str(e)}), 500


def _parse_ai_provider(form):
    """Parse the compose AI-provider selection into ``(provider, model, use_local)``.

    ``provider`` is "openai"/"anthropic" for an explicit cloud choice, or ``None``
    to use the configured default (``LLM_PROVIDER``). ``use_local`` is True for the
    local (Ollama) option. Falls back to the legacy ``use_local`` form field when
    no ``provider`` field is sent.
    """
    provider = (form.get('provider') or '').strip().lower() or None
    model = (form.get('model') or '').strip() or None
    if provider == 'cloud':          # legacy value => configured cloud default
        provider = None
    use_local = provider in ('local', 'ollama') or (
        provider is None
        and form.get('use_local', 'false').lower() in ('true', '1', 'yes')
    )
    if use_local:
        provider = None              # local is signalled via use_local, not provider
    return provider, model, use_local


def _condense_notice(meta):
    """Build a user-facing notice when long source content was condensed.

    ``meta`` is the dict returned by ``condense_document_text``; returns ``None``
    when no condensation happened (short input) so callers can skip the toast.
    """
    if not meta or not meta.get('condensed'):
        return None
    chunks = meta.get('chunks', 0)
    notice = (
        f"Condensed {meta.get('original_chars', 0):,} characters of source content into "
        f"{chunks} section summar{'y' if chunks == 1 else 'ies'} before generating, "
        "so the whole thing informs the posts."
    )
    if meta.get('chunks_dropped'):
        notice += " Only the first part of a very large source was used."
    return notice


@app.route('/compose/generate', methods=['POST'])
def compose_generate():
    """Generate social media posts using LLM based on source type."""
    source_type = request.form.get('source_type', 'freeform')
    content = request.form.get('content', '').strip()
    platforms = request.form.getlist('platforms')
    tone = request.form.get('tone', 'professional')
    posts_per_platform = request.form.get('posts_per_platform', 10, type=int)
    extra_context = request.form.get('extra_context', '').strip() or None
    topic = request.form.get('topic', '').strip() or None
    image_url = request.form.get('image_url', '').strip() or None
    ai_provider, ai_model, use_local = _parse_ai_provider(request.form)

    if source_type not in ('image', 'file') and not content:
        return jsonify({"error": "Content is required"}), 400

    if not platforms:
        platforms = ['linkedin', 'threads', 'twitter']
    
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        # Generate posts based on source type
        source_data = None
        notice = None
        if source_type == 'freeform':
            generated = generate_posts_from_prompt(
                prompt=content,
                platforms=platforms,
                tone=tone,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
                use_local=use_local,
                provider=ai_provider,
                model=ai_model,
            )
        elif source_type == 'url':
            if is_github_repo_url(content):
                owner, repo_name = parse_github_repo_url(content)
                repo_data = fetch_github_repo(owner, repo_name)
                gh_extra = extra_context or ""
                gh_extra = f"This content is from the GitHub repository {repo_data['title']}. Include the repo URL ({content}) as credit.\n{gh_extra}".strip()
                # Condense long repo content (README, etc.) instead of truncating.
                gh_text, condense_meta = condense_document_text(
                    repo_data['content'],
                    provider=ai_provider,
                    model=ai_model,
                    use_local=use_local,
                )
                notice = _condense_notice(condense_meta)
                generated = generate_posts_from_text(
                    text=gh_text,
                    platforms=platforms,
                    tone=tone,
                    topic=repo_data['title'],
                    posts_per_platform=posts_per_platform,
                    extra_context=gh_extra,
                    use_local=use_local,
                    source_url=content,
                    provider=ai_provider,
                    model=ai_model,
                )
                source_data = {
                    "url": content,
                    "title": repo_data['title'],
                    "description": repo_data['description'],
                    "content": repo_data['content'],
                    "og_image": repo_data['og_image'],
                }
                source_id = add_url_source(
                    url=content,
                    title=repo_data['title'],
                    description=repo_data['description'],
                    content=repo_data['content'],
                    og_image=repo_data['og_image'],
                )
                source_data["source_id"] = source_id
            else:
                result = generate_posts_from_url(
                    url=content,
                    platforms=platforms,
                    tone=tone,
                    posts_per_platform=posts_per_platform,
                    extra_context=extra_context,
                    use_local=use_local,
                    provider=ai_provider,
                    model=ai_model,
                )
                # New structure: {"posts": {...}, "source_data": {...}}
                generated = result.get("posts", result)
                source_data = result.get("source_data")
                notice = _condense_notice(result.get("condense_meta"))

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
                if not image_url and source_data and source_data.get("og_image"):
                    image_url = source_data["og_image"]
        elif source_type == 'text':
            # Long pasted text is condensed (map-reduce) rather than truncated to
            # the first ~5000 chars, so the whole thing informs generation.
            text_for_gen, condense_meta = condense_document_text(
                content,
                provider=ai_provider,
                model=ai_model,
                use_local=use_local,
            )
            notice = _condense_notice(condense_meta)
            generated = generate_posts_from_text(
                text=text_for_gen,
                platforms=platforms,
                tone=tone,
                topic=topic,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
                use_local=use_local,
                provider=ai_provider,
                model=ai_model,
            )
        elif source_type == 'file':
            upload = request.files.get('file')
            if not upload or not upload.filename:
                return jsonify({"error": "Please choose a document to upload"}), 400
            if not document_extractor.is_supported(upload.filename):
                return jsonify({
                    "error": f"Unsupported file type. Supported formats: "
                             f"{document_extractor.accept_attribute()}"
                }), 400
            try:
                file_bytes = upload.read()
                extracted_text = document_extractor.extract_text(file_bytes, upload.filename)
            except document_extractor.ExtractionError as exc:
                return jsonify({"error": str(exc)}), 400

            file_display_name = secure_filename(upload.filename) or upload.filename
            # Use the document's name as the topic unless the user gave one.
            file_topic = topic or os.path.splitext(file_display_name)[0].replace('_', ' ').replace('-', ' ')
            # Long documents are condensed (map-reduce) rather than truncated, so
            # the whole document informs generation instead of just the first page.
            condensed_text, condense_meta = condense_document_text(
                extracted_text,
                provider=ai_provider,
                model=ai_model,
                use_local=use_local,
            )
            notice = _condense_notice(condense_meta)
            generated = generate_posts_from_text(
                text=condensed_text,
                platforms=platforms,
                tone=tone,
                topic=file_topic,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
                use_local=use_local,
                provider=ai_provider,
                model=ai_model,
            )
        elif source_type == 'image':
            import base64 as b64module
            import urllib.request as _urlreq
            import mimetypes
            files = request.files.getlist('images')
            recall_urls = request.form.getlist('recall_image_urls')

            images_payload = []
            first_image_bytes = None

            if files and files[0].filename:
                # Fresh file uploads
                if len(files) > 4:
                    return jsonify({"error": "Maximum 4 images allowed"}), 400
                for f in files:
                    if not allowed_file(f.filename):
                        return jsonify({"error": f"File type not allowed: {f.filename}"}), 400
                    raw = f.read()
                    if len(raw) > 16 * 1024 * 1024:
                        return jsonify({"error": f"File too large: {f.filename} (max 16MB)"}), 400
                    if first_image_bytes is None:
                        first_image_bytes = raw
                        first_image_ext = f.filename.rsplit('.', 1)[1].lower() if '.' in f.filename else 'jpg'
                    mime = f.content_type or 'image/jpeg'
                    images_payload.append({
                        "base64": b64module.b64encode(raw).decode(),
                        "mime_type": mime,
                    })
            elif recall_urls:
                # Recalled from recent image prompts -- fetch by URL through SSRF guard
                for url in recall_urls[:4]:
                    try:
                        raw, ctype, _final = _fetch_safely(
                            url,
                            max_bytes=16 * 1024 * 1024,
                            timeout=15,
                            allowed_content_types=("image/",),
                        )
                        mime = ctype or mimetypes.guess_type(url)[0] or 'image/jpeg'
                        if first_image_bytes is None:
                            first_image_bytes = raw
                        image_url = image_url or url
                        images_payload.append({
                            "base64": b64module.b64encode(raw).decode(),
                            "mime_type": mime,
                        })
                    except (UnsafeURLError, RuntimeError, requests.RequestException) as exc:
                        app.logger.warning("Could not fetch recalled image %s: %s", url, exc)
                    except Exception:
                        app.logger.warning("Could not fetch recalled image: %s", url)
                if not images_payload:
                    return jsonify({"error": "Could not fetch the recalled image(s)"}), 400
            else:
                return jsonify({"error": "At least one image is required"}), 400

            generated = generate_posts_from_images(
                images=images_payload,
                prompt=content or None,
                platforms=platforms,
                tone=tone,
                posts_per_platform=posts_per_platform,
                extra_context=extra_context,
                use_local=use_local,
                provider=ai_provider,
                model=ai_model,
            )

            if first_image_bytes and not image_url:
                try:
                    from io import BytesIO
                    from PIL import Image
                    img = Image.open(BytesIO(first_image_bytes))
                    img.verify()
                    if CLOUDINARY_CONFIGURED:
                        result = cloudinary.uploader.upload(
                            first_image_bytes,
                            folder="insights",
                            resource_type="image",
                        )
                        image_url = result['secure_url']
                    else:
                        unique_fn = f"{uuid.uuid4().hex}.{first_image_ext}"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_fn)
                        with open(filepath, 'wb') as fp:
                            fp.write(first_image_bytes)
                        image_url = f"{request.host_url}static/uploads/{unique_fn}"
                except Exception:
                    app.logger.debug("Could not auto-upload first image for attachment")
        else:
            return jsonify({"error": f"Unknown source type: {source_type}"}), 400
        
        # Save generated posts to database
        if source_type == 'file':
            source_label = f"[File: {file_display_name}]"
        elif content:
            source_label = content[:1000]
        else:
            source_label = f"[{len(request.files.getlist('images'))} image(s)]"
        requested_platforms = {p.lower() for p in platforms}
        saved_posts = {}
        for platform, post_data in generated.items():
            if platform == 'raw':
                continue
            
            # Normalize platform name (LLMs sometimes return "Threads" or "X")
            norm_platform = platform.lower().strip()
            platform_aliases = {"x": "twitter"}
            norm_platform = platform_aliases.get(norm_platform, norm_platform)

            # Drop platforms the user didn't ask for (local models hallucinate extras)
            if norm_platform not in requested_platforms:
                continue

            posts_list = post_data if isinstance(post_data, list) else [post_data]
            if norm_platform not in saved_posts:
                saved_posts[norm_platform] = []
            
            for post_content in posts_list:
                post_id = add_standalone_post(
                    source_type=source_type,
                    source_content=source_label,
                    platform=norm_platform,
                    content=post_content,
                    image_url=image_url,
                )
                saved_posts[norm_platform].append({
                    'id': post_id,
                    'content': post_content,
                    'image_url': image_url,
                })
                if not image_url:
                    _maybe_attach_link_image(post_id, post_content)
        
        response_data = {
            "success": True,
            "generated": generated,
            "saved_posts": saved_posts,
        }
        if source_data:
            response_data["source_data"] = source_data
        if notice:
            response_data["notice"] = notice

        return jsonify(response_data)
        
    except Exception as e:
        app.logger.exception("Failed to generate posts")
        return jsonify({"error": str(e)}), 500


@app.route('/compose/ollama-status', methods=['GET'])
def compose_ollama_status():
    """Return Ollama availability and installed vision models."""
    return jsonify(check_ollama_status())


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

    valid_platforms = ['linkedin', 'threads', 'twitter', 'facebook', 'instagram']
    if platform not in valid_platforms:
        return jsonify({"error": f"Invalid platform. Must be one of: {', '.join(valid_platforms)}"}), 400

    post_id = add_standalone_post(
        source_type='manual',
        source_content='Manual post',
        platform=platform,
        content=content,
        image_url=image_url,
    )

    if not image_url:
        _maybe_attach_link_image(post_id, content)

    return jsonify({
        "success": True,
        "post": {
            "id": post_id,
            "platform": platform,
            "content": content,
            "image_url": image_url,
        }
    })


@app.route('/compose/import', methods=['POST'])
def compose_import_file():
    """Import posts from a CSV or XLSX file into the Command Center."""
    import csv as csv_mod

    IMPORT_PLATFORMS = {'threads', 'linkedin', 'facebook', 'twitter', 'instagram'}
    TRUTHY = {'true', '1', 'yes', 'y', 't'}

    def _is_truthy(val) -> bool:
        return str(val or '').strip().lower() in TRUTHY

    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
        return jsonify({"error": "Unsupported file type. Please upload a .csv or .xlsx file."}), 400

    rows = []
    try:
        if filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode('utf-8-sig'))
            reader = csv_mod.DictReader(stream)
            header_map = {k.strip().lower(): k for k in (reader.fieldnames or [])}
            platform_col = header_map.get('platform')
            copy_col = header_map.get('copy')
            video_col = header_map.get('video')
            repost_col = header_map.get('repost')
            if not platform_col or not copy_col:
                return jsonify({"error": "CSV must have 'Platform' and 'copy' columns. "
                                f"Found: {', '.join(reader.fieldnames or [])}"}), 400
            for row in reader:
                rows.append({
                    'platform': (row.get(platform_col) or '').strip(),
                    'copy': (row.get(copy_col) or '').strip(),
                    'video': (row.get(video_col) or '').strip() if video_col else '',
                    'repost': (row.get(repost_col) or '').strip() if repost_col else '',
                })
        else:
            import openpyxl
            wb = openpyxl.load_workbook(file.stream, read_only=True, data_only=True)
            ws = wb.active
            raw_rows = list(ws.iter_rows(values_only=True))
            wb.close()
            if not raw_rows:
                return jsonify({"error": "The spreadsheet is empty."}), 400
            headers = [str(h).strip().lower() if h else '' for h in raw_rows[0]]
            try:
                platform_idx = headers.index('platform')
            except ValueError:
                return jsonify({"error": "Spreadsheet must have a 'Platform' column. "
                                f"Found: {', '.join(str(h) for h in raw_rows[0])}"}), 400
            try:
                copy_idx = headers.index('copy')
            except ValueError:
                return jsonify({"error": "Spreadsheet must have a 'copy' column. "
                                f"Found: {', '.join(str(h) for h in raw_rows[0])}"}), 400
            video_idx = headers.index('video') if 'video' in headers else None
            repost_idx = headers.index('repost') if 'repost' in headers else None
            for row in raw_rows[1:]:
                rows.append({
                    'platform': str(row[platform_idx]).strip() if row[platform_idx] else '',
                    'copy': str(row[copy_idx]).strip() if row[copy_idx] else '',
                    'video': str(row[video_idx]).strip() if video_idx is not None and row[video_idx] else '',
                    'repost': str(row[repost_idx]).strip() if repost_idx is not None and row[repost_idx] is not None else '',
                })
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400

    existing = get_existing_standalone_content()

    imported = 0
    skipped = 0
    skipped_details = []
    by_platform = {}
    original_filename = file.filename

    for i, row in enumerate(rows, start=2):
        raw_platform = row['platform']
        content = row['copy']

        content_preview = (content[:80] + '...') if len(content) > 80 else content

        if not content:
            skipped += 1
            skipped_details.append({
                "row": i,
                "reason": "empty_content",
                "message": "Content is empty",
                "fix": "Add post text to the 'copy' column in this row.",
                "platform": raw_platform or "(blank)",
                "preview": "",
            })
            continue

        tokens = [t.strip() for t in re.split(r'[,;|]', raw_platform) if t.strip()]
        seen = set()
        platforms = []
        for token in tokens:
            normalized = token.lower()
            if normalized == 'x':
                normalized = 'twitter'
            if normalized in seen:
                continue
            seen.add(normalized)
            platforms.append((normalized, token))

        if not platforms:
            skipped += 1
            skipped_details.append({
                "row": i,
                "reason": "unsupported_platform",
                "message": f"Unsupported platform '{raw_platform}'",
                "fix": "Change the Platform value to one of: Threads, LinkedIn, Facebook, Twitter/X, Instagram. Use commas to list multiple (e.g. 'Twitter,Threads').",
                "platform": raw_platform or "(blank)",
                "preview": content_preview,
            })
            continue

        source_content = original_filename
        if row.get('video'):
            source_content = f"{original_filename} | {row['video']}"

        is_repost = _is_truthy(row.get('repost'))

        for platform, original_token in platforms:
            if platform not in IMPORT_PLATFORMS:
                skipped += 1
                skipped_details.append({
                    "row": i,
                    "reason": "unsupported_platform",
                    "message": f"Unsupported platform '{original_token}'",
                    "fix": "Change the Platform value to one of: Threads, LinkedIn, Facebook, Twitter/X, Instagram. Use commas to list multiple (e.g. 'Twitter,Threads').",
                    "platform": original_token or "(blank)",
                    "preview": content_preview,
                })
                continue

            existing_id = existing.get((platform, content))
            if existing_id is not None and not is_repost:
                skipped += 1
                skipped_details.append({
                    "row": i,
                    "reason": "duplicate",
                    "message": f"Duplicate — identical {platform} post already exists (Post #{existing_id})",
                    "fix": "This post is already in the Command Center. Edit the copy in the file to make it unique, delete the existing post first, or set the 'repost' column to true to allow a duplicate.",
                    "platform": platform,
                    "preview": content_preview,
                    "duplicate_post_id": existing_id,
                })
                continue

            new_id = add_standalone_post(
                source_type='import',
                source_content=source_content,
                platform=platform,
                content=content,
                repost=is_repost,
            )
            if not is_repost:
                existing[(platform, content)] = new_id
            imported += 1
            by_platform[platform] = by_platform.get(platform, 0) + 1
            _maybe_attach_link_image(new_id, content)

    return jsonify({
        "success": True,
        "imported": imported,
        "skipped": skipped,
        "skipped_details": skipped_details,
        "by_platform": by_platform,
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

    # If the edit introduced a URL and the post still has no image, kick off a
    # background fetch of the og:image. Skip when the body did not change or
    # already contained the same URLs.
    post_keys = post.keys() if hasattr(post, 'keys') else []
    existing_image = post['image_url'] if 'image_url' in post_keys else None
    existing_content = post['content'] if 'content' in post_keys else ''
    if not existing_image:
        old_urls = set(extract_urls_from_text(existing_content or ''))
        new_urls = extract_urls_from_text(new_content)
        if any(u not in old_urls for u in new_urls):
            _maybe_attach_link_image(post_id, new_content)

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


@app.route('/compose/post/<int:post_id>/media', methods=['POST'])
def compose_set_post_media(post_id: int):
    """Set a standalone post's Instagram media format and media list.

    Form/JSON: ig_post_type in {feed,carousel,reel,story}; media_items = JSON
    string list of {"url","kind":"image"|"video"}.
    """
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    payload = request.get_json(silent=True) or request.form
    ig_post_type = (payload.get('ig_post_type') or 'feed').strip().lower()
    if ig_post_type not in ('feed', 'carousel', 'reel', 'story'):
        return jsonify({"error": f"Invalid post type: {ig_post_type}"}), 400

    raw_items = payload.get('media_items') or '[]'
    if isinstance(raw_items, str):
        try:
            items = json.loads(raw_items)
        except (ValueError, TypeError):
            return jsonify({"error": "media_items must be valid JSON"}), 400
    else:
        items = raw_items
    if not isinstance(items, list):
        return jsonify({"error": "media_items must be a list"}), 400

    # Normalize items to {url, kind}
    clean = []
    for it in items:
        if not isinstance(it, dict):
            continue
        url = (it.get('url') or '').strip()
        if not url:
            continue
        kind = 'video' if it.get('kind') == 'video' else 'image'
        clean.append({"url": url, "kind": kind})

    # Validate item counts per type (light server-side guard; publish re-validates)
    if ig_post_type == 'carousel' and clean and not (2 <= len(clean) <= 10):
        return jsonify({"error": "Carousels need between 2 and 10 items."}), 400
    if ig_post_type in ('reel', 'story') and len(clean) > 1:
        return jsonify({"error": f"A {ig_post_type} takes a single media item."}), 400

    set_standalone_post_media(post_id, ig_post_type, clean)
    return jsonify({"success": True, "ig_post_type": ig_post_type, "media_items": clean})


@app.route('/compose/post/<int:post_id>/user-tags', methods=['POST'])
def compose_set_post_user_tags(post_id: int):
    """Set the Instagram people-tags for a standalone post (feed photos only).

    JSON/Form: user_tags = JSON list of {"username", "x", "y"} with x/y in 0..1.
    An empty list clears the tags.
    """
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    payload = request.get_json(silent=True) or request.form
    raw_tags = payload.get('user_tags')
    if raw_tags is None:
        raw_tags = []
    if isinstance(raw_tags, str):
        try:
            tags = json.loads(raw_tags or '[]')
        except (ValueError, TypeError):
            return jsonify({"error": "user_tags must be valid JSON"}), 400
    else:
        tags = raw_tags
    if not isinstance(tags, list):
        return jsonify({"error": "user_tags must be a list"}), 400
    if len(tags) > 20:
        return jsonify({"error": "At most 20 people can be tagged."}), 400

    clean = []
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        username = (tag.get('username') or '').strip().lstrip('@')
        if not username or not re.fullmatch(r'[A-Za-z0-9._]{1,30}', username):
            return jsonify({"error": f"Invalid Instagram username: {tag.get('username')!r}"}), 400
        try:
            x = min(max(float(tag.get('x', 0.5)), 0.0), 1.0)
            y = min(max(float(tag.get('y', 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            return jsonify({"error": "Tag x/y must be numbers"}), 400
        clean.append({"username": username, "x": round(x, 4), "y": round(y, 4)})

    set_standalone_post_user_tags(post_id, clean)
    return jsonify({"success": True, "user_tags": clean})


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


@app.route('/compose/post/<int:post_id>/link-image', methods=['GET', 'POST'])
def compose_apply_link_image(post_id: int):
    """Fetch the og:image of a URL referenced by the post and attach it.
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
            url:
              type: string
              description: Optional override URL. Must be one of the URLs detected in the post body unless `allow_external` is true.
            allow_external:
              type: boolean
              default: false
    responses:
      200:
        description: Image attached successfully (POST) or detected URLs (GET)
      400:
        description: No URL available or override URL not in post body
      404:
        description: Post not found
      422:
        description: Could not extract an og:image from the URL
    """
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    detected = extract_urls_from_text(post.get('content') or '')

    if request.method == 'GET':
        return jsonify({
            "success": True,
            "detected_urls": detected,
            "image_url": post.get('image_url'),
        })

    data = request.get_json(silent=True) or {}
    override = (data.get('url') or '').strip()
    allow_external = bool(data.get('allow_external'))

    target = override or (detected[0] if detected else None)
    if not target:
        return jsonify({
            "error": "No URL found in post and none provided.",
            "detected_urls": detected,
        }), 400

    if override and override not in detected and not allow_external:
        return jsonify({
            "error": "Override URL is not present in the post body.",
            "detected_urls": detected,
        }), 400

    try:
        _assert_safe_url(target)
    except UnsafeURLError as exc:
        return jsonify({"error": f"URL rejected: {exc}", "detected_urls": detected}), 400

    og_image = fetch_og_image_for_url(target)
    if not og_image:
        return jsonify({
            "error": "Could not find an og:image at that URL.",
            "detected_urls": detected,
            "source_url": target,
        }), 422

    try:
        saved_url = save_stock_image_to_library(og_image)
    except (UnsafeURLError, RuntimeError, ValueError, requests.RequestException) as exc:
        app.logger.warning("Failed to save og:image %s for post %s: %s", og_image, post_id, exc)
        return jsonify({
            "error": f"Failed to download og:image: {exc}",
            "detected_urls": detected,
            "source_url": target,
            "og_image_url": og_image,
        }), 422

    update_standalone_post_image(post_id, saved_url)
    return jsonify({
        "success": True,
        "image_url": saved_url,
        "source_url": target,
        "og_image_url": og_image,
        "detected_urls": detected,
    })


@app.route('/compose/post/<int:post_id>/refresh-source-image', methods=['POST'])
def compose_refresh_source_image(post_id: int):
    """Re-pull the image from the post's original source (YouTube/article).

    Posts generated from a saved source store the source URL in
    ``source_content`` (with ``source_type`` of 'saved_source' or 'url'). This
    re-fetches the YouTube thumbnail or article og:image so an updated source
    image can be applied to the post.
    """
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    source_type = post['source_type'] if 'source_type' in post.keys() else None
    source_content = post['source_content'] if 'source_content' in post.keys() else None
    url = (source_content or '').strip()
    if source_type not in ('saved_source', 'url') or not url.startswith(('http://', 'https://')):
        return jsonify({"error": "This post is not tied to a YouTube/article source."}), 400

    og_image = None
    if is_youtube_url(url):
        video_id = get_youtube_video_id(url)
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                og_image = info.get("thumbnail")
        except Exception:
            og_image = None
        if not og_image and video_id:
            og_image = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    else:
        og_image = fetch_og_image_for_url(url)

    if not og_image:
        return jsonify({
            "error": "Could not find an image at the source.",
            "source_url": url,
        }), 422

    try:
        _assert_safe_url(og_image)
    except UnsafeURLError as exc:
        return jsonify({"error": f"Source image rejected: {exc}", "source_url": url}), 422

    try:
        saved_url = save_stock_image_to_library(og_image)
    except (UnsafeURLError, RuntimeError, ValueError, requests.RequestException) as exc:
        app.logger.warning("Failed to save source image %s for post %s: %s", og_image, post_id, exc)
        return jsonify({
            "error": f"Failed to download source image: {exc}",
            "source_url": url,
            "og_image_url": og_image,
        }), 422

    update_standalone_post_image(post_id, saved_url)
    return jsonify({
        "success": True,
        "image_url": saved_url,
        "source_url": url,
        "og_image_url": og_image,
    })


def _maybe_attach_link_image(post_id: int, content: str | None, kind: str = 'standalone') -> None:
    """Schedule a background fetch of the first URL's og:image for the post.

    No-op if the post already has an image, has no URLs, or all fetches fail.
    Designed to be called from request handlers without blocking the response.
    `kind` is 'standalone' (compose) or 'social' (per-article posts).
    """
    if not content:
        return
    urls = extract_urls_from_text(content)
    if not urls:
        return

    if kind == 'social':
        loader, updater = get_social_post, update_social_post_image
    else:
        loader, updater = get_standalone_post, update_standalone_post_image

    def _worker(pid: int, candidate_urls: list) -> None:
        try:
            current = loader(pid)
            if not current:
                return
            if current['image_url'] if 'image_url' in current.keys() else None:
                return
            for candidate in candidate_urls:
                try:
                    _assert_safe_url(candidate)
                except UnsafeURLError as exc:
                    app.logger.info("Skipping unsafe URL %s for post %s: %s", candidate, pid, exc)
                    continue
                og_image = fetch_og_image_for_url(candidate)
                if not og_image:
                    continue
                try:
                    saved_url = save_stock_image_to_library(og_image)
                except (UnsafeURLError, RuntimeError, ValueError, requests.RequestException) as exc:
                    app.logger.info(
                        "Could not save og:image %s for post %s: %s", og_image, pid, exc
                    )
                    continue
                latest = loader(pid)
                latest_image = latest['image_url'] if latest and 'image_url' in latest.keys() else None
                if latest and not latest_image:
                    updater(pid, saved_url)
                    app.logger.info(
                        "Auto-attached og:image to %s post %s from %s", kind, pid, candidate
                    )
                return
        except Exception:
            app.logger.exception("Background link-image worker failed for post %s", pid)

    try:
        _link_image_executor.submit(_worker, post_id, urls)
    except RuntimeError:
        # Executor shut down (e.g. during reload); run inline as a best effort
        _worker(post_id, urls)


def _ensure_instagram_image(
    content: str | None,
    image_url: str | None,
    *,
    standalone_post_id: int | None = None,
    social_post_id: int | None = None,
) -> tuple[str | None, str | None]:
    """Instagram feed posts require an image; return (image_url, error).

    If the post has no image, try to attach a stock image based on the content
    and persist it back to the source post so the record reflects what gets
    published. error is set only when no image could be obtained.
    """
    if image_url:
        return image_url, None

    stock_url = None
    if content:
        try:
            stock_url = get_image_for_post(content)
        except Exception as exc:
            app.logger.warning("Stock image lookup for Instagram post failed: %s", exc)

    if stock_url:
        try:
            if standalone_post_id:
                update_standalone_post_image(standalone_post_id, stock_url)
            elif social_post_id:
                update_social_post_image(social_post_id, stock_url)
        except Exception as exc:
            app.logger.warning("Could not persist auto-attached Instagram image: %s", exc)
        return stock_url, None

    return None, (
        "Instagram requires an image and no stock image could be found. "
        "Attach an image to the post and retry."
    )


def _ensure_instagram_media(
    content: str | None,
    image_url: str | None,
    ig_post_type: str | None,
    media_items: list | None,
    *,
    standalone_post_id: int | None = None,
    social_post_id: int | None = None,
) -> tuple[list | None, str | None]:
    """Validate/prepare Instagram media by format. Returns (resolved_items, error).

    resolved_items is the list of {"url","kind"} to publish. error is set on failure.
    - feed: delegates to _ensure_instagram_image (stock auto-attach); 1 image.
    - carousel: 2-10 items.
    - reel: exactly one video item.
    - story: exactly one media item (image or video).
    """
    ig_post_type = ig_post_type or 'feed'
    items = [it for it in (media_items or []) if it]

    if ig_post_type == 'feed':
        url, err = _ensure_instagram_image(
            content, image_url,
            standalone_post_id=standalone_post_id, social_post_id=social_post_id,
        )
        if err:
            return None, err
        return [{"url": url, "kind": "image"}], None

    if ig_post_type == 'carousel':
        if not (2 <= len(items) <= 10):
            return None, "Instagram carousels need between 2 and 10 items. Add media and retry."
        if any(not it.get("url") for it in items):
            return None, "A carousel item is missing its media URL."
        return items, None

    if ig_post_type == 'reel':
        videos = [it for it in items if it.get("kind") == "video" and it.get("url")]
        if len(videos) != 1:
            return None, "Instagram Reels require exactly one video. Attach a video and retry."
        return [videos[0]], None

    if ig_post_type == 'story':
        valid = [it for it in items if it.get("url")]
        if len(valid) != 1:
            return None, "A story needs exactly one image or video. Attach one and retry."
        return [valid[0]], None

    return None, f"Unknown Instagram post type: {ig_post_type}"


def _instagram_publish_for_post(
    access_token: str,
    *,
    content: str | None,
    image_url: str | None,
    standalone_post_id: int | None = None,
    social_post_id: int | None = None,
) -> dict:
    """Publish an Instagram post honoring its media format (feed/carousel/reel/story).

    Reads ig_post_type + media_items from the standalone post (feed when absent),
    validates via _ensure_instagram_media, routes to the right client method, and
    returns the client result dict unchanged. On a validation failure the result
    carries ``guard_error: True`` so the worker can fail fast instead of retrying.
    """
    ig_post_type = 'feed'
    media_items: list = []
    user_tags: list = []
    if standalone_post_id:
        row = get_standalone_post(standalone_post_id)
        if row:
            row = dict(row)
            ig_post_type = row.get('ig_post_type') or 'feed'
            raw = row.get('media_items')
            if raw:
                try:
                    media_items = json.loads(raw)
                except (ValueError, TypeError):
                    media_items = []
            raw_tags = row.get('ig_user_tags')
            if raw_tags:
                try:
                    user_tags = json.loads(raw_tags)
                except (ValueError, TypeError):
                    user_tags = []

    resolved, err = _ensure_instagram_media(
        content, image_url, ig_post_type, media_items,
        standalone_post_id=standalone_post_id, social_post_id=social_post_id,
    )
    if err:
        return {"success": False, "error": {"message": err}, "friendly": err, "guard_error": True}

    client = get_instagram_client()
    caption = (content or "")[:2200]

    if ig_post_type == 'carousel':
        return client.publish_carousel_post(access_token, caption, resolved)
    if ig_post_type == 'reel':
        return client.publish_reel_post(access_token, caption, resolved[0]["url"])
    if ig_post_type == 'story':
        item = resolved[0]
        if item.get("kind") == "video":
            return client.publish_story_post(access_token, video_url=item["url"])
        return client.publish_story_post(access_token, image_url=item["url"])
    # feed (default)
    return client.publish_image_post(
        access_token, caption, resolved[0]["url"], user_tags=user_tags or None,
    )


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

    # Manual size cap for images (the global MAX_CONTENT_LENGTH is raised for video)
    file.seek(0, os.SEEK_END)
    if file.tell() > MAX_IMAGE_BYTES:
        file.seek(0)
        return jsonify({"error": "Image too large (max 16MB)."}), 400
    file.seek(0)

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
                folder="insights",
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


@app.route('/compose/upload-video', methods=['POST'])
def compose_upload_video():
    """Upload a video for Instagram Reels / video Stories / video carousel items.

    Requires Cloudinary (Instagram's fetcher can't reach local /static/uploads in
    most deploys). No transcoding — Instagram rejects wrong codec/aspect/duration,
    surfaced back through the publish flow. For large files, paste a public URL
    instead.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_video_file(file.filename):
        return jsonify({"error": f"Video type not allowed. Allowed types: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"}), 400

    if not CLOUDINARY_CONFIGURED:
        return jsonify({
            "error": "Video upload needs Cloudinary. Configure Cloudinary, or paste a "
                     "public video URL instead.",
        }), 400

    video_bytes = file.read()
    if not video_bytes:
        return jsonify({"error": "Empty video file"}), 400

    try:
        result = cloudinary.uploader.upload(
            video_bytes,
            folder="insights",
            resource_type="video",
        )
    except Exception as e:
        app.logger.error("Cloudinary video upload failed: %s", str(e))
        return jsonify({
            "error": f"Video upload failed: {e}. Try a smaller/shorter clip or paste a public URL.",
        }), 400

    video_url = result['secure_url']
    filename = result['public_id'].split('/')[-1]
    add_uploaded_image(
        filename=filename,
        url=video_url,
        storage='cloudinary',
        size=result.get('bytes', len(video_bytes)),
        media_type='video',
    )
    return jsonify({
        "success": True,
        "video_url": video_url,
        "filename": filename,
        "storage": "cloudinary",
    })


def _load_image_for_fit(url: str) -> bytes:
    """Load raw image bytes for the media-fit endpoint.

    Local /static/uploads URLs are read from disk; everything else goes
    through the SSRF-hardened fetcher.
    """
    parsed = urlparse(url)
    path = parsed.path or ''
    if path.startswith('/static/uploads/'):
        filename = os.path.basename(path)
        if not allowed_file(filename):
            raise ValueError("unsupported local image type")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(filepath):
            raise ValueError("local image not found")
        if os.path.getsize(filepath) > MAX_IMAGE_BYTES:
            raise ValueError("image too large (max 16MB)")
        with open(filepath, 'rb') as f:
            return f.read()
    data, _content_type, _final_url = _fetch_safely(
        url,
        max_bytes=MAX_IMAGE_BYTES,
        timeout=15,
        allowed_content_types=("image/",),
    )
    return data


def _ig_crop_to_canvas(img: Image.Image, tw: int, th: int,
                       focus_x: float = 0.5, focus_y: float = 0.5) -> Image.Image:
    """Scale the image to cover the canvas, then crop around the focus point."""
    w, h = img.size
    scale = max(tw / w, th / h)
    new_w = max(tw, round(w * scale))
    new_h = max(th, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = min(max(round(focus_x * new_w - tw / 2), 0), new_w - tw)
    top = min(max(round(focus_y * new_h - th / 2), 0), new_h - th)
    return img.crop((left, top, left + tw, top + th))


def _ig_pad_to_canvas(img: Image.Image, tw: int, th: int,
                      pad_style: str = 'blur') -> Image.Image:
    """Fit the image inside the canvas and fill the borders.

    pad_style 'blur' mimics Instagram's look (blurred, darkened cover of the
    same image); 'black'/'white' use a solid background.
    """
    w, h = img.size
    if pad_style == 'blur':
        canvas = _ig_crop_to_canvas(img, tw, th).filter(ImageFilter.GaussianBlur(40))
        canvas = ImageEnhance.Brightness(canvas).enhance(0.65)
    else:
        color = (0, 0, 0) if pad_style == 'black' else (255, 255, 255)
        canvas = Image.new('RGB', (tw, th), color)
    scale = min(tw / w, th / h)
    fit_w = max(1, round(w * scale))
    fit_h = max(1, round(h * scale))
    fitted = img.resize((fit_w, fit_h), Image.LANCZOS)
    canvas.paste(fitted, ((tw - fit_w) // 2, (th - fit_h) // 2))
    return canvas


def _store_fitted_image(jpeg_bytes: bytes, prefix: str) -> str:
    """Persist a processed JPEG to Cloudinary (if configured) or local uploads.

    Returns the URL to use for the post — absolute for local storage, matching
    the upload routes so Instagram's fetcher can reach it.
    """
    if CLOUDINARY_CONFIGURED:
        try:
            result = cloudinary.uploader.upload(
                jpeg_bytes,
                folder="insights",
                resource_type="image",
            )
            saved_url = result['secure_url']
            add_uploaded_image(
                filename=result['public_id'].split('/')[-1],
                url=saved_url,
                storage='cloudinary',
                size=result.get('bytes', len(jpeg_bytes)),
            )
            return saved_url
        except Exception as e:
            app.logger.error("Cloudinary upload of fitted image failed: %s", e)
            # Fall back to local storage
    unique_filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    with open(os.path.join(UPLOAD_FOLDER, unique_filename), 'wb') as f:
        f.write(jpeg_bytes)
    add_uploaded_image(
        filename=unique_filename,
        url=f"/static/uploads/{unique_filename}",
        storage='local',
        size=len(jpeg_bytes),
    )
    return f"{request.host_url}static/uploads/{unique_filename}"


@app.route('/compose/media/fit', methods=['POST'])
def compose_fit_media():
    """Resize an image to an exact Instagram canvas (crop or pad).
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
            url:
              type: string
              description: Source image URL (upload, library, or public URL)
            target:
              type: string
              enum: [story, reel, feed_square, feed_portrait, feed_landscape]
            mode:
              type: string
              enum: [crop, pad]
              description: crop = fill canvas and cut overflow; pad = fit inside and fill borders
            focus_x:
              type: number
              description: Crop anchor 0..1 (0.5 = center), crop mode only
            focus_y:
              type: number
            pad_style:
              type: string
              enum: [blur, black, white]
    responses:
      200:
        description: New image URL and dimensions
      400:
        description: Validation or processing error
    """
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()
    target = (data.get('target') or '').strip()
    mode = (data.get('mode') or 'crop').strip()
    pad_style = (data.get('pad_style') or 'blur').strip()

    if not url:
        return jsonify({"error": "url is required"}), 400
    if target not in IG_FIT_TARGETS:
        return jsonify({
            "error": f"target must be one of: {', '.join(sorted(IG_FIT_TARGETS))}"
        }), 400
    if mode not in ('crop', 'pad'):
        return jsonify({"error": "mode must be crop or pad"}), 400
    if pad_style not in ('blur', 'black', 'white'):
        return jsonify({"error": "pad_style must be blur, black, or white"}), 400
    try:
        focus_x = min(max(float(data.get('focus_x', 0.5)), 0.0), 1.0)
        focus_y = min(max(float(data.get('focus_y', 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        return jsonify({"error": "focus_x/focus_y must be numbers"}), 400

    try:
        raw = _load_image_for_fit(url)
    except (UnsafeURLError, ValueError, RuntimeError, requests.RequestException) as e:
        return jsonify({"error": f"Could not load image: {e}"}), 400

    tw, th = IG_FIT_TARGETS[target]
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert('RGB')
        if mode == 'crop':
            fitted = _ig_crop_to_canvas(img, tw, th, focus_x, focus_y)
        else:
            fitted = _ig_pad_to_canvas(img, tw, th, pad_style)
        output = io.BytesIO()
        fitted.save(output, format='JPEG', quality=90, optimize=True)
        jpeg_bytes = output.getvalue()
    except Exception as e:
        app.logger.error("Media fit processing failed for %s: %s", url, e)
        return jsonify({"error": f"Could not process image: {e}"}), 400

    try:
        new_url = _store_fitted_image(jpeg_bytes, f"igfit_{target}")
    except OSError as e:
        app.logger.error("Could not store fitted image: %s", e)
        return jsonify({"error": "Could not save the adjusted image."}), 500

    return jsonify({"success": True, "image_url": new_url, "width": tw, "height": th})


# ── Text overlays on story/feed media ──────────────────────────────────────

# Candidate fonts for baked-in text, best first (bold reads best on stories).
_OVERLAY_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",   # macOS
    "/System/Library/Fonts/Helvetica.ttc",                 # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",        # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_OVERLAY_BG_STYLES = {
    "none": None,
    "dark": (0, 0, 0, 153),        # 60% black — the classic story text pill
    "light": (255, 255, 255, 191),  # 75% white
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
}
MAX_OVERLAY_LAYERS = 10
MAX_VIDEO_FETCH_BYTES = 100 * 1024 * 1024  # matches MAX_CONTENT_LENGTH


def _find_overlay_font() -> str | None:
    """Return the first available TTF/TTC font path for text overlays."""
    for path in _OVERLAY_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _clean_overlay_layers(raw_layers) -> tuple[list | None, str | None]:
    """Validate/normalize annotate layers. Returns (layers, error)."""
    if not isinstance(raw_layers, list) or not raw_layers:
        return None, "layers must be a non-empty list"
    if len(raw_layers) > MAX_OVERLAY_LAYERS:
        return None, f"At most {MAX_OVERLAY_LAYERS} text layers are supported."
    clean = []
    for layer in raw_layers:
        if not isinstance(layer, dict):
            return None, "each layer must be an object"
        text = (layer.get("text") or "").replace("\r\n", "\n").strip("\n").rstrip()
        if not text or len(text) > 300:
            return None, "layer text must be 1-300 characters"
        color = (layer.get("color") or "#ffffff").strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            return None, f"invalid color: {color!r}"
        bg = (layer.get("bg") or "none").strip()
        if bg not in _OVERLAY_BG_STYLES:
            return None, f"invalid bg style: {bg!r}"
        try:
            x = min(max(float(layer.get("x", 0.5)), 0.0), 1.0)
            y = min(max(float(layer.get("y", 0.5)), 0.0), 1.0)
            size = min(max(float(layer.get("size", 0.06)), 0.02), 0.3)
        except (TypeError, ValueError):
            return None, "layer x/y/size must be numbers"
        clean.append({"text": text, "x": x, "y": y, "size": size, "color": color, "bg": bg})
    return clean, None


def _annotate_image(img: Image.Image, layers: list, font_path: str | None) -> Image.Image:
    """Bake text layers onto an image (centered at x/y, IG-style pill option)."""
    img = img.convert("RGB")
    W, H = img.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for layer in layers:
        font_px = max(12, round(layer["size"] * W))
        try:
            font = ImageFont.truetype(font_path, font_px) if font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        cx, cy = layer["x"] * W, layer["y"] * H
        spacing = round(font_px * 0.25)
        stroke_w = 0 if layer["bg"] != "none" else max(2, font_px // 22)
        kwargs = dict(font=font, anchor="mm", align="center", spacing=spacing)
        bbox = draw.multiline_textbbox((cx, cy), layer["text"], stroke_width=stroke_w, **kwargs)
        bg_fill = _OVERLAY_BG_STYLES[layer["bg"]]
        if bg_fill:
            pad_x, pad_y = round(font_px * 0.45), round(font_px * 0.28)
            draw.rounded_rectangle(
                [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y],
                radius=round(font_px * 0.35),
                fill=bg_fill,
            )
        draw.multiline_text(
            (cx, cy), layer["text"], fill=layer["color"],
            stroke_width=stroke_w, stroke_fill=(0, 0, 0, 210) if stroke_w else None,
            **kwargs,
        )
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _load_video_for_annotate(url: str, workdir: str) -> str:
    """Materialize the source video as a local file path inside workdir.

    Local /static/uploads URLs are used directly; remote URLs go through the
    SSRF-hardened fetcher. Raises ValueError/UnsafeURLError/RuntimeError.
    """
    parsed = urlparse(url)
    path = parsed.path or ""
    if path.startswith("/static/uploads/"):
        filename = os.path.basename(path)
        if not allowed_video_file(filename):
            raise ValueError("unsupported local video type")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(filepath):
            raise ValueError("local video not found")
        return filepath
    data, _content_type, _final = _fetch_safely(
        url,
        max_bytes=MAX_VIDEO_FETCH_BYTES,
        timeout=60,
        allowed_content_types=("video/", "application/octet-stream", "binary/"),
    )
    ext = ".mov" if path.lower().endswith(".mov") else ".mp4"
    local = os.path.join(workdir, f"src{ext}")
    with open(local, "wb") as f:
        f.write(data)
    return local


def _annotate_video(src_path: str, layers: list, font_path: str, workdir: str) -> str:
    """Burn text layers into a video with ffmpeg drawtext. Returns output path."""
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg is not installed on the server")

    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", src_path],
        capture_output=True, text=True, timeout=60,
    )
    try:
        w, _h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    except (ValueError, IndexError):
        raise RuntimeError(f"could not read video dimensions: {probe.stderr.strip()[:200]}")

    filters = []
    for i, layer in enumerate(layers):
        font_px = max(12, round(layer["size"] * w))
        # textfile= sidesteps drawtext's escaping rules entirely
        textfile = os.path.join(workdir, f"layer{i}.txt")
        with open(textfile, "w", encoding="utf-8") as f:
            f.write(layer["text"])
        parts = [
            f"fontfile='{font_path}'",
            f"textfile='{textfile}'",
            f"fontsize={font_px}",
            f"fontcolor=0x{layer['color'].lstrip('#')}",
            f"x=w*{layer['x']:.4f}-text_w/2",
            f"y=h*{layer['y']:.4f}-text_h/2",
            f"line_spacing={round(font_px * 0.25)}",
        ]
        bg = layer["bg"]
        if bg != "none":
            boxcolor = {
                "dark": "black@0.6", "light": "white@0.75",
                "black": "black", "white": "white",
            }[bg]
            parts += ["box=1", f"boxcolor={boxcolor}", f"boxborderw={round(font_px * 0.35)}"]
        else:
            shadow = max(1, font_px // 22)
            parts += ["shadowcolor=black@0.7", f"shadowx={shadow}", f"shadowy={shadow}"]
        filters.append("drawtext=" + ":".join(parts))

    out_path = os.path.join(workdir, "annotated.mp4")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", src_path, "-vf", ",".join(filters),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "copy", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
         out_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0 or not os.path.isfile(out_path):
        app.logger.error("ffmpeg drawtext failed: %s", result.stderr[-800:])
        raise RuntimeError("video text rendering failed (see server log)")
    return out_path


def _store_annotated_video(video_path: str) -> str:
    """Persist an annotated video like an upload; returns its serving URL."""
    size = os.path.getsize(video_path)
    if CLOUDINARY_CONFIGURED:
        result = cloudinary.uploader.upload(
            video_path, folder="insights", resource_type="video",
        )
        video_url = result["secure_url"]
        add_uploaded_image(
            filename=result["public_id"].split("/")[-1],
            url=video_url,
            storage="cloudinary",
            size=result.get("bytes", size),
            media_type="video",
        )
        return video_url
    unique_filename = f"igtext_{uuid.uuid4().hex}.mp4"
    dest = os.path.join(UPLOAD_FOLDER, unique_filename)
    shutil.copyfile(video_path, dest)
    add_uploaded_image(
        filename=unique_filename,
        url=f"/static/uploads/{unique_filename}",
        storage="local",
        size=size,
        media_type="video",
    )
    return f"{request.host_url}static/uploads/{unique_filename}"


@app.route('/compose/media/annotate', methods=['POST'])
def compose_annotate_media():
    """Bake text layers onto an image or burn them into a video.
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
            url:
              type: string
              description: Source media URL (upload, library, or public URL)
            kind:
              type: string
              enum: [image, video]
            layers:
              type: array
              description: >
                Text layers: {text, x, y (0..1 center), size (fraction of
                media width), color (#rrggbb), bg (none|dark|light|black|white)}
    responses:
      200:
        description: New media URL
      400:
        description: Validation or processing error
    """
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    kind = (data.get("kind") or "image").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    if kind not in ("image", "video"):
        return jsonify({"error": "kind must be image or video"}), 400
    layers, err = _clean_overlay_layers(data.get("layers"))
    if err:
        return jsonify({"error": err}), 400

    font_path = _find_overlay_font()

    if kind == "image":
        try:
            raw = _load_image_for_fit(url)
            img = Image.open(io.BytesIO(raw))
            img.load()
            annotated = _annotate_image(img, layers, font_path)
            output = io.BytesIO()
            annotated.save(output, format="JPEG", quality=90, optimize=True)
        except (UnsafeURLError, ValueError, RuntimeError, requests.RequestException) as e:
            return jsonify({"error": f"Could not load image: {e}"}), 400
        except Exception as e:
            app.logger.error("Image annotate failed for %s: %s", url, e)
            return jsonify({"error": f"Could not add text: {e}"}), 400
        try:
            new_url = _store_fitted_image(output.getvalue(), "igtext")
        except OSError as e:
            app.logger.error("Could not store annotated image: %s", e)
            return jsonify({"error": "Could not save the image."}), 500
        return jsonify({"success": True, "image_url": new_url, "kind": "image"})

    # Video path — needs ffmpeg on the server
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return jsonify({
            "error": "Adding text to video needs ffmpeg installed on the server "
                     "(brew install ffmpeg).",
        }), 400
    if not font_path:
        return jsonify({"error": "No usable system font found for video text."}), 400
    workdir = tempfile.mkdtemp(prefix="igtext_")
    try:
        try:
            src = _load_video_for_annotate(url, workdir)
            out_path = _annotate_video(src, layers, font_path, workdir)
        except (UnsafeURLError, ValueError, RuntimeError, requests.RequestException,
                subprocess.TimeoutExpired) as e:
            return jsonify({"error": f"Could not add text to video: {e}"}), 400
        try:
            new_url = _store_annotated_video(out_path)
        except Exception as e:
            app.logger.error("Could not store annotated video: %s", e)
            return jsonify({"error": f"Could not save the video: {e}"}), 500
        return jsonify({"success": True, "video_url": new_url, "kind": "video"})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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


@app.route('/compose/posts/bulk-toggle-used', methods=['POST'])
def compose_bulk_toggle_used():
    """Bulk mark/unmark standalone posts as used."""
    data = request.get_json()
    post_ids = data.get('post_ids', [])
    used = data.get('used', True)

    if not post_ids:
        return jsonify({"error": "No posts selected"}), 400

    post_ids = [int(pid) for pid in post_ids]
    updated = 0
    for post_id in post_ids:
        post = get_standalone_post(post_id)
        if post:
            mark_standalone_post_used(post_id, used)
            updated += 1

    return jsonify({
        "success": True,
        "updated_count": updated,
        "used": used,
    })


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


@app.route('/compose/post/<int:post_id>/instagram', methods=['POST'])
def compose_post_to_instagram(post_id: int):
    """Post a standalone post to Instagram immediately."""
    post = get_standalone_post(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    # Get Instagram token
    token = get_instagram_token()
    if not token:
        return jsonify({"error": "Instagram not connected. Please connect your account first."}), 401

    # Check if token is expired and try to refresh
    if instagram_is_token_expired(token['expires_at']):
        client = get_instagram_client()
        try:
            new_token = client.refresh_access_token(token['access_token'])
            expires_at = instagram_calculate_token_expiry(new_token.get('expires_in', 5184000))
            update_instagram_token(
                access_token=new_token['access_token'],
                expires_at=expires_at,
            )
            token = get_instagram_token()
        except Exception as e:
            app.logger.warning("Failed to refresh Instagram token: %s", e)
            return jsonify({"error": "Instagram token expired. Please reconnect."}), 401

    client = get_instagram_client()

    try:
        # Publish honoring the post's Instagram format (feed/carousel/reel/story)
        image_url = post['image_url'] if 'image_url' in post.keys() else None
        result = _instagram_publish_for_post(
            token['access_token'],
            content=post['content'],
            image_url=image_url,
            standalone_post_id=post_id,
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
                platform='instagram',
                scheduled_for=now,
                status='posted',
                linkedin_post_urn=result.get('permalink'),  # Store permalink for view link
            )

            return jsonify({
                "success": True,
                "post_id": result.get('post_id'),
                "permalink": result.get('permalink'),
                "message": "Posted to Instagram successfully!",
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('friendly') or result.get('error', 'Unknown error'),
            }), 400

    except Exception as e:
        app.logger.exception("Failed to post to Instagram")
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
    if platform not in ['linkedin', 'threads', 'facebook', 'twitter', 'instagram']:
        return jsonify({"error": f"Platform {platform} does not support scheduling yet"}), 400

    # Instagram posts require media; validate by format (and auto-attach a stock
    # image for feed) now so the user can review or fix it before the scheduled time.
    if platform == 'instagram':
        existing_image = post['image_url'] if 'image_url' in post.keys() else None
        ig_post_type = post['ig_post_type'] if 'ig_post_type' in post.keys() else None
        raw_items = post['media_items'] if 'media_items' in post.keys() else None
        try:
            media_items = json.loads(raw_items) if raw_items else []
        except (ValueError, TypeError):
            media_items = []
        _, media_err = _ensure_instagram_media(
            post['content'],
            existing_image,
            ig_post_type,
            media_items,
            standalone_post_id=post_id,
        )
        if media_err:
            return jsonify({"error": media_err}), 400

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
    post_counts = count_standalone_posts_by_source_urls(
        [s['url'] for s in sources if s['url']]
    )
    sources_with_status = []
    for s in sources:
        s_dict = dict(s)
        s_dict['is_youtube'] = is_youtube_url(s_dict.get('url', ''))
        s_dict['is_github'] = is_github_repo_url(s_dict.get('url', ''))
        s_dict['post_count'] = post_counts.get(s_dict.get('url', ''), 0)
        if s_dict['is_youtube']:
            ep = get_episode(s_dict['url'])
            if ep:
                s_dict['episode_status'] = ep['status']
                s_dict['episode_id'] = ep['id']
            else:
                s_dict['episode_status'] = None
                s_dict['episode_id'] = None
        sources_with_status.append(s_dict)
    return render_template('sources.html', sources=sources_with_status)


@app.route('/sources/<int:source_id>/posts')
def source_posts(source_id: int):
    """Return the standalone posts generated from a saved source as JSON."""
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404

    posts = list_standalone_posts_by_source_url(source['url'])
    post_ids = [p['id'] for p in posts]
    posted_info = get_posted_info_for_standalone_posts(post_ids) if post_ids else {}

    result = []
    for p in posts:
        p_keys = p.keys()
        platform = p['platform']
        posted = None
        platform_posted = posted_info.get(p['id'], {}).get(platform)
        if platform_posted:
            posted = {
                'url': platform_posted.get('url'),
                'posted_at': platform_posted.get('posted_at'),
            }
        result.append({
            'id': p['id'],
            'platform': platform,
            'content': p['content'],
            'image_url': p['image_url'] if 'image_url' in p_keys else None,
            'created_at': p['created_at'],
            'used': bool(p['used']) if 'used' in p_keys else False,
            'posted': posted,
        })

    return jsonify({
        'source': {'id': source['id'], 'url': source['url'], 'title': source['title']},
        'posts': result,
    })


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
        if is_youtube_url(url):
            video_id = get_youtube_video_id(url)
            title, description, og_image = url, "", None
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title", title)
                    description = info.get("description", "")
                    og_image = info.get("thumbnail")
            except Exception:
                pass
            if not og_image and video_id:
                og_image = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            source_id = add_url_source(
                url=url, title=title,
                description=description, content="", og_image=og_image,
            )
            return jsonify({
                "success": True,
                "is_youtube": True,
                "source": {
                    "id": source_id,
                    "url": url,
                    "title": title,
                    "description": description,
                    "content": "",
                    "og_image": og_image,
                }
            })

        if is_github_repo_url(url):
            owner, repo_name = parse_github_repo_url(url)
            repo_data = fetch_github_repo(owner, repo_name)
            source_id = add_url_source(
                url=url,
                title=repo_data['title'],
                description=repo_data['description'],
                content=repo_data['content'],
                og_image=repo_data['og_image'],
            )
            return jsonify({
                "success": True,
                "is_github": True,
                "source": {
                    "id": source_id,
                    "url": url,
                    "title": repo_data['title'],
                    "description": repo_data['description'],
                    "content": repo_data['content'],
                    "og_image": repo_data['og_image'],
                }
            })

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


@app.route('/sources/<int:source_id>/transcribe', methods=['POST'])
def transcribe_source(source_id: int):
    """Queue a YouTube source for background audio transcription."""
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    if not is_youtube_url(source['url']):
        return jsonify({"error": "Only YouTube sources can be transcribed"}), 400
    existing = get_episode(source['url'])
    if existing and existing['status'] in ('queued', 'processing'):
        return redirect(url_for('status_page'))
    queue_episode(source['url'], source['title'] or source['url'], None, None)
    task_queue.put({
        "url": source['url'],
        "title": source['title'] or source['url'],
        "feed_id": None,
        "published": None,
    })
    return redirect(url_for('status_page'))


@app.route('/sources/<int:source_id>/reextract', methods=['POST'])
def reextract_source(source_id: int):
    """Re-extract content from a URL source using improved extraction."""
    import trafilatura
    
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    
    url = source['url']
    
    try:
        if is_youtube_url(url):
            video_id = get_youtube_video_id(url)
            title, description, og_image = source['title'] or url, source['description'] or "", source['og_image']
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title", title)
                    description = info.get("description", "")
                    og_image = info.get("thumbnail")
            except Exception:
                pass
            if not og_image and video_id:
                og_image = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            updated = update_url_source_content(
                source_id=source_id,
                title=title,
                description=description,
                content="",
                og_image=og_image,
            )
            if not updated:
                return jsonify({"error": "Failed to update source"}), 500
            return jsonify({
                "success": True,
                "source": {
                    "id": source_id,
                    "url": url,
                    "title": title,
                    "description": description,
                    "content": "",
                    "og_image": og_image,
                }
            })

        if is_github_repo_url(url):
            owner, repo_name = parse_github_repo_url(url)
            repo_data = fetch_github_repo(owner, repo_name)
            updated = update_url_source_content(
                source_id=source_id,
                title=repo_data['title'],
                description=repo_data['description'],
                content=repo_data['content'],
                og_image=repo_data['og_image'],
            )
            if not updated:
                return jsonify({"error": "Failed to update source"}), 500
            return jsonify({
                "success": True,
                "source": {
                    "id": source_id,
                    "url": url,
                    "title": repo_data['title'],
                    "description": repo_data['description'],
                    "content": repo_data['content'],
                    "og_image": repo_data['og_image'],
                }
            })

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
    posts_per_platform = request.form.get('posts_per_platform', 10, type=int)
    extra_context = request.form.get('extra_context', '').strip() or None
    image_url = request.form.get('image_url', '').strip() or None
    ai_provider, ai_model, use_local = _parse_ai_provider(request.form)

    if not source_id:
        return jsonify({"error": "Source ID is required"}), 400
    
    source = get_url_source(source_id)
    if not source:
        return jsonify({"error": "Source not found"}), 404
    
    if not image_url and source['og_image'] and not is_github_repo_url(source['url']):
        image_url = source['og_image']

    if is_youtube_url(source['url']):
        credit = f"IMPORTANT: Include this YouTube video URL as a reference/credit in every post: {source['url']}"
        extra_context = f"{credit}\n{extra_context}" if extra_context else credit
    elif is_github_repo_url(source['url']):
        credit = f"This content is from the GitHub repository {source['title']}. Include the repo URL ({source['url']}) as credit."
        extra_context = f"{credit}\n{extra_context}" if extra_context else credit

    if not platforms:
        platforms = ['linkedin', 'threads', 'twitter']
    
    posts_per_platform = max(1, min(posts_per_platform, 10))
    
    try:
        from insights import generate_posts_from_text
        
        source_text = f"TITLE: {source['title']}\n\n"
        if source['description']:
            source_text += f"DESCRIPTION: {source['description']}\n\n"
        source_text += f"CONTENT: {source['content']}\n\n"
        source_text += f"ORIGINAL URL: {source['url']}"
        
        generated = generate_posts_from_text(
            text=source_text,
            platforms=platforms,
            tone=tone,
            topic=source['title'],
            posts_per_platform=posts_per_platform,
            extra_context=extra_context,
            use_local=use_local,
            source_url=source['url'],
            provider=ai_provider,
            model=ai_model,
        )

        update_url_source_last_used(source_id)
        
        # Normalize platform names and filter hallucinated platforms
        requested_platforms = {p.lower() for p in platforms}
        platform_aliases = {"x": "twitter"}
        saved_posts = {}
        for platform, post_data in generated.items():
            if platform == 'raw':
                continue
            
            norm_platform = platform.lower().strip()
            norm_platform = platform_aliases.get(norm_platform, norm_platform)

            if norm_platform not in requested_platforms:
                continue

            posts_list = post_data if isinstance(post_data, list) else [post_data]
            if norm_platform not in saved_posts:
                saved_posts[norm_platform] = []
            
            for post_content in posts_list:
                post_id = add_standalone_post(
                    source_type='saved_source',
                    source_content=source['url'][:1000],
                    platform=norm_platform,
                    content=post_content,
                    image_url=image_url,
                )
                saved_posts[norm_platform].append({
                    'id': post_id,
                    'content': post_content,
                    'image_url': image_url,
                })
                if not image_url:
                    _maybe_attach_link_image(post_id, post_content)
        
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
# YouTube Thumbnail Generator
# ============================================================================


THUMBNAIL_UPLOAD_DIR = os.path.join(UPLOAD_FOLDER, 'youtube_thumbnails')
os.makedirs(THUMBNAIL_UPLOAD_DIR, exist_ok=True)


@app.route('/thumbnails')
def thumbnails_page():
    """YouTube Thumbnail Generator page."""
    return render_template('thumbnails.html')


@app.route('/thumbnails/metadata', methods=['POST'])
def thumbnails_fetch_metadata():
    """Fetch metadata for a YouTube video URL."""
    url = (request.form.get('url') or '').strip()
    if not url:
        return jsonify({"error": "YouTube URL is required"}), 400

    if not is_youtube_url(url):
        return jsonify({"error": "Please enter a valid YouTube URL"}), 400

    if classify_youtube_url(url) != 'video':
        return jsonify({"error": "Please enter a single YouTube video URL (not a channel or playlist)"}), 400

    try:
        metadata = fetch_youtube_metadata(url)
        return jsonify({"success": True, "metadata": metadata})
    except Exception as e:
        app.logger.exception("Failed to fetch YouTube metadata")
        return jsonify({"error": str(e)}), 500


@app.route('/thumbnails/suggest-prompt', methods=['POST'])
def thumbnails_suggest_prompt():
    """Return the auto-generated prompt for given URL / aspect / style (no image cost)."""
    url = (request.form.get('url') or '').strip()
    aspect = request.form.get('aspect', '16:9')
    style = request.form.get('style', 'bold')

    if not url:
        return jsonify({"error": "YouTube URL is required"}), 400
    if not is_youtube_url(url):
        return jsonify({"error": "Please enter a valid YouTube URL"}), 400
    if aspect not in ('16:9', '9:16'):
        return jsonify({"error": "Aspect ratio must be 16:9 or 9:16"}), 400
    if style not in ('bold', 'minimal', 'cinematic'):
        return jsonify({"error": "Style must be bold, minimal, or cinematic"}), 400

    try:
        metadata = fetch_youtube_metadata(url)
        prompt = suggested_thumbnail_prompt(metadata, aspect, style)
        return jsonify({"success": True, "prompt": prompt})
    except Exception as e:
        app.logger.exception("Failed to build suggested prompt")
        return jsonify({"error": str(e)}), 500


@app.route('/thumbnails/generate', methods=['POST'])
def thumbnails_generate():
    """Generate a thumbnail for a YouTube video, persist it, and return the result."""
    import base64 as b64module
    import uuid as uuid_mod

    url = (request.form.get('url') or '').strip()
    aspect = request.form.get('aspect', '16:9')
    style = request.form.get('style', 'bold')
    custom_prompt = (request.form.get('prompt') or '').strip() or None

    if not url:
        return jsonify({"error": "YouTube URL is required"}), 400
    if not is_youtube_url(url):
        return jsonify({"error": "Please enter a valid YouTube URL"}), 400
    if aspect not in ('16:9', '9:16'):
        return jsonify({"error": "Aspect ratio must be 16:9 or 9:16"}), 400
    if style not in ('bold', 'minimal', 'cinematic'):
        return jsonify({"error": "Style must be bold, minimal, or cinematic"}), 400

    try:
        result = generate_youtube_thumbnail(
            url=url,
            aspect=aspect,
            style=style,
            custom_prompt=custom_prompt,
        )

        # Persist the PNG to disk
        filename = f"{uuid_mod.uuid4().hex}.png"
        filepath = os.path.join(THUMBNAIL_UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(b64module.b64decode(result["image_base64"]))

        image_relpath = f"static/uploads/youtube_thumbnails/{filename}"
        image_url = f"/static/uploads/youtube_thumbnails/{filename}"

        meta = result["metadata"]
        saved_id = add_generated_thumbnail(
            youtube_url=url,
            video_id=meta.get("video_id"),
            title=meta.get("title", ""),
            channel=meta.get("channel", ""),
            aspect=aspect,
            style=style,
            prompt=result["prompt"],
            image_relpath=image_relpath,
        )

        return jsonify({
            "success": True,
            "image_base64": result["image_base64"],
            "prompt": result["prompt"],
            "metadata": meta,
            "saved_id": saved_id,
            "image_url": image_url,
        })
    except Exception as e:
        app.logger.exception("Failed to generate thumbnail")
        return jsonify({"error": str(e)}), 500


@app.route('/thumbnails/saved')
def thumbnails_saved_list():
    """Return all saved thumbnails as JSON."""
    rows = list_generated_thumbnails(limit=100)
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "youtube_url": r["youtube_url"],
            "video_id": r["video_id"],
            "title": r["title"],
            "channel": r["channel"],
            "aspect": r["aspect"],
            "style": r["style"],
            "prompt": r["prompt"],
            "image_url": f"/{r['image_relpath']}" if r["image_relpath"] else None,
            "created_at": r["created_at"],
        })
    return jsonify({"success": True, "thumbnails": items})


@app.route('/thumbnails/saved/<int:thumb_id>', methods=['DELETE'])
def thumbnails_saved_delete(thumb_id: int):
    """Delete a saved thumbnail (DB row + file on disk)."""
    row = get_generated_thumbnail(thumb_id)
    if not row:
        return jsonify({"error": "Thumbnail not found"}), 404

    if row["image_relpath"]:
        try:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), row["image_relpath"])
            if os.path.isfile(filepath):
                os.remove(filepath)
        except OSError:
            app.logger.warning("Could not delete thumbnail file: %s", row["image_relpath"])

    delete_generated_thumbnail(thumb_id)
    return jsonify({"success": True})


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
            facebook_token = None
            twitter_token = None
            instagram_token = None
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
                    
                    elif platform == 'facebook':
                        # Handle Facebook posting
                        if facebook_token is None:
                            facebook_token = get_facebook_token()
                        
                        if not facebook_token:
                            app.logger.warning("Scheduled Facebook post %d due but Facebook not connected", post['id'])
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message='Facebook not connected',
                            )
                            continue
                        
                        if not facebook_token['page_id'] or not facebook_token['page_access_token']:
                            app.logger.warning("Scheduled Facebook post %d due but no Page selected", post['id'])
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message='No Facebook Page selected',
                            )
                            continue
                        
                        if facebook_is_token_expired(facebook_token['expires_at']):
                            fb_client = get_facebook_client()
                            try:
                                new_token = fb_client.refresh_access_token(facebook_token['access_token'])
                                expires_at = facebook_calculate_token_expiry(new_token.get('expires_in', 5184000))
                                update_facebook_token(
                                    access_token=new_token['access_token'],
                                    expires_at=expires_at,
                                )
                                facebook_token = get_facebook_token()
                            except Exception as e:
                                app.logger.error("Failed to refresh Facebook token: %s", e)
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message='Facebook token expired',
                                )
                                continue
                        
                        fb_client = get_facebook_client()
                        
                        result = fb_client.publish_smart_post(
                            page_access_token=facebook_token['page_access_token'],
                            page_id=facebook_token['page_id'],
                            text=content[:5000],
                            image_url=image_url,
                        )
                        
                        if result['success']:
                            update_scheduled_post_status(
                                post['id'],
                                status='posted',
                                linkedin_post_urn=result.get('permalink'),
                            )
                            if post['social_post_id']:
                                mark_social_post_used(post['social_post_id'], True)
                            if post['standalone_post_id']:
                                mark_standalone_post_used(post['standalone_post_id'], True)
                            app.logger.info("Scheduled Facebook post %d published successfully", post['id'])
                        else:
                            error_msg = str(result.get('error', 'Unknown error'))[:500]
                            retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                            if retry_count < MAX_RETRIES:
                                increment_retry_count(post['id'])
                                redistribute_scheduled_posts(platform)
                                redistributed_platforms.add(platform)
                                app.logger.warning(
                                    "Facebook post %d failed (%s), rescheduled to next slot (retry %d/%d)",
                                    post['id'], error_msg, retry_count + 1, MAX_RETRIES,
                                )
                                break
                            else:
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message=f'{error_msg} (max retries exhausted)',
                                )
                                app.logger.error("Scheduled Facebook post %d failed permanently: %s", post['id'], error_msg)
                    
                    elif platform == 'twitter':
                        if twitter_token is None:
                            twitter_token = get_twitter_token()
                        
                        if not twitter_token:
                            app.logger.warning("Scheduled Twitter post %d due but Twitter not connected", post['id'])
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message='Twitter not connected',
                            )
                            continue
                        
                        if twitter_is_token_expired(twitter_token['expires_at']):
                            tw_client = get_twitter_client()
                            try:
                                new_token = tw_client.refresh_access_token(twitter_token['refresh_token'])
                                expires_at = twitter_calculate_token_expiry(new_token.get('expires_in', 7200))
                                update_twitter_token(
                                    access_token=new_token['access_token'],
                                    expires_at=expires_at,
                                    refresh_token=new_token.get('refresh_token'),
                                )
                                twitter_token = get_twitter_token()
                            except Exception as e:
                                app.logger.error("Failed to refresh Twitter token: %s", e)
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message='Twitter token expired',
                                )
                                continue
                        
                        tw_client = get_twitter_client()
                        
                        if image_url:
                            app.logger.info("Posting Twitter with image: %s", image_url)
                            result = tw_client.create_image_post(
                                access_token=twitter_token['access_token'],
                                text=content[:280],
                                image_url=image_url,
                            )
                        else:
                            result = tw_client.create_post(
                                access_token=twitter_token['access_token'],
                                text=content[:280],
                            )
                        
                        if result['success']:
                            update_scheduled_post_status(
                                post['id'],
                                status='posted',
                                linkedin_post_urn=result.get('permalink'),
                            )
                            if post['social_post_id']:
                                mark_social_post_used(post['social_post_id'], True)
                            if post['standalone_post_id']:
                                mark_standalone_post_used(post['standalone_post_id'], True)
                            app.logger.info("Scheduled Twitter post %d published successfully", post['id'])
                        else:
                            error_msg = str(result.get('error', 'Unknown error'))[:500]
                            retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                            if retry_count < MAX_RETRIES:
                                increment_retry_count(post['id'])
                                redistribute_scheduled_posts(platform)
                                redistributed_platforms.add(platform)
                                app.logger.warning(
                                    "Twitter post %d failed (%s), rescheduled to next slot (retry %d/%d)",
                                    post['id'], error_msg, retry_count + 1, MAX_RETRIES,
                                )
                                break
                            else:
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message=f'{error_msg} (max retries exhausted)',
                                )
                                app.logger.error("Scheduled Twitter post %d failed permanently: %s", post['id'], error_msg)

                    elif platform == 'instagram':
                        if instagram_token is None:
                            instagram_token = get_instagram_token()

                        if not instagram_token:
                            app.logger.warning("Scheduled Instagram post %d due but Instagram not connected", post['id'])
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message='Instagram not connected',
                            )
                            continue

                        if instagram_is_token_expired(instagram_token['expires_at']):
                            ig_client = get_instagram_client()
                            try:
                                new_token = ig_client.refresh_access_token(instagram_token['access_token'])
                                expires_at = instagram_calculate_token_expiry(new_token.get('expires_in', 5184000))
                                update_instagram_token(
                                    access_token=new_token['access_token'],
                                    expires_at=expires_at,
                                )
                                instagram_token = get_instagram_token()
                            except Exception as e:
                                app.logger.error("Failed to refresh Instagram token: %s", e)
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message='Instagram token expired',
                                )
                                continue

                        # Publish honoring the post's Instagram format
                        # (feed/carousel/reel/story). A media-validation failure
                        # (guard_error) fails immediately with no retry — a missing
                        # image/video won't fix itself.
                        result = _instagram_publish_for_post(
                            instagram_token['access_token'],
                            content=content,
                            image_url=image_url,
                            standalone_post_id=post['standalone_post_id'],
                            social_post_id=post['social_post_id'],
                        )

                        if result.get('guard_error'):
                            media_err = result.get('friendly', 'Instagram media requirement not met')
                            app.logger.warning("Scheduled Instagram post %d media invalid: %s", post['id'], media_err)
                            update_scheduled_post_status(
                                post['id'],
                                status='failed',
                                error_message=media_err,
                            )
                            continue

                        if result['success']:
                            update_scheduled_post_status(
                                post['id'],
                                status='posted',
                                linkedin_post_urn=result.get('permalink'),
                            )
                            if post['social_post_id']:
                                mark_social_post_used(post['social_post_id'], True)
                            if post['standalone_post_id']:
                                mark_standalone_post_used(post['standalone_post_id'], True)
                            app.logger.info("Scheduled Instagram post %d published successfully", post['id'])
                        else:
                            error_msg = str(result.get('friendly') or result.get('error', 'Unknown error'))[:500]
                            retry_count = post['retry_count'] if 'retry_count' in post.keys() and post['retry_count'] else 0
                            if retry_count < MAX_RETRIES:
                                increment_retry_count(post['id'])
                                redistribute_scheduled_posts(platform)
                                redistributed_platforms.add(platform)
                                app.logger.warning(
                                    "Instagram post %d failed (%s), rescheduled to next slot (retry %d/%d)",
                                    post['id'], error_msg, retry_count + 1, MAX_RETRIES,
                                )
                                break
                            else:
                                update_scheduled_post_status(
                                    post['id'],
                                    status='failed',
                                    error_message=f'{error_msg} (max retries exhausted)',
                                )
                                app.logger.error("Scheduled Instagram post %d failed permanently: %s", post['id'], error_msg)

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


def _backfill_youtube_channels_bg():
    """One-time background task to populate channel for existing YouTube episodes."""
    episodes = get_youtube_episodes_missing_channel()
    if not episodes:
        return
    app.logger.info("Backfilling channel for %d YouTube episodes", len(episodes))
    for ep in episodes:
        try:
            meta = fetch_youtube_metadata(ep["url"])
            channel = meta.get("channel") or None
            if channel:
                set_episode_channel(ep["id"], channel)
        except Exception:
            continue
    app.logger.info("YouTube channel backfill complete")


# ── Content Agent (agentic content preparation) ──────────────────────────────

def _parse_brief_form(form) -> dict:
    """Extract + validate content-brief fields from a POST form."""
    def _b(name: str) -> bool:
        return form.get(name, 'false').lower() in ('true', '1', 'yes', 'on')

    keywords = [k.strip() for k in re.split(r'[\n,]', form.get('must_include_keywords', '')) if k.strip()]
    focus = [s.strip() for s in re.split(r'[\n,]', form.get('focus_sources', '')) if s.strip()]
    run_days = [int(x) for x in form.getlist('run_days') if x.isdigit()]
    ppp = form.get('posts_per_platform', 3, type=int) or 3
    return {
        'name': (form.get('name') or '').strip() or 'Untitled brief',
        'instructions': (form.get('instructions') or '').strip(),
        'content_type': form.get('content_type', 'posts'),
        'platforms': form.getlist('platforms'),
        'tone': form.get('tone', 'professional'),
        'posts_per_platform': max(1, min(ppp, 10)),
        'article_count': max(0, form.get('article_count', 0, type=int) or 0),
        'article_style': form.get('article_style', 'blog'),
        'focus_sources': focus,
        'must_include_keywords': keywords,
        'audience_persona': (form.get('audience_persona') or '').strip() or None,
        'use_web_search': _b('use_web_search'),
        'use_saved_sources': _b('use_saved_sources'),
        'cadence': form.get('cadence', 'manual'),
        'run_time': (form.get('run_time') or '').strip() or None,
        'run_days': run_days,
        'auto_queue': _b('auto_queue'),
        'review_window_hours': max(0, form.get('review_window_hours', 24, type=int) or 24),
        'max_sources_per_run': max(1, form.get('max_sources_per_run', 5, type=int) or 5),
        'max_cost_usd': max(0.0, form.get('max_cost_usd', 0.5, type=float) or 0.5),
        'max_drafts_per_run': max(1, form.get('max_drafts_per_run', 30, type=int) or 30),
    }


@app.route('/briefs')
def briefs_page():
    """List content briefs with their latest run status."""
    briefs = []
    for row in list_content_briefs():
        d = dict(row)
        try:
            d['platforms_list'] = json.loads(d.get('platforms') or '[]')
        except Exception:
            d['platforms_list'] = []
        try:
            d['keywords_list'] = json.loads(d.get('must_include_keywords') or '[]')
        except Exception:
            d['keywords_list'] = []
        try:
            d['focus_list'] = json.loads(d.get('focus_sources') or '[]')
        except Exception:
            d['focus_list'] = []
        try:
            d['run_days_list'] = json.loads(d.get('run_days') or '[]')
        except Exception:
            d['run_days_list'] = []
        runs = list_brief_runs(d['id'], limit=1)
        d['last_run'] = dict(runs[0]) if runs else None
        briefs.append(d)
    return render_template('briefs.html', briefs=briefs)


@app.route('/briefs/create', methods=['POST'])
def brief_create():
    data = _parse_brief_form(request.form)
    if not data['instructions']:
        return jsonify({"error": "Instructions (the prompt) are required"}), 400
    brief_id = create_content_brief(**data)
    if data['cadence'] != 'manual':
        set_content_brief_schedule(brief_id, next_run_at=_compute_next_run_at(get_content_brief(brief_id)))
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"success": True, "id": brief_id})
    return redirect(url_for('briefs_page'))


@app.route('/briefs/<int:brief_id>/edit', methods=['POST'])
def brief_edit(brief_id):
    if not get_content_brief(brief_id):
        return jsonify({"error": "Brief not found"}), 404
    data = _parse_brief_form(request.form)
    update_content_brief(brief_id, **data)
    next_at = _compute_next_run_at(get_content_brief(brief_id)) if data['cadence'] != 'manual' else None
    set_content_brief_schedule(brief_id, next_run_at=next_at)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"success": True, "id": brief_id})
    return redirect(url_for('briefs_page'))


@app.route('/briefs/<int:brief_id>/delete', methods=['POST'])
def brief_delete(brief_id):
    delete_content_brief(brief_id)
    return jsonify({"success": True})


@app.route('/briefs/<int:brief_id>/toggle', methods=['POST'])
def brief_toggle(brief_id):
    brief = get_content_brief(brief_id)
    if not brief:
        return jsonify({"error": "Brief not found"}), 404
    enabled = request.form.get('enabled', 'true').lower() in ('1', 'true', 'yes', 'on')
    set_content_brief_enabled(brief_id, enabled)
    if enabled and (dict(brief).get('cadence') or 'manual') != 'manual':
        set_content_brief_schedule(brief_id, next_run_at=_compute_next_run_at(brief))
    return jsonify({"success": True, "enabled": enabled})


@app.route('/briefs/<int:brief_id>/run', methods=['POST'])
def brief_run_now(brief_id):
    """Kick off an on-demand run in a background thread; UI polls for status."""
    if not get_content_brief(brief_id):
        return jsonify({"error": "Brief not found"}), 404
    if get_active_brief_run(brief_id):
        return jsonify({"error": "A run is already in progress for this brief"}), 409
    run_id = create_brief_run(brief_id, trigger='manual')

    def _bg():
        with usage_meter.usage_context("proactive"):
            try:
                content_agent.run_brief(brief_id, trigger='manual', run_id=run_id)
            except Exception:
                app.logger.exception("Manual brief run %s failed", brief_id)
                try:
                    finalize_brief_run(run_id, status='error', error_message='unhandled')
                except Exception:
                    pass

    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({"success": True, "run_id": run_id, "status": "running"})


@app.route('/briefs/<int:brief_id>/runs')
def brief_runs(brief_id):
    return jsonify({"runs": [dict(r) for r in list_brief_runs(brief_id, limit=20)]})


@app.route('/briefs/runs/<int:run_id>')
def brief_run_status(run_id):
    run = get_brief_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(dict(run))


def _compute_next_run_at(brief) -> "str | None":
    """Next local ISO run time for a recurring brief, or None for manual/paused.

    ``run_days`` uses Python weekday ints (Monday=0 .. Sunday=6). Local time is
    used throughout to stay consistent with get_next_available_slot.
    """
    d = dict(brief)
    cadence = (d.get("cadence") or "manual").lower()
    if cadence == "manual":
        return None
    try:
        hh, mm = (int(x) for x in (d.get("run_time") or "09:00").split(":")[:2])
    except Exception:
        hh, mm = 9, 0
    now = datetime.now()
    if cadence == "daily":
        candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate.isoformat(timespec="seconds")
    if cadence == "weekly":
        try:
            days = [int(x) for x in json.loads(d.get("run_days") or "[]") if 0 <= int(x) <= 6]
        except Exception:
            days = []
        if not days:
            days = [now.weekday()]
        for offset in range(0, 8):
            cand = (now + timedelta(days=offset)).replace(hour=hh, minute=mm, second=0, microsecond=0)
            if cand.weekday() in days and cand > now:
                return cand.isoformat(timespec="seconds")
        return (now + timedelta(days=7)).replace(
            hour=hh, minute=mm, second=0, microsecond=0
        ).isoformat(timespec="seconds")
    return None


def content_agent_worker() -> None:
    """Poll for due content briefs and run them; usage metered as proactive.

    Mirrors scheduled_post_worker: a resilient sleep loop that never dies on a
    single brief's failure. A per-brief active-run guard prevents a scheduled run
    from overlapping an in-flight manual run.
    """
    tick = int(os.getenv("CONTENT_AGENT_TICK_SECONDS", "60") or 60)
    with usage_meter.usage_context("proactive"):
        while True:
            try:
                time.sleep(tick)
                now_iso = datetime.now().isoformat(timespec="seconds")
                for brief in get_due_content_briefs(now_iso):
                    bid = brief["id"]
                    if get_active_brief_run(bid):
                        continue  # a manual run is already in flight
                    run_id = create_brief_run(bid, trigger="scheduled")
                    try:
                        content_agent.run_brief(bid, trigger="scheduled", run_id=run_id)
                    except Exception:
                        app.logger.exception("Scheduled brief %s failed", bid)
                    finally:
                        run = get_brief_run(run_id)
                        status = dict(run)["status"] if run else "error"
                        set_content_brief_schedule(
                            bid, next_run_at=_compute_next_run_at(brief),
                            last_run_at=now_iso, last_run_status=status,
                        )
            except Exception:
                app.logger.exception("content_agent_worker loop error")


def start_workers():
    """Start all background worker threads."""
    # Episode processing worker (usage metered as proactive/background)
    episode_worker = threading.Thread(target=_proactive_worker, daemon=True)
    episode_worker.start()
    app.logger.info("Episode processing worker started")

    # Scheduled post worker
    scheduled_worker = threading.Thread(target=scheduled_post_worker, daemon=True)
    scheduled_worker.start()
    app.logger.info("Scheduled post worker started")

    # Content agent worker (recurring briefs). On-demand runs work even if disabled.
    if os.getenv("CONTENT_AGENT_ENABLED", "true").lower() in ("1", "true", "yes"):
        content_worker = threading.Thread(target=content_agent_worker, daemon=True)
        content_worker.start()
        app.logger.info("Content agent worker started")

    # One-time backfill for YouTube channel names
    backfill_thread = threading.Thread(target=_backfill_youtube_channels_bg, daemon=True)
    backfill_thread.start()


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

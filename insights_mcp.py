"""Insights as an MCP server: let Claude or Codex write your posts on your plan.

Run by an MCP client (Claude Code, Claude Desktop, Codex) over stdio::

    python insights_mcp.py --db /path/to/insights.db

The assistant then does the writing itself, inside the chat you are already
paying for through a Claude or ChatGPT plan, and uses these tools to read what
Insights knows and to hand drafts back:

* ``get_writing_guide``  each platform's rules, the tones and the house style,
  plus a few posts you already published, as examples of your voice;
* ``list_sources`` / ``get_source``  podcast episodes (transcript and summary),
  saved web pages and articles;
* ``list_posts``  existing posts as Compose shows them, one card per piece of
  copy with the platforms it goes to;
* ``save_drafts``  new unused drafts, which appear in Compose marked
  "Claude or Codex" for you to review, edit, queue or post;
* ``update_draft``  rewrite a draft that is not queued or published yet.

What it can never do is publish, schedule, queue, delete, or read an account
token: those stay in the app, behind your own click. Everything a tool returns
is your own data, and everything sent to ``save_drafts`` is checked first (known
platform, non-empty, no em or en dashes) and refused without writing anything if
any post fails.

The protocol is JSON-RPC 2.0 over newline-delimited stdin/stdout (MCP's stdio
transport), implemented here without a dependency because the app needs only
``initialize``, ``tools/list`` and ``tools/call``. Nothing but protocol messages
is ever written to stdout; logs go to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import database  # noqa: E402

logger = logging.getLogger("insights_mcp")

SERVER_NAME = "insights"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SOURCE_TYPE = "mcp"
PLATFORMS = ("linkedin", "threads", "twitter", "facebook", "instagram")
PLATFORM_ALIASES = {"x": "twitter", "fb": "facebook", "ig": "instagram"}
CHAR_LIMITS = {"twitter": 280, "threads": 500, "instagram": 2200, "linkedin": 3000}
FORBIDDEN_DASHES = {chr(0x2014): "an em dash", chr(0x2013): "an en dash"}
MAX_DRAFTS_PER_CALL = 50
MAX_SOURCE_CHARS = 100_000


class ToolError(Exception):
    """A refusal the assistant should read and act on (shown as isError)."""


# ── helpers ──────────────────────────────────────────────────────────────

def _connect(db):
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _int(value, name, default=None, low=None, high=None):
    if value is None:
        if default is None:
            raise ToolError(f"{name} is required.")
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a whole number.")
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _platform(name):
    key = str(name or "").strip().lower()
    key = PLATFORM_ALIASES.get(key, key)
    if key not in PLATFORMS:
        raise ToolError(f"Unknown platform {name!r}. Use one of: {', '.join(PLATFORMS)}.")
    return key


def _card_rows(conn, post_id):
    """Every row of the Compose card that ``post_id`` belongs to.

    Compose shows one card per ``(content, image_url)``, so an edit must reach
    all of them or the card splits in two.
    """
    row = conn.execute("SELECT * FROM standalone_posts WHERE id = ?", (post_id,)).fetchone()
    if row is None:
        raise ToolError(f"There is no post {post_id}.")
    return conn.execute(
        "SELECT * FROM standalone_posts WHERE content = ? AND "
        "COALESCE(image_url, '') = COALESCE(?, '') ORDER BY id",
        (row["content"], row["image_url"])).fetchall()


def _state(conn, ids):
    """``{id: (queued, posted)}`` for standalone post ids."""
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    queued = {r[0] for r in conn.execute(
        f"SELECT standalone_post_id FROM scheduled_posts WHERE status = 'pending' "
        f"AND standalone_post_id IN ({marks})", ids)}
    posted = {r[0] for r in conn.execute(
        f"SELECT standalone_post_id FROM scheduled_posts WHERE status = 'posted' "
        f"AND standalone_post_id IN ({marks})", ids)}
    return {i: (i in queued, i in posted) for i in ids}


# ── tools ────────────────────────────────────────────────────────────────

def tool_get_writing_guide(db, args):
    import insights

    wanted = [_platform(p) for p in (args.get("platforms") or PLATFORMS)]
    examples = {}
    with _connect(db) as conn:
        for platform in wanted:
            rows = conn.execute(
                "SELECT content FROM standalone_posts WHERE platform = ? AND used = 1 "
                "ORDER BY created_at DESC LIMIT 3", (platform,)).fetchall()
            examples[platform] = [r["content"] for r in rows]
    return {
        "house_style": insights.NO_EM_DASH_RULE,
        "platforms": {p: {"guideline": insights.PLATFORM_GUIDELINES.get(p, ""),
                          "max_characters": CHAR_LIMITS.get(p)} for p in wanted},
        "tones": insights.TONE_GUIDES,
        "examples_of_published_posts": examples,
        "how_drafts_work": (
            "save_drafts stores each post as an unused draft in the Insights Compose page, "
            "one card per piece of copy with every platform it is for. The owner reviews "
            "them there and decides what to queue or publish. Write one post per call "
            "entry; give the same copy several platforms only when it fits all of them."),
    }


def tool_list_sources(db, args):
    kind = (args.get("kind") or "all").strip().lower()
    if kind not in ("all", "episodes", "urls", "articles"):
        raise ToolError("kind must be one of: all, episodes, urls, articles.")
    query = (args.get("query") or "").strip().lower()
    limit = _int(args.get("limit"), "limit", default=20, low=1, high=100)
    like = f"%{query}%"
    items = []
    with _connect(db) as conn:
        if kind in ("all", "episodes"):
            for r in conn.execute(
                    "SELECT id, title, url, published, status, "
                    "length(COALESCE(transcript, '')) AS t, length(COALESCE(summary, '')) AS s "
                    "FROM episodes WHERE (? = '' OR lower(COALESCE(title,'')) LIKE ?) "
                    "ORDER BY id DESC LIMIT ?", (query, like, limit)):
                items.append({"kind": "episode", "id": r["id"], "title": r["title"], "url": r["url"],
                              "date": r["published"], "status": r["status"],
                              "transcript_chars": r["t"], "summary_chars": r["s"]})
        if kind in ("all", "urls"):
            for r in conn.execute(
                    "SELECT id, title, url, created_at, length(COALESCE(content, '')) AS c "
                    "FROM url_sources WHERE (? = '' OR lower(COALESCE(title,'') || ' ' || url) LIKE ?) "
                    "ORDER BY COALESCE(last_used_at, created_at) DESC LIMIT ?", (query, like, limit)):
                items.append({"kind": "url", "id": r["id"], "title": r["title"], "url": r["url"],
                              "date": r["created_at"], "content_chars": r["c"]})
        if kind in ("all", "articles"):
            for r in conn.execute(
                    "SELECT id, topic, style, created_at, length(COALESCE(content, '')) AS c "
                    "FROM articles WHERE (? = '' OR lower(COALESCE(topic,'')) LIKE ?) "
                    "ORDER BY id DESC LIMIT ?", (query, like, limit)):
                items.append({"kind": "article", "id": r["id"], "title": r["topic"],
                              "style": r["style"], "date": r["created_at"], "content_chars": r["c"]})
    return {"sources": items[: limit if kind != "all" else limit * 3]}


def tool_get_source(db, args):
    kind = (args.get("kind") or "").strip().lower()
    source_id = _int(args.get("id"), "id")
    max_chars = _int(args.get("max_chars"), "max_chars", default=20000, low=500, high=MAX_SOURCE_CHARS)
    offset = _int(args.get("offset"), "offset", default=0, low=0)
    with _connect(db) as conn:
        if kind == "episode":
            r = conn.execute("SELECT * FROM episodes WHERE id = ?", (source_id,)).fetchone()
            if r is None:
                raise ToolError(f"There is no episode {source_id}.")
            title, url = r["title"], r["url"]
            text = ""
            if r["summary"]:
                text += "SUMMARY\n" + r["summary"] + "\n\n"
            text += "TRANSCRIPT\n" + (r["transcript"] or "(no transcript yet)")
        elif kind == "url":
            r = conn.execute("SELECT * FROM url_sources WHERE id = ?", (source_id,)).fetchone()
            if r is None:
                raise ToolError(f"There is no saved web page {source_id}.")
            title, url = r["title"], r["url"]
            text = ((r["description"] or "") + "\n\n" + (r["content"] or "")).strip()
        elif kind == "article":
            r = conn.execute("SELECT * FROM articles WHERE id = ?", (source_id,)).fetchone()
            if r is None:
                raise ToolError(f"There is no article {source_id}.")
            title, url = r["topic"], None
            text = r["content"] or ""
        else:
            raise ToolError("kind must be one of: episode, url, article.")
    chunk = text[offset: offset + max_chars]
    following = offset + len(chunk)
    return {"kind": kind, "id": source_id, "title": title, "url": url,
            "total_chars": len(text), "offset": offset, "text": chunk,
            "next_offset": following if following < len(text) else None}


def tool_list_posts(db, args):
    status = (args.get("status") or "drafts").strip().lower()
    if status not in ("drafts", "queued", "posted", "all"):
        raise ToolError("status must be one of: drafts, queued, posted, all.")
    platform = _platform(args["platform"]) if args.get("platform") else None
    query = (args.get("query") or "").strip().lower()
    limit = _int(args.get("limit"), "limit", default=20, low=1, high=100)
    with _connect(db) as conn:
        rows = conn.execute(
            "SELECT * FROM standalone_posts WHERE (? = '' OR lower(content) LIKE ?) "
            "ORDER BY created_at DESC, id DESC LIMIT 5000",
            (query, f"%{query}%")).fetchall()
        state = _state(conn, [r["id"] for r in rows])
    cards, order = {}, []
    for r in rows:
        key = (r["content"], r["image_url"] or "")
        if key not in cards:
            cards[key] = {"id": r["id"], "content": r["content"], "image_url": r["image_url"],
                          "source": r["source_type"], "created_at": r["created_at"],
                          "platforms": [], "post_ids": [], "used": False,
                          "queued": False, "posted": False}
            order.append(key)
        card = cards[key]
        card["platforms"].append(r["platform"])
        card["post_ids"].append(r["id"])
        queued, posted = state.get(r["id"], (False, False))
        card["used"] = card["used"] or bool(r["used"])
        card["queued"] = card["queued"] or queued
        card["posted"] = card["posted"] or posted
    out = []
    for key in order:
        card = cards[key]
        # Rows come newest first; a card lists its platforms in the order saved.
        pairs = sorted(zip(card["post_ids"], card["platforms"]))
        card["post_ids"] = [i for i, _ in pairs]
        card["platforms"] = [p for _, p in pairs]
        card["id"] = card["post_ids"][0]
        if platform and platform not in card["platforms"]:
            continue
        is_draft = not (card["used"] or card["queued"] or card["posted"])
        if status == "drafts" and not is_draft:
            continue
        if status == "queued" and not card["queued"]:
            continue
        if status == "posted" and not (card["posted"] or card["used"]):
            continue
        out.append(card)
        if len(out) >= limit:
            break
    return {"posts": out}


def _check_copy(content, platforms, where):
    if not isinstance(content, str) or not content.strip():
        raise ToolError(f"{where}: content is empty.")
    for char, name in FORBIDDEN_DASHES.items():
        if char in content:
            raise ToolError(f"{where}: contains {name}. The house style never uses them; "
                            "rewrite with a comma, colon, period or parentheses.")
    warnings = []
    for platform in platforms:
        limit = CHAR_LIMITS.get(platform)
        if limit and len(content) > limit:
            warnings.append(f"{where}: {len(content)} characters is over {platform}'s {limit}.")
    return warnings


def tool_save_drafts(db, args):
    posts = args.get("posts")
    if not isinstance(posts, list) or not posts:
        raise ToolError("posts must be a non-empty list of {content, platforms}.")
    if len(posts) > MAX_DRAFTS_PER_CALL:
        raise ToolError(f"Save at most {MAX_DRAFTS_PER_CALL} posts per call.")
    label = str(args.get("source_label") or "Written by Claude or Codex")[:1000]

    # Check every post before writing any, so a refusal leaves nothing half-saved.
    plan, warnings = [], []
    for index, post in enumerate(posts, start=1):
        where = f"post {index}"
        if not isinstance(post, dict):
            raise ToolError(f"{where}: must be an object with content and platforms.")
        raw = post.get("platforms") or ([post["platform"]] if post.get("platform") else [])
        if isinstance(raw, str):
            raw = [raw]
        platforms = []
        for name in raw:
            key = _platform(name)
            if key not in platforms:
                platforms.append(key)
        if not platforms:
            raise ToolError(f"{where}: name at least one platform.")
        content = post.get("content")
        warnings += _check_copy(content, platforms, where)
        image = post.get("image_url")
        if image is not None and not (isinstance(image, str) and image.startswith(("http://", "https://"))):
            raise ToolError(f"{where}: image_url must be an http(s) URL.")
        plan.append((content.strip(), platforms, image or None))

    saved = []
    for content, platforms, image in plan:
        ids = [database.add_standalone_post(
            source_type=SOURCE_TYPE, source_content=label, platform=platform,
            content=content, image_url=image, db_path=db) for platform in platforms]
        saved.append({"id": ids[0], "post_ids": ids, "platforms": platforms})
    return {"saved": saved, "warnings": warnings,
            "next": "The drafts are in Insights > Compose, marked 'Claude or Codex', "
                    "for the owner to review, queue or publish."}


def tool_update_draft(db, args):
    post_id = _int(args.get("post_id"), "post_id")
    content = args.get("content")
    with _connect(db) as conn:
        rows = _card_rows(conn, post_id)
        ids = [r["id"] for r in rows]
        state = _state(conn, ids)
    platforms = [r["platform"] for r in rows]
    _check_copy(content, platforms, f"post {post_id}")
    for r in rows:
        queued, posted = state[r["id"]]
        if r["used"] or queued or posted:
            why = "published" if (posted or r["used"]) else "queued"
            raise ToolError(f"Post {post_id} is already {why} on {r['platform']}, so it is not "
                            "changed here. Edit it in Compose if it really should change.")
    for r in rows:
        database.update_standalone_post(r["id"], content.strip(), db_path=db)
    return {"updated": ids, "platforms": platforms}


TOOLS = {
    "get_writing_guide": (tool_get_writing_guide, {
        "description": ("The house style, each platform's rules and length limit, the tones, and a few "
                        "already-published posts per platform as examples of the owner's voice. Read "
                        "this before writing posts."),
        "inputSchema": {"type": "object", "properties": {
            "platforms": {"type": "array", "items": {"type": "string", "enum": list(PLATFORMS)},
                          "description": "Limit to these platforms (default: all)."}},
            "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    }),
    "list_sources": (tool_list_sources, {
        "description": "List material posts can be written from: podcast episodes, saved web pages and articles.",
        "inputSchema": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["all", "episodes", "urls", "articles"]},
            "query": {"type": "string", "description": "Words to look for in titles."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    }),
    "get_source": (tool_get_source, {
        "description": ("Read one source's text: an episode's summary and transcript, a saved page's "
                        "content, or an article. Long text comes in pages; pass next_offset back as offset."),
        "inputSchema": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["episode", "url", "article"]},
            "id": {"type": "integer"},
            "offset": {"type": "integer", "minimum": 0},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": MAX_SOURCE_CHARS}},
            "required": ["kind", "id"], "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    }),
    "list_posts": (tool_list_posts, {
        "description": ("List posts the way Compose shows them: one card per piece of copy, with its "
                        "platforms and whether it is a draft, queued or published. Use it to avoid "
                        "repeating what already exists."),
        "inputSchema": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["drafts", "queued", "posted", "all"]},
            "platform": {"type": "string", "enum": list(PLATFORMS)},
            "query": {"type": "string", "description": "Words to look for in the copy."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    }),
    "save_drafts": (tool_save_drafts, {
        "description": ("Save new posts as unused drafts in Compose for the owner to review. Nothing is "
                        "queued or published. Each entry is one piece of copy and the platforms it is "
                        "for. Copy with an em or en dash, an unknown platform or no text is refused, and "
                        "then nothing from the call is saved."),
        "inputSchema": {"type": "object", "properties": {
            "posts": {"type": "array", "minItems": 1, "maxItems": MAX_DRAFTS_PER_CALL, "items": {
                "type": "object", "properties": {
                    "content": {"type": "string"},
                    "platforms": {"type": "array", "minItems": 1,
                                  "items": {"type": "string", "enum": list(PLATFORMS)}},
                    "image_url": {"type": "string", "description": "Optional http(s) image to attach."}},
                "required": ["content", "platforms"], "additionalProperties": False}},
            "source_label": {"type": "string",
                             "description": "What the posts were written from, shown with the draft."}},
            "required": ["posts"], "additionalProperties": False},
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    }),
    "update_draft": (tool_update_draft, {
        "description": ("Rewrite a draft's copy on every platform of its card. Refused for a post that "
                        "is queued or already published."),
        "inputSchema": {"type": "object", "properties": {
            "post_id": {"type": "integer"}, "content": {"type": "string"}},
            "required": ["post_id", "content"], "additionalProperties": False},
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    }),
}


# ── JSON-RPC over stdio ──────────────────────────────────────────────────

def _result(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message, db):
    """Answer one JSON-RPC message; ``None`` for a notification."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Invalid Request")
    method = message.get("method")
    request_id = message.get("id")
    is_notification = "id" not in message
    params = message.get("params") or {}

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _result(request_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "Insights", "version": SERVER_VERSION},
            "instructions": (
                "Insights holds the owner's podcast episodes, saved pages and social posts. Write "
                "posts yourself, then hand them to save_drafts: they land in Compose as drafts for "
                "the owner to review. Read get_writing_guide first. Never use em or en dashes. "
                "These tools cannot publish, schedule or delete anything."),
        })
    if is_notification:
        return None
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": [dict(spec, name=name) for name, (_, spec) in TOOLS.items()]})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            return _error(request_id, -32602, f"Unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "arguments must be an object")
        try:
            payload = TOOLS[name][0](db, arguments)
            text, failed = json.dumps(payload, ensure_ascii=False, indent=1, default=str), False
        except ToolError as exc:
            text, failed = str(exc), True
        except Exception as exc:  # report, never crash the server
            logger.exception("tool %s failed", name)
            text, failed = f"{name} failed: {exc}", True
        result = {"content": [{"type": "text", "text": text}], "isError": failed}
        if not failed:
            result["structuredContent"] = payload
        return _result(request_id, result)
    return _error(request_id, -32601, f"Method not found: {method}")


def resolve_db(path):
    db = os.path.abspath(os.path.expanduser(
        path or os.environ.get("INSIGHTS_DB") or os.path.join(HERE, "insights.db")))
    if not os.path.isfile(db):
        raise SystemExit(f"insights_mcp: no database at {db}. Pass --db with the path of the "
                         "insights.db the app uses (or set INSIGHTS_DB).")
    with sqlite3.connect(db) as conn:
        if not conn.execute("SELECT name FROM sqlite_master WHERE name = 'standalone_posts'").fetchone():
            raise SystemExit(f"insights_mcp: {db} is not an Insights database.")
    return db


def serve(db, stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = _error(None, -32700, "Parse error")
        else:
            if isinstance(message, list):
                replies = [r for r in (handle(m, db) for m in message) if r is not None]
                reply = replies or None
            else:
                reply = handle(message, db)
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
            stdout.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Insights MCP server (stdio).")
    parser.add_argument("--db", help="Path to the app's insights.db (default: INSIGHTS_DB, "
                                     "else insights.db beside this file).")
    options = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    db = resolve_db(options.db)
    serve(db)


if __name__ == "__main__":
    main()

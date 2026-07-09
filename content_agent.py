"""Content agent orchestrator.

Runs a single content brief end to end: research (procure source material) ->
generate draft social posts and/or long-form articles by REUSING the existing
``insights.py`` generators -> persist as UNUSED drafts (``source_type='agent'``,
``used=0``) so they surface in the Compose Command Center / Articles page for
review -> optionally auto-queue approved-window drafts into the schedule.

Called by:
* the on-demand ``/briefs/<id>/run`` route (spawns a thread), and
* the recurring ``content_agent_worker`` daemon.

Both wrap the call in ``usage_meter.usage_context("proactive")`` so all AI usage
is attributed to proactive/automated work. Generation reuse means identical
prompting, batching, JSON parsing, and metering as the manual Compose flow.

Import rule: this module is imported at the top of ``insights_web``, so it must
NOT import ``insights_web`` at module load. The one helper it needs from there
(``_maybe_attach_link_image``) is late-imported inside a function.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from urllib.parse import urlsplit

import database
import insights
import research_engine

logger = logging.getLogger(__name__)

DEFAULT_PLATFORMS = ["linkedin", "threads", "twitter"]
_PLATFORM_ALIASES = {"x": "twitter"}
# Platforms the scheduler can auto-queue (instagram is generable but not schedulable).
SCHEDULABLE = {"linkedin", "threads", "facebook", "twitter"}


# ── Brief parsing ───────────────────────────────────────────────────────────

def _json_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [s.strip() for s in str(val).split(",") if s.strip()]


def _domain_or_self(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return ""
    if "://" in x or "/" in x:
        host = urlsplit(x if "://" in x else "https://" + x).netloc.lower()
    else:
        host = x.lower()
    return host[4:] if host.startswith("www.") else host


def _brief_to_dict(row) -> dict:
    d = dict(row)
    d["platforms"] = _json_list(d.get("platforms")) or []
    d["must_include_keywords"] = _json_list(d.get("must_include_keywords")) or []
    focus = _json_list(d.get("focus_sources")) or []
    d["focus_sources_list"] = focus
    d["focus_sources_domains"] = [x for x in (_domain_or_self(f) for f in focus) if x]
    d["run_days"] = _json_list(d.get("run_days")) or []
    return d


# ── Context building ────────────────────────────────────────────────────────

def _creative_context(brief: dict) -> str:
    parts = []
    if brief.get("audience_persona"):
        parts.append(f"Target audience/persona: {brief['audience_persona']}.")
    kws = brief.get("must_include_keywords") or []
    if kws:
        parts.append(
            "Naturally include these keywords/hashtags/CTA in each post: "
            + ", ".join(str(k) for k in kws) + "."
        )
    return "\n".join(parts)


def _combine_context(*chunks) -> str | None:
    joined = "\n\n".join(c for c in chunks if c)
    return joined or None


# ── Cost / cap helpers ──────────────────────────────────────────────────────

def _cost_snapshot() -> str:
    """UTC timestamp in the exact format stored in usage_events.ts (space, not 'T')."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _current_cost(since_ts: str) -> float:
    try:
        return float(database.usage_totals(mode="proactive", start_ts=since_ts)["cost_usd"])
    except Exception:
        return 0.0


def _caps_hit(draft_count: int, max_drafts: int, since_ts: str, max_cost: float | None) -> bool:
    if draft_count >= max_drafts:
        return True
    if max_cost and _current_cost(since_ts) >= max_cost:
        return True
    return False


# ── Persistence ─────────────────────────────────────────────────────────────

def _attach_image(post_id: int, content: str) -> None:
    try:
        from insights_web import _maybe_attach_link_image  # late import: avoid cycle
        _maybe_attach_link_image(post_id, content)
    except Exception:
        logger.debug("link-image attach unavailable", exc_info=True)


def _persist_posts(generated, platforms, *, source_label, image_url,
                   brief_id, run_id, remaining) -> list:
    """Normalize generator output and save each post as an unused agent draft.

    Mirrors the platform normalization in ``compose_generate`` (lowercase, x->twitter,
    drop unrequested platforms, coerce str->list). Returns the new post ids.
    """
    requested = {p.lower() for p in platforms}
    ids: list = []
    for platform, post_data in (generated or {}).items():
        if platform == "raw":
            continue
        norm = _PLATFORM_ALIASES.get(platform.lower().strip(), platform.lower().strip())
        if norm not in requested:
            continue
        posts_list = post_data if isinstance(post_data, list) else [post_data]
        for content in posts_list:
            if len(ids) >= remaining:
                return ids
            if not content or not str(content).strip():
                continue
            pid = database.add_standalone_post(
                source_type="agent",
                source_content=(source_label or "")[:1000],
                platform=norm,
                content=str(content),
                image_url=image_url,
                brief_id=brief_id,
                brief_run_id=run_id,
            )
            ids.append(pid)
            if not image_url:
                _attach_image(pid, str(content))
    return ids


def _auto_queue(brief: dict, post_ids: list) -> int:
    """Schedule schedulable drafts into future slots >= review_window_hours out.

    Drafts stay used=0 (still reviewable/editable/unqueue-able in Compose); only
    the scheduled_post_worker will post them at slot time.
    """
    window_hours = int(brief.get("review_window_hours") or 24)
    now = datetime.now()  # local, to match get_next_available_slot
    queued = 0
    for pid in post_ids:
        row = database.get_standalone_post(pid)
        if not row:
            continue
        platform = dict(row).get("platform")
        if platform not in SCHEDULABLE:
            continue
        slot = database.get_next_available_slot(platform)
        if not slot:
            continue
        try:
            slot_dt = datetime.fromisoformat(slot)
        except Exception:
            continue
        if (slot_dt - now).total_seconds() < window_hours * 3600:
            continue
        database.add_scheduled_post(
            scheduled_for=slot, post_type="standalone",
            standalone_post_id=pid, platform=platform, status="pending",
        )
        queued += 1
    return queued


# ── Public entrypoint ───────────────────────────────────────────────────────

def run_brief(brief_id: int, *, trigger: str = "manual", run_id: int | None = None) -> dict:
    """Execute one run of a content brief. Never raises after the run row exists.

    Returns a summary dict: ``{status, posts_created, articles_created,
    sources_found, sources_used, queued, cost_usd, run_id, warnings}``.
    """
    row = database.get_content_brief(brief_id)
    if not row:
        if run_id is not None:
            database.finalize_brief_run(run_id, status="error",
                                        error_message=f"brief {brief_id} not found")
        return {"status": "error", "error": "brief not found", "run_id": run_id}

    brief = _brief_to_dict(row)
    if run_id is None:
        run_id = database.create_brief_run(brief_id, trigger=trigger)

    since_ts = _cost_snapshot()
    max_cost = float(brief.get("max_cost_usd") or 0) or None
    max_drafts = int(brief.get("max_drafts_per_run") or 30)
    max_sources = int(brief.get("max_sources_per_run") or 5)

    warnings: list = []
    status = "success"
    err = None
    post_ids: list = []
    article_ids: list = []
    sources_found = 0
    sources_used = 0
    queued = 0
    query_plan: list = []

    try:
        platforms = brief["platforms"] or DEFAULT_PLATFORMS
        tone = brief.get("tone") or "professional"
        ppp = max(1, min(int(brief.get("posts_per_platform") or 3), 10))
        content_type = brief.get("content_type") or "posts"
        want_posts = content_type in ("posts", "both")
        want_articles = content_type in ("articles", "both")
        instructions = brief.get("instructions") or ""
        creative = _creative_context(brief)
        research_attempted = bool(brief.get("use_web_search") or brief.get("use_saved_sources"))

        # ── Research ─────────────────────────────────────────────────────
        rres = None
        if research_attempted:
            rbrief = research_engine.ResearchBrief(
                topic=instructions,
                keywords=brief["must_include_keywords"],
                focus_domains=brief["focus_sources_domains"],
                audience=brief.get("audience_persona") or None,
                use_web_search=bool(brief.get("use_web_search")),
                use_saved_sources=bool(brief.get("use_saved_sources")),
                max_items=max_sources,
            )
            rres = research_engine.run_research(rbrief)
            warnings.extend(rres.warnings)
            query_plan = rres.query_plan
            sources_found = len(rres.items)

        items = rres.items if rres else []
        research_text = rres.research_brief if rres else ""

        # ── Social posts ─────────────────────────────────────────────────
        if want_posts and items:
            for item in items:
                if _caps_hit(len(post_ids), max_drafts, since_ts, max_cost):
                    status = "partial"
                    warnings.append("cap_reached")
                    break
                ctx = _combine_context(
                    creative,
                    f"Editorial angle: {instructions}" if instructions else "",
                    research_text,
                    f"Why this source is relevant: {item.why_relevant}" if item.why_relevant else "",
                )
                try:
                    generated = insights.generate_posts_from_text(
                        text=item.text, platforms=platforms, tone=tone,
                        topic=item.title, posts_per_platform=ppp,
                        extra_context=ctx, source_url=item.url,
                    )
                except Exception:
                    logger.exception("post generation failed for %s", item.url)
                    continue
                ids = _persist_posts(
                    generated, platforms,
                    source_label=(item.url or brief["name"]),
                    image_url=item.image_url, brief_id=brief_id, run_id=run_id,
                    remaining=max_drafts - len(post_ids),
                )
                post_ids.extend(ids)
                if ids:
                    sources_used += 1
        elif want_posts and not items and not research_attempted:
            # Pure prompt brief (no sourcing configured): draft straight from the prompt.
            try:
                generated = insights.generate_posts_from_prompt(
                    prompt=instructions, platforms=platforms, tone=tone,
                    posts_per_platform=ppp, extra_context=creative or None,
                )
                post_ids.extend(_persist_posts(
                    generated, platforms, source_label=brief["name"],
                    image_url=None, brief_id=brief_id, run_id=run_id,
                    remaining=max_drafts,
                ))
            except Exception:
                logger.exception("prompt-only post generation failed for brief %s", brief_id)
        elif want_posts and not items and research_attempted:
            # Research ran but surfaced nothing new (e.g. all duplicates): skip
            # rather than emit repetitive generic posts on every scheduled run.
            warnings.append("no_new_sources")

        # ── Long-form articles ───────────────────────────────────────────
        if want_articles and items:
            n_articles = int(brief.get("article_count") or 0) or 1
            for item in items[:n_articles]:
                if _caps_hit(len(post_ids) + len(article_ids), max_drafts, since_ts, max_cost):
                    status = "partial"
                    warnings.append("cap_reached")
                    break
                ctx = _combine_context(creative, research_text)
                try:
                    content = insights.generate_article(
                        transcript=item.text, summary=item.snippet or "",
                        topic=(instructions or item.title),
                        podcast_title=item.domain or "web",
                        episode_title=item.title,
                        style=brief.get("article_style") or "blog",
                        extra_context=ctx, is_text_source=True,
                    )
                except Exception:
                    logger.exception("article generation failed for %s", item.url)
                    continue
                if not content:
                    continue
                aid = database.add_article(
                    None, (instructions[:200] or item.title),
                    brief.get("article_style") or "blog", content,
                    brief_id=brief_id, brief_run_id=run_id, source_type="agent",
                )
                article_ids.append(aid)
                sources_used += 1

        # ── Auto-queue ───────────────────────────────────────────────────
        if brief.get("auto_queue") and post_ids:
            try:
                queued = _auto_queue(brief, post_ids)
            except Exception:
                logger.exception("auto-queue failed for brief %s", brief_id)

    except Exception as exc:
        logger.exception("content brief run %s failed", brief_id)
        status = "error"
        err = str(exc)

    cost = _current_cost(since_ts)
    database.finalize_brief_run(
        run_id, status=status,
        sources_found=sources_found, sources_used=sources_used,
        posts_created=len(post_ids), articles_created=len(article_ids),
        cost_usd=cost, error_message=err,
        log={"warnings": warnings, "query_plan": query_plan, "queued": queued},
    )

    return {
        "status": status,
        "run_id": run_id,
        "posts_created": len(post_ids),
        "articles_created": len(article_ids),
        "sources_found": sources_found,
        "sources_used": sources_used,
        "queued": queued,
        "cost_usd": cost,
        "warnings": warnings,
    }

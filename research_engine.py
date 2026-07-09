"""Deterministic, cost-capped research pipeline for the content agent.

Given a :class:`ResearchBrief`, gather relevant, deduplicated source material from
(a) live web search (via :mod:`web_search`) and (b) the user's saved sources
(``url_sources`` / ``episodes``), then return a bounded, ranked set of
:class:`SourceItem` plus a synthesized research brief for the generators.

This is intentionally a fixed pipeline (query-plan -> search -> safe-fetch ->
rank -> select -> synthesize), NOT an open-ended agent loop: predictable cost and
latency matter for unattended recurring runs. All web fetches go through the
SSRF-safe ``fetch_article_content_safe`` in ``insights_web`` (late-imported to
avoid an import cycle).
"""

from __future__ import annotations

import os
import re
import json
import difflib
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit

import database
import web_search

logger = logging.getLogger(__name__)


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


_MAX_TEXT = _envi("RESEARCH_MAX_TEXT_CHARS", 8000)
_CANDIDATE_CAP = _envi("RESEARCH_CANDIDATE_CAP", 60)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "are", "how", "what", "why", "your", "you", "this", "that", "from", "by",
    "at", "as", "it", "be", "about", "into", "our", "we", "they", "their",
}
_TRACKING = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src",
}


# ── Public data structures ──────────────────────────────────────────────────

@dataclass
class ResearchBrief:
    """Input to :func:`run_research` (parsed from a content_briefs row)."""

    topic: str
    keywords: List[str] = field(default_factory=list)
    focus_domains: List[str] = field(default_factory=list)
    audience: Optional[str] = None
    use_web_search: bool = True
    use_saved_sources: bool = True
    strict_focus: bool = False
    max_items: int = 5
    freshness_days: Optional[int] = None


@dataclass
class SourceItem:
    """A ranked, ready-to-generate-from source document."""

    title: str
    url: Optional[str]
    text: str
    snippet: str = ""
    image_url: Optional[str] = None
    why_relevant: str = ""
    origin: str = "web"          # web | url_source | episode
    score: float = 0.0
    published: Optional[str] = None
    domain: str = ""
    source_id: Optional[int] = None


@dataclass
class ResearchStats:
    searches_used: int = 0
    fetches_used: int = 0
    candidates_seen: int = 0
    dropped_already_posted: int = 0
    dropped_duplicates: int = 0
    provider: str = "none"


@dataclass
class ResearchResult:
    topic: str
    query_plan: List[str]
    research_brief: str
    items: List[SourceItem]
    stats: ResearchStats
    warnings: List[str] = field(default_factory=list)


# ── Small helpers ───────────────────────────────────────────────────────────

def _tokens(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def _content_tokens(text: str) -> set:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 2}


def _norm_alnum(text: str) -> str:
    """Lowercase, strip all non-alphanumerics. Lets '#AIsecurity' match 'AI security'."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _canonicalize_url(url: str) -> str:
    """Normalize a URL for dedup/caching (lowercase host, strip tracking + trailing slash)."""
    try:
        parts = urlsplit((url or "").strip())
        scheme = (parts.scheme or "https").lower()
        host = parts.netloc.lower()
        query = "&".join(
            q for q in parts.query.split("&")
            if q and q.split("=")[0].lower() not in _TRACKING
        )
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((scheme, host, path, query, ""))
    except Exception:
        return url or ""


def _domain_of(url: str) -> str:
    try:
        host = urlsplit(url or "").netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _loads_json(text: str):
    """Best-effort JSON extraction from an LLM reply (fences, embedded array/object)."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    for candidate in (stripped, text):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                pass
    return None


def _relevant(blob: str, topic_tokens: set, kw_norms: list) -> bool:
    """Lenient prefilter (favor recall; scoring/selection handles precision).

    Keeps a candidate if any keyword (normalized) appears in the blob, or the blob
    shares at least one content token with the topic.
    """
    if kw_norms:
        blob_norm = _norm_alnum(blob)
        if any(k and k in blob_norm for k in kw_norms):
            return True
    return bool(topic_tokens & _tokens(blob))


# ── Stage 1: query planning ─────────────────────────────────────────────────

def _concise_query(text: str, max_words: int = 14) -> str:
    """Reduce a long brief/topic to a short search query (first sentence, capped).

    Content briefs can be long multi-paragraph prompts; search providers reject
    very long queries, so the heuristic fallback must not use the raw brief.
    """
    text = (text or "").strip()
    if not text:
        return ""
    first = re.split(r"[.!?\n]", text, maxsplit=1)[0].strip()
    words = (first or text).split()
    return " ".join(words[:max_words])


def _build_query_plan(brief: ResearchBrief, use_local: bool, max_queries: int) -> List[str]:
    base = (brief.topic or "").strip()
    short = _concise_query(base)
    heuristic = [short] if short else []
    if brief.keywords:
        heuristic.append(f"{short} {' '.join(str(k) for k in brief.keywords[:3])}".strip())

    try:
        import insights
        client, model, provider = insights._get_llm_client(
            use_local, model=os.getenv("CONTENT_AGENT_MODEL"))
        aud = f" for an audience of {brief.audience}" if brief.audience else ""
        kw = f" Must relate to: {', '.join(brief.keywords)}." if brief.keywords else ""
        msgs = [
            {"role": "system", "content": (
                "You generate diverse, high-signal web search queries. "
                "Reply with a JSON array of short query strings only."
            )},
            {"role": "user", "content": (
                f"Topic: {base[:1000]}{aud}.{kw} Produce {max_queries} distinct search "
                "queries, each under 12 words, that together cover the topic broadly. "
                "Reply with a JSON array of strings only."
            )},
        ]
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=0.4, max_tokens=300
        )
        _meter_chat(resp, "research_query_plan", provider, model)
        parsed = _loads_json(resp.choices[0].message.content.strip())
        queries: List[str] = []
        if isinstance(parsed, list):
            queries = [str(q).strip() for q in parsed if str(q).strip()]
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    queries = [str(q).strip() for q in v if str(q).strip()]
                    break
        if queries:
            return _dedupe_str(queries)[:max_queries]
    except Exception:
        logger.debug("query plan LLM failed; using heuristic", exc_info=True)

    return _dedupe_str([q for q in heuristic if q])[:max_queries]


def _dedupe_str(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


# ── Stage 2: web search ─────────────────────────────────────────────────────

def _gather_web(brief, query_plan, stats, warnings) -> List[web_search.SearchResult]:
    provider = web_search.get_provider()
    stats.provider = provider
    if provider == "none" or not brief.use_web_search:
        return []

    max_searches = _envi("RESEARCH_MAX_SEARCHES", 6)
    per_query = _envi("WEB_SEARCH_MAX_RESULTS", 5)

    # Main queries run without a domain filter for breadth (unless strict focus);
    # add one focused query per focus-domain so those sources are represented.
    plan = [(q, brief.focus_domains if brief.strict_focus else None) for q in query_plan]
    plan += [(brief.topic, [d]) for d in brief.focus_domains[:2]]

    hits: List[web_search.SearchResult] = []
    searches = 0
    for query, domains in plan:
        if searches >= max_searches:
            break
        try:
            found = web_search.search(
                query, max_results=per_query, allowed_domains=domains,
                days=brief.freshness_days, topic="general",
            )
            searches += 1
            hits.extend(found)
        except web_search.SearchError as exc:
            warnings.append("search_unavailable")
            logger.info("web search failed for %r: %s", query, exc)
            break  # provider is failing hard; stop and fall back to saved sources

    stats.searches_used = searches
    return hits


# ── Stage 3: fetch/extract web hits into SourceItems (SSRF-safe) ────────────

def _web_hits_to_items(hits, brief, stats, warnings) -> List[SourceItem]:
    if not hits:
        return []

    max_fetches = _envi("RESEARCH_MAX_FETCHES", 15)

    try:
        from insights_web import fetch_article_content_safe, fetch_og_image_for_url
    except Exception:
        logger.warning("SSRF-safe fetch helpers unavailable; skipping web extraction")
        return []

    try:
        from github_client import is_github_repo_url, parse_github_repo_url, fetch_github_repo
    except Exception:
        is_github_repo_url = None

    items: List[SourceItem] = []
    seen = set()
    fetches = 0

    for h in hits:
        canon = _canonicalize_url(h.url)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        domain = _domain_of(canon)
        text = (h.content or "").strip()
        image = None

        # 1) Reuse cached url_sources content when present (no network).
        cached = database.get_url_source_by_url(canon)
        if cached is not None:
            cd = dict(cached)
            if not text and cd.get("content"):
                text = cd["content"]
            image = cd.get("og_image")

        # 2) GitHub short-circuit.
        if not text and is_github_repo_url and is_github_repo_url(canon):
            try:
                owner, repo = parse_github_repo_url(canon)
                repo_data = fetch_github_repo(owner, repo)
                text = (repo_data.get("content") or "").strip()
                image = image or repo_data.get("og_image")
            except Exception:
                logger.debug("GitHub fetch failed for %s", canon, exc_info=True)

        # 3) SSRF-safe HTML fetch + extract.
        if not text and fetches < max_fetches:
            extracted, _final = fetch_article_content_safe(canon)
            fetches += 1
            if extracted:
                text = extracted

        # 4) Fall back to the search snippet if it's substantial.
        if not text:
            if len((h.snippet or "")) < 160:
                continue
            text = h.snippet

        # Optional og:image (already SSRF-safe) within the fetch budget.
        if not image and fetches < max_fetches:
            try:
                image = fetch_og_image_for_url(canon)
            except Exception:
                image = None

        # Cache procured content for reuse on future runs.
        try:
            sid = database.add_url_source(
                url=canon, title=h.title or canon,
                description=(h.snippet or "")[:500],
                content=text[:_MAX_TEXT], og_image=image,
            )
        except Exception:
            sid = None

        items.append(SourceItem(
            title=h.title or canon, url=canon, text=text[:_MAX_TEXT],
            snippet=h.snippet or "", image_url=image, origin="web",
            score=float(h.score or 0.0), published=h.published,
            domain=domain, source_id=sid,
        ))

    stats.fetches_used = fetches
    return items


# ── Stage 4: existing saved sources ─────────────────────────────────────────

def _gather_existing(brief) -> List[SourceItem]:
    topic_tokens = _content_tokens(brief.topic)
    kw_norms = [_norm_alnum(k) for k in brief.keywords if _norm_alnum(k)]
    focus = {d.lower() for d in brief.focus_domains}
    items: List[SourceItem] = []

    try:
        url_rows = database.list_url_sources()[:_CANDIDATE_CAP]
    except Exception:
        url_rows = []
    for row in url_rows:
        d = dict(row)
        url = d.get("url") or ""
        domain = _domain_of(url)
        if focus and brief.strict_focus and not any(f in domain for f in focus):
            continue
        blob = f"{d.get('title', '')} {d.get('description', '')} {d.get('content', '')}"
        if not _relevant(blob, topic_tokens, kw_norms):
            continue
        content = (d.get("content") or "").strip()
        if not content:
            continue
        items.append(SourceItem(
            title=d.get("title") or url, url=url, text=content[:_MAX_TEXT],
            snippet=(d.get("description") or "")[:400], image_url=d.get("og_image"),
            origin="url_source", domain=domain, source_id=d.get("id"),
        ))

    try:
        ep_rows = database.list_all_episodes("published")[:_CANDIDATE_CAP]
    except Exception:
        ep_rows = []
    for row in ep_rows:
        d = dict(row)
        blob = f"{d.get('title', '')} {d.get('summary', '')}"
        if not _relevant(blob, topic_tokens, kw_norms):
            continue
        text = (d.get("summary") or d.get("transcript") or "").strip()
        if not text:
            continue
        items.append(SourceItem(
            title=d.get("title") or "Episode", url=d.get("url"),
            text=text[:_MAX_TEXT], snippet=(d.get("summary") or "")[:400],
            origin="episode", domain=_domain_of(d.get("url") or ""),
            source_id=d.get("id"),
        ))

    return items


# ── Stage 5: dedup + rank + select ──────────────────────────────────────────

def _drop_already_posted(items: List[SourceItem], stats) -> List[SourceItem]:
    urls = [it.url for it in items if it.url]
    if not urls:
        return items
    try:
        counts = database.count_standalone_posts_by_source_urls(urls)
    except Exception:
        return items
    kept = []
    for it in items:
        if it.url and counts.get(it.url, 0) > 0:
            stats.dropped_already_posted += 1
            continue
        kept.append(it)
    return kept


def _score_and_select(items: List[SourceItem], brief: ResearchBrief, max_items: int) -> List[SourceItem]:
    topic_tokens = _content_tokens(brief.topic)
    kw_norms = [_norm_alnum(k) for k in brief.keywords if _norm_alnum(k)]
    focus = {d.lower() for d in brief.focus_domains}

    for it in items:
        blob = f"{it.title} {it.snippet} {it.text[:500]}"
        bt = _tokens(blob)
        blob_norm = _norm_alnum(blob)
        kw_hits = sum(1 for k in kw_norms if k in blob_norm)
        topic_hits = len(topic_tokens & bt)
        in_focus = bool(focus) and any(f in it.domain for f in focus)
        score = 2.0 * kw_hits + min(topic_hits, 5) * 1.0
        if in_focus:
            score += 3.0
        if it.origin in ("url_source", "episode"):
            score += 1.0
        if it.score:
            score += 2.0 * float(it.score)
        if it.published:
            score += 0.5
        it.score = score

        reasons = []
        if kw_hits:
            reasons.append(f"matches {kw_hits} keyword(s)")
        if in_focus:
            reasons.append("from a focus source")
        if it.origin != "web":
            reasons.append("from your saved sources")
        it.why_relevant = "; ".join(reasons) or f"relevant to '{brief.topic}'"

    items.sort(key=lambda x: x.score, reverse=True)

    per_domain = _envi("RESEARCH_PER_DOMAIN_CAP", 2)
    domain_counts: dict = {}
    out: List[SourceItem] = []
    for it in items:
        d = it.domain or "?"
        if domain_counts.get(d, 0) >= per_domain:
            continue
        domain_counts[d] = domain_counts.get(d, 0) + 1
        out.append(it)
        if len(out) >= max_items:
            break
    return out


def _dedupe_titles(items: List[SourceItem], stats) -> List[SourceItem]:
    out: List[SourceItem] = []
    norms: List[str] = []
    for it in items:
        norm = " ".join(sorted(_content_tokens(it.title)))
        if any(difflib.SequenceMatcher(None, norm, prev).ratio() > 0.85 for prev in norms):
            stats.dropped_duplicates += 1
            continue
        norms.append(norm)
        out.append(it)
    return out


# ── Stage 6: synthesize research brief ──────────────────────────────────────

def _synthesize_brief(brief: ResearchBrief, items: List[SourceItem], use_local: bool) -> str:
    if not items:
        return ""
    joined = "\n\n".join(
        f"- {it.title} ({it.domain}): {(it.snippet or it.text[:200]).strip()}"
        for it in items[:8]
    )
    try:
        import insights
        client, model, provider = insights._get_llm_client(
            use_local, model=os.getenv("CONTENT_AGENT_MODEL"))
        aud = f" The audience is {brief.audience}." if brief.audience else ""
        msgs = [
            {"role": "system", "content": (
                "You synthesize research notes into a concise brief a writer can use. "
                "Write 4 to 6 sentences. " + insights.NO_EM_DASH_RULE
            )},
            {"role": "user", "content": (
                f"Topic: {brief.topic}.{aud}\n\nSources:\n{joined}\n\n"
                "Write a short synthesis highlighting the key angles, facts, and themes to cover."
            )},
        ]
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=0.5, max_tokens=500
        )
        _meter_chat(resp, "research_brief", provider, model)
        return resp.choices[0].message.content.strip()
    except Exception:
        logger.debug("research brief synthesis failed", exc_info=True)
        return "Key sources:\n" + joined


def _meter_chat(resp, category: str, provider: str, model: str) -> None:
    try:
        import usage_meter
        usage_meter.record_chat(resp, category=category, provider=provider, model=model)
    except Exception:
        logger.debug("usage metering unavailable", exc_info=True)


# ── Orchestration entrypoint ────────────────────────────────────────────────

def run_research(brief: ResearchBrief, *, use_local: bool = False) -> ResearchResult:
    """Run the full research pipeline and return a bounded, ranked result."""
    stats = ResearchStats()
    warnings: List[str] = []
    max_items = brief.max_items or _envi("RESEARCH_MAX_ITEMS", 12)

    query_plan = (
        _build_query_plan(brief, use_local, _envi("RESEARCH_QUERY_COUNT", 4))
        if brief.use_web_search else [brief.topic]
    )

    web_hits = _gather_web(brief, query_plan, stats, warnings)
    web_items = _web_hits_to_items(web_hits, brief, stats, warnings)
    existing_items = _gather_existing(brief) if brief.use_saved_sources else []

    candidates = web_items + existing_items
    stats.candidates_seen = len(candidates)
    if not candidates:
        warnings.append("no_candidates")
        return ResearchResult(brief.topic, query_plan, "", [], stats, warnings)

    candidates = _drop_already_posted(candidates, stats)
    selected = _score_and_select(candidates, brief, max_items)
    selected = _dedupe_titles(selected, stats)

    if not selected:
        warnings.append("all_duplicates")

    research_brief = _synthesize_brief(brief, selected, use_local)
    return ResearchResult(brief.topic, query_plan, research_brief, selected, stats, warnings)

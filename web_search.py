"""Pluggable web-search provider abstraction for the content agent.

The content agent's research engine (``research_engine.py``) calls :func:`search`
to discover fresh source material. Provider selection is controlled by the
``WEB_SEARCH_PROVIDER`` env var:

* ``auto`` (default): Tavily if ``TAVILY_API_KEY`` is set, else Brave if
  ``BRAVE_SEARCH_API_KEY`` is set, else OpenAI native web search (needs only the
  existing ``OPENAI_API_KEY``), else ``none``.
* ``tavily`` / ``brave`` / ``openai`` / ``none``: force a specific provider.

Design rules mirror ``usage_meter``: failures degrade gracefully. A hard provider
failure raises :class:`SearchError` so the caller can fall back to
existing-sources-only research rather than crash an unattended run.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


@dataclass
class SearchResult:
    """A single normalized web-search hit."""

    title: str
    url: str
    snippet: str = ""
    content: Optional[str] = None      # provider-extracted full text if available
    score: Optional[float] = None
    published: Optional[str] = None
    source: str = ""                    # provider name


class SearchError(RuntimeError):
    """Raised when the resolved provider fails hard (network, auth, quota)."""


def get_provider() -> str:
    """Resolve the effective provider name from env + key availability."""
    choice = (os.getenv("WEB_SEARCH_PROVIDER", "auto") or "auto").strip().lower()
    if choice and choice != "auto":
        return choice
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        return "brave"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "none"


def search(
    query: str,
    *,
    max_results: int = 5,
    allowed_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    topic: str = "general",
    days: Optional[int] = None,
    include_content: bool = True,
    timeout: float = 20.0,
) -> List[SearchResult]:
    """Run a single web search and return normalized results.

    Returns ``[]`` when the provider is ``none`` or the query is blank. Raises
    :class:`SearchError` on a hard provider failure so callers can degrade.
    """
    query = (query or "").strip()
    if not query:
        return []
    # Providers (Tavily especially) reject over-long queries with a 400 Bad
    # Request; cap length and avoid cutting mid-word.
    max_chars = int(os.getenv("WEB_SEARCH_MAX_QUERY_CHARS", "380") or 380)
    if len(query) > max_chars:
        query = query[:max_chars].rsplit(" ", 1)[0]

    provider = get_provider()
    if provider == "none":
        return []

    try:
        if provider == "tavily":
            results = _search_tavily(
                query, max_results, allowed_domains, exclude_domains,
                topic, days, include_content, timeout,
            )
        elif provider == "brave":
            results = _search_brave(query, max_results, allowed_domains, timeout)
        elif provider == "openai":
            results = _search_openai(query, max_results, allowed_domains, timeout)
        else:
            raise SearchError(f"Unknown web search provider: {provider}")
    except SearchError:
        raise
    except Exception as exc:  # normalize any provider-specific error
        raise SearchError(f"{provider} search failed: {exc}") from exc

    # Best-effort cost metering for paid SERP providers.
    if provider in ("tavily", "brave"):
        try:
            import usage_meter
            usage_meter.record_search(provider, calls=1)
        except Exception:
            logger.debug("search metering unavailable", exc_info=True)

    return results[:max_results]


# ── Tavily ──────────────────────────────────────────────────────────────────

def _search_tavily(query, max_results, allowed_domains, exclude_domains,
                   topic, days, include_content, timeout) -> List[SearchResult]:
    import requests

    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise SearchError("TAVILY_API_KEY is not configured")

    payload = {
        "api_key": key,
        "query": query,
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
        "max_results": max_results,
        "include_raw_content": bool(include_content),
        "include_images": True,
        "topic": "news" if topic == "news" else "general",
    }
    if allowed_domains:
        payload["include_domains"] = list(allowed_domains)
    if exclude_domains:
        payload["exclude_domains"] = list(exclude_domains)
    if days:
        payload["days"] = int(days)

    resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=timeout)
    if resp.status_code == 401:
        raise SearchError("Tavily rejected the API key (401)")
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in data.get("results", []):
        url = r.get("url")
        if not url:
            continue
        results.append(SearchResult(
            title=r.get("title") or url,
            url=url,
            snippet=r.get("content") or "",
            content=(r.get("raw_content") or None) if include_content else None,
            score=r.get("score"),
            published=r.get("published_date"),
            source="tavily",
        ))
    return results


# ── Brave ───────────────────────────────────────────────────────────────────

def _search_brave(query, max_results, allowed_domains, timeout) -> List[SearchResult]:
    import requests

    key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not key:
        raise SearchError("BRAVE_SEARCH_API_KEY is not configured")

    # Brave has no strict include_domains param; bias via site: filters in the query.
    q = query
    if allowed_domains:
        sites = " OR ".join(f"site:{d}" for d in allowed_domains)
        q = f"{query} ({sites})"

    headers = {"X-Subscription-Token": key, "Accept": "application/json"}
    params = {"q": q, "count": max(1, min(max_results, 20))}
    resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=timeout)
    if resp.status_code == 401:
        raise SearchError("Brave rejected the API key (401)")
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in (data.get("web", {}) or {}).get("results", []):
        url = r.get("url")
        if not url:
            continue
        results.append(SearchResult(
            title=r.get("title") or url,
            url=url,
            snippet=r.get("description") or "",
            content=None,
            score=None,
            published=r.get("age"),
            source="brave",
        ))
    return results


# ── OpenAI native web search (fallback; no new key) ─────────────────────────

def _search_openai(query, max_results, allowed_domains, timeout) -> List[SearchResult]:
    """Use OpenAI's native web-search tool via the Responses API.

    Requires a reasonably modern ``openai`` SDK. On older SDKs this raises,
    which :func:`search` converts to ``SearchError`` so research falls back to
    existing sources. Returns citation URLs (no extracted body -> the research
    engine must fetch/extract each one).
    """
    import insights  # reuse the configured client factory (OPENAI_API_KEY)

    # Force the OpenAI backend: the web_search tool is OpenAI-specific, so this
    # must not follow LLM_PROVIDER (which may be "anthropic").
    client, model, _provider = insights._get_llm_client(provider="openai")
    search_model = os.getenv("OPENAI_SEARCH_MODEL", model)

    tool = {"type": "web_search"}
    if allowed_domains:
        tool["filters"] = {"allowed_domains": list(allowed_domains)}

    if not hasattr(client, "responses"):
        raise SearchError("Installed openai SDK has no Responses API (web_search)")

    resp = client.responses.create(
        model=search_model,
        tools=[tool],
        input=(
            f"Find {max_results} recent, authoritative, diverse sources about: {query}. "
            "Return the most relevant results; cite each source URL."
        ),
        include=["web_search_call.action.sources"],
    )

    # Meter the underlying model call (tokens -> cost) like any chat call.
    try:
        import usage_meter
        usage_meter.record_chat(resp, category="research_search", provider="openai", model=search_model)
    except Exception:
        logger.debug("openai search metering unavailable", exc_info=True)

    results = _extract_openai_citations(resp)
    if not results:
        raise SearchError("OpenAI web search returned no citations")
    return results


def _extract_openai_citations(resp) -> List[SearchResult]:
    """Pull url/title citations out of a Responses API result, defensively."""
    out: List[SearchResult] = []
    seen = set()

    def _add(url, title):
        if url and url not in seen:
            seen.add(url)
            out.append(SearchResult(title=title or url, url=url, source="openai"))

    try:
        for item in getattr(resp, "output", []) or []:
            # web_search_call action sources
            action = getattr(item, "action", None)
            for src in (getattr(action, "sources", None) or []):
                _add(getattr(src, "url", None) or (src.get("url") if isinstance(src, dict) else None),
                     getattr(src, "title", None) or (src.get("title") if isinstance(src, dict) else None))
            # message content annotations
            for content in (getattr(item, "content", None) or []):
                for ann in (getattr(content, "annotations", None) or []):
                    url = getattr(ann, "url", None) or (ann.get("url") if isinstance(ann, dict) else None)
                    title = getattr(ann, "title", None) or (ann.get("title") if isinstance(ann, dict) else None)
                    _add(url, title)
    except Exception:
        logger.debug("Failed to parse OpenAI citations", exc_info=True)
    return out

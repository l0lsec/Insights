"""
GitHub Repository client for extracting repo metadata, README, and release notes
from public GitHub repositories via the REST API.
"""

import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
GITHUB_REPO_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9\-_.]+)/([a-zA-Z0-9\-_.]+)(?:/.*)?$"
)

_REQUEST_TIMEOUT = 15


def is_github_repo_url(url: str) -> bool:
    """Return True if *url* points to a GitHub repository (not gists, orgs, etc.)."""
    if not url:
        return False
    match = GITHUB_REPO_PATTERN.match(url.strip().rstrip("/"))
    if not match:
        return False
    owner, repo = match.group(1), match.group(2)
    # Exclude GitHub special paths that aren't repos
    excluded_owners = {
        "settings", "notifications", "explore", "topics",
        "trending", "collections", "sponsors", "login", "signup",
        "orgs", "marketplace", "features", "security", "enterprise",
        "pricing", "about", "team", "customer-stories", "readme",
    }
    if owner.lower() in excluded_owners:
        return False
    return True


def parse_github_repo_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub repo URL.

    Raises ValueError if the URL doesn't match.
    """
    match = GITHUB_REPO_PATTERN.match(url.strip().rstrip("/"))
    if not match:
        raise ValueError(f"Not a valid GitHub repo URL: {url}")
    owner = match.group(1)
    repo = match.group(2)
    # Strip .git suffix if present
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _api_get(path: str, accept: str = "application/vnd.github+json") -> requests.Response:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    return requests.get(
        f"{GITHUB_API_BASE}{path}",
        headers=headers,
        timeout=_REQUEST_TIMEOUT,
    )


def _fetch_repo_metadata(owner: str, repo: str) -> Optional[dict]:
    try:
        resp = _api_get(f"/repos/{owner}/{repo}")
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch repo metadata for %s/%s: %s", owner, repo, exc)
        return None


def _fetch_readme(owner: str, repo: str) -> Optional[str]:
    try:
        resp = _api_get(f"/repos/{owner}/{repo}/readme", accept="application/vnd.github.raw")
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.debug("No README for %s/%s: %s", owner, repo, exc)
        return None


def _fetch_latest_release(owner: str, repo: str) -> Optional[dict]:
    try:
        resp = _api_get(f"/repos/{owner}/{repo}/releases/latest")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("No releases for %s/%s: %s", owner, repo, exc)
        return None


def fetch_github_repo(owner: str, repo: str) -> dict:
    """Fetch repo metadata, README, and latest release in parallel.

    Returns a dict with keys: title, description, content, og_image, metadata.
    Raises RuntimeError if the repo metadata cannot be fetched.
    """
    futures = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures["meta"] = pool.submit(_fetch_repo_metadata, owner, repo)
        futures["readme"] = pool.submit(_fetch_readme, owner, repo)
        futures["release"] = pool.submit(_fetch_latest_release, owner, repo)

    meta = futures["meta"].result()
    readme_text = futures["readme"].result()
    release = futures["release"].result()

    if meta is None:
        raise RuntimeError(f"Could not fetch repository {owner}/{repo}. It may not exist or may be private.")

    full_name = meta.get("full_name", f"{owner}/{repo}")
    description = meta.get("description") or ""
    language = meta.get("language") or "Unknown"
    stars = meta.get("stargazers_count", 0)
    forks = meta.get("forks_count", 0)
    topics = meta.get("topics") or []
    homepage = meta.get("homepage") or ""
    avatar = meta.get("owner", {}).get("avatar_url") or ""

    # Build the assembled content blob
    lines = [
        f"Repository: {full_name}",
        f"Description: {description}",
        f"Language: {language} | Stars: {stars:,} | Forks: {forks:,}",
    ]
    if topics:
        lines.append(f"Topics: {', '.join(topics)}")
    if homepage:
        lines.append(f"Homepage: {homepage}")

    lines.append("")

    if readme_text:
        lines.append("--- README ---")
        lines.append(readme_text.strip())
        lines.append("")

    if release:
        tag = release.get("tag_name", "")
        release_name = release.get("name", tag)
        release_body = release.get("body") or ""
        header = f"--- Latest Release: {release_name} ({tag}) ---" if release_name != tag else f"--- Latest Release: {tag} ---"
        lines.append(header)
        if release_body:
            lines.append(release_body.strip())
        lines.append("")

    content = "\n".join(lines)

    # Truncate extremely large READMEs to keep content manageable
    max_content = 50_000
    if len(content) > max_content:
        content = content[:max_content] + "\n\n[Content truncated]"

    return {
        "title": full_name,
        "description": description,
        "content": content,
        "og_image": avatar,
        "metadata": {
            "language": language,
            "stars": stars,
            "forks": forks,
            "topics": topics,
            "homepage": homepage,
            "latest_release": release.get("tag_name") if release else None,
        },
    }

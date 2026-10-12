"""Web tools backed by the self-hosted Firecrawl instance.

Exposes the full Firecrawl surface: web search, single-page scrape, site map,
full crawl (blocking), and LLM-backed structured extract. All hit
FIRECRAWL_API_URL.

`web_scrape` falls back to the CamoFox stealth browser (CAMOFOX_URL) when
Firecrawl times out, errors, comes back empty or gets an error page: Firecrawl's
Playwright waits for the page's `load` event, which ad-heavy sites (Yahoo
Finance, AP) never fire, and it gets bot-blocked where CamoFox does not.

`image_search` hits the search middleware directly (IMAGE_SEARCH_URL — the
searchmw service that also backs Firecrawl's web search), which pools Brave's
image API and Tavily's include_images behind 429-driven failover.
"""
from __future__ import annotations

import json
import time

import httpx

from ..config import settings
from .registry import Tool

MAX_OUTPUT = 30_000  # cap chars returned to the model per tool call
FIRECRAWL_TIMEOUT_MS = 15_000  # web_scrape's Firecrawl budget before the CamoFox fallback


def _err(stage: str, exc: Exception) -> str:
    return f"Error during {stage}: {type(exc).__name__}: {exc}"


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"
    return text


def web_search(query: str, limit: int = 5) -> str:
    if not settings.FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    try:
        resp = httpx.post(
            f"{settings.FIRECRAWL_API_URL}/v1/search",
            json={"query": query, "limit": max(1, min(limit, 10))},
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as exc:
        return _err("web_search", exc)
    if not data:
        return f"No results for: {query}"
    lines = []
    for i, r in enumerate(data, 1):
        lines.append(f"{i}. {r.get('title', '(no title)')}\n   {r.get('url', '')}\n   {r.get('description', '')}")
    return "\n".join(lines)


def image_search(query: str, count: int = 4) -> str:
    if not settings.IMAGE_SEARCH_URL:
        return "Error: IMAGE_SEARCH_URL is not configured."
    try:
        resp = httpx.get(
            f"{settings.IMAGE_SEARCH_URL}/images",
            params={"q": query, "count": max(1, min(count, 10))},
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as exc:
        return _err("image_search", exc)
    if not results:
        return f"No images found for: {query}"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        line = f"{i}. {title}\n   image: {r.get('image_url', '')}"
        if r.get("source_url"):
            line += f"\n   from: {r['source_url']}"
        lines.append(line)
    return "\n".join(lines)


def _page(url: str, title: str, body: str) -> str:
    header = f"# {title}\n({url})\n\n" if title else f"({url})\n\n"
    return header + (_truncate(body) or "(no content extracted)")


def _firecrawl_scrape(url: str) -> tuple[str | None, str]:
    """(page, "") on a usable result, else (None, why it wasn't usable)."""
    try:
        resp = httpx.post(
            f"{settings.FIRECRAWL_API_URL}/v1/scrape",
            # Firecrawl's own per-page budget. Kept short: a page it can't
            # render by then is one CamoFox usually renders in seconds.
            json={"url": url, "formats": ["markdown"], "timeout": FIRECRAWL_TIMEOUT_MS},
            timeout=FIRECRAWL_TIMEOUT_MS / 1000 + 30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not payload.get("success"):
        return None, str(payload.get("error", "unknown error"))[:200]
    data = payload.get("data", {})
    meta = data.get("metadata", {})
    status = meta.get("statusCode") or 200
    md = (data.get("markdown") or "").strip()
    if status >= 400:
        return None, f"site returned HTTP {status}"
    if not md:
        return None, "no content extracted"
    return _page(url, meta.get("title", ""), md), ""


# CamoFox user for scrape fallbacks — separate from browser_use's tabs.
_CAMOFOX_USER = "scrape"


def _camofox_scrape(url: str) -> tuple[str | None, str]:
    """Render the page in CamoFox and read its visible text."""
    base = settings.CAMOFOX_URL
    tab_id = None
    try:
        r = httpx.post(
            f"{base}/tabs",
            json={"userId": _CAMOFOX_USER, "sessionKey": _CAMOFOX_USER, "url": url},
            timeout=60.0,
        )
        r.raise_for_status()
        tab_id = r.json().get("tabId")
        if not tab_id:
            return None, "no tab opened"
        r = httpx.post(
            f"{base}/tabs/{tab_id}/evaluate",
            json={
                "userId": _CAMOFOX_USER,
                "expression": "JSON.stringify([document.title, document.body ? document.body.innerText : ''])",
            },
            timeout=30.0,
        )
        r.raise_for_status()
        title, text = json.loads(r.json().get("result") or '["", ""]')
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if tab_id:
            try:
                httpx.delete(f"{base}/tabs/{tab_id}", params={"userId": _CAMOFOX_USER}, timeout=10.0)
            except Exception:
                pass
    text = (text or "").strip()
    if not text:
        return None, "no content extracted"
    return _page(url, title, text), ""


def web_scrape(url: str) -> str:
    if not settings.FIRECRAWL_API_URL and not settings.CAMOFOX_URL:
        return "Error: neither FIRECRAWL_API_URL nor CAMOFOX_URL is configured."
    why = "FIRECRAWL_API_URL is not configured"
    if settings.FIRECRAWL_API_URL:
        page, why = _firecrawl_scrape(url)
        if page:
            return page
    if not settings.CAMOFOX_URL:
        return f"Scrape failed: {why}"
    page, camo_why = _camofox_scrape(url)
    if page:
        return page
    return f"Scrape failed: Firecrawl ({why}); CamoFox ({camo_why})"


def web_map(url: str, limit: int = 100, search: str | None = None) -> str:
    """List all reachable URLs on a site (Firecrawl /v1/map)."""
    if not settings.FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    body = {"url": url, "limit": max(1, min(limit, 500))}
    if search:
        body["search"] = search
    try:
        resp = httpx.post(f"{settings.FIRECRAWL_API_URL}/v1/map", json=body, timeout=60.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return _err("web_map", exc)
    links = payload.get("data") or payload.get("links") or []
    if not links:
        return f"No pages found for {url}"
    # Firecrawl returns either string URLs or {url, ...} objects.
    flat = []
    for item in links:
        if isinstance(item, str):
            flat.append(item)
        elif isinstance(item, dict):
            flat.append(item.get("url") or item.get("link") or "")
    flat = [u for u in flat if u]
    head = f"Found {len(flat)} page(s) on {url}:\n"
    return _truncate(head + "\n".join(f"- {u}" for u in flat[:500]))


def web_crawl(url: str, limit: int = 10, max_wait: int = 90) -> str:
    """Crawl a site (blocking) and return aggregated markdown per page.

    Starts an async crawl, then polls until it completes or `max_wait` seconds
    elapse. Best for 'get me everything under this section' — for one page use
    web_scrape.
    """
    if not settings.FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    try:
        resp = httpx.post(
            f"{settings.FIRECRAWL_API_URL}/v1/crawl",
            json={"url": url, "limit": max(1, min(limit, 50))},
            timeout=30.0,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            return f"Crawl did not start: {job}"
        deadline = time.time() + max(10, min(max_wait, 300))
        status = None
        while time.time() < deadline:
            time.sleep(2)
            s = httpx.get(f"{settings.FIRECRAWL_API_URL}/v1/crawl/{job_id}", timeout=30.0)
            s.raise_for_status()
            status = s.json()
            if status.get("status") in ("completed", "failed", "cancelled"):
                break
        if not status:
            return f"Crawl {job_id} still running after timeout — partial results may be available later."
        if status.get("status") != "completed":
            return f"Crawl {job_id} ended with status '{status.get('status')}'."
        pages = status.get("data") or []
        if not pages:
            return f"Crawl of {url} returned no pages."
    except Exception as exc:
        return _err("web_crawl", exc)
    parts = [f"Crawled {len(pages)} page(s) from {url}:\n"]
    for p in pages:
        md = p.get("markdown", "") or ""
        meta = p.get("metadata") or {}
        purl = meta.get("sourceURL") or p.get("url") or ""
        parts.append(f"\n## {purl}\n{md[:4000]}")
    return _truncate("\n".join(parts))


def web_tools() -> list[Tool]:
    return [
        Tool(
            "web_search",
            "Search the web and get a ranked list of results (title, url, snippet). "
            "Use for current information or to find pages to scrape.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "Number of results (1-10, default 5)."},
                },
                "required": ["query"],
            },
            web_search,
        ),
        Tool(
            "image_search",
            "Search the web for IMAGES and get direct image URLs (with source pages). "
            "Use when the user asks what something looks like or wants a photo/picture. "
            "Include 1-2 of the returned image URLs in your reply, each pasted bare on "
            "its own line — every surface auto-embeds a lone image URL.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find images of, e.g. 'fisher cat animal'."},
                    "count": {"type": "integer", "description": "Number of images (1-10, default 4)."},
                },
                "required": ["query"],
            },
            image_search,
        ),
        Tool(
            "web_scrape",
            "Fetch a URL and return its main content as clean markdown. "
            "Ground your answer in the actual scraped content — don't claim "
            "the page said something it didn't.",
            {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            web_scrape,
        ),
        Tool(
            "web_map",
            "List all reachable URLs on a site (Firecrawl /v1/map). Use to discover "
            "pages under a domain before scraping specific ones. Optionally filter "
            "with `search` (only URLs matching that substring are returned).",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Root URL to map."},
                    "limit": {"type": "integer", "description": "Max URLs to return (1-500, default 100)."},
                    "search": {"type": "string", "description": "Optional substring filter for returned URLs."},
                },
                "required": ["url"],
            },
            web_map,
        ),
        Tool(
            "web_crawl",
            "Crawl an entire site or section and return aggregated markdown from each "
            "page (Firecrawl /v1/crawl, polled to completion). Slower than web_scrape — "
            "use when you need many pages under one path. `limit` caps page count.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Root URL to crawl from."},
                    "limit": {"type": "integer", "description": "Max pages (1-50, default 10)."},
                    "max_wait": {"type": "integer", "description": "Max seconds to wait for completion (10-300, default 90)."},
                },
                "required": ["url"],
            },
            web_crawl,
        ),
    ]

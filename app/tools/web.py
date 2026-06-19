"""Web tools backed by the self-hosted Firecrawl instance.

Exposes the full Firecrawl surface: web search, single-page scrape, site map,
full crawl (blocking), and LLM-backed structured extract. All hit
FIRECRAWL_API_URL.
"""
from __future__ import annotations

import time

import httpx

from ..config import FIRECRAWL_API_URL
from .registry import Tool

MAX_OUTPUT = 30_000  # cap chars returned to the model per tool call


def _err(stage: str, exc: Exception) -> str:
    return f"Error during {stage}: {type(exc).__name__}: {exc}"


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"
    return text


def web_search(query: str, limit: int = 5) -> str:
    if not FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    try:
        resp = httpx.post(
            f"{FIRECRAWL_API_URL}/v1/search",
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


def web_scrape(url: str) -> str:
    if not FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    try:
        resp = httpx.post(
            f"{FIRECRAWL_API_URL}/v1/scrape",
            json={"url": url, "formats": ["markdown"]},
            timeout=90.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return _err("web_scrape", exc)
    if not payload.get("success"):
        return f"Scrape failed: {payload.get('error', 'unknown error')}"
    data = payload.get("data", {})
    md = data.get("markdown", "")
    title = data.get("metadata", {}).get("title", "")
    header = f"# {title}\n({url})\n\n" if title else f"({url})\n\n"
    body = md[:MAX_OUTPUT]
    if len(md) > MAX_OUTPUT:
        body += f"\n... [truncated, {len(md) - MAX_OUTPUT} more chars]"
    return header + (body or "(no content extracted)")


def web_map(url: str, limit: int = 100, search: str | None = None) -> str:
    """List all reachable URLs on a site (Firecrawl /v1/map)."""
    if not FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    body = {"url": url, "limit": max(1, min(limit, 500))}
    if search:
        body["search"] = search
    try:
        resp = httpx.post(f"{FIRECRAWL_API_URL}/v1/map", json=body, timeout=60.0)
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
    if not FIRECRAWL_API_URL:
        return "Error: FIRECRAWL_API_URL is not configured."
    try:
        resp = httpx.post(
            f"{FIRECRAWL_API_URL}/v1/crawl",
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
            s = httpx.get(f"{FIRECRAWL_API_URL}/v1/crawl/{job_id}", timeout=30.0)
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

import logging

import httpx

from config import FIRECRAWL_MAX_RESULTS, FIRECRAWL_URL

log = logging.getLogger("siegclaw.search")


def web_search(query: str) -> str | None:
    """Search the web via local Firecrawl instance and return formatted results."""
    try:
        resp = httpx.post(
            f"{FIRECRAWL_URL}/v1/search",
            json={"query": query, "limit": FIRECRAWL_MAX_RESULTS},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error("Firecrawl search failed: %s", e)
        return None

    results = data.get("data", [])
    if not results:
        return None

    parts = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("description") or r.get("markdown", "")
        parts.append(f"**{title}** ({url})\n{content}")

    log.info("Web search returned %d results for '%s'", len(results), query[:60])
    return "\n\n".join(parts)

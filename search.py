import logging

from tavily import TavilyClient

from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

log = logging.getLogger("siegclaw.search")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def web_search(query: str) -> str | None:
    """Search the web via Tavily and return formatted results."""
    if not tavily_client:
        log.warning("Tavily not configured, skipping web search")
        return None

    try:
        response = tavily_client.search(
            query=query,
            max_results=TAVILY_MAX_RESULTS,
            search_depth="basic",
        )
    except Exception as e:
        log.error("Tavily search failed: %s", e)
        return None

    results = response.get("results", [])
    if not results:
        return None

    parts = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"**{title}** ({url})\n{content}")

    log.info("Web search returned %d results for '%s'", len(results), query[:60])
    return "\n\n".join(parts)

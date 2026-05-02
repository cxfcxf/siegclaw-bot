import logging

import httpx

from config import CAMOFOX_URL

log = logging.getLogger("siegclaw.browser")

# tabId keyed by user_id — persists across tool calls within a message
_sessions: dict[str, str] = {}

_TIMEOUT = 20
_SNAP_MAX_CHARS = 6000


def browse_page(url: str, user_id: str) -> str:
    """Open URL in real browser, return page title + ARIA snapshot."""
    if user_id in _sessions:
        close_browser_session(user_id)
    resp = httpx.post(
        f"{CAMOFOX_URL}/tabs/open",
        json={"userId": user_id, "url": url},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    tab_id = resp.json()["tabId"]
    _sessions[user_id] = tab_id

    # Wait for network idle
    httpx.post(
        f"{CAMOFOX_URL}/tabs/{tab_id}/wait",
        json={"userId": user_id, "timeout": 8000, "waitForNetwork": True},
        timeout=15,
    )

    return _snapshot(tab_id, user_id)


def browser_click(ref: str, user_id: str) -> str:
    """Click an element by its ARIA ref ID, return updated page snapshot."""
    tab_id = _sessions.get(user_id)
    if not tab_id:
        return "No active browser session — call browse_page first."

    resp = httpx.post(
        f"{CAMOFOX_URL}/tabs/{tab_id}/click",
        json={"userId": user_id, "ref": ref},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()

    httpx.post(
        f"{CAMOFOX_URL}/tabs/{tab_id}/wait",
        json={"userId": user_id, "timeout": 3000, "waitForNetwork": True},
        timeout=10,
    )
    return _snapshot(tab_id, user_id)


def browser_type(ref: str, text: str, user_id: str, submit: bool = False) -> str:
    """Type text into an element by its ARIA ref ID."""
    tab_id = _sessions.get(user_id)
    if not tab_id:
        return "No active browser session — call browse_page first."

    resp = httpx.post(
        f"{CAMOFOX_URL}/tabs/{tab_id}/type",
        json={"userId": user_id, "ref": ref, "text": text, "submit": submit},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return f"Typed successfully. Call browser_snapshot to see the updated page."


def browser_snapshot(user_id: str) -> str:
    """Get the current page state as an ARIA snapshot."""
    tab_id = _sessions.get(user_id)
    if not tab_id:
        return "No active browser session — call browse_page first."
    return _snapshot(tab_id, user_id)


def screenshot_bytes(user_id: str) -> bytes | None:
    """Return raw PNG screenshot of the current tab, or None on failure."""
    tab_id = _sessions.get(user_id)
    if not tab_id:
        return None
    try:
        resp = httpx.get(
            f"{CAMOFOX_URL}/tabs/{tab_id}/screenshot",
            params={"userId": user_id},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        log.warning("Screenshot failed: %s", e)
    return None


def close_browser_session(user_id: str):
    """Close the tab and clean up session."""
    tab_id = _sessions.pop(user_id, None)
    if tab_id:
        try:
            httpx.delete(
                f"{CAMOFOX_URL}/tabs/{tab_id}",
                params={"userId": user_id},
                timeout=5,
            )
        except Exception as e:
            log.warning("Failed to close browser tab: %s", e)


def _snapshot(tab_id: str, user_id: str) -> str:
    resp = httpx.get(
        f"{CAMOFOX_URL}/tabs/{tab_id}/snapshot",
        params={"userId": user_id},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    url = data.get("url", "")
    snapshot = data.get("snapshot", "")

    if len(snapshot) > _SNAP_MAX_CHARS:
        snapshot = snapshot[:_SNAP_MAX_CHARS] + "\n... (truncated)"

    return f"URL: {url}\n\n{snapshot}"

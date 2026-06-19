"""Stealth browser tool backed by the self-hosted CamoFox (camoufox) server.

CamoFox renders JS and bypasses bot detection (Cloudflare, fingerprinting). This
exposes it as a single polymorphic `browser_use` tool driven by an `action`
argument. The flow is an agent loop:

  1. action=open   -> open a URL, get a tab_id + an accessibility snapshot
                      (a token-efficient text view of the page with stable
                      element refs like `e1`, `e2`, `e3`).
  2. action=click/type/scroll/press/navigate -> mutate, then return the NEW
                      snapshot so you can see what changed and grab fresh refs.
  3. action=links  -> dump the links on the current page.
  4. action=close  -> close the tab when done.

Reach for this when web_scrape (Firecrawl) returns empty/garbage because the
page is JS-rendered or bot-blocked.
"""
from __future__ import annotations

import httpx

from ..config import CAMOFOX_URL
from .registry import Tool

# Single shared session for the harness (cookies persist across conversations,
# which is usually what you want for logins). Tabs auto-recycle when idle.
_USER = "harness"
_SESSION = "default"
_SNAPSHOT_CAP = 20_000  # chars of accessibility snapshot returned per action


def _err(msg: str) -> str:
    return f"Error: {msg}"


def _req(method: str, path: str, **kwargs):
    return httpx.request(method, f"{CAMOFOX_URL}{path}", timeout=60.0, **kwargs)


def _snapshot(tab_id: str) -> str:
    """Fetch the accessibility snapshot (text + element refs) for a tab."""
    r = _req("GET", f"/tabs/{tab_id}/snapshot", params={"userId": _USER})
    if r.status_code != 200:
        return f"(snapshot unavailable: HTTP {r.status_code} {r.text[:160]})"
    data = r.json()
    snap = data.get("snapshot") or data.get("text") or ""
    if not snap:
        snap = str(data)
    if len(snap) > _SNAPSHOT_CAP:
        snap = snap[:_SNAPSHOT_CAP] + f"\n... [snapshot truncated, {len(snap) - _SNAPSHOT_CAP} more chars]"
    return snap


def browser_use(
    action: str,
    tab_id: str | None = None,
    url: str | None = None,
    ref: str | None = None,
    text: str | None = None,
    key: str | None = None,
    direction: str = "down",
    press_enter: bool = False,
    macro: str | None = None,
    query: str | None = None,
) -> str:
    if not CAMOFOX_URL:
        return _err("CAMOFOX_URL is not configured.")

    try:
        # --- open: create a tab, optionally via a search macro ----------------
        if action == "open":
            if not url and not macro:
                return _err("open requires `url` (or `macro` + `query`).")
            body: dict = {"userId": _USER, "sessionKey": _SESSION}
            if macro:
                body["macro"] = macro
                if query:
                    body["query"] = query
            else:
                body["url"] = url
            r = _req("POST", "/tabs", json=body)
            if r.status_code != 200:
                return _err(f"open failed: HTTP {r.status_code} {r.text[:160]}")
            tab_id = r.json().get("tabId")
            if not tab_id:
                return _err(f"open: no tabId in response: {r.text[:160]}")
            return f"Opened tab {tab_id}.\n\n" + _snapshot(tab_id)

        # Every other action needs an existing tab.
        if not tab_id:
            return _err(
                f"`tab_id` is required for action '{action}' "
                "(open one first with action=open)."
            )

        # --- snapshot: just read the current page state ----------------------
        if action == "snapshot":
            return _snapshot(tab_id)

        # --- navigate: go to a new URL or run a search macro -----------------
        if action == "navigate":
            body = {"userId": _USER}
            if macro:
                body["macro"] = macro
                if query:
                    body["query"] = query
            elif url:
                body["url"] = url
            else:
                return _err("navigate requires `url` or `macro` + `query`.")
            r = _req("POST", f"/tabs/{tab_id}/navigate", json=body)
            if r.status_code != 200:
                return _err(f"navigate failed: HTTP {r.status_code} {r.text[:160]}")
            return _snapshot(tab_id)

        # --- click / type / press / scroll: mutate then re-snapshot ----------
        if action == "click":
            if not ref:
                return _err("click requires `ref` (an element ref from the snapshot, e.g. e3).")
            r = _req("POST", f"/tabs/{tab_id}/click", json={"userId": _USER, "ref": ref})
            if r.status_code != 200:
                return _err(f"click failed: HTTP {r.status_code} {r.text[:160]}")
            return _snapshot(tab_id)

        if action == "type":
            if not ref:
                return _err("type requires `ref`.")
            if text is None:
                return _err("type requires `text`.")
            r = _req(
                "POST", f"/tabs/{tab_id}/type",
                json={"userId": _USER, "ref": ref, "text": text, "pressEnter": bool(press_enter)},
            )
            if r.status_code != 200:
                return _err(f"type failed: HTTP {r.status_code} {r.text[:160]}")
            return _snapshot(tab_id)

        if action == "press":
            if not key:
                return _err("press requires `key` (e.g. Enter, Tab, Escape).")
            r = _req("POST", f"/tabs/{tab_id}/press", json={"userId": _USER, "key": key})
            if r.status_code != 200:
                return _err(f"press failed: HTTP {r.status_code} {r.text[:160]}")
            return _snapshot(tab_id)

        if action == "scroll":
            r = _req(
                "POST", f"/tabs/{tab_id}/scroll",
                json={"userId": _USER, "direction": direction},
            )
            if r.status_code != 200:
                return _err(f"scroll failed: HTTP {r.status_code} {r.text[:160]}")
            return _snapshot(tab_id)

        # --- links: dump anchor text + href for the current page -------------
        if action == "links":
            r = _req("GET", f"/tabs/{tab_id}/links", params={"userId": _USER})
            if r.status_code != 200:
                return _err(f"links failed: HTTP {r.status_code} {r.text[:160]}")
            data = r.json()
            raw = data.get("links") if isinstance(data, dict) else data
            if not raw:
                return "(no links)"
            out = []
            for i, l in enumerate(raw[:200], 1):
                if isinstance(l, dict):
                    label = (l.get("text") or l.get("aria-label") or "").strip()[:50]
                    href = l.get("href") or l.get("url") or ""
                    out.append(f"{i}. {label}{' — ' if label else ''}{href}")
                else:
                    out.append(f"{i}. {l}")
            return "\n".join(out)

        # --- close: release the tab -----------------------------------------
        if action == "close":
            r = _req("DELETE", f"/tabs/{tab_id}", params={"userId": _USER})
            if r.status_code in (200, 204):
                return f"Closed tab {tab_id}."
            return _err(f"close failed: HTTP {r.status_code} {r.text[:160]}")

        return _err(
            f"unknown action '{action}'. Valid: open, snapshot, navigate, "
            "click, type, press, scroll, links, close."
        )
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def browser_tools() -> list[Tool]:
    return [
        Tool(
            "browser_use",
            "Drive a stealth (camoufox) browser that renders JS and bypasses bot "
            "detection. Use when web_scrape comes back empty or blocked. "
            "STATEFUL AGENT LOOP: open a URL -> read the returned accessibility "
            "snapshot (text view of the page with element refs like e1/e2/e3) -> "
            "click/type/scroll using those refs -> each action returns the NEW "
            "snapshot so you can keep going. Always close the tab when finished. "
            "Supported macros for navigate/open: @google_search, @youtube_search, "
            "@amazon_search, @reddit_search, @reddit_subreddit, @wikipedia_search, "
            "@twitter_search, @yelp_search, @spotify_search, @netflix_search, "
            "@linkedin_search, @instagram_search, @tiktok_search, @twitch_search.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "snapshot", "navigate", "click", "type", "press", "scroll", "links", "close"],
                        "description": "What to do. open returns a tab_id + first snapshot; all others need tab_id.",
                    },
                    "tab_id": {"type": "string", "description": "Tab id from `open`. Required for everything except `open`."},
                    "url": {"type": "string", "description": "URL for `open`/`navigate` (mutually exclusive with `macro`)."},
                    "ref": {"type": "string", "description": "Element ref from a snapshot (e.g. e3) for `click`/`type`."},
                    "text": {"type": "string", "description": "Text to type (for `type`)."},
                    "key": {"type": "string", "description": "Keyboard key for `press` (Enter, Tab, Escape, ...)."},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Scroll direction (default down)."},
                    "press_enter": {"type": "boolean", "description": "Press Enter after typing (for `type`)."},
                    "macro": {"type": "string", "description": "Search macro for `open`/`navigate` (e.g. @google_search)."},
                    "query": {"type": "string", "description": "Search query when using a `macro`."},
                },
                "required": ["action"],
            },
            browser_use,
        ),
    ]

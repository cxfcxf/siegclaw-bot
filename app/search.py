"""Web and image search over pooled Brave and Tavily API keys.

This used to be a separate service (searchmw) that spoke SearXNG's API to a
self-hosted Firecrawl. Nothing else calls it any more, so it lives in-process:
`web_search` and `image_search` call `do_search` / `do_image_search` directly.

Keys are tried in order, key by key, following SEARCH_PROVIDER_ORDER. When a
provider returns a quota/rate error (429/402/403/432), that key is sidelined
and the next key/provider is tried — pooling free tiers into one allowance.
The sidelining lasts the provider's Retry-After when given (a transient rate
limit); otherwise it escalates with each consecutive miss (60s -> 120s -> 240s
-> ... capped at 1h) so a blip recovers fast while a genuinely exhausted key
isn't re-probed constantly. A success resets the escalation; a restart clears
all cooldowns. Network/server errors (timeout, DNS, 5xx) sideline the slot
briefly (ERROR_COOLDOWN) so a dead provider doesn't stall every query; a 401
(bad key) sidelines for COOLDOWN_MAX.

Queries are sanitized (quotes/backslashes/backticks removed) because those make
Brave do strict exact-match and return nothing.

`usage()` reports quota per key for the status page: Tavily slots are
live-queried against Tavily's free usage endpoint (exact used/limit); Brave
exposes no such endpoint, so Brave slots show our own count of successful calls
this calendar month, persisted to DATA_DIR/search_usage.json so restarts don't
zero it, against BRAVE_MONTHLY_LIMIT. The local count only sees traffic through
this bot; `set_count` seeds it with the dashboard's real number. All counts
(including seeds) zero at calendar-month rollover.

Keys, order, the Brave limit and the cache TTL are settings, read live on every
call, so a key pasted into the UI works on the next search.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from hashlib import sha1

from .config import DATA_DIR, settings

log = logging.getLogger("siegclaw.search")

SANITIZE = '"`\\'
RESULTS = 20            # results asked of a provider per query
TIMEOUT = 10.0          # upstream timeout (s)
# Escalating backoff for a 429 without a Retry-After: first miss sidelines for
# COOLDOWN_BASE, doubling each consecutive miss up to COOLDOWN_MAX.
COOLDOWN_BASE = 60.0
COOLDOWN_MAX = 3600.0
# Sideline for network/server errors (timeout, DNS, 5xx). Without this, a dead
# provider adds a full TIMEOUT to every query before failover; with it, the
# slot is skipped for a short while and only re-probed occasionally.
ERROR_COOLDOWN = 30.0
# When a provider answers with nothing, ask the next one before giving up.
FALLBACK_ON_EMPTY = True
# Cache size cap: distinct queries otherwise accumulate forever (expired entries
# are only dropped on re-lookup of the same key). At the cap, expired entries are
# swept; if still full, the soonest-to-expire entries are evicted.
CACHE_MAX = 1000
# Time-sensitive queries skip the cache entirely (always fresh) when they contain
# one of these words, or a year near the current one. Best-effort: agents often
# strip these down to bare keywords, so the TTL is the real staleness bound.
CACHE_BYPASS = frozenset(
    "now today tonight yesterday tomorrow live current currently latest recent"
    " score scores standings price prices stock stocks weather forecast temperature"
    " breaking news headlines trending update updates".split()
)

STATE_FILE = str(DATA_DIR / "search_usage.json")

# Connect directly to the provider APIs, never through an injected proxy.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _split(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _keys(provider: str) -> list[str]:
    return _split(settings.BRAVE_API_KEYS if provider == "brave" else settings.TAVILY_API_KEYS)


def _order() -> list[str]:
    """Providers in the order to try them: the configured ones first, then any
    known provider the setting left out."""
    order = [p for p in _split(settings.SEARCH_PROVIDER_ORDER.lower()) if p in WEB_FNS]
    return order + [p for p in WEB_FNS if p not in order]


def configured() -> bool:
    """True when at least one search key is set."""
    return any(_keys(p) for p in WEB_FNS)


class QuotaError(Exception):
    """Provider signalled rate/quota exhaustion (HTTP 429/402/403/432).

    cooldown: seconds to sideline this key, from the provider's Retry-After when
    available (a 1s per-second rate-limit then only sidelines Brave for ~1s,
    instead of the flat default), else None -> the caller escalates.
    """
    def __init__(self, msg, cooldown=None):
        super().__init__(msg)
        self.cooldown = cooldown


def _retry_after(e):
    try:
        return float(e.headers.get("Retry-After"))
    except (TypeError, ValueError, AttributeError):
        return None


def _http_json(req):
    with _opener.open(req, timeout=TIMEOUT) as r:
        return json.load(r)


# ---- backends: each returns a list of {url, title, content} ----
def call_brave(key, query, count, page):
    params = urllib.parse.urlencode({"q": query, "count": min(count, 20), "offset": max(0, page - 1)})
    req = urllib.request.Request(
        "https://api.search.brave.com/res/v1/web/search?" + params,
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    )
    try:
        data = _http_json(req)
    except urllib.error.HTTPError as e:
        if e.code == 401:  # bad/revoked key: never recovers without a restart
            raise QuotaError("brave HTTP 401 (bad key)", COOLDOWN_MAX)
        if e.code in (429, 402, 403):
            raise QuotaError(f"brave HTTP {e.code}", _retry_after(e))
        raise
    return [{"url": r.get("url"), "title": r.get("title"), "content": r.get("description", "")}
            for r in ((data.get("web") or {}).get("results") or [])]


def call_tavily(key, query, count, page):
    if page > 1:
        return []  # Tavily has no page-based pagination
    body = json.dumps({"query": query, "max_results": min(count, 20), "search_depth": "basic"}).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        data = _http_json(req)
    except urllib.error.HTTPError as e:
        if e.code == 401:  # bad/revoked key: never recovers without a restart
            raise QuotaError("tavily HTTP 401 (bad key)", COOLDOWN_MAX)
        if e.code in (429, 432, 402, 403):
            raise QuotaError(f"tavily HTTP {e.code}", _retry_after(e))
        raise
    return [{"url": r.get("url"), "title": r.get("title"), "content": r.get("content", "")}
            for r in (data.get("results") or [])]


# ---- image backends: each returns a list of {image_url, thumbnail, title, source_url} ----
def call_brave_images(key, query, count, page):
    if page > 1:
        return []
    params = urllib.parse.urlencode({"q": query, "count": min(count, 20)})
    req = urllib.request.Request(
        "https://api.search.brave.com/res/v1/images/search?" + params,
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    )
    try:
        data = _http_json(req)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise QuotaError("brave HTTP 401 (bad key)", COOLDOWN_MAX)
        if e.code in (429, 402, 403):
            raise QuotaError(f"brave HTTP {e.code}", _retry_after(e))
        raise
    out = []
    for r in data.get("results") or []:
        img = (r.get("properties") or {}).get("url") or (r.get("thumbnail") or {}).get("src")
        if not img:
            continue
        out.append({
            "image_url": img,
            "thumbnail": (r.get("thumbnail") or {}).get("src", ""),
            "title": r.get("title", ""),
            "source_url": r.get("url", ""),
        })
    return out


def call_tavily_images(key, query, count, page):
    if page > 1:
        return []
    body = json.dumps({
        "query": query, "max_results": 1, "search_depth": "basic",
        "include_images": True, "include_image_descriptions": True,
    }).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        data = _http_json(req)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise QuotaError("tavily HTTP 401 (bad key)", COOLDOWN_MAX)
        if e.code in (429, 432, 402, 403):
            raise QuotaError(f"tavily HTTP {e.code}", _retry_after(e))
        raise
    out = []
    for im in (data.get("images") or [])[:count]:
        if isinstance(im, str):
            im = {"url": im}
        if not im.get("url"):
            continue
        out.append({
            "image_url": im["url"], "thumbnail": "",
            "title": im.get("description", ""), "source_url": "",
        })
    return out


# Handlers per provider, in their natural fallback order.
WEB_FNS = {"brave": call_brave, "tavily": call_tavily}

# Image handlers keyed by the same provider names/keys/slots as web search —
# a key sidelined by a web-search 429 is (correctly) skipped for images too.
IMAGE_FNS = {"brave": call_brave_images, "tavily": call_tavily_images}


def slot_id(provider, key):
    return f"{provider}:{sha1(key.encode()).hexdigest()[:8]}"


def build_slots():
    """(provider, key) pairs in the order they are tried."""
    return [(p, k) for p in _order() for k in _keys(p)]


# In-memory per-slot state (cleared on restart; that naturally re-tries the
# first provider). Guarded by _state_lock: tool calls run on
# multiple threads and the _fails escalation is a read-modify-write.
#   _cooldowns[sid] = unix ts until which to skip this slot
#   _fails[sid]     = consecutive 429s without a Retry-After (drives escalation)
_cooldowns = {}
_fails = {}
_state_lock = threading.Lock()

# Per-slot count of successful upstream calls this calendar month — the only
# quota signal we can get for Brave (no usage API; billed against credit).
# Persisted to STATE_FILE (atomic tmp+rename) so a restart doesn't zero it;
# a new month does. Calendar-month reset is an approximation of the provider's
# billing cycle. Guarded by _state_lock like the cooldown state.
_counts = {}
_counts_month = time.strftime("%Y-%m")


def _load_counts():
    global _counts, _counts_month
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        if d.get("month") == _counts_month and isinstance(d.get("counts"), dict):
            _counts = d["counts"]
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("search: usage file unreadable (%s); counts start at 0", e)


def _roll_month_locked():
    """Zero all counts when the calendar month changes (fresh allowance —
    manual corrections via set_count are discarded too). Caller holds
    _state_lock."""
    global _counts, _counts_month
    month = time.strftime("%Y-%m")
    if month != _counts_month:
        _counts = {}
        _counts_month = month


def _save_counts_locked():
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"month": _counts_month, "counts": _counts}, f)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        log.warning("search: usage file write failed: %s", e)


def _bump_count(sid):
    """Record one successful upstream call. Caller holds _state_lock."""
    _roll_month_locked()
    _counts[sid] = _counts.get(sid, 0) + 1
    _save_counts_locked()

# Query cache: key "<sanitized query>|<page>" -> (expiry_ts, results, provider).
# Guarded by a lock because tool calls run on multiple threads.
_cache = {}
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0, "bypass": 0}


def sanitize(q):
    q = re.sub("[" + re.escape(SANITIZE) + "]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")  # a year, e.g. 2026
# Relative-time phrases anchored to "now" -> answer changes over time, so bypass.
# Matched as phrases (not bare words) so "work week"/"gap year" stay cacheable
# while "next week"/"last month"/"3 days ago" do not.
_REL_TIME_RE = re.compile(
    r"\b(next|last|this|past|coming|upcoming|previous)\s+"
    r"(week|weekend|month|year|quarter|season|day|days|night|hour|hours|"
    r"mon|tue|wed|thu|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago\b"
)


def is_fresh_query(q):
    """True if q looks time-sensitive and should bypass the cache."""
    ql = q.lower()
    if _REL_TIME_RE.search(ql):
        return True
    # Only the current year +/- 1 signals freshness; "world cup 2018" is stable
    # and cacheable, "election 2026" is not.
    this_year = time.localtime().tm_year
    if any(abs(int(m.group(0)) - this_year) <= 1 for m in _YEAR_RE.finditer(ql)):
        return True
    words = set(re.findall(r"[a-z0-9]+", ql))
    return any(w in words for w in CACHE_BYPASS)


def _search_providers(q, page):
    """Run the web-search failover chain over an already-sanitized query."""
    return _run_failover(WEB_FNS, q, RESULTS, page)


def _run_failover(fns, q, count, page):
    """The failover chain, parameterized by a {provider: handler} map so web
    and image search share the same slots, cooldowns, and escalation state."""
    now = time.time()
    used = None
    for provider, key in build_slots():
        fn = fns.get(provider)
        if fn is None:
            continue
        sid = slot_id(provider, key)
        with _state_lock:
            if _cooldowns.get(sid, 0) > now:
                continue
        try:
            results = fn(key, q, count, page)
        except QuotaError as e:
            with _state_lock:
                if e.cooldown is not None:
                    cd = e.cooldown                   # provider sent Retry-After: transient
                else:
                    n = _fails.get(sid, 0)            # escalate on each consecutive miss
                    cd = min(COOLDOWN_BASE * (2 ** n), COOLDOWN_MAX)
                    _fails[sid] = n + 1
                _cooldowns[sid] = now + cd
            log.warning("search: %s quota/rate: %s -> next key (cooldown %.0fs)", sid, e, cd)
            continue
        except Exception as e:
            # Network/server trouble (timeout, DNS, 5xx): sideline briefly so a
            # dead provider doesn't add TIMEOUT to every query.
            with _state_lock:
                _cooldowns[sid] = now + ERROR_COOLDOWN
            log.warning("search: %s error: %s: %s -> next key (cooldown %.0fs)",
                        sid, type(e).__name__, e, ERROR_COOLDOWN)
            continue
        with _state_lock:
            _fails.pop(sid, None)  # success: reset escalation
            _bump_count(sid)
        used = provider
        if results or not FALLBACK_ON_EMPTY:
            return results, used
    return [], used


def do_search(query, page):
    """Returns (results, provider, cached). Caches non-empty results per the TTL,
    unless the query looks time-sensitive (then it always goes live)."""
    q = sanitize(query)
    ttl = settings.SEARCH_CACHE_TTL
    cacheable = ttl > 0 and not is_fresh_query(q)
    key = f"{q}|{page}"

    if cacheable:
        now = time.time()
        with _cache_lock:
            hit = _cache.get(key)
            if hit and hit[0] > now:
                _cache_stats["hits"] += 1
                return hit[1], hit[2], True
            if hit:
                _cache.pop(key, None)  # expired
            _cache_stats["misses"] += 1
    elif ttl > 0:
        with _cache_lock:
            _cache_stats["bypass"] += 1

    results, used = _search_providers(q, page)

    if cacheable and results:  # never cache empties
        _cache_put(key, results, used, ttl)
    return results, used, False


def _cache_put(key, results, used, ttl):
    now = time.time()
    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            for k in [k for k, v in _cache.items() if v[0] <= now]:
                del _cache[k]  # sweep expired
            while len(_cache) >= CACHE_MAX:  # still full: evict soonest-to-expire
                del _cache[min(_cache, key=lambda k: _cache[k][0])]
        _cache[key] = (now + ttl, results, used)


def _tavily_quota(key):
    """Live used/limit from Tavily's free usage endpoint (doesn't burn a
    credit). Returns None if unreachable — usage() should never fail because
    an upstream is down."""
    req = urllib.request.Request(
        "https://api.tavily.com/usage",
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with _opener.open(req, timeout=5) as r:
            acct = (json.load(r).get("account") or {})
        return {"used": acct.get("plan_usage"), "limit": acct.get("plan_limit"),
                "source": "tavily-api"}
    except Exception:
        return None


def do_image_search(query, count):
    """Image search over the same failover chain. Returns (results, provider,
    cached). Image results are stable, so they're always cached (no freshness
    bypass) under an 'img|' cache-key prefix."""
    q = sanitize(query)
    key = f"img|{q}|{count}"
    ttl = settings.SEARCH_CACHE_TTL
    if ttl > 0:
        now = time.time()
        with _cache_lock:
            hit = _cache.get(key)
            if hit and hit[0] > now:
                _cache_stats["hits"] += 1
                return hit[1], hit[2], True
            if hit:
                _cache.pop(key, None)
            _cache_stats["misses"] += 1
    results, used = _run_failover(IMAGE_FNS, q, count, 1)
    if ttl > 0 and results:
        _cache_put(key, results, used, ttl)
    return results, used, False


def usage() -> dict:
    """Per-key quota, cooldown state and cache stats, for the status page."""
    now = time.time()
    slots = []
    for provider, key in build_slots():
        sid = slot_id(provider, key)
        with _state_lock:
            _roll_month_locked()
            cd_until = _cooldowns.get(sid, 0)
            fails = _fails.get(sid, 0)
            calls = _counts.get(sid, 0)
        if provider == "tavily":
            quota = _tavily_quota(key)
        else:  # brave: no usage API -> our own count vs the configured cap
            quota = {"used": calls, "limit": settings.BRAVE_MONTHLY_LIMIT or None,
                     "source": "local-count"}
        slots.append({
            "provider": provider, "slot": sid,
            "cooling_down": cd_until > now,
            "cooldown_until": cd_until if cd_until > now else None,
            "consecutive_fails": fails,
            "calls_this_month": calls,
            "quota": quota,
        })
    with _cache_lock:
        cache = {**_cache_stats, "size": len(_cache), "ttl": settings.SEARCH_CACHE_TTL}
    total = cache["hits"] + cache["misses"]
    cache["hit_rate"] = round(cache["hits"] / total, 3) if total else None
    return {"order": _order(), "slots": slots, "cache": cache}


def set_count(count: int, provider: str | None = None, slot: str | None = None) -> dict:
    """Seed one slot's monthly call count with the real number from the
    provider's dashboard (the local count only starts when the key is added).
    `slot` disambiguates when a provider has several keys. Raises ValueError
    unless exactly one slot matches."""
    if count < 0:
        raise ValueError("count must be >= 0")
    all_slots = [(slot_id(p, k), p) for p, k in build_slots()]
    matched = [s for s, p in all_slots if s == slot or (slot is None and p == provider)]
    if len(matched) != 1:
        raise ValueError("need exactly one matching slot; slots: "
                         + (", ".join(s for s, _p in all_slots) or "none configured"))
    sid = matched[0]
    with _state_lock:
        _roll_month_locked()
        _counts[sid] = count
        _save_counts_locked()
    log.info("search: counter set: %s = %d", sid, count)
    return {"slot": sid, "calls_this_month": count}


_load_counts()

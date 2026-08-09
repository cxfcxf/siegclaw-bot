"""Configuration and provider auto-detection.

Everything is driven by environment variables (see .env.example). A provider is
"available" (and therefore offered in the web UI) when its API key is set, or,
for keyless local engines, when its OpenAI-compatible /models endpoint responds.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def _path(env: str, default: str) -> Path:
    return (BASE_DIR / os.getenv(env, default)).resolve()


WORKSPACE_DIR = _path("WORKSPACE_DIR", "./workspace")
DATA_DIR = _path("DATA_DIR", "./data")
WIKI_DIR = _path("WIKI_DIR", "./wiki")
# The Discord-channel wiki: a SEPARATE corpus with its own home page, index and
# pages. Channel users can reach only this one; the owner's private wiki above
# is never read, searched or indexed on a channel turn. See app/wiki.py.
WIKI_PUBLIC_DIR = _path("WIKI_PUBLIC_DIR", "./wiki-public")
MCP_CONFIG_PATH = _path("MCP_CONFIG_PATH", "./mcp.json")

UPLOADS_DIR = DATA_DIR / "uploads"

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
WIKI_DIR.mkdir(parents=True, exist_ok=True)
WIKI_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --- Research stack --------------------------------------------------------
FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "").rstrip("/")
CAMOFOX_URL = os.getenv("CAMOFOX_URL", "").rstrip("/")
# The search middleware behind Firecrawl (searchmw), hit directly for /images.
IMAGE_SEARCH_URL = os.getenv("IMAGE_SEARCH_URL", "").rstrip("/")

# --- Locale ----------------------------------------------------------------
# IANA timezone injected into the system prompt so the model always knows "now".
HARNESS_TZ = os.getenv("HARNESS_TZ", "America/Los_Angeles")

# --- Limits ----------------------------------------------------------------
BASH_TIMEOUT = int(os.getenv("BASH_TIMEOUT", "120"))
# Newest cron-run conversations kept per job (older ones are auto-deleted, so a
# frequent job can't grow the DB forever). Moving a run out of its "Cron: <job>"
# group exempts it from pruning.
CRON_KEEP_RUNS = int(os.getenv("CRON_KEEP_RUNS", "30"))
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "25"))

# Chat-template kwarg that toggles model reasoning per request (model-dependent;
# e.g. "enable_thinking" for this gemma/Qwen-style template).
THINK_KWARG = os.getenv("THINK_KWARG", "enable_thinking")

# --- Discord ---------------------------------------------------------------
# When a valid token is set, the app connects to Discord on startup (in the same
# process as the web UI). Leave empty to run web-UI-only.
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
# Discord is multi-user; the builtin shell/file tools run in this container, so
# they're withheld from Discord unless explicitly enabled.
DISCORD_ENABLE_SHELL = os.getenv("DISCORD_ENABLE_SHELL", "false").lower() in ("1", "true", "yes")
# DMs are the OWNER surface: they carry the private wiki, read-write. Anyone who
# shares a server with the bot can DM it (Discord has no owner-only-DM setting),
# so non-owner DMs are ignored. Empty = resolve the application owner from
# Discord at runtime; set an id here to override (e.g. a second account).
DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID", "").strip()
# Stream DM replies by editing the message in place (~1s per edit — Discord has
# no real streaming, so this is send-then-overwrite). Off by default: the reply
# lands as one complete message when the turn finishes.
DISCORD_STREAM_DMS = os.getenv("DISCORD_STREAM_DMS", "false").lower() in ("1", "true", "yes")
MAX_DISCORD_LENGTH = 2000

# --- Default model order (shared by every surface) -------------------------
# Every NEW conversation — web UI, Discord DM, Discord channel mention — starts
# on this model. The preferred default is tried first; if its provider isn't
# available right now (e.g. the local llama.cpp server is down), the fallback is
# used instead. DEFAULT_MODEL blank means "whatever the provider serves first"
# (handy for llama.cpp, which serves one model). EFFORT is the reasoning effort
# for providers that support it (DeepSeek/OpenRouter); ignored elsewhere.
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "llamacpp").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip()
DEFAULT_EFFORT = os.getenv("DEFAULT_EFFORT", "").strip() or None
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "deepseek").strip()
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "deepseek-v4-flash").strip()
FALLBACK_EFFORT = os.getenv("FALLBACK_EFFORT", "high").strip() or None

# --- Discord context window ------------------------------------------------
# Hybrid time/count window over live Discord channel history (Discord is the
# source of truth — these turns are not stored in the SQLite conversation db).
CONTEXT_MESSAGE_COUNT = int(os.getenv("CONTEXT_MESSAGE_COUNT", "50"))
CONTEXT_TIME_WINDOW_HOURS = int(os.getenv("CONTEXT_TIME_WINDOW_HOURS", "24"))
CONTEXT_ACTIVITY_THRESHOLD = int(os.getenv("CONTEXT_ACTIVITY_THRESHOLD", "30"))
CONTEXT_MAX_MESSAGES = int(os.getenv("CONTEXT_MAX_MESSAGES", "150"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "16000"))


@dataclass
class ProviderSpec:
    id: str
    name: str
    base_url_env: str | None
    base_url_default: str
    key_env: str | None  # None => keyless (local engine)

    def base_url(self) -> str:
        override = os.getenv(self.base_url_env) if self.base_url_env else None
        if override:
            return override.rstrip("/")
        return self.base_url_default.rstrip("/")

    def api_key(self) -> str | None:
        if self.key_env:
            return os.getenv(self.key_env) or None
        return None


# Registry of known OpenAI-compatible providers.
KNOWN_PROVIDERS: list[ProviderSpec] = [
    ProviderSpec("openai", "OpenAI", "OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ProviderSpec("openrouter", "OpenRouter", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ProviderSpec("deepseek", "DeepSeek", "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    ProviderSpec("xiaomi", "Xiaomi MiMo", "XIAOMI_BASE_URL", "https://api.xiaomimimo.com/v1", "XIAOMI_API_KEY"),
    ProviderSpec("llamacpp", "llama.cpp (local)", "LLAMACPP_BASE_URL", "http://localhost:8080/v1", None),
]


def get_provider(provider_id: str) -> ProviderSpec | None:
    return next((p for p in KNOWN_PROVIDERS if p.id == provider_id), None)


def _models_reachable(base_url: str, api_key: str | None, timeout: float = 4.0) -> list[dict] | None:
    """Return [{id, context}] if the OpenAI-compatible /models endpoint responds,
    else None. `context` is the model's max context window when reported."""
    if not base_url:
        return None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(f"{base_url}/models", headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [
            {"id": m["id"], "context": _context_of(m)}
            for m in data
            if "id" in m
        ]
    except Exception:
        return None


@dataclass
class AvailableProvider:
    id: str
    name: str
    base_url: str
    models: list[str] = field(default_factory=list)
    model_context: dict[str, int] = field(default_factory=dict)
    effort_levels: list[str] = field(default_factory=list)


# Reasoning-effort levels a provider accepts when thinking is on. Absent =>
# on/off only. (DeepSeek: high/max; low/medium map to high, xhigh to max.)
EFFORT_LEVELS: dict[str, list[str]] = {
    "deepseek": ["high", "max"],
}

def _openrouter_context_index() -> dict[str, int]:
    """Context windows harvested from OpenRouter's /models, which — unlike most
    direct provider APIs (DeepSeek, MiMo return only id/owner) — reports
    context_length for every model. Used to fill context for those providers so
    the UI meter works without any hardcoded values.

    Indexed by both the full OpenRouter id (``vendor/model``) and the bare model
    id, so a direct provider's plain id (e.g. ``deepseek-v4-flash``) matches
    OpenRouter's ``deepseek/deepseek-v4-flash``. Requires OPENROUTER_API_KEY;
    returns {} otherwise. Reuses the day-TTL model cache (with background
    refresh), so it adds no extra fetch beyond detecting OpenRouter itself."""
    spec = get_provider("openrouter")
    if spec is None or not spec.api_key():
        return {}
    index: dict[str, int] = {}
    for m in _cached_models("openrouter", spec.base_url(), spec.api_key()):
        ctx = m.get("context")
        if not ctx:
            continue
        mid = m["id"]
        index[mid] = ctx
        bare = mid.split("/", 1)[1] if "/" in mid else mid
        index.setdefault(bare, ctx)  # full id wins on a bare-name collision
    return index


def _context_of(m: dict) -> int | None:
    """Pull a max-context-window value out of a /models entry across providers.

    Field name/shape varies: llama.cpp exposes it under meta.n_ctx, OpenRouter as
    a top-level context_length, others (e.g. LM Studio, various gateways) use context_window.
    """
    for key in ("context_length", "context_window", "max_context_length"):
        v = m.get(key)
        if isinstance(v, int):
            return v
    meta = m.get("meta") or {}
    for key in ("n_ctx", "context_length", "context_window"):
        v = meta.get(key)
        if isinstance(v, int):
            return v
    return None


# Provider detection, two-tier for speed:
# - Keyless local engines (llama.cpp) are probed via a real HTTP GET /models on
#   a SHORT cache (PROVIDER_LIVENESS_CACHE_TTL). A TCP connect isn't enough — a
#   proxy/port-forward (or the server process with no model loaded) happily
#   accepts TCP but never answers the request, so we require an actual HTTP
#   response to consider it up. The probe is cached briefly so repeated refreshes
#   stay fast while still noticing an outage within seconds.
# - The (large, rarely-changing) model lists of keyed cloud providers are fetched
#   at most once per PROVIDER_MODELS_CACHE_TTL (a day) via _cached_models.
PROVIDER_MODELS_CACHE_TTL = float(os.getenv("PROVIDER_MODELS_CACHE_TTL", str(86400)))
PROVIDER_LIVENESS_CACHE_TTL = float(os.getenv("PROVIDER_LIVENESS_CACHE_TTL", "10"))
_LIVENESS_TIMEOUT = float(os.getenv("PROVIDER_LIVENESS_TIMEOUT", "2"))
_models_cache: dict[str, tuple[float, list[dict]]] = {}
_liveness_cache: dict[str, tuple[float, list[dict] | None]] = {}

# Stale-while-revalidate: once a cache entry exists, an expired one is served
# immediately while a daemon thread refreshes it in the background, so a slow or
# down upstream never blocks /api/providers after the first warm load. Refreshes
# are deduped by key so overlapping requests don't spawn duplicate probes.
_refresh_inflight: set[str] = set()
_refresh_lock = threading.Lock()


def _refresh_async(key: str, work: Callable[[], object]) -> None:
    with _refresh_lock:
        if key in _refresh_inflight:
            return
        _refresh_inflight.add(key)

    def run() -> None:
        try:
            work()
        finally:
            with _refresh_lock:
                _refresh_inflight.discard(key)

    threading.Thread(target=run, daemon=True).start()

# Send-time fallback: if the chosen provider isn't serving, retry this many
# times (transient blips) before switching the conversation to the fallback
# model. The switch is persisted on the conversation, so it sticks for that
# session and won't flap back if the original returns mid-conversation.
SEND_FALLBACK_RETRIES = int(os.getenv("SEND_FALLBACK_RETRIES", "3"))
SEND_FALLBACK_RETRY_DELAY = float(os.getenv("SEND_FALLBACK_RETRY_DELAY", "0.75"))


def provider_serving(provider_id: str) -> bool:
    """True if the provider can serve a request right now — used for send-time
    fallback. Keyed (cloud) providers count as serving when their key is set
    (they're effectively always up). Keyless local engines get a fresh, UNCACHED
    HTTP /models probe — the only check that sees through a port-forward or a
    server with no model loaded (a TCP connect would falsely succeed). The fresh
    result is written back into the liveness cache, so once a stopped engine is
    caught here, detect_providers/resolve_default_model immediately stop
    offering it instead of serving the stale "up" entry until a background
    refresh happens to run."""
    spec = get_provider(provider_id)
    if spec is None:
        return False
    if spec.key_env:
        return bool(spec.api_key())
    return _do_probe_keyless(provider_id, spec.base_url()) is not None


def model_valid_for(provider_id: str, model: str) -> bool:
    """True if `model` is plausibly usable on `provider`. Keyless local engines
    accept anything (blank = "whatever's loaded"; a custom id is the server's
    call). Keyed cloud providers must use a name from their known model list — a
    blank or foreign name (e.g. a llama.cpp model id sent to DeepSeek) is rejected
    so we fall back instead of 400-ing and persisting a bad provider/model pair.
    An empty/unknown list is treated as "can't validate" (allow)."""
    spec = get_provider(provider_id)
    if spec is None:
        return False
    if not spec.key_env:
        return True
    if not model:
        return False
    known = next((p.models for p in detect_providers() if p.id == provider_id), None)
    return not known or model in known


def _do_probe_keyless(provider_id: str, base_url: str) -> list[dict] | None:
    """Perform the actual liveness probe and store the result in the cache."""
    result = _models_reachable(base_url, None, timeout=_LIVENESS_TIMEOUT)
    _liveness_cache[provider_id] = (time.monotonic(), result)
    return result


def _probe_keyless(provider_id: str, base_url: str) -> list[dict] | None:
    """Short-TTL-cached HTTP probe of a keyless local engine's /models endpoint.
    Returns the model list when the server is actually serving, or None when
    it's down. This IS the liveness check for keyless engines: a TCP connect
    can't be trusted here — a port-forward (or the server process with no model
    loaded) will accept TCP but never answer the request. Requiring a real HTTP
    /models response catches both.

    A cold cache probes synchronously so the first detection is accurate; an
    expired entry is served stale and refreshed in the background, so a down or
    slow engine never makes /api/providers pay the probe timeout on the request
    path. (Send-time correctness is unaffected: provider_serving() always probes
    fresh and uncached.)"""
    now = time.monotonic()
    cached = _liveness_cache.get(provider_id)
    if cached is not None:
        if (now - cached[0]) >= PROVIDER_LIVENESS_CACHE_TTL:
            _refresh_async(f"live:{provider_id}", lambda: _do_probe_keyless(provider_id, base_url))
        return cached[1]
    return _do_probe_keyless(provider_id, base_url)


def _do_fetch_models(provider_id: str, base_url: str, api_key: str | None) -> list[dict] | None:
    """Fetch the provider's model list and cache it on success (failures are not
    cached, so a transient outage is retried rather than locked in)."""
    fetched = _models_reachable(base_url, api_key)
    if fetched is not None:
        _models_cache[provider_id] = (time.monotonic(), fetched)
    return fetched


def _cached_models(provider_id: str, base_url: str, api_key: str | None) -> list[dict]:
    """The provider's model list, fetched at most once per
    PROVIDER_MODELS_CACHE_TTL (a day by default). A cold cache fetches
    synchronously so the first load has a list; an expired entry is served stale
    and refreshed in the background, so the day-boundary expiry never blocks a
    request on a slow cloud /models call."""
    now = time.monotonic()
    cached = _models_cache.get(provider_id)
    if cached is not None:
        if (now - cached[0]) >= PROVIDER_MODELS_CACHE_TTL:
            _refresh_async(f"models:{provider_id}", lambda: _do_fetch_models(provider_id, base_url, api_key))
        return cached[1]
    fetched = _do_fetch_models(provider_id, base_url, api_key)
    return fetched if fetched is not None else []


def detect_providers() -> list[AvailableProvider]:
    """Detect which providers are usable right now.

    - Keyed (cloud) providers: included when the API key is set; models come
      from the day-long model cache (_cached_models). Cloud providers are
      effectively always up when keyed, so they aren't probed.
    - Keyless local engines: included only when the short-TTL HTTP /models probe
      (_probe_keyless) answers — that probe IS the liveness check, and (unlike a
      TCP ping) it sees through port-forwards / a server with no model loaded.
    """
    available: list[AvailableProvider] = []
    or_ctx = _openrouter_context_index()  # context source for APIs that omit it
    for spec in KNOWN_PROVIDERS:
        base_url = spec.base_url()
        key = spec.api_key()
        effort = EFFORT_LEVELS.get(spec.id, [])

        if spec.key_env:  # cloud / keyed provider
            if not key:
                continue
            models = _cached_models(spec.id, base_url, key)
        else:  # keyless local engine — only if the HTTP /models probe answers
            models = _probe_keyless(spec.id, base_url)
            if models is None:
                continue

        ids = [m["id"] for m in models]
        # Prefer the context the API actually reports; otherwise look it up in
        # OpenRouter's catalog (by full vendor/model id, then bare id) for
        # providers (DeepSeek, MiMo) whose own /models omits the field.
        ctx = {}
        for m in models:
            c = m["context"] or or_ctx.get(f"{spec.id}/{m['id']}") or or_ctx.get(m["id"])
            if c:
                ctx[m["id"]] = c
        available.append(AvailableProvider(spec.id, spec.name, base_url, ids, ctx, effort))
    return available


def resolve_default_model(
    providers: dict[str, AvailableProvider] | None = None,
) -> tuple[str, str, str | None] | None:
    """The model a NEW conversation starts on, for every surface.

    Order: the preferred default (DEFAULT_PROVIDER/MODEL) if that provider is
    available right now; otherwise the fallback (FALLBACK_PROVIDER/MODEL/EFFORT);
    last resort, the first detected provider + its first model. A blank model
    means "use whatever the provider lists first". Returns
    (provider_id, model, effort) or None if nothing usable is available.

    Pass an already-detected provider map to avoid a second detection pass (the
    /api/providers endpoint detects once and shares it).
    """
    if providers is None:
        providers = {p.id: p for p in detect_providers()}

    pref = providers.get(DEFAULT_PROVIDER)
    if pref is not None:
        model = DEFAULT_MODEL or (pref.models[0] if pref.models else None)
        if model:
            return pref.id, model, DEFAULT_EFFORT

    fb = providers.get(FALLBACK_PROVIDER)
    if fb is not None:
        model = FALLBACK_MODEL or (fb.models[0] if fb.models else None)
        if model:
            return fb.id, model, FALLBACK_EFFORT

    for p in providers.values():
        if p.models:
            return p.id, p.models[0], None
    return None


def effort_for(provider: str, model: str) -> str | None:
    """The reasoning effort to use for an already-chosen (provider, model) — so a
    conversation resumed onto the fallback model keeps its configured effort.
    Matches the default/fallback entries; otherwise None (let the model default)."""
    if provider == DEFAULT_PROVIDER and (not DEFAULT_MODEL or model == DEFAULT_MODEL):
        return DEFAULT_EFFORT
    if provider == FALLBACK_PROVIDER and model == FALLBACK_MODEL:
        return FALLBACK_EFFORT
    return None

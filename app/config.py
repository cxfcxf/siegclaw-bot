"""Configuration and provider auto-detection.

Paths come from the process environment because the settings store lives inside
one of them — a path cannot be read from the database that a path locates. They
are the only configuration that does; everything else is a live setting,
declared in `app.settings`, stored in SQLite and edited from the web UI. Read
those through the re-exported ``settings`` object (``settings.HARNESS_TZ``) —
never copy one into a module constant, or it will stop tracking changes.

There is no .env file and no dotenv loader: on first run the store imports
whatever the environment holds, and after that the database is the only source.

Models come from a single OpenAI-compatible endpoint (LLM_BASE_URL, with an
optional LLM_API_KEY): OpenRouter by default, but any server speaking the same
API works — a local llama.cpp included. It is offered in the web UI whenever a
base URL is set.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import settings as settings_store
from .settings import settings

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def _path(env: str, default: str) -> Path:
    """Paths are read from the real environment (e.g. compose's `environment:`),
    not from the settings store — the store's database lives inside DATA_DIR."""
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

# Settings live beside the conversation database, in the data directory above.
settings_store.init(DATA_DIR / "settings.db")

# Discord's hard per-message limit — a protocol fact, not a preference.
MAX_DISCORD_LENGTH = 2000


def _blank_to_none(value: str | None) -> str | None:
    """A blank effort means "let the model decide", which the API expresses as
    an absent field rather than an empty string."""
    return value or None


@dataclass
class ProviderSpec:
    id: str

    def base_url(self) -> str:
        return settings.LLM_BASE_URL.strip().rstrip("/")

    def api_key(self) -> str | None:
        return settings.LLM_API_KEY or None

    def name(self) -> str:
        """Display name: "OpenRouter" for OpenRouter, else the endpoint's host
        (e.g. "localhost:8080" for a local llama.cpp)."""
        if is_openrouter(self.base_url()):
            return "OpenRouter"
        return urlparse(self.base_url()).netloc or self.base_url()


# The one endpoint. Its id is what conversations store as their provider.
PROVIDER_ID = "llm"
PROVIDER = ProviderSpec(PROVIDER_ID)


def get_provider(provider_id: str) -> ProviderSpec | None:
    return PROVIDER if provider_id == PROVIDER_ID else None


def is_openrouter(base_url: str) -> bool:
    """Whether the endpoint is OpenRouter, whose request extensions (unified
    reasoning, cache_control, sticky sessions, host routing) apply only there."""
    host = urlparse(base_url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def probe_models(base_url: str, api_key: str | None, timeout: float = 4.0) -> list[dict] | None:
    """Return [{id, context}] if the OpenAI-compatible /models endpoint responds,
    else None. `context` is the model's max context window when reported.

    Uncached, so the settings UI can use it to validate a base URL + key before
    they are saved."""
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


# Reasoning-effort levels offered when thinking is on. OpenRouter normalizes
# effort across models (a level a model lacks maps to its nearest one); other
# servers get OpenAI's standard low/medium/high.
OPENROUTER_EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]
STANDARD_EFFORT_LEVELS = ["low", "medium", "high"]

def _context_of(m: dict) -> int | None:
    """Pull a max-context-window value out of a /models entry across providers.

    Field name/shape varies: providers may report context_length, and gateways
    may use context_window or nested metadata.
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


# Cloud model catalogs are cached (PROVIDER_MODELS_CACHE_TTL) and refreshed in
# the background.
_models_cache: dict[str, tuple[float, list[dict]]] = {}

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


def _on_settings_changed(changed: set[str]) -> None:
    """Drop cached model catalogs when the credentials or base URL behind them
    change — otherwise a newly entered key would keep serving the catalog the
    old one fetched (or no catalog at all), for up to a day."""
    if any(k.endswith(("_API_KEY", "_BASE_URL")) for k in changed):
        _models_cache.clear()


settings_store.on_change(_on_settings_changed)


def provider_serving(provider_id: str) -> bool:
    """Whether the endpoint is configured; completion failures trigger fallback.
    The key is optional — a local server usually takes none."""
    spec = get_provider(provider_id)
    return spec is not None and bool(spec.base_url())


def _do_fetch_models(provider_id: str, base_url: str, api_key: str | None) -> list[dict] | None:
    """Fetch the provider's model list and cache it on success (failures are not
    cached, so a transient outage is retried rather than locked in)."""
    fetched = probe_models(base_url, api_key)
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
        if (now - cached[0]) >= settings.PROVIDER_MODELS_CACHE_TTL:
            _refresh_async(f"models:{provider_id}", lambda: _do_fetch_models(provider_id, base_url, api_key))
        return cached[1]
    fetched = _do_fetch_models(provider_id, base_url, api_key)
    return fetched if fetched is not None else []


def model_matches_filter(model_id: str) -> bool:
    """MODEL_FILTER is comma-separated, case-insensitive substrings; a model is
    listed when any one matches. Blank lists every model."""
    terms = [t.strip().lower() for t in settings.MODEL_FILTER.split(",") if t.strip()]
    return not terms or any(t in model_id.lower() for t in terms)


def detect_providers() -> list[AvailableProvider]:
    """The configured endpoint and its cached model catalog, narrowed by
    MODEL_FILTER — a one-element list, or empty when no base URL is set."""
    base_url = PROVIDER.base_url()
    if not base_url:
        return []
    effort = OPENROUTER_EFFORT_LEVELS if is_openrouter(base_url) else STANDARD_EFFORT_LEVELS
    models = [
        m for m in _cached_models(PROVIDER_ID, base_url, PROVIDER.api_key())
        if model_matches_filter(m["id"])
    ]
    ids = [m["id"] for m in models]
    ctx = {m["id"]: m["context"] for m in models if m["context"]}
    return [AvailableProvider(PROVIDER_ID, PROVIDER.name(), base_url, ids, ctx, effort)]


def resolve_default_model(
    providers: dict[str, AvailableProvider] | None = None,
) -> tuple[str, str, str | None] | None:
    """The model a NEW conversation starts on, for every surface: DEFAULT_MODEL
    at DEFAULT_EFFORT, where a blank model means "whatever the endpoint lists
    first". Returns (provider_id, model, effort), or None when the endpoint is
    unconfigured or nothing usable is listed.

    Pass an already-detected provider map to avoid a second detection pass (the
    /api/providers endpoint detects once and shares it).
    """
    if providers is None:
        providers = {p.id: p for p in detect_providers()}
    p = providers.get(PROVIDER_ID)
    if p is None:
        return None
    model = settings.DEFAULT_MODEL or (p.models[0] if p.models else None)
    if not model:
        return None
    return PROVIDER_ID, model, _blank_to_none(settings.DEFAULT_EFFORT)



def effort_for(provider: str, model: str) -> str | None:
    """The reasoning effort to use for an already-chosen (provider, model):
    DEFAULT_EFFORT for the default model, otherwise None (let the model decide)."""
    if provider != PROVIDER_ID:
        return None
    if not settings.DEFAULT_MODEL or model == settings.DEFAULT_MODEL:
        return _blank_to_none(settings.DEFAULT_EFFORT)
    return None

"""Configuration and provider auto-detection.

Paths come from the environment (see .env.example) because the settings store
lives inside one of them. Everything else is a live setting: declared in
`app.settings`, resolved stored-value > environment > default, and editable at
runtime from the web UI. Read those through the re-exported ``settings`` object
(``settings.HARNESS_TZ``) — never copy one into a module constant, or it will
stop tracking changes.

A provider is "available" (and therefore offered in the web UI) when its API
key is set, from either source.
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

from . import settings as settings_store
from .settings import settings

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

# Settings live beside the conversation database, in the data directory above.
settings_store.init(DATA_DIR / "settings.db")

# Discord's hard per-message limit — a protocol fact, not a preference.
MAX_DISCORD_LENGTH = 2000

# The Discord client is built once at startup, so its token stays env-only
# rather than pretending to be live-editable.
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()


def _blank_to_none(value: str | None) -> str | None:
    """A blank effort means "let the model decide", which the API expresses as
    an absent field rather than an empty string."""
    return value or None


@dataclass
class ProviderSpec:
    id: str
    name: str
    base_url_env: str | None
    base_url_default: str
    key_env: str

    def base_url(self) -> str:
        override = getattr(settings, self.base_url_env, "") if self.base_url_env else ""
        if override:
            return override.rstrip("/")
        return self.base_url_default.rstrip("/")

    def api_key(self) -> str | None:
        return getattr(settings, self.key_env, "") or None


# Registry of known OpenAI-compatible providers.
KNOWN_PROVIDERS: list[ProviderSpec] = [
    ProviderSpec("openai", "OpenAI", "OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ProviderSpec("deepseek", "DeepSeek", "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
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
# on/off only. DeepSeek V4.1 Flash supports low/high/max.
EFFORT_LEVELS: dict[str, list[str]] = {
    "deepseek": ["low", "high", "max"],
}

# Context windows for models whose provider's /models omits the field —
# DeepSeek returns only id/object/owned_by, so its published figures are kept
# here. A value the API actually reports always wins over this table.
STATIC_MODEL_CONTEXT: dict[str, dict[str, int]] = {
    "deepseek": {"deepseek-flash": 1_000_000, "deepseek-v4-pro": 1_000_000},
}


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
    """Whether the provider has credentials; completion failures trigger fallback."""
    spec = get_provider(provider_id)
    return spec is not None and bool(spec.api_key())


def canonical_model(provider_id: str, model: str) -> str:
    """Resolve retired Flash aliases that DeepSeek no longer lists in /models."""
    if provider_id == "deepseek" and model in (
        "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
    ):
        return "deepseek-flash"
    return model


def model_valid_for(provider_id: str, model: str) -> bool:
    """Validate against the provider catalog, allowing an unavailable catalog."""
    spec = get_provider(provider_id)
    if spec is None:
        return False
    if not model:
        return False
    known = next((p.models for p in detect_providers() if p.id == provider_id), None)
    return not known or canonical_model(provider_id, model) in known


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
        if (now - cached[0]) >= settings.PROVIDER_MODELS_CACHE_TTL:
            _refresh_async(f"models:{provider_id}", lambda: _do_fetch_models(provider_id, base_url, api_key))
        return cached[1]
    fetched = _do_fetch_models(provider_id, base_url, api_key)
    return fetched if fetched is not None else []


def detect_providers() -> list[AvailableProvider]:
    """List providers with API keys and their cached model catalogs."""
    available: list[AvailableProvider] = []
    for spec in KNOWN_PROVIDERS:
        base_url = spec.base_url()
        key = spec.api_key()
        effort = EFFORT_LEVELS.get(spec.id, [])

        if not key:
            continue
        models = _cached_models(spec.id, base_url, key)

        ids = [m["id"] for m in models]
        # Prefer the context the API reports; fall back to the static table for
        # providers that omit it.
        static = STATIC_MODEL_CONTEXT.get(spec.id, {})
        ctx = {}
        for m in models:
            c = m["context"] or static.get(canonical_model(spec.id, m["id"]))
            if c:
                ctx[m["id"]] = c
        available.append(AvailableProvider(spec.id, spec.name, base_url, ids, ctx, effort))
    return available


def resolve_default_model(
    providers: dict[str, AvailableProvider] | None = None,
) -> tuple[str, str, str | None] | None:
    """The model a NEW conversation starts on, for every surface.

    Order: the preferred default (DEFAULT_PROVIDER/MODEL settings) if that provider is
    available right now; otherwise the fallback (FALLBACK_PROVIDER/MODEL/EFFORT);
    last resort, the first detected provider + its first model. A blank model
    means "use whatever the provider lists first". Returns
    (provider_id, model, effort) or None if nothing usable is available.

    Pass an already-detected provider map to avoid a second detection pass (the
    /api/providers endpoint detects once and shares it).
    """
    if providers is None:
        providers = {p.id: p for p in detect_providers()}

    pref = providers.get(settings.DEFAULT_PROVIDER)
    if pref is not None:
        model = settings.DEFAULT_MODEL or (pref.models[0] if pref.models else None)
        if model:
            return pref.id, canonical_model(pref.id, model), _blank_to_none(settings.DEFAULT_EFFORT)

    fb = providers.get(settings.FALLBACK_PROVIDER)
    if fb is not None:
        model = settings.FALLBACK_MODEL or (fb.models[0] if fb.models else None)
        if model:
            return fb.id, canonical_model(fb.id, model), _blank_to_none(settings.FALLBACK_EFFORT)

    for p in providers.values():
        if p.models:
            return p.id, p.models[0], None
    return None


def effort_for(provider: str, model: str) -> str | None:
    """The reasoning effort to use for an already-chosen (provider, model) — so a
    conversation resumed onto the fallback model keeps its configured effort.
    Matches the default/fallback entries; otherwise None (let the model default)."""
    model = canonical_model(provider, model)
    default_model = settings.DEFAULT_MODEL
    if provider == settings.DEFAULT_PROVIDER and (
        not default_model or model == canonical_model(provider, default_model)
    ):
        return _blank_to_none(settings.DEFAULT_EFFORT)
    fallback_model = settings.FALLBACK_MODEL
    if provider == settings.FALLBACK_PROVIDER and model == canonical_model(provider, fallback_model):
        return _blank_to_none(settings.FALLBACK_EFFORT)
    return None

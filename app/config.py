"""Configuration and provider auto-detection.

Everything is driven by environment variables (see .env.example). A provider is
"available" (and therefore offered in the web UI) when its API key is set, or,
for keyless local engines, when its OpenAI-compatible /models endpoint responds.
"""
from __future__ import annotations

import os
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
SKILLS_DIR = _path("SKILLS_DIR", "./skills")
SOUL_PATH = _path("SOUL_PATH", "./soul.md")
MCP_CONFIG_PATH = _path("MCP_CONFIG_PATH", "./mcp.json")

UPLOADS_DIR = DATA_DIR / "uploads"

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SKILLS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --- Research stack --------------------------------------------------------
FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "").rstrip("/")
CAMOFOX_URL = os.getenv("CAMOFOX_URL", "").rstrip("/")

# --- Locale ----------------------------------------------------------------
# IANA timezone injected into the system prompt so the model always knows "now".
HARNESS_TZ = os.getenv("HARNESS_TZ", "America/Los_Angeles")

# --- Limits ----------------------------------------------------------------
BASH_TIMEOUT = int(os.getenv("BASH_TIMEOUT", "120"))
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "25"))

# Chat-template kwarg that toggles model reasoning per request (model-dependent;
# e.g. "enable_thinking" for this gemma/Qwen-style template).
THINK_KWARG = os.getenv("THINK_KWARG", "enable_thinking")

# --- Memory service --------------------------------------------------------
# mem0 runs as its own container (REST server). When set, the bot talks to it
# over HTTP instead of importing the mem0 library (keeps this image lean).
MEM0_API_URL = os.getenv("MEM0_API_URL", "").rstrip("/")

# --- Discord ---------------------------------------------------------------
# When a valid token is set, the app connects to Discord on startup (in the same
# process as the web UI). Leave empty to run web-UI-only.
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
# Discord is multi-user; the builtin shell/file tools run in this container, so
# they're withheld from Discord unless explicitly enabled.
DISCORD_ENABLE_SHELL = os.getenv("DISCORD_ENABLE_SHELL", "false").lower() in ("1", "true", "yes")
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


def _models_reachable(base_url: str, api_key: str | None) -> list[dict] | None:
    """Return [{id, context}] if the OpenAI-compatible /models endpoint responds,
    else None. `context` is the model's max context window when reported."""
    if not base_url:
        return None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(f"{base_url}/models", headers=headers, timeout=4.0)
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


def detect_providers() -> list[AvailableProvider]:
    """Detect which providers are usable right now.

    - Keyed providers: included when the key env is set. Models are fetched if
      reachable, but the provider is still listed even if the listing fails.
    - Keyless local providers: included only when /models actually responds.
    """
    available: list[AvailableProvider] = []
    for spec in KNOWN_PROVIDERS:
        base_url = spec.base_url()
        key = spec.api_key()

        effort = EFFORT_LEVELS.get(spec.id, [])
        if spec.key_env:  # cloud / keyed provider
            if not key:
                continue
            models = _models_reachable(base_url, key) or []
            ids = [m["id"] for m in models]
            ctx = {m["id"]: m["context"] for m in models if m["context"]}
            available.append(AvailableProvider(spec.id, spec.name, base_url, ids, ctx, effort))
        else:  # keyless local engine — only if reachable
            models = _models_reachable(base_url, None)
            if models is None:
                continue
            ids = [m["id"] for m in models]
            ctx = {m["id"]: m["context"] for m in models if m["context"]}
            available.append(AvailableProvider(spec.id, spec.name, base_url, ids, ctx, effort))
    return available


def resolve_default_model() -> tuple[str, str, str | None] | None:
    """The model a NEW conversation starts on, for every surface.

    Order: the preferred default (DEFAULT_PROVIDER/MODEL) if that provider is
    available right now; otherwise the fallback (FALLBACK_PROVIDER/MODEL/EFFORT);
    last resort, the first detected provider + its first model. A blank model
    means "use whatever the provider lists first". Returns
    (provider_id, model, effort) or None if nothing usable is available.
    """
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

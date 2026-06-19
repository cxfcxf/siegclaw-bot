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
# Provider/model the Discord bot uses. If unset, falls back to the first detected
# provider + its first model (see default_provider_model()).
DISCORD_PROVIDER = os.getenv("DISCORD_PROVIDER", "").strip() or None
DISCORD_MODEL = os.getenv("DISCORD_MODEL", "").strip() or None
# Discord is multi-user; the builtin shell/file tools run in this container, so
# they're withheld from Discord unless explicitly enabled.
DISCORD_ENABLE_SHELL = os.getenv("DISCORD_ENABLE_SHELL", "false").lower() in ("1", "true", "yes")
MAX_DISCORD_LENGTH = 2000

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
    ProviderSpec("groq", "Groq", "GROQ_BASE_URL", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    ProviderSpec("llamacpp", "llama.cpp (local)", "LLAMACPP_BASE_URL", "http://localhost:8080/v1", None),
    ProviderSpec("ollama", "Ollama (local)", "OLLAMA_BASE_URL", "http://localhost:11434/v1", None),
    ProviderSpec("lmstudio", "LM Studio (local)", "LMSTUDIO_BASE_URL", "http://localhost:1234/v1", None),
    ProviderSpec("local", "Custom (local/remote)", "LOCAL_OPENAI_BASE_URL", "", "LOCAL_OPENAI_API_KEY"),
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


def _context_of(m: dict) -> int | None:
    """Pull a max-context-window value out of a /models entry across providers.

    Field name/shape varies: llama.cpp exposes it under meta.n_ctx, OpenRouter as
    a top-level context_length, others (LM Studio, etc.) use context_window.
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

        if spec.key_env:  # cloud / keyed provider
            if not key:
                continue
            if spec.id == "local" and not base_url:
                continue
            models = _models_reachable(base_url, key) or []
            ids = [m["id"] for m in models]
            ctx = {m["id"]: m["context"] for m in models if m["context"]}
            available.append(AvailableProvider(spec.id, spec.name, base_url, ids, ctx))
        else:  # keyless local engine — only if reachable
            models = _models_reachable(base_url, None)
            if models is None:
                continue
            ids = [m["id"] for m in models]
            ctx = {m["id"]: m["context"] for m in models if m["context"]}
            available.append(AvailableProvider(spec.id, spec.name, base_url, ids, ctx))
    return available


def default_provider_model() -> tuple[str, str] | None:
    """Pick the (provider_id, model) the Discord bot should use.

    Honors DISCORD_PROVIDER/DISCORD_MODEL when set; otherwise falls back to the
    first detected provider and its first listed model. Returns None if nothing
    usable is available (web-UI-only deployments may still have no provider).
    """
    providers = detect_providers()
    if not providers:
        return None
    chosen = None
    if DISCORD_PROVIDER:
        chosen = next((p for p in providers if p.id == DISCORD_PROVIDER), None)
    chosen = chosen or providers[0]
    model = DISCORD_MODEL or (chosen.models[0] if chosen.models else None)
    if not model:
        return None
    return chosen.id, model

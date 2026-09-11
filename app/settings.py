"""Live, DB-backed settings with an environment floor.

Every tunable is declared once in ``SPECS`` — name, type, default, category and
whether it holds a secret. A value is resolved in three layers:

    stored (SQLite)  >  environment (.env)  >  spec default

so an existing deployment keeps running untouched (nothing is stored yet, so
every read falls through to `.env`), and anything changed in the web UI wins
from the next read onward. Clearing a stored value re-exposes the env/default
underneath it, which is the escape hatch when a value set through the UI turns
out to be wrong.

Reads are served from an in-memory cache refreshed on write, so a per-turn
lookup costs a dict access rather than a query. Writes go through ``set_many``,
which coerces and validates against the spec before touching the DB — an
invalid value is rejected rather than stored and read back as a crash later.

This module imports nothing from the rest of the app (``init`` is handed a DB
path by `config`), so it sits below everything else in the import graph.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("siegclaw.settings")

# Value written into a secret field to mean "leave the stored secret alone".
# The API never sends a secret back to the browser, so an unchanged form field
# round-trips this sentinel instead of the real key.
SECRET_UNCHANGED = "__unchanged__"


@dataclass(frozen=True)
class Spec:
    key: str                      # canonical name, matches the env var
    type: str                     # "str" | "int" | "float" | "bool"
    default: Any
    category: str                 # groups the settings UI
    label: str
    help: str = ""
    secret: bool = False          # write-only: never returned to a client
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    allow_blank: bool = True      # a blank string is a meaningful value


def _s(key, default, category, label, help="", **kw) -> Spec:
    return Spec(key, "str", default, category, label, help, **kw)


def _i(key, default, category, label, help="", **kw) -> Spec:
    return Spec(key, "int", default, category, label, help, **kw)


def _f(key, default, category, label, help="", **kw) -> Spec:
    return Spec(key, "float", default, category, label, help, **kw)


def _b(key, default, category, label, help="", **kw) -> Spec:
    return Spec(key, "bool", default, category, label, help, **kw)


# --- The registry ----------------------------------------------------------
# Paths (DATA_DIR, WIKI_DIR, …) are deliberately absent: the settings DB lives
# inside DATA_DIR, so a path cannot be resolved from the store that a path
# locates. Those stay environment-only. DISCORD_BOT_TOKEN is likewise env-only —
# the client is constructed once at startup from it.
SPECS: tuple[Spec, ...] = (
    # --- Providers (credentials are secret; base URLs are not) -------------
    _s("OPENAI_API_KEY", "", "providers", "OpenAI API key", secret=True),
    _s("OPENAI_BASE_URL", "https://api.openai.com/v1", "providers", "OpenAI base URL",
       "Override for Azure or a proxy."),
    _s("DEEPSEEK_API_KEY", "", "providers", "DeepSeek API key", secret=True),
    _s("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1", "providers", "DeepSeek base URL"),

    # --- Default model order ----------------------------------------------
    _s("DEFAULT_PROVIDER", "deepseek", "models", "Default provider",
       "Every new conversation — web, DM, channel — starts here."),
    _s("DEFAULT_MODEL", "deepseek-flash", "models", "Default model",
       "Blank selects whatever the provider lists first."),
    _s("DEFAULT_EFFORT", "high", "models", "Default reasoning effort",
       "DeepSeek accepts low, high or max. Blank lets the model decide."),
    _s("FALLBACK_PROVIDER", "deepseek", "models", "Fallback provider",
       "Used when the default provider has no credentials or stops serving."),
    _s("FALLBACK_MODEL", "deepseek-flash", "models", "Fallback model"),
    _s("FALLBACK_EFFORT", "high", "models", "Fallback reasoning effort"),
    _i("SEND_FALLBACK_RETRIES", 3, "models", "Send retries before fallback",
       "Transient blips are retried this many times before the conversation is"
       " switched to the fallback model.", minimum=0, maximum=20),
    _f("SEND_FALLBACK_RETRY_DELAY", 0.75, "models", "Delay between retries (s)",
       minimum=0, maximum=30),
    _f("PROVIDER_MODELS_CACHE_TTL", 86400.0, "models", "Model catalog cache (s)",
       "How long a provider's /models list is reused before a background"
       " refresh. Saving provider settings clears the cache regardless.",
       minimum=0),

    # --- Agent limits ------------------------------------------------------
    _i("MAX_AGENT_ITERATIONS", 25, "agent", "Max agent iterations",
       "Tool-call rounds allowed in a single turn before it is cut off.",
       minimum=1, maximum=200),
    _i("BASH_TIMEOUT", 120, "agent", "Shell tool timeout (s)", minimum=1, maximum=3600),
    _i("CRON_KEEP_RUNS", 30, "agent", "Cron runs kept per job",
       "Older cron-run conversations are pruned so a frequent job cannot grow"
       " the database forever.", minimum=1, maximum=1000),

    # --- Locale ------------------------------------------------------------
    _s("HARNESS_TZ", "America/Los_Angeles", "locale", "Timezone",
       "IANA name, injected into the system prompt so the model knows 'now'.",
       allow_blank=False),

    # --- Research stack ----------------------------------------------------
    _s("FIRECRAWL_API_URL", "", "research", "Firecrawl URL"),
    _s("CAMOFOX_URL", "", "research", "Camofox URL"),
    _s("IMAGE_SEARCH_URL", "", "research", "Image search URL",
       "The search middleware behind Firecrawl, hit directly for /images."),

    # --- Discord -----------------------------------------------------------
    _b("DISCORD_ENABLE_SHELL", False, "discord", "Enable shell tools in Discord",
       "Discord is multi-user and the builtin shell/file tools run in this"
       " container, so they are withheld unless this is on."),
    _s("DISCORD_OWNER_ID", "", "discord", "Owner user id",
       "Blank resolves the application owner from Discord at runtime."),
    _b("DISCORD_STREAM_DMS", False, "discord", "Stream DM replies",
       "Edits the message in place about once a second instead of sending one"
       " complete reply at the end of the turn."),

    # --- Discord context window -------------------------------------------
    _i("CONTEXT_MESSAGE_COUNT", 50, "discord", "Channel history messages",
       minimum=1, maximum=500),
    _i("CONTEXT_TIME_WINDOW_HOURS", 24, "discord", "Channel history window (h)",
       minimum=1, maximum=720),
    _i("CONTEXT_ACTIVITY_THRESHOLD", 30, "discord", "Busy-channel threshold",
       minimum=1, maximum=500),
    _i("CONTEXT_MAX_MESSAGES", 150, "discord", "Channel history hard cap",
       minimum=1, maximum=1000),
    _i("CONTEXT_MAX_CHARS", 16000, "discord", "Channel history char cap",
       minimum=500, maximum=500_000),
)

BY_KEY: dict[str, Spec] = {s.key: s for s in SPECS}


class SettingsError(ValueError):
    """A rejected write — the message is safe to show a user."""


# --- Coercion --------------------------------------------------------------
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def coerce(spec: Spec, raw: Any) -> Any:
    """Turn a stored/env/submitted value into the spec's type, or raise."""
    if spec.type == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise SettingsError(f"{spec.key}: expected a true/false value, got {raw!r}")

    if spec.type in ("int", "float"):
        try:
            value = int(raw) if spec.type == "int" else float(raw)
        except (TypeError, ValueError):
            raise SettingsError(f"{spec.key}: expected a number, got {raw!r}") from None
        if spec.minimum is not None and value < spec.minimum:
            raise SettingsError(f"{spec.key}: must be at least {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise SettingsError(f"{spec.key}: must be at most {spec.maximum}")
        return value

    text = "" if raw is None else str(raw).strip()
    if not text and not spec.allow_blank:
        raise SettingsError(f"{spec.key}: cannot be blank")
    if spec.choices and text and text not in spec.choices:
        raise SettingsError(f"{spec.key}: must be one of {', '.join(spec.choices)}")
    return text


# --- Store -----------------------------------------------------------------
_db_path: str | None = None
_cache: dict[str, Any] | None = None
_lock = threading.RLock()
_listeners: list[Callable[[set[str]], None]] = []


def init(db_path) -> None:
    """Point the store at its SQLite file and warm the cache. Called by
    `config` once the data directory is known."""
    global _db_path
    with _lock:
        _db_path = str(db_path)
        with _conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS settings ("
                " key TEXT PRIMARY KEY,"
                " value TEXT NOT NULL,"   # JSON-encoded
                " updated_at REAL)"
            )
        _reload()


def _conn() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("settings.init() has not been called")
    return sqlite3.connect(_db_path)


def _stored() -> dict[str, Any]:
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
    except sqlite3.Error:
        log.exception("settings: read failed; falling back to environment")
        return {}
    out: dict[str, Any] = {}
    for key, value in rows:
        if key not in BY_KEY:
            continue  # a setting retired in a later version; ignore, don't crash
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            log.warning("settings: %s holds invalid JSON; ignoring", key)
    return out


def _resolve(spec: Spec, stored: dict[str, Any]) -> Any:
    """stored > environment > default, with a bad value never fatal."""
    for source, raw in (("stored", stored.get(spec.key)), ("env", os.getenv(spec.key))):
        if raw is None:
            continue
        if source == "env" and not raw.strip() and spec.default != "":
            continue  # an empty env var means "unset", not "blank"
        try:
            return coerce(spec, raw)
        except SettingsError as exc:
            log.warning("settings: ignoring %s value (%s)", source, exc)
    return spec.default


def _reload() -> None:
    global _cache
    stored = _stored()
    _cache = {spec.key: _resolve(spec, stored) for spec in SPECS}


def _values() -> dict[str, Any]:
    with _lock:
        if _cache is None:
            _reload()
        return _cache  # type: ignore[return-value]


def get(key: str) -> Any:
    """The live value of one setting."""
    if key not in BY_KEY:
        raise KeyError(key)
    return _values()[key]


def _write(mutate: Callable[[sqlite3.Connection], None]) -> set[str]:
    """Run a DB mutation, refresh the cache and announce what actually changed.
    Holding the lock across the write and the reload keeps a concurrent reader
    from seeing the old cache after the row is gone."""
    with _lock:
        before = dict(_values())
        with _conn() as conn:
            mutate(conn)
        _reload()
        changed = {k for k, v in _values().items() if before.get(k) != v}
    if changed:
        _notify(changed)
    return changed


def set_many(updates: dict[str, Any]) -> set[str]:
    """Validate and persist several settings at once, returning the keys that
    actually changed. Nothing is written unless every value validates, so a
    rejected field cannot leave the rest half-applied.

    A secret whose value is SECRET_UNCHANGED is skipped, which is how the UI
    submits a form it never received the real key for. Writing a blank string to
    a secret deletes the override and re-exposes whatever `.env` holds.
    """
    cleaned: dict[str, Any] = {}
    deletes: list[str] = []
    for key, raw in updates.items():
        spec = BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"unknown setting {key!r}")
        if spec.secret:
            if raw == SECRET_UNCHANGED:
                continue
            if not str(raw or "").strip():
                deletes.append(key)
                continue
        cleaned[key] = coerce(spec, raw)

    def mutate(conn: sqlite3.Connection) -> None:
        for key, value in cleaned.items():
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, strftime('%s','now'))"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value)),
            )
        conn.executemany("DELETE FROM settings WHERE key = ?", [(k,) for k in deletes])

    return _write(mutate)


def reset(keys: list[str] | None = None) -> set[str]:
    """Drop stored overrides so the environment/default shows through again."""
    def mutate(conn: sqlite3.Connection) -> None:
        if keys is None:
            conn.execute("DELETE FROM settings")
        else:
            conn.executemany("DELETE FROM settings WHERE key = ?", [(k,) for k in keys])

    return _write(mutate)


# --- Change notification ---------------------------------------------------
def on_change(fn: Callable[[set[str]], None]) -> None:
    """Register a callback run after settings change (e.g. to bust a cache)."""
    _listeners.append(fn)


def _notify(changed: set[str]) -> None:
    for fn in list(_listeners):
        try:
            fn(changed)
        except Exception:  # a bad listener must not fail the write
            log.exception("settings: change listener failed")


# --- Introspection for the API ---------------------------------------------
def describe() -> list[dict[str, Any]]:
    """The settings as the UI needs them: spec metadata plus the current value,
    with secrets reduced to a set/unset flag. Never returns a secret's value —
    that is the whole reason this is a separate shape from `_values()`."""
    values = _values()
    stored = _stored()
    out: list[dict[str, Any]] = []
    for spec in SPECS:
        row: dict[str, Any] = {
            "key": spec.key,
            "type": spec.type,
            "category": spec.category,
            "label": spec.label,
            "help": spec.help,
            "secret": spec.secret,
            "overridden": spec.key in stored,
            "choices": list(spec.choices),
            "minimum": spec.minimum,
            "maximum": spec.maximum,
        }
        if spec.secret:
            row["is_set"] = bool(values[spec.key])
            row["value"] = None
        else:
            row["value"] = values[spec.key]
            row["default"] = spec.default
        out.append(row)
    return out


class _Live:
    """Attribute access to live values: ``settings.HARNESS_TZ``.

    Deliberately an object rather than module constants — ``from config import
    HARNESS_TZ`` would bind a value once at import and never see a change.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return get(name)
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return sorted(BY_KEY)


settings = _Live()

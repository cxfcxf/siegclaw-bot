"""Durable memory with interchangeable backends.

- Mem0ServiceBackend: talks to a mem0 OSS REST server (its own container) over
  HTTP — automatic fact extraction, semantic retrieval, and add/update/delete.
  The extraction LLM + embedder live inside that service, so this image stays
  lean. Selected whenever MEM0_API_URL is set.
- Mem0Backend: the in-process mem0 library (optional; only if `mem0ai` is
  installed and MEM0_ENABLED is on). Kept for non-service deployments.
- SimpleBackend: SQLite list of facts, all injected into the prompt. Explicit
  remember/forget. Zero extra deps. The always-available fallback.

`scope` selects whose memory a call touches (mem0's `user_id`). The web UI passes
no scope (→ the default user); the Discord bot passes a per-user scope like
`discord:<user_id>` so people's facts don't bleed across users.

Module-level functions delegate to the active backend so callers stay simple.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from typing import Any, Protocol

import httpx

from .config import DATA_DIR, MEM0_API_URL
from .tools.registry import Tool

USER_ID = os.getenv("MEM0_USER_ID", "default")
MAX_INJECTED = 200
SEARCH_LIMIT = int(os.getenv("MEM0_SEARCH_LIMIT", "6"))


# --------------------------------------------------------------------------- #
# Backend protocol
# --------------------------------------------------------------------------- #
class Backend(Protocol):
    kind: str

    def relevant_block(self, query: str, scope: str | None = None) -> str: ...
    def list_memories(self) -> list[dict[str, Any]]: ...
    def add_memory(self, content: str, scope: str | None = None) -> str: ...
    def delete_memory(self, mem_id: str) -> bool: ...
    def search(self, query: str, scope: str | None = None) -> str: ...
    # Returns extraction events: [{"event": "ADD"|"UPDATE"|"DELETE", "memory": str}, ...]
    def record_turn(self, user_text: str, assistant_text: str, scope: str | None = None) -> list[dict[str, Any]]: ...
    def tools(self) -> list[Tool]: ...


# --------------------------------------------------------------------------- #
# Simple SQLite backend
# --------------------------------------------------------------------------- #
class SimpleBackend:
    kind = "simple"

    def __init__(self) -> None:
        self.db_path = DATA_DIR / "memory.db"
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, created_at REAL)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_memory(self, content: str, scope: str | None = None) -> str:
        content = content.strip()
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM memories WHERE content=?", (content,)).fetchone()
            if row:
                return str(row["id"])
            cur = conn.execute(
                "INSERT INTO memories (content, created_at) VALUES (?, ?)", (content, time.time())
            )
            return str(cur.lastrowid)

    def list_memories(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, content FROM memories ORDER BY id ASC").fetchall()
        return [{"id": str(r["id"]), "content": r["content"]} for r in rows]

    def delete_memory(self, mem_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            return cur.rowcount > 0

    def search(self, query: str, scope: str | None = None) -> str:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE content LIKE ? ORDER BY id ASC",
                (f"%{query}%",),
            ).fetchall()
        if not rows:
            return f"No memories matching '{query}'."
        return "\n".join(f"[#{r['id']}] {r['content']}" for r in rows)

    def relevant_block(self, query: str, scope: str | None = None) -> str:
        mems = self.list_memories()
        if not mems:
            return (
                "# Memory\nNo durable facts stored yet. When the user shares something worth "
                "remembering, call `remember`."
            )
        lines = ["# Memory (durable facts about the user — refer to these when relevant)"]
        lines += [f"- [#{m['id']}] {m['content']}" for m in mems[:MAX_INJECTED]]
        lines.append("\nSave new facts with `remember`; remove outdated ones with `forget` by [#id].")
        return "\n".join(lines)

    def record_turn(self, user_text: str, assistant_text: str, scope: str | None = None) -> list[dict[str, Any]]:
        # No automatic extraction in the simple backend.
        return []

    def tools(self) -> list[Tool]:
        def remember(content: str) -> str:
            return f"Saved to memory as [#{self.add_memory(content)}]: {content.strip()}"

        def forget(id: int) -> str:
            return f"Removed memory [#{id}]." if self.delete_memory(str(id)) else f"No memory [#{id}]."

        def search_memory(query: str) -> str:
            return self.search(query)

        return [
            _remember_tool(remember),
            Tool("forget", "Delete a stored memory by its numeric [#id].",
                 {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}, forget),
            _search_tool(search_memory),
        ]


# Stricter extraction prompt. The mem0 default is permissive and local models
# are over-eager, so without this you get conversation logs ("User was told X")
# and world facts ("War Y ended in Z") saved as "memories". This restricts
# extraction to durable USER-centric attributes and leans hard toward skipping.
# (Used only by the in-process Mem0Backend; the mem0 service container carries
# its own equivalent via `custom_instructions` in mem0svc/server.py.)
_EXTRACTION_PROMPT = """You are a strict, conservative memory curator for a personal assistant. From the conversation below, extract ONLY durable facts about the USER that meet ALL of these criteria:

1. About the USER specifically — their identity, stable preferences, work, environment, or ongoing projects.
2. Stable — would still be true a month from now.
3. Useful — would plausibly change how the assistant should respond in the future.

GOOD examples (worth saving):
- "User prefers concise, direct answers without preamble"
- "User is a software engineer working primarily in Python"
- "User runs local LLMs (gemma, qwen3) via llama.cpp on a Mac mini"
- "User is building a self-hosted chat application called harness"
- "User's preferred written language is English"

BAD examples — NEVER save these or anything shaped like them:
- "User asked about X" / "User inquired about Y" — that is a question, not a user fact
- "User was informed that…" / "User was told…" — that is what the assistant said, not a user attribute
- "User mentioned that X happened" — usually a world event, not user info
- "The war ended on DATE" / "X was signed" — world events are general knowledge, not memories
- Any one-sentence summary of what was discussed

Default to NOT saving. An empty result is preferred over noise. If the conversation is mostly Q&A about a topic (news, history, technical help), there is usually nothing worth saving.

Conversation:
{content}

Return JSON: {{"facts": ["...", "..."]}}
"""


# Heuristic backstop: even with the strict prompt, local models sometimes slip
# conversation-log style "facts" through. Drop those before counting them.
import re as _re
# Any leading subject ("User", "the assistant", "John") followed by a
# communication verb is a conversation-log entry, not a durable fact.
_BAD_MEMORY_RE = _re.compile(
    r"^(?:the\s+)?[\w'-]+\s+"
    r"(?:"
    r"wa?s\s+(?:informed|told|shown|given|explained|taught|asked|inquired|wondered|noted|mentioned)|"
    r"were\s+(?:informed|told|shown|given|explained|taught|asked|inquired|wondered|noted|mentioned)|"
    r"asked|inquired|wondered|talked|spoke|chatted|discussed"
    r")\b",
    _re.IGNORECASE,
)


def _is_user_fact(memory_text: str) -> bool:
    """Heuristic filter — drop obvious conversation-log entries that slipped
    past the LLM filter. Returns True if the memory is plausibly a user fact.
    Conservative: matches past-tense communication verbs ("User was told…",
    "User asked about…") but not present-tense state ("User is …", "User runs …")."""
    if not memory_text or not memory_text.strip():
        return False
    if _BAD_MEMORY_RE.match(memory_text.strip()):
        return False
    return True


# --------------------------------------------------------------------------- #
# mem0 REST service backend
# --------------------------------------------------------------------------- #
class Mem0ServiceBackend:
    """Talks to a self-hosted mem0 OSS REST server. Every call degrades to an
    empty/no-op result on failure so a slow or restarting service never breaks
    the chat or crashes startup."""

    kind = "mem0-service"

    def __init__(self) -> None:
        self.base = MEM0_API_URL

    def _uid(self, scope: str | None) -> str:
        return scope or USER_ID

    @staticmethod
    def _results(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, dict):
            return raw.get("results") or raw.get("memories") or []
        return raw or []

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        r = httpx.post(f"{self.base}{path}", json=body, timeout=60.0)
        r.raise_for_status()
        return r.json()

    def _search(self, query: str, scope: str | None) -> list[dict[str, Any]]:
        try:
            data = self._post("/search", {"query": query, "user_id": self._uid(scope), "limit": SEARCH_LIMIT})
            return self._results(data)
        except Exception as exc:
            print(f"[mem0] search failed: {type(exc).__name__}: {exc}")
            return []

    def add_memory(self, content: str, scope: str | None = None) -> str:
        try:
            data = self._post("/memories", {
                "messages": [{"role": "user", "content": content.strip()}],
                "user_id": self._uid(scope),
            })
            items = self._results(data)
            return str(items[0].get("id", "")) if items else ""
        except Exception as exc:
            print(f"[mem0] add failed: {type(exc).__name__}: {exc}")
            return ""

    def list_memories(self) -> list[dict[str, Any]]:
        try:
            r = httpx.get(f"{self.base}/memories", params={"user_id": USER_ID}, timeout=30.0)
            r.raise_for_status()
            items = self._results(r.json())
        except Exception as exc:
            print(f"[mem0] list failed: {type(exc).__name__}: {exc}")
            items = []
        return [{"id": i.get("id", ""), "content": i.get("memory", "")} for i in items]

    def delete_memory(self, mem_id: str) -> bool:
        try:
            r = httpx.delete(f"{self.base}/memories/{mem_id}", timeout=30.0)
            return r.status_code in (200, 204)
        except Exception:
            return False

    def search(self, query: str, scope: str | None = None) -> str:
        items = self._search(query, scope)
        if not items:
            return f"No memories matching '{query}'."
        return "\n".join(f"- {i.get('memory', '')}" for i in items)

    def relevant_block(self, query: str, scope: str | None = None) -> str:
        items = self._search(query, scope)
        if not items:
            return (
                "# Memory\nNothing relevant remembered for this query. Durable facts are "
                "captured automatically; you may also call `remember` to force-save one."
            )
        lines = ["# Memory (relevant facts — refer to these when useful)"]
        lines += [f"- {i.get('memory', '')}" for i in items]
        return "\n".join(lines)

    def record_turn(self, user_text: str, assistant_text: str, scope: str | None = None) -> list[dict[str, Any]]:
        try:
            data = self._post("/memories", {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "user_id": self._uid(scope),
            })
        except Exception as exc:  # never let extraction break the chat
            print(f"[mem0] extraction failed: {type(exc).__name__}: {exc}")
            return []
        events: list[dict[str, Any]] = []
        for item in self._results(data):
            event = (item.get("event") or "").upper()
            memory_text = item.get("memory", "")
            if not memory_text:
                continue
            # Heuristic filter: drop conversation-log entries that aren't user
            # facts. If the service already saved it, delete the bad memory.
            if not _is_user_fact(memory_text):
                mem_id = item.get("id")
                if mem_id:
                    self.delete_memory(mem_id)
                continue
            events.append({"event": event or "ADD", "memory": memory_text})
        return events

    def tools(self) -> list[Tool]:
        return scoped_tools(None)


# --------------------------------------------------------------------------- #
# mem0 library backend (optional, in-process)
# --------------------------------------------------------------------------- #
def _mem0_config() -> dict[str, Any]:
    base_url = os.getenv("MEM0_LLM_BASE_URL") or "http://localhost:8080/v1"
    model = os.getenv("MEM0_LLM_MODEL") or "local"  # llama.cpp ignores the name
    api_key = os.getenv("MEM0_LLM_API_KEY") or "not-needed"

    embed_provider = os.getenv("MEM0_EMBEDDER_PROVIDER", "huggingface")
    embed_model = os.getenv("MEM0_EMBEDDER_MODEL", "multi-qa-MiniLM-L6-cos-v1")
    embedder_cfg: dict[str, Any] = {"model": embed_model}
    if embed_provider == "openai":  # e.g. a separate embeddings server
        embedder_cfg["openai_base_url"] = os.getenv("MEM0_EMBEDDER_BASE_URL", base_url)
        embedder_cfg["api_key"] = os.getenv("MEM0_EMBEDDER_API_KEY", api_key)

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "openai_base_url": base_url,
                "api_key": api_key,
                "temperature": 0,
                "max_tokens": 2000,
            },
        },
        "embedder": {"provider": embed_provider, "config": embedder_cfg},
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "mem0", "path": str(DATA_DIR / "mem0_chroma")},
        },
        "history_db_path": str(DATA_DIR / "mem0_history.db"),
        "custom_prompt": _EXTRACTION_PROMPT,
    }


class Mem0Backend:
    kind = "mem0"

    def __init__(self) -> None:
        from mem0 import Memory  # imported lazily so the dep is optional

        # mem0's OpenAI client reads OPENAI_API_KEY from the env in some paths.
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("MEM0_LLM_API_KEY") or "not-needed")
        self.memory = Memory.from_config(_mem0_config())

    def _uid(self, scope: str | None) -> str:
        return scope or USER_ID

    @staticmethod
    def _results(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, dict):
            return raw.get("results", [])
        return raw or []

    def add_memory(self, content: str, scope: str | None = None) -> str:
        # Explicit fact: store verbatim (no extraction).
        res = self.memory.add(content.strip(), user_id=self._uid(scope), infer=False)
        items = self._results(res)
        return str(items[0]["id"]) if items else ""

    def list_memories(self) -> list[dict[str, Any]]:
        items = self._results(self.memory.get_all(filters={"user_id": USER_ID}, top_k=MAX_INJECTED))
        return [{"id": i.get("id", ""), "content": i.get("memory", "")} for i in items]

    def delete_memory(self, mem_id: str) -> bool:
        try:
            self.memory.delete(mem_id)
            return True
        except Exception:
            return False

    def search(self, query: str, scope: str | None = None) -> str:
        items = self._results(self.memory.search(query, filters={"user_id": self._uid(scope)}, top_k=SEARCH_LIMIT))
        if not items:
            return f"No memories matching '{query}'."
        return "\n".join(f"- {i.get('memory', '')}" for i in items)

    def relevant_block(self, query: str, scope: str | None = None) -> str:
        try:
            items = self._results(self.memory.search(query, filters={"user_id": self._uid(scope)}, top_k=SEARCH_LIMIT))
        except Exception:
            items = []
        if not items:
            return (
                "# Memory\nNothing relevant remembered for this query. Durable facts are "
                "captured automatically; you may also call `remember` to force-save one."
            )
        lines = ["# Memory (relevant facts — refer to these when useful)"]
        lines += [f"- {i.get('memory', '')}" for i in items]
        return "\n".join(lines)

    def record_turn(self, user_text: str, assistant_text: str, scope: str | None = None) -> list[dict[str, Any]]:
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        try:
            res = self.memory.add(messages, user_id=self._uid(scope), infer=True)
        except Exception as exc:  # never let extraction break the chat
            print(f"[mem0] extraction failed: {type(exc).__name__}: {exc}")
            return []
        events: list[dict[str, Any]] = []
        for item in self._results(res):
            event = (item.get("event") or "").upper()
            memory_text = item.get("memory", "")
            if not memory_text:
                continue
            if not _is_user_fact(memory_text):
                mem_id = item.get("id")
                if mem_id:
                    try:
                        self.memory.delete(mem_id)
                    except Exception:
                        pass
                continue
            events.append({"event": event or "ADD", "memory": memory_text})
        return events

    def tools(self) -> list[Tool]:
        return scoped_tools(None)


# --------------------------------------------------------------------------- #
# Shared tool schemas
# --------------------------------------------------------------------------- #
def _remember_tool(handler) -> Tool:
    return Tool(
        "remember",
        "Force-save a durable fact about the user to long-term memory. Most facts are "
        "captured automatically, so use this only to be certain something is kept.",
        {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        handler,
    )


def _search_tool(handler) -> Tool:
    return Tool(
        "search_memory",
        "Search long-term memory for relevant facts about the user.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler,
    )


def scoped_tools(scope: str | None) -> list[Tool]:
    """remember / search_memory bound to a memory scope (mem0 user_id). The
    Discord registry uses this so each user's tools touch only their memory."""
    def remember(content: str) -> str:
        mem_id = add_memory(content, scope)
        return f"Saved to memory: {content.strip()}" + (f" (id {mem_id})" if mem_id else "")

    def search_memory(query: str) -> str:
        return search(query, scope)

    return [_remember_tool(remember), _search_tool(search_memory)]


# --------------------------------------------------------------------------- #
# Active backend (singleton)
# --------------------------------------------------------------------------- #
_backend: Backend | None = None


def init_memory() -> None:
    global _backend
    if MEM0_API_URL:
        try:
            _backend = Mem0ServiceBackend()
            print(f"[memory] backend: mem0 service ({MEM0_API_URL})")
            return
        except Exception as exc:
            print(f"[memory] mem0 service unavailable ({type(exc).__name__}: {exc}); using simple store")
            _backend = SimpleBackend()
            return
    enabled = os.getenv("MEM0_ENABLED", "true").lower() not in ("0", "false", "no")
    if enabled:
        try:
            _backend = Mem0Backend()
            print("[memory] backend: mem0 (in-process)")
            return
        except Exception as exc:
            print(f"[memory] mem0 unavailable ({type(exc).__name__}: {exc}); using simple store")
    _backend = SimpleBackend()
    print("[memory] backend: simple")


def backend() -> Backend:
    if _backend is None:
        init_memory()
    assert _backend is not None
    return _backend


# Delegating module-level API used across the app.
def relevant_block(query: str, scope: str | None = None) -> str:
    return backend().relevant_block(query, scope)


def list_memories() -> list[dict[str, Any]]:
    return backend().list_memories()


def add_memory(content: str, scope: str | None = None) -> str:
    return backend().add_memory(content, scope)


def delete_memory(mem_id: str) -> bool:
    return backend().delete_memory(mem_id)


def search(query: str, scope: str | None = None) -> str:
    return backend().search(query, scope)


def record_turn(user_text: str, assistant_text: str, scope: str | None = None) -> list[dict[str, Any]]:
    return backend().record_turn(user_text, assistant_text, scope)


def memory_tools() -> list[Tool]:
    return backend().tools()


# --------------------------------------------------------------------------- #
# Debounced background extraction
# --------------------------------------------------------------------------- #
# Extraction should NOT block the chat composer, and it shouldn't run on every
# turn — the user might be about to ask a follow-up that changes what's worth
# remembering. So instead of extracting inline, schedule a single extraction
# MEMORY_DEBOUNCE_SECONDS after the last turn. Each new turn cancels the pending
# timer and starts a new one, so back-to-back turns batch into one run.
_EXTRACT_DELAY = float(os.getenv("MEMORY_DEBOUNCE_SECONDS", "300"))
_pending_task: asyncio.Task | None = None
_pending_turns: list[tuple[str, str, str | None]] = []


def schedule_extraction(user_text: str, assistant_text: str, scope: str | None = None) -> None:
    """Schedule a memory extraction to fire after _EXTRACT_DELAY of chat quiet.
    Re-schedules on every call — busy stretches coalesce into a single run that
    fires once the user stops asking. Each turn carries its own scope (mem0
    user_id): None for the web UI's default user, `discord:<id>` for DM turns."""
    global _pending_task
    if not assistant_text.strip():
        return
    _pending_turns.append((user_text, assistant_text, scope))
    if _pending_task is not None and not _pending_task.done():
        _pending_task.cancel()
    try:
        _pending_task = asyncio.create_task(_run_extraction())
    except RuntimeError:
        # No running loop (e.g. called outside the app) — fall back to inline.
        _run_extraction_sync()


def _run_extraction_sync() -> None:
    global _pending_turns
    turns, _pending_turns = _pending_turns, []
    for u, a, scope in turns:
        try:
            record_turn(u, a, scope)
        except Exception as exc:  # never let extraction break anything
            print(f"[memory] extraction failed: {type(exc).__name__}: {exc}")


async def _run_extraction() -> None:
    global _pending_turns
    try:
        await asyncio.sleep(_EXTRACT_DELAY)
    except asyncio.CancelledError:
        return
    # Rebind (don't clear in place): slice-assigning [] would also empty `turns`,
    # which aliases the same list — that bug silently disabled all extraction.
    turns, _pending_turns = _pending_turns, []
    if not turns:
        return
    for user_text, assistant_text, scope in turns:
        try:
            await asyncio.to_thread(record_turn, user_text, assistant_text, scope)
        except Exception as exc:
            print(f"[memory] background extraction failed: {type(exc).__name__}: {exc}")

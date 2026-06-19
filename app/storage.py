"""SQLite persistence for conversations and messages."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from .config import DATA_DIR

DB_PATH = DATA_DIR / "conversations.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                provider TEXT,
                model TEXT,
                created_at REAL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                name TEXT,
                images TEXT,
                reasoning TEXT,
                created_at REAL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            """
        )
        # Migrations for DBs created before later columns were added.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "images" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN images TEXT")
        if "reasoning" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT")
        ccols = [r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
        if "prompt_tokens" not in ccols:
            conn.execute("ALTER TABLE conversations ADD COLUMN prompt_tokens INTEGER")


def create_conversation(provider: str, model: str, title: str = "New chat") -> str:
    cid = uuid.uuid4().hex
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, provider, model, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (cid, title, provider, model, now, now),
        )
    return cid


def list_conversations() -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, provider, model, prompt_tokens, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: str) -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    return dict(row) if row else None


def delete_conversation(cid: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))


def rename_conversation(cid: str, title: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))


def touch_conversation(cid: str, provider: str | None = None, model: str | None = None) -> None:
    with _conn() as conn:
        if provider and model:
            conn.execute(
                "UPDATE conversations SET updated_at=?, provider=?, model=? WHERE id=?",
                (time.time(), provider, model, cid),
            )
        else:
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (time.time(), cid))


def set_prompt_tokens(cid: str, tokens: int) -> None:
    """Record the last-reported prompt-token count for a conversation so the UI
    can show context-window usage on load, before the next turn confirms it."""
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET prompt_tokens=? WHERE id=?", (tokens, cid)
        )


def add_message(
    cid: str,
    role: str,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
    images: list[str] | None = None,
    reasoning: str | None = None,
) -> str:
    msg_id = uuid.uuid4().hex
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_call_id, name, images, reasoning, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                msg_id,
                cid,
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                name,
                json.dumps(images) if images else None,
                reasoning or None,
                time.time(),
            ),
        )
    return msg_id


def rewind_from(cid: str, message_id: str) -> None:
    """Delete the given message and every message that came after it in the
    conversation. Used by the retry/regenerate flow."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM messages WHERE id=? AND conversation_id=?",
            (message_id, cid),
        ).fetchone()
        if not row:
            return
        conn.execute(
            "DELETE FROM messages WHERE conversation_id=? AND created_at >= ?",
            (cid, row["created_at"]),
        )


def get_messages(cid: str) -> list[dict[str, Any]]:
    """Return messages in OpenAI chat format, oldest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, tool_calls, tool_call_id, name, images, reasoning FROM messages"
            " WHERE conversation_id=? ORDER BY created_at ASC",
            (cid,),
        ).fetchall()
    messages: list[dict[str, Any]] = []
    for r in rows:
        msg: dict[str, Any] = {"role": r["role"], "id": r["id"]}
        if r["content"] is not None:
            msg["content"] = r["content"]
        if r["tool_calls"]:
            msg["tool_calls"] = json.loads(r["tool_calls"])
        if r["tool_call_id"]:
            msg["tool_call_id"] = r["tool_call_id"]
        if r["name"]:
            msg["name"] = r["name"]
        if r["images"]:
            # Non-API field; the agent expands these into multimodal content.
            msg["images"] = json.loads(r["images"])
        if r["reasoning"]:
            # Non-API field; shown in the UI, stripped before sending to the model.
            msg["reasoning"] = r["reasoning"]
        messages.append(msg)
    return messages

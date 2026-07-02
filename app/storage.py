"""SQLite persistence for conversations and messages."""
from __future__ import annotations

import json
import re
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
            CREATE TABLE IF NOT EXISTS discord_dm_state (
                user_id TEXT PRIMARY KEY,
                active_cid TEXT
            );
            CREATE TABLE IF NOT EXISTS scheduled_jobs (
                id TEXT PRIMARY KEY,
                name TEXT,
                prompt TEXT,
                cron TEXT,
                target_type TEXT,       -- 'channel' | 'dm'
                target_id TEXT,
                enabled INTEGER,
                created_at REAL,
                next_run REAL,
                last_run REAL,
                last_status TEXT,       -- 'ok' | 'error' | NULL (never run)
                last_result TEXT
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
        # ref: a short, human-typable number identifying a conversation (used by
        # Discord /list and /resume so webui UUID conversations can be resumed
        # from Discord). Backfilled for existing rows.
        if "ref" not in ccols:
            conn.execute("ALTER TABLE conversations ADD COLUMN ref INTEGER")
            rows = conn.execute("SELECT id FROM conversations ORDER BY created_at ASC").fetchall()
            for i, r in enumerate(rows, start=1):
                conn.execute("UPDATE conversations SET ref=? WHERE id=?", (i, r["id"]))
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_ref ON conversations(ref)"
        )
        # discord_dm_state once had (active_session, last_session) columns; if the
        # current (active_cid) shape isn't present, drop and recreate. The table
        # only holds an ephemeral per-user pointer, so dropping it is safe.
        dcols = [r["name"] for r in conn.execute("PRAGMA table_info(discord_dm_state)").fetchall()]
        if "active_cid" not in dcols:
            conn.execute("DROP TABLE IF EXISTS discord_dm_state")
            conn.execute(
                "CREATE TABLE discord_dm_state (user_id TEXT PRIMARY KEY, active_cid TEXT)"
            )
        # Full-text search over message bodies: an external-content FTS5 index
        # kept in sync by triggers (messages are insert/delete only, never
        # updated). Backfilled once when first added to an existing DB.
        had_fts = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone() is not None
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, content='messages', content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;
            """
        )
        if not had_fts:
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")


def create_conversation(provider: str, model: str, title: str = "New chat") -> str:
    cid = uuid.uuid4().hex
    now = time.time()
    with _conn() as conn:
        ref = (conn.execute("SELECT COALESCE(MAX(ref), 0) + 1 FROM conversations").fetchone()[0])
        conn.execute(
            "INSERT INTO conversations (id, title, provider, model, ref, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (cid, title, provider, model, ref, now, now),
        )
    return cid


# --------------------------------------------------------------------------- #
# Discord DM ↔ web UI shared conversations
#
# DMs and the web UI share one pool of conversations. Each conversation carries
# a short numeric `ref` so Discord users can `/list` and `/resume <ref>` any
# conversation (including ones started in the web UI). We track each Discord
# user's currently-active conversation id so a plain DM appends to the right one.
# --------------------------------------------------------------------------- #
def dm_active_cid(user_id: str) -> str | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT active_cid FROM discord_dm_state WHERE user_id=?", (user_id,)
        ).fetchone()
    return row["active_cid"] if row else None


def dm_set_active_cid(user_id: str, cid: str) -> bool:
    """Point a user's active DM conversation at `cid`. Returns False if no such
    conversation exists."""
    with _conn() as conn:
        if not conn.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone():
            return False
        if conn.execute("SELECT 1 FROM discord_dm_state WHERE user_id=?", (user_id,)).fetchone():
            conn.execute(
                "UPDATE discord_dm_state SET active_cid=? WHERE user_id=?", (cid, user_id)
            )
        else:
            conn.execute(
                "INSERT INTO discord_dm_state (user_id, active_cid) VALUES (?,?)", (user_id, cid)
            )
    return True


def dm_clear_active(user_id: str) -> None:
    """Drop the active-conversation pointer so the next DM starts a fresh one.
    Used by /new — no conversation row is created until the user actually sends a
    message (mirrors the web UI, which only persists a conversation on first send)."""
    with _conn() as conn:
        conn.execute("DELETE FROM discord_dm_state WHERE user_id=?", (user_id,))


def message_count(cid: str) -> int:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?", (cid,)
        ).fetchone()
    return row["n"]


def conversation_by_ref(ref: int) -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE ref=?", (ref,)).fetchone()
    return dict(row) if row else None


def list_all_conversations() -> list[dict[str, Any]]:
    """All conversations with message counts, newest first (for Discord /list)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.ref, c.title, c.provider, c.model, c.updated_at,"
            " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count"
            " FROM conversations c ORDER BY c.updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_conversation_model(cid: str, provider: str, model: str) -> bool:
    with _conn() as conn:
        if not conn.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone():
            return False
        conn.execute(
            "UPDATE conversations SET provider=?, model=?, updated_at=? WHERE id=?",
            (provider, model, time.time(), cid),
        )
    return True


def list_conversations() -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, provider, model, ref, prompt_tokens, created_at, updated_at"
            " FROM conversations ORDER BY updated_at DESC"
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


def rewind_last_user_turn(cid: str) -> None:
    """Delete the most recent user message and everything after it. Used when a
    Discord DM turn failed mid-flight after the user message was already stored,
    so the conversation doesn't end on an unanswered question."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM messages WHERE conversation_id=? AND role='user'"
            " ORDER BY created_at DESC LIMIT 1",
            (cid,),
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


# --- Full-text search --------------------------------------------------------- #
def _fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 prefix query ('pgvector mem' →
    '"pgvector"* "mem"*') so user input can't hit MATCH syntax errors."""
    return " ".join(f'"{t}"*' for t in re.findall(r"\w+", q))


def search_messages(query: str, limit: int = 30) -> list[dict[str, Any]]:
    """Full-text search over user/assistant message bodies. Returns one row per
    conversation — its best-ranked match, best first — with a snippet whose
    matched terms are delimited by \\x01…\\x02 for the UI to highlight."""
    fq = _fts_query(query)
    if not fq:
        return []
    with _conn() as conn:
        rows = conn.execute(
            # bm25()/snippet() only work while the FTS table drives the query, so
            # the match runs in a MATERIALIZED CTE (flattened into the aggregate,
            # SQLite errors with "unable to use function bm25"). The outer
            # bare-columns-with-MIN pick keeps, per conversation, the snippet of
            # its best-ranked matching message.
            """
            WITH hits AS MATERIALIZED (
                SELECT c.id, c.title, c.updated_at,
                       snippet(messages_fts, 0, char(1), char(2), ' … ', 12) AS snippet,
                       bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages m ON m.rowid = messages_fts.rowid
                JOIN conversations c ON c.id = m.conversation_id
                WHERE messages_fts MATCH ? AND m.role IN ('user', 'assistant')
            )
            SELECT id, title, updated_at, snippet, MIN(rank) AS rank
            FROM hits
            GROUP BY id
            ORDER BY rank
            LIMIT ?
            """,
            (fq, limit),
        ).fetchall()
    return [
        {"id": r["id"], "title": r["title"], "updated_at": r["updated_at"], "snippet": r["snippet"]}
        for r in rows
    ]


# --- Scheduled jobs --------------------------------------------------------- #
def _job_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "name": r["name"],
        "prompt": r["prompt"],
        "cron": r["cron"],
        "target_type": r["target_type"],
        "target_id": r["target_id"],
        "enabled": bool(r["enabled"]),
        "created_at": r["created_at"],
        "next_run": r["next_run"],
        "last_run": r["last_run"],
        "last_status": r["last_status"],
        "last_result": r["last_result"],
    }


def create_job(
    name: str, prompt: str, cron: str, target_type: str, target_id: str,
    next_run: float, enabled: bool = True,
) -> str:
    jid = uuid.uuid4().hex
    with _conn() as conn:
        conn.execute(
            "INSERT INTO scheduled_jobs (id, name, prompt, cron, target_type, "
            "target_id, enabled, created_at, next_run) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (jid, name, prompt, cron, target_type, target_id,
             1 if enabled else 0, time.time(), next_run),
        )
    return jid


def list_jobs() -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scheduled_jobs ORDER BY created_at DESC"
        ).fetchall()
    return [_job_row(r) for r in rows]


def get_job(jid: str) -> dict[str, Any] | None:
    with _conn() as conn:
        r = conn.execute("SELECT * FROM scheduled_jobs WHERE id = ?", (jid,)).fetchone()
    return _job_row(r) if r else None


def update_job(jid: str, **fields: Any) -> None:
    """Update arbitrary columns. `enabled` is coerced to 0/1."""
    if "enabled" in fields:
        fields["enabled"] = 1 if fields["enabled"] else 0
    allowed = {
        "name", "prompt", "cron", "target_type", "target_id",
        "enabled", "next_run", "last_run", "last_status", "last_result",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    cols = ", ".join(f"{k} = ?" for k in sets)
    with _conn() as conn:
        conn.execute(
            f"UPDATE scheduled_jobs SET {cols} WHERE id = ?", (*sets.values(), jid)
        )


def delete_job(jid: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (jid,))


def due_jobs(now: float) -> list[dict[str, Any]]:
    """Enabled jobs whose next_run has passed."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scheduled_jobs WHERE enabled = 1 AND next_run IS NOT NULL "
            "AND next_run <= ? ORDER BY next_run ASC",
            (now,),
        ).fetchall()
    return [_job_row(r) for r in rows]

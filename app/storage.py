"""SQLite persistence for conversations and messages."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import time
import uuid
from typing import Any

from .config import DATA_DIR, UPLOADS_DIR

log = logging.getLogger("siegclaw.storage")

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
        # Per-reply provenance: which "provider/model" produced an assistant
        # message (the conversations.model column only keeps the last one).
        if "model" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN model TEXT")
        # audio: '/uploads/...' URL of an audio clip tied to the message — the
        # user's voice recording on user rows, the TTS reading on assistant rows.
        if "audio" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN audio TEXT")
        # docs: JSON [{"url", "name"}] of document attachments (PDF/text) on a
        # user message; their extracted text is injected at prompt-build time.
        if "docs" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN docs TEXT")
        ccols = [r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
        if "prompt_tokens" not in ccols:
            conn.execute("ALTER TABLE conversations ADD COLUMN prompt_tokens INTEGER")
        # grp: optional user-assigned group name for organizing the sidebar /
        # /resume picker (NULL = ungrouped). "grp" because GROUP is an SQL keyword.
        if "grp" not in ccols:
            conn.execute("ALTER TABLE conversations ADD COLUMN grp TEXT")
        # Groups are first-class (a group can exist empty); membership is still
        # the conversations.grp column. Seed from any legacy labels.
        conn.execute("CREATE TABLE IF NOT EXISTS groups (name TEXT PRIMARY KEY, created_at REAL)")
        conn.execute(
            "INSERT OR IGNORE INTO groups (name, created_at)"
            " SELECT DISTINCT grp, ? FROM conversations WHERE grp IS NOT NULL",
            (time.time(),),
        )
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
        # Lifetime counters (e.g. total tokens consumed). created_at records
        # when counting began — old turns predating the feature aren't counted.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS counters"
            " (name TEXT PRIMARY KEY, value INTEGER, created_at REAL)"
        )
        # system: 1 = auto-created group owned by a feature (Research, Cron: *)
        # — protected from rename/delete; cron pruning depends on the name.
        gcols = [r["name"] for r in conn.execute("PRAGMA table_info(groups)").fetchall()]
        if "system" not in gcols:
            conn.execute("ALTER TABLE groups ADD COLUMN system INTEGER DEFAULT 0")
            conn.execute(
                "UPDATE groups SET system=1 WHERE name='Research' OR name LIKE 'Cron: %'"
            )
        _migrate_upload_layout(conn)


def _migrate_upload_layout(conn: sqlite3.Connection) -> None:
    """One-time move of media files from the original flat uploads/ into
    per-conversation directories (uploads/<cid>/<file>), rewriting the stored
    '/uploads/<file>' URLs to match. No-ops instantly once everything is
    nested. Flat files no row references (already-orphaned) are left in place
    — they're candidates for manual cleanup, not silent deletion."""
    rows = conn.execute(
        "SELECT id, conversation_id, images, audio FROM messages"
        " WHERE images LIKE '%\"/uploads/%' OR audio LIKE '/uploads/%'"
    ).fetchall()
    moved = 0

    def fix(url: str, cid: str) -> str:
        nonlocal moved
        m = re.fullmatch(r"/uploads/([^/]+)", url)
        if not m:
            return url
        src = UPLOADS_DIR / m.group(1)
        dest_dir = UPLOADS_DIR / cid
        if src.is_file():
            dest_dir.mkdir(exist_ok=True)
            src.rename(dest_dir / m.group(1))
            moved += 1
        elif not (dest_dir / m.group(1)).is_file():
            return url  # file lost before the migration; keep the row honest
        return f"/uploads/{cid}/{m.group(1)}"

    for r in rows:
        cid = r["conversation_id"]
        updates: dict[str, str] = {}
        if r["audio"]:
            fixed = fix(r["audio"], cid)
            if fixed != r["audio"]:
                updates["audio"] = fixed
        if r["images"]:
            imgs = json.loads(r["images"])
            fixed_imgs = [fix(u, cid) for u in imgs]
            if fixed_imgs != imgs:
                updates["images"] = json.dumps(fixed_imgs)
        for col, val in updates.items():
            conn.execute(f"UPDATE messages SET {col}=? WHERE id=?", (val, r["id"]))
    if moved:
        log.info("uploads migration: moved %d file(s) into per-conversation dirs", moved)


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
            "SELECT c.id, c.ref, c.title, c.provider, c.model, c.updated_at, c.grp,"
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
            "SELECT id, title, provider, model, ref, prompt_tokens, grp, created_at, updated_at"
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
    # Media (images, voice clips, TTS readings) lives under uploads/<cid>/ —
    # the conversation owns its files, so they go with it.
    shutil.rmtree(UPLOADS_DIR / cid, ignore_errors=True)


def prune_group(grp: str, keep: int) -> int:
    """Delete the oldest conversations in a group beyond the newest `keep`.
    Used by the scheduler so frequent cron jobs don't accumulate runs forever;
    moving a conversation out of the group exempts it. Returns deleted count."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id FROM conversations WHERE grp=? ORDER BY created_at DESC",
            (grp,),
        ).fetchall()
    stale = [r["id"] for r in rows[keep:]]
    for cid in stale:
        delete_conversation(cid)
    return len(stale)


def rename_conversation(cid: str, title: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))


def set_conversation_group(cid: str, grp: str | None, system: bool = False) -> None:
    """Assign a conversation to a group, creating it if needed. `system=True`
    marks a feature-owned group (Research, Cron: *) as protected — used by the
    auto-creation paths, never by user actions."""
    with _conn() as conn:
        if grp:
            conn.execute(
                "INSERT OR IGNORE INTO groups (name, created_at, system) VALUES (?,?,?)",
                (grp, time.time(), 1 if system else 0),
            )
        conn.execute("UPDATE conversations SET grp=? WHERE id=?", (grp or None, cid))


def group_is_system(name: str) -> bool:
    with _conn() as conn:
        row = conn.execute("SELECT system FROM groups WHERE name=?", (name,)).fetchone()
    return bool(row and row["system"])


def list_groups() -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT g.name, g.created_at, g.system,"
            " (SELECT COUNT(*) FROM conversations c WHERE c.grp = g.name) AS count,"
            " (SELECT MAX(c.updated_at) FROM conversations c WHERE c.grp = g.name) AS last_active"
            " FROM groups g ORDER BY last_active DESC NULLS LAST, g.name"
        ).fetchall()
    return [{**dict(r), "system": bool(r["system"])} for r in rows]


def create_group(name: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT OR IGNORE INTO groups (name, created_at, system) VALUES (?,?,0)", (name, time.time()))


def rename_group(old: str, new: str) -> bool:
    """Rename a (custom) group; if the new name already exists, the two merge.
    System groups refuse — features find their conversations by group name
    (cron pruning), so renaming would silently break them."""
    with _conn() as conn:
        row = conn.execute("SELECT system FROM groups WHERE name=?", (old,)).fetchone()
        if row and row["system"]:
            return False
        conn.execute("INSERT OR IGNORE INTO groups (name, created_at, system) VALUES (?,?,0)", (new, time.time()))
        conn.execute("UPDATE conversations SET grp=? WHERE grp=?", (new, old))
        if old != new:
            conn.execute("DELETE FROM groups WHERE name=?", (old,))
    return True


def delete_group(name: str) -> int | None:
    """Delete a (custom) group. Its conversations are NOT deleted — they're
    ungrouped (back to the date-sectioned root list). Returns how many were
    ungrouped, or None if the group is system-owned (refused)."""
    with _conn() as conn:
        row = conn.execute("SELECT system FROM groups WHERE name=?", (name,)).fetchone()
        if row and row["system"]:
            return None
        cur = conn.execute("UPDATE conversations SET grp=NULL WHERE grp=?", (name,))
        conn.execute("DELETE FROM groups WHERE name=?", (name,))
        return cur.rowcount


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
    model: str | None = None,
    audio: str | None = None,
    docs: list[dict] | None = None,
) -> str:
    msg_id = uuid.uuid4().hex
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_call_id, name, images, reasoning, model, audio, docs, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                model,
                audio,
                json.dumps(docs) if docs else None,
                time.time(),
            ),
        )
    return msg_id


def get_message(msg_id: str) -> dict[str, Any] | None:
    with _conn() as conn:
        r = conn.execute(
            "SELECT id, conversation_id, role, content, audio FROM messages WHERE id=?",
            (msg_id,),
        ).fetchone()
    return dict(r) if r else None


def set_message_audio(msg_id: str, url: str) -> None:
    """Attach an audio URL to an existing message (the TTS reading of an
    assistant reply is generated on demand, long after the row is written)."""
    with _conn() as conn:
        conn.execute("UPDATE messages SET audio=? WHERE id=?", (url, msg_id))


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
            "SELECT id, role, content, tool_calls, tool_call_id, name, images, reasoning, model, audio, docs FROM messages"
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
        if r["model"]:
            # Non-API field; which provider/model produced this reply.
            msg["model"] = r["model"]
        if r["audio"]:
            # Non-API field; voice clip (user) or TTS reading (assistant).
            msg["audio"] = r["audio"]
        if r["docs"]:
            # Non-API field; the agent injects the extracted text at prompt time.
            msg["docs"] = json.loads(r["docs"])
        messages.append(msg)
    return messages


def bump_counter(name: str, delta: int) -> None:
    if not delta:
        return
    with _conn() as conn:
        conn.execute(
            "INSERT INTO counters (name, value, created_at) VALUES (?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
            (name, delta, time.time()),
        )


def add_token_usage(prompt: int, completion: int) -> None:
    """Record what one turn consumed. Prompt tokens are summed across every
    model call of the turn (history is re-sent each tool round — that's what
    the provider actually bills), completion likewise."""
    bump_counter("tokens_prompt", prompt)
    bump_counter("tokens_completion", completion)


def get_counters() -> dict[str, dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute("SELECT name, value, created_at FROM counters").fetchall()
    return {r["name"]: {"value": r["value"], "since": r["created_at"]} for r in rows}


def usage_stats() -> dict[str, Any]:
    """Aggregate numbers for the status page: table sizes and which models have
    been answering (messages.model provenance), all-time and last 7 days."""
    week_ago = time.time() - 7 * 86400
    with _conn() as conn:
        conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        by_model = conn.execute(
            "SELECT model, COUNT(*) AS total, SUM(created_at >= ?) AS last7"
            " FROM messages WHERE role='assistant' AND model IS NOT NULL"
            " GROUP BY model ORDER BY total DESC",
            (week_ago,),
        ).fetchall()
    return {
        "conversations": conversations,
        "messages": messages,
        "tokens": get_counters(),
        "models": [
            {"model": r["model"], "replies": r["total"], "replies_7d": r["last7"] or 0}
            for r in by_model
        ],
    }


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

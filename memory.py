import asyncio
import json
import logging
import os
import sqlite3
import time

import numpy as np
from google.genai import types

from config import (
    MEMORY_DB_PATH,
    MEMORY_DECAY_DAYS,
    MEMORY_EXTRACTION_PROMPT,
    MODEL,
    gemini_client,
)

log = logging.getLogger("siegclaw.memory")

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 3072

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON memories(channel_id)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id)")
        _conn.commit()
    return _conn


def _get_embedding(text: str) -> np.ndarray:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)


def _get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [np.array(e.values, dtype=np.float32) for e in response.embeddings]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def init_db():
    _get_conn()
    log.info("SQLite memory DB initialized at %s", MEMORY_DB_PATH)


def search_memories(
    channel_id: str,
    query: str,
    user_id: str | None = None,
    limit: int = 10,
) -> list[str]:
    """Cosine similarity search with recency weighting."""
    conn = _get_conn()

    if user_id:
        rows = conn.execute(
            "SELECT fact, created_at, embedding FROM memories "
            "WHERE channel_id = ? OR user_id = ?",
            (channel_id, user_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT fact, created_at, embedding FROM memories WHERE channel_id = ?",
            (channel_id,),
        ).fetchall()

    if not rows:
        return []

    query_embedding = _get_embedding(query)
    now = time.time()
    decay_seconds = MEMORY_DECAY_DAYS * 86400

    scored = []
    for fact, created_at, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        similarity = _cosine_similarity(query_embedding, emb)
        age = now - created_at
        decay = max(0.3, 1.0 - (age / decay_seconds))
        scored.append((fact, similarity * decay))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = [fact for fact, _ in scored[:limit]]

    log.debug("Memory search for '%s': %d results", query[:50], len(results))
    return results


def _store_memories(channel_id: str, user_id: str, facts: list[str]):
    if not facts:
        return
    conn = _get_conn()
    now = time.time()

    embeddings = _get_embeddings_batch(facts)
    conn.executemany(
        "INSERT INTO memories (fact, channel_id, user_id, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
        [
            (fact, channel_id, user_id, now, emb.tobytes())
            for fact, emb in zip(facts, embeddings)
        ],
    )
    conn.commit()
    log.info("Stored %d memories for channel=%s user=%s", len(facts), channel_id, user_id)


async def extract_and_store_memories(
    channel_id: str, user_id: str, conversation: str, bot_reply: str
):
    try:
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            conversation=conversation, bot_reply=bot_reply
        )
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You extract facts from conversations. Return only valid JSON arrays.",
                response_mime_type="application/json",
            ),
        )
        facts = json.loads(response.text)
        if not isinstance(facts, list) or not facts:
            return

        valid_facts = [f for f in facts if isinstance(f, str) and f.strip()]
        if not valid_facts:
            return

        await asyncio.to_thread(_store_memories, channel_id, user_id, valid_facts)

    except Exception as e:
        log.error("Memory extraction failed: %s", e)

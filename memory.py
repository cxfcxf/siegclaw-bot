import asyncio
import json
import logging
import time

import pyarrow as pa
from google.genai import types

from config import (
    LANCEDB_PATH,
    MEMORY_DECAY_DAYS,
    MEMORY_EXTRACTION_PROMPT,
    MODEL,
    gemini_client,
)

log = logging.getLogger("siegclaw.memory")

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 3072
DEDUP_THRESHOLD = 0.95

_table = None


def _get_table():
    global _table
    if _table is None:
        raise RuntimeError("Memory DB not initialized — call init_db() first")
    return _table


def _get_embedding(text: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in response.embeddings]


def init_db():
    import lancedb

    global _table
    db = lancedb.connect(LANCEDB_PATH)

    schema = pa.schema([
        pa.field("fact", pa.utf8()),
        pa.field("channel_id", pa.utf8()),
        pa.field("user_id", pa.utf8()),
        pa.field("created_at", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])

    if "memories" in db.table_names():
        _table = db.open_table("memories")
    else:
        _table = db.create_table("memories", schema=schema)

    log.info("LanceDB memory initialized at %s", LANCEDB_PATH)


def search_memories(
    channel_id: str,
    query: str,
    user_id: str | None = None,
    limit: int = 10,
) -> list[str]:
    """Vector search with recency weighting."""
    table = _get_table()

    query_embedding = _get_embedding(query)

    if user_id:
        results = (
            table.search(query_embedding)
            .where(f"channel_id = '{channel_id}' OR user_id = '{user_id}'", prefilter=True)
            .limit(limit * 3)
            .to_list()
        )
    else:
        results = (
            table.search(query_embedding)
            .where(f"channel_id = '{channel_id}'", prefilter=True)
            .limit(limit * 3)
            .to_list()
        )

    if not results:
        return []

    now = time.time()
    decay_seconds = MEMORY_DECAY_DAYS * 86400

    scored = []
    for row in results:
        # LanceDB returns _distance (lower = more similar), convert to similarity
        similarity = max(0.0, 1.0 - row.get("_distance", 1.0))
        age = now - row["created_at"]
        decay = max(0.3, 1.0 - (age / decay_seconds))
        scored.append((row["fact"], similarity * decay))

    scored.sort(key=lambda x: x[1], reverse=True)
    facts = [fact for fact, _ in scored[:limit]]

    log.debug("Memory search for '%s': %d results", query[:50], len(facts))
    return facts


def _is_duplicate(embedding: list[float], channel_id: str) -> bool:
    """Return True if a near-identical fact already exists (similarity >= DEDUP_THRESHOLD)."""
    table = _get_table()

    try:
        results = (
            table.search(embedding)
            .where(f"channel_id = '{channel_id}'", prefilter=True)
            .limit(1)
            .to_list()
        )
    except Exception:
        return False

    if not results:
        return False

    distance = results[0].get("_distance", 1.0)
    similarity = max(0.0, 1.0 - distance)
    return similarity >= DEDUP_THRESHOLD


def _store_memories(channel_id: str, user_id: str, facts: list[str]):
    if not facts:
        return

    table = _get_table()
    now = time.time()

    embeddings = _get_embeddings_batch(facts)
    rows = []
    skipped = 0
    for fact, emb in zip(facts, embeddings):
        if _is_duplicate(emb, channel_id):
            log.debug("Skipping duplicate memory: '%s'", fact[:60])
            skipped += 1
            continue
        rows.append({
            "fact": fact,
            "channel_id": channel_id,
            "user_id": user_id,
            "created_at": now,
            "vector": emb,
        })

    if rows:
        table.add(rows)
        log.info("Stored %d memories (skipped %d duplicates) for channel=%s", len(rows), skipped, channel_id)


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

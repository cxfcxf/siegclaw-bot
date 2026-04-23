import asyncio
import json
import logging
import math
import time

import pyarrow as pa

from config import (
    EMBEDDING_MODEL,
    LANCEDB_PATH,
    MEMORY_DECAY_DAYS,
    MEMORY_EXTRACTION_PROMPT,
    MODEL,
    openai_client,
    openai_generate,
)

log = logging.getLogger("siegclaw.memory")

EMBEDDING_DIM = 1536
DEDUP_THRESHOLD = 0.95

_table = None


def _get_table():
    global _table
    if _table is None:
        raise RuntimeError("Memory DB not initialized — call init_db() first")
    return _table


def _validate_id(id_str: str) -> str:
    if not id_str.isdigit():
        raise ValueError(f"Invalid ID format: {id_str!r}")
    return id_str


def _get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


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
        existing = db.open_table("memories")
        existing_dim = existing.schema.field("vector").type.list_size
        if existing_dim != EMBEDDING_DIM:
            log.warning(
                "Embedding dimension mismatch (existing=%d, new=%d) — recreating memories table",
                existing_dim, EMBEDDING_DIM,
            )
            db.drop_table("memories")
            _table = db.create_table("memories", schema=schema)
        else:
            _table = existing
    else:
        _table = db.create_table("memories", schema=schema)

    log.info("LanceDB memory initialized at %s", LANCEDB_PATH)


def search_memories(
    channel_id: str,
    query: str,
    user_id: str | None = None,
    limit: int = 10,
) -> list[str]:
    table = _get_table()
    query_embedding = _get_embedding(query)
    channel_id = _validate_id(channel_id)

    if user_id:
        user_id = _validate_id(user_id)
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
        similarity = max(0.0, 1.0 - row.get("_distance", 1.0))
        age = now - row["created_at"]
        decay = math.exp(-age / decay_seconds)
        scored.append((row["fact"], similarity * decay))

    scored.sort(key=lambda x: x[1], reverse=True)
    facts = [fact for fact, _ in scored[:limit]]

    log.debug("Memory search for '%s': %d results", query[:50], len(facts))
    return facts


def _is_duplicate(embedding: list[float], channel_id: str) -> bool:
    table = _get_table()
    try:
        channel_id = _validate_id(channel_id)
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

    channel_id = _validate_id(channel_id)
    user_id = _validate_id(user_id)
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
            openai_generate,
            MODEL,
            [
                {"role": "system", "content": "You extract facts from conversations. Return only valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        try:
            facts = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            log.warning("Memory extraction returned invalid JSON: %s", e)
            return
        if not isinstance(facts, list) or not facts:
            return

        valid_facts = [f for f in facts if isinstance(f, str) and f.strip()]
        if not valid_facts:
            return

        await asyncio.to_thread(_store_memories, channel_id, user_id, valid_facts)

    except Exception as e:
        log.error("Memory extraction failed: %s", e)

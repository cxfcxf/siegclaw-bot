"""Tiny mem0 REST service, configured for a fully self-hosted, lightweight stack:

- extraction LLM  -> your llama.cpp (OpenAI-compatible)   [MEM0_LLM_BASE_URL]
- embedder        -> the fastembed service (OpenAI-compatible) [MEM0_EMBED_BASE_URL]
- vector store    -> pgvector (Postgres)                  [POSTGRES_*]
- NO graph store  (the stock mem0-api-server forces neo4j; we don't)

Because embeddings are fetched over HTTP from fastembed, this image needs no
torch/sentence-transformers — just mem0ai + psycopg. It exposes the subset of the
mem0 REST API the bot's Mem0ServiceBackend calls.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI
from mem0 import Memory
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mem0svc] %(levelname)s: %(message)s")

# Disable mem0's outbound telemetry.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

EMBED_DIMS = int(os.getenv("MEM0_EMBED_DIMS", "768"))

CONFIG: dict[str, Any] = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_LLM_MODEL", "local"),
            "openai_base_url": os.environ["MEM0_LLM_BASE_URL"],
            "api_key": os.getenv("MEM0_LLM_API_KEY", "not-needed"),
            "temperature": 0,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_EMBED_MODEL", "nomic-embed-text"),
            "openai_base_url": os.environ["MEM0_EMBED_BASE_URL"],
            "api_key": "not-needed",
            "embedding_dims": EMBED_DIMS,
        },
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": os.getenv("POSTGRES_HOST", "pgvector"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "dbname": os.getenv("POSTGRES_DB", "mem0"),
            "user": os.getenv("POSTGRES_USER", "mem0"),
            "password": os.getenv("POSTGRES_PASSWORD", "mem0"),
            "collection_name": os.getenv("POSTGRES_COLLECTION_NAME", "memories"),
            "embedding_model_dims": EMBED_DIMS,
        },
    },
    "history_db_path": os.getenv("HISTORY_DB_PATH", "/app/history/history.db"),
}

# mem0's OpenAI clients also read OPENAI_API_KEY/OPENAI_BASE_URL from env in some
# paths — keep them harmless/empty so they don't override the per-section config.
os.environ.setdefault("OPENAI_API_KEY", "not-needed")

mem = Memory.from_config(CONFIG)
app = FastAPI(title="mem0-lite")


class Message(BaseModel):
    role: str
    content: str


class AddRequest(BaseModel):
    messages: list[Message]
    user_id: Optional[str] = None
    infer: bool = True
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    limit: int = 10
    filters: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/memories")
def add_memories(req: AddRequest) -> Any:
    messages = [m.model_dump() for m in req.messages]
    return mem.add(
        messages,
        user_id=req.user_id or "default",
        infer=req.infer,
        metadata=req.metadata,
    )


@app.post("/search")
def search(req: SearchRequest) -> Any:
    # Newer mem0 requires entity scoping via filters, not a top-level user_id.
    return mem.search(req.query, filters={"user_id": req.user_id or "default"}, limit=req.limit)


@app.get("/memories")
def get_all(user_id: str = "default") -> Any:
    return mem.get_all(filters={"user_id": user_id})


@app.delete("/memories/{memory_id}")
def delete(memory_id: str) -> dict:
    mem.delete(memory_id)
    return {"deleted": memory_id}

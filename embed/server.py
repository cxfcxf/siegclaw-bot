"""Featherweight OpenAI-compatible embeddings server.

Uses fastembed (ONNX Runtime — no torch, no GPU needed) so a tiny ~140M model
like nomic-embed-text runs on CPU in single-digit milliseconds. This is the same
class of runtime browsers use (ONNX Runtime Web), which is why no Ollama/LM Studio
is needed for embeddings. mem0 (or anything) talks to it via /v1/embeddings.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastembed import TextEmbedding
from pydantic import BaseModel

MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

app = FastAPI(title="fastembed")
_model = TextEmbedding(MODEL)  # loads the ONNX model (cached in the image)


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = None  # accepted, ignored


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL}


@app.post("/v1/embeddings")
def embeddings(body: EmbedRequest) -> dict:
    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    vectors = [v.tolist() for v in _model.embed(texts)]
    return {
        "object": "list",
        "model": body.model or MODEL,
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

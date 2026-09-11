"""OpenAI-compatible client factory.

A single client type (the `openai` SDK) talks to every provider by pointing
`base_url` at the right place. Each provider requires an API key.
"""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from .config import get_provider


@lru_cache(maxsize=16)
def client_for(provider_id: str) -> AsyncOpenAI:
    spec = get_provider(provider_id)
    if spec is None:
        raise ValueError(f"Unknown provider: {provider_id}")
    key = spec.api_key()
    if not key:
        raise ValueError(f"API key is not configured for provider: {provider_id}")
    return AsyncOpenAI(
        base_url=spec.base_url(),
        api_key=key,
        max_retries=2,
    )

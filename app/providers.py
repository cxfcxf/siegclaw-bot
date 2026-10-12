"""OpenAI-compatible client factory.

A single client type (the `openai` SDK) talks to the configured endpoint by
pointing `base_url` at it. The key is optional — a local server takes none.
"""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from . import settings as settings_store
from .config import get_provider


@lru_cache(maxsize=16)
def client_for(provider_id: str) -> AsyncOpenAI:
    spec = get_provider(provider_id)
    if spec is None:
        raise ValueError(f"Unknown provider: {provider_id}")
    base_url = spec.base_url()
    if not base_url:
        raise ValueError("No base URL is configured")
    return AsyncOpenAI(
        base_url=base_url,
        # The SDK refuses an empty key; keyless servers ignore the placeholder.
        api_key=spec.api_key() or "none",
        # Retries are agent.with_retries' job (SEND_RETRIES), so the SDK's
        # own would only multiply them.
        max_retries=0,
    )


def _on_settings_changed(changed: set[str]) -> None:
    """A new key or base URL takes effect on the next call, not the next restart."""
    if {"LLM_API_KEY", "LLM_BASE_URL"} & changed:
        client_for.cache_clear()


settings_store.on_change(_on_settings_changed)

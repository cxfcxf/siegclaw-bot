"""The agent loop: streaming chat with tool-call orchestration.

`run_turn` is an async generator that yields event dicts the web layer serializes
as SSE. It accumulates streamed tool_calls, executes them, appends results, and
loops until the model stops or the iteration cap is hit. Provider-agnostic: it
relies only on the OpenAI-compatible chat-completions + function-calling contract.
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from datetime import datetime
from typing import Any, AsyncGenerator
from zoneinfo import ZoneInfo

from . import memory, storage
from .config import (
    HARNESS_TZ,
    MAX_AGENT_ITERATIONS,
    SEND_FALLBACK_RETRIES,
    SEND_FALLBACK_RETRY_DELAY,
    SOUL_PATH,
    THINK_KWARG,
    UPLOADS_DIR,
    model_valid_for,
    provider_serving,
    resolve_default_model,
)
from .providers import client_for
from .skills import discover_skills, load_skill_tool, skills_index
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
from .tools.clock import clock_tools
from .tools.registry import Registry
from .tools.web import web_tools


# Providers that require the chain-of-thought to be replayed on tool-call turns
# (they 400 otherwise). The stored `reasoning` is emitted back as
# `reasoning_content` for these; for everyone else it's dropped.
_REASONING_REPLAY_PROVIDERS = ("deepseek", "xiaomi")


def needs_reasoning_replay(provider: str) -> bool:
    return provider in _REASONING_REPLAY_PROVIDERS


def read_soul() -> str:
    if SOUL_PATH.exists():
        return SOUL_PATH.read_text(errors="replace").strip()
    return "You are a helpful assistant."


def build_registry(mcp_tools: list | None = None) -> tuple[Registry, dict]:
    """Assemble all tools. Returns (registry, skills) — skills feed the prompt."""
    registry = Registry()
    registry.extend(clock_tools())
    registry.extend(builtin_tools())
    registry.extend(web_tools())
    registry.extend(browser_tools())
    registry.extend(memory.memory_tools())
    skills = discover_skills()
    if skills:
        registry.add(load_skill_tool(skills))
    if mcp_tools:
        registry.extend(mcp_tools)
    return registry, skills


def conversation_time_block(started_at: float | None = None) -> str:
    """A 'today' captured ONCE per conversation (its start date, DAY precision
    only) and placed in the cacheable system prefix. Frozen on purpose: only the
    date is given (not the time of day), so the prefix stays byte-identical
    across every turn and the prompt cache never invalidates — the date drifts
    stale by at most a day within a long conversation, which is fine (and far
    better than the model's training cutoff). `started_at` is the conversation's
    created_at; None means use now() (single-shot surfaces: channel, cron). For
    the precise time of day, to the second, the model calls `current_time`."""
    try:
        tz = ZoneInfo(HARNESS_TZ)
        dt = datetime.fromtimestamp(started_at, tz) if started_at else datetime.now(tz)
    except Exception:
        dt = datetime.now().astimezone()
    stamp = dt.strftime("%A, %B %-d, %Y")
    return (
        f"This conversation started on {stamp} ({HARNESS_TZ}). Treat that as the "
        "current date — you already know it, so do not search the web just to "
        "determine today's date or year. Only the date is provided here (kept "
        "stable so the prompt can be cached); for the precise time of day, to the "
        "second, call the `current_time` tool."
    )


def assemble_system_prompt(skills: dict, *parts: str) -> str:
    """Join the given prompt sections plus the skills index (when non-empty),
    dropping any blanks. Shared by the web, Discord, and cron system prompts."""
    sections = [*parts, skills_index(skills)]
    return "\n\n".join(p for p in sections if p)


def static_system_prompt(skills: dict, *extra: str) -> str:
    """The cacheable prefix of the system prompt: soul + any caller-supplied
    sections (preamble, day-level time, FROZEN memory) + skills index. Every one
    of these is stable for the life of a conversation, so the prefix — and all
    prior turns — stay in the prompt cache and never get re-evaluated. There is
    deliberately NO per-turn-volatile content here; nothing rides in the tail."""
    return assemble_system_prompt(skills, read_soul(), *extra)


def conversation_memory_block(conversation: dict | None, query: str, scope: str | None = None) -> str:
    """The memory section of the (cacheable) system prefix, FROZEN for the life
    of the conversation. On the first turn (no stored frozen block yet) we
    retrieve the memories relevant to the opening user message and persist the
    resulting block; every later turn reuses that exact string so the prefix —
    and thus the prompt cache — never changes. The model can pull fresher or
    differently-relevant memories on demand via the `search_memory` tool.

    For stateless surfaces (no `conversation` row) the block is just computed
    from the query each time — those turns are single-shot, so the prefix is
    still stable across the turn's own tool-loop iterations."""
    if conversation is None:
        return memory.relevant_block(query, scope)
    frozen = conversation.get("frozen_memory")
    if frozen is None:
        frozen = memory.relevant_block(query, scope)
        storage.set_frozen_memory(conversation["id"], frozen)
    return frozen


async def resolve_for_turn(
    conversation_id: str | None,
    provider: str,
    model: str,
    effort: str | None,
) -> tuple[str, str, str | None]:
    """Pick the (provider, model, effort) to actually run this turn with, with
    retry-then-fallback.

    Two reasons to fall back: the provider isn't currently serving (e.g. the
    local engine was stopped — see config.provider_serving), or the model isn't
    valid for that provider (e.g. a llama.cpp model id left on a conversation
    that got switched to a cloud provider). Retries cover transient blips; once
    it falls back, the switch is PERSISTED on the conversation so the rest of
    the session stays on the fallback — it won't flap back if the original
    returns mid-conversation. A brand-new conversation resolves the original
    again via the default (which picks it once it's back up).

    Returns the (provider, model, effort) to run with — unchanged only if the
    requested provider is serving AND the model is valid for it (or no different
    fallback exists)."""
    # Fast path: model is valid for the provider -> just confirm it's serving,
    # retrying a few times in case the provider blipped.
    if model_valid_for(provider, model):
        last_try = max(SEND_FALLBACK_RETRIES - 1, 0)
        for attempt in range(SEND_FALLBACK_RETRIES):
            if provider_serving(provider):
                return provider, model, effort
            if attempt < last_try:
                await asyncio.sleep(SEND_FALLBACK_RETRY_DELAY)
    # Provider down after retries, or model wrong for it -> switch to the
    # default (fallback) model and persist it for the session.
    fb = resolve_default_model()
    if fb and (fb[0] != provider or fb[1] != model):
        if conversation_id:
            storage.update_conversation_model(conversation_id, fb[0], fb[1])
        return fb
    # No different fallback available — return as-is and let the turn surface the error.
    return provider, model, effort


def reasoning_extra_body(provider: str, think: bool, effort: str | None = None) -> dict[str, Any]:
    """Per-request reasoning toggle + effort. The mechanism differs per provider:
    - llama.cpp: chat_template_kwargs flag (server-side), on/off only.
    - DeepSeek/Xiaomi MiMo: {"thinking": {"type": enabled|disabled}} (OpenAI-compat).
      DeepSeek also honors reasoning_effort (high/max).
    - OpenRouter: reasoning.enabled (gateway-level); optional reasoning.effort.
    - Others (OpenAI): no known toggle; let the model default.
    Shared by the web turn (run_turn) and the Discord turn."""
    if provider == "llamacpp":
        return {"chat_template_kwargs": {THINK_KWARG: think}}
    if needs_reasoning_replay(provider):
        body: dict[str, Any] = {"thinking": {"type": "enabled" if think else "disabled"}}
        if think and effort and provider == "deepseek":
            body["reasoning_effort"] = effort
        return body
    if provider == "openrouter":
        body = {"reasoning": {"enabled": bool(think)}}
        if think and effort:
            body["reasoning"]["effort"] = effort
        return body
    return {}


def _new_tool_call() -> dict[str, Any]:
    return {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}


def _image_data_url(url: str) -> str | None:
    """Turn a stored '/uploads/<file>' reference into a base64 data URL the model
    can read, regardless of network topology. Pass through real data/http URLs."""
    if url.startswith(("data:", "http://", "https://")):
        return url
    name = url.rsplit("/", 1)[-1]
    path = UPLOADS_DIR / name
    if not path.exists():
        return None
    mime = mimetypes.guess_type(name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _to_api_messages(history: list[dict[str, Any]], provider: str | None = None) -> list[dict[str, Any]]:
    """Convert stored messages into OpenAI-ready ones, expanding any attached
    images into multimodal `content` parts and dropping the non-API `images` key.

    For providers that require the chain-of-thought to be replayed on tool-call
    turns (DeepSeek, Xiaomi MiMo), the stored `reasoning` is emitted back as
    `reasoning_content`; for everyone else it's dropped."""
    drop = ("images", "reasoning")  # non-API display fields
    keep_reasoning = needs_reasoning_replay(provider)
    out: list[dict[str, Any]] = []
    for msg in history:
        images = msg.get("images")
        base = {k: v for k, v in msg.items() if k not in drop}
        if images:
            parts: list[dict[str, Any]] = []
            text = msg.get("content")
            if text:
                parts.append({"type": "text", "text": text})
            for url in images:
                data_url = _image_data_url(url)
                if data_url:
                    parts.append({"type": "image_url", "image_url": {"url": data_url}})
            base["content"] = parts
        if keep_reasoning and msg.get("reasoning") and msg.get("role") == "assistant":
            base["reasoning_content"] = msg["reasoning"]
        out.append(base)
    return out


async def run_turn(
    conversation_id: str,
    provider: str,
    model: str,
    user_message: str,
    registry: Registry,
    skills: dict,
    images: list[str] | None = None,
    think: bool = True,
    effort: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    client = client_for(provider)
    replay_reasoning = needs_reasoning_replay(provider)

    user_msg_id = storage.add_message(
        conversation_id, "user", content=user_message, images=images
    )
    storage.touch_conversation(conversation_id, provider, model)
    # Tell the client the stored id of this user message so the UI can offer
    # retry-from-here (which needs to rewind by id on the server).
    yield {"type": "user_message", "id": user_msg_id}

    # Fully static system prefix (soul + conversation-fixed day-level time +
    # FROZEN memory + skills) — identical every turn, so it (and all prior turns)
    # stay in the prompt cache. Memory is snapshotted on the first turn and
    # reused verbatim thereafter; the model fetches fresher facts via
    # `search_memory`, and the precise time via `current_time` (both are tool
    # calls that happen after the cached prefix, so they never bust it).
    convo = storage.get_conversation(conversation_id)
    started_at = convo.get("created_at") if convo else None
    time_block = conversation_time_block(started_at)
    memory_block = await asyncio.to_thread(conversation_memory_block, convo, user_message)
    sys_content = await asyncio.to_thread(static_system_prompt, skills, time_block, memory_block)
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
    messages.extend(_to_api_messages(storage.get_messages(conversation_id), provider))

    tool_schemas = registry.schemas()
    extra_body = reasoning_extra_body(provider, think, effort)
    final_answer = ""
    total_tokens = 0  # completion tokens across all model calls this turn (for tok/s)
    last_prompt_tokens = 0  # prompt tokens of the final model call (context usage)

    for _ in range(MAX_AGENT_ITERATIONS):
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas or None,
                stream=True,
                # Ask the server to emit a final usage chunk so the UI can show tok/s.
                stream_options={"include_usage": True},
                extra_body=extra_body,
            )
        except Exception as exc:
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return

        content_buf = ""
        reasoning_buf = ""
        tool_calls: list[dict[str, Any]] = []
        finish_reason = None

        async for chunk in stream:
            # The usage chunk arrives last and carries no choices — read it first.
            usage = getattr(chunk, "usage", None)
            if usage:
                if getattr(usage, "completion_tokens", None):
                    total_tokens += usage.completion_tokens
                pt = getattr(usage, "prompt_tokens", None)
                if pt:
                    last_prompt_tokens = pt
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Reasoning text. Field name differs by provider/engine:
            # - reasoning_content: DeepSeek/llama.cpp convention.
            # - reasoning: OpenRouter's field (flat string per chunk).
            # - reasoning_details: OpenRouter's structured array fallback.
            reasoning_piece = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if not reasoning_piece:
                for rd in getattr(delta, "reasoning_details", None) or []:
                    reasoning_piece = (reasoning_piece or "") + (
                        getattr(rd, "text", None) or getattr(rd, "summary", None) or ""
                    )
            if reasoning_piece:
                reasoning_buf += reasoning_piece
                yield {"type": "reasoning", "text": reasoning_piece}

            if getattr(delta, "content", None):
                content_buf += delta.content
                yield {"type": "token", "text": delta.content}

            for tc in getattr(delta, "tool_calls", None) or []:
                while len(tool_calls) <= tc.index:
                    tool_calls.append(_new_tool_call())
                slot = tool_calls[tc.index]
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["function"]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments

        # Persist the assistant message (content and/or tool calls). Reasoning is
        # stored for the UI; for DeepSeek/MiMo it's also kept on the in-memory
        # assistant message (as reasoning_content) so tool-call turns replay it
        # back to the API on the next iteration (else those providers 400).
        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content_buf or None,
                "tool_calls": tool_calls,
            }
            if replay_reasoning and reasoning_buf:
                assistant_msg["reasoning_content"] = reasoning_buf
            messages.append(assistant_msg)
            storage.add_message(
                conversation_id, "assistant", content=content_buf or None,
                tool_calls=tool_calls, reasoning=reasoning_buf,
            )
        elif content_buf or reasoning_buf:
            messages.append({"role": "assistant", "content": content_buf})
            storage.add_message(conversation_id, "assistant", content=content_buf, reasoning=reasoning_buf)
        if content_buf:
            final_answer = content_buf

        if finish_reason == "tool_calls" or tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"] or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_call", "id": tc["id"], "name": name, "arguments": raw_args}
                result = await registry.call(name, args)
                yield {"type": "tool_result", "id": tc["id"], "name": name, "result": result}
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "name": name, "content": result}
                messages.append(tool_msg)
                storage.add_message(
                    conversation_id, "tool", content=result, tool_call_id=tc["id"], name=name
                )
            continue  # loop back for the model's next step

        # No tool calls => the turn is complete. Schedule debounced background
        # memory extraction (fires MEMORY_DEBOUNCE_SECONDS after the last turn)
        # and unblock the composer immediately — extraction no longer holds the
        # stream open.
        memory.schedule_extraction(user_message, final_answer)
        if last_prompt_tokens:
            storage.set_prompt_tokens(conversation_id, last_prompt_tokens)
        yield {"type": "done", "tokens": total_tokens or None, "prompt_tokens": last_prompt_tokens or None}
        return

    yield {"type": "error", "message": f"Stopped after {MAX_AGENT_ITERATIONS} tool iterations."}

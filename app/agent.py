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
import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator
from zoneinfo import ZoneInfo

from . import docs, storage, wiki
from .config import (
    HARNESS_TZ,
    MAX_AGENT_ITERATIONS,
    SEND_FALLBACK_RETRIES,
    SEND_FALLBACK_RETRY_DELAY,
    THINK_KWARG,
    UPLOADS_DIR,
    model_valid_for,
    provider_serving,
    resolve_default_model,
)
from .providers import client_for
from .research import RESEARCH_MODE_PREAMBLE, research_tools, set_origin as set_research_origin
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
from .tools.clock import clock_tools
from .tools.jobs import job_tools
from .tools.registry import Registry
from .tools.web import web_tools


# Providers that require the chain-of-thought to be replayed on tool-call turns
# (they 400 otherwise). The stored `reasoning` is emitted back as
# `reasoning_content` for these; for everyone else it's dropped.
_REASONING_REPLAY_PROVIDERS = ("deepseek", "xiaomi")


def needs_reasoning_replay(provider: str) -> bool:
    return provider in _REASONING_REPLAY_PROVIDERS


def build_registry(mcp_tools: list | None = None) -> Registry:
    """Assemble all tools for owner surfaces (web UI + Discord DMs). The wiki
    tools are WRITABLE here — the wiki feeds every future system prompt, so
    only the owner's surfaces may write it."""
    registry = Registry()
    registry.extend(clock_tools())
    registry.extend(builtin_tools())
    registry.extend(web_tools())
    registry.extend(browser_tools())
    registry.extend(wiki.wiki_tools(writable=True))
    registry.extend(job_tools())
    registry.extend(research_tools())
    if mcp_tools:
        registry.extend(mcp_tools)
    return registry


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


def static_system_prompt(*extra: str) -> str:
    """The system prompt: the wiki's home page (the model's own root page) +
    any caller-supplied sections (surface preamble, day-level time) + the wiki
    index. Stable within a conversation unless a wiki page changes — a page
    write busts the prompt cache once, which is the price of learning."""
    sections = [wiki.home_text(), *extra, wiki.wiki_index()]
    return "\n\n".join(p for p in sections if p)


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


def _upload_path(url: str) -> Path | None:
    """Resolve a stored '/uploads/...' URL (flat or '<cid>/<file>') to its file
    on disk, refusing anything that escapes UPLOADS_DIR."""
    rel = url.removeprefix("/uploads/")
    path = (UPLOADS_DIR / rel).resolve()
    if not path.is_relative_to(UPLOADS_DIR.resolve()):
        return None
    return path


def _image_data_url(url: str) -> str | None:
    """Turn a stored '/uploads/...' reference into a base64 data URL the model
    can read, regardless of network topology. Pass through real data/http URLs."""
    if url.startswith(("data:", "http://", "https://")):
        return url
    path = _upload_path(url)
    if path is None or not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _adopt_upload(conversation_id: str, url: str) -> str:
    """Move a freshly uploaded flat '/uploads/<file>' into this conversation's
    directory ('/uploads/<cid>/<file>') and return the rewritten URL. Uploads
    land flat because the uploader often doesn't know the conversation yet
    (web images are uploaded before the first message creates the chat; a DM
    voice clip is transcribed before its conversation is resolved); adopting
    at attach time makes uploads/ self-describing — every file lives under the
    conversation that owns it, and deleting the conversation deletes its dir.
    Already-adopted ('/uploads/<cid>/…') and external URLs pass through."""
    m = re.fullmatch(r"/uploads/([^/]+)", url)
    if not m:
        return url
    src = UPLOADS_DIR / m.group(1)
    if not src.is_file():
        return url
    dest_dir = UPLOADS_DIR / conversation_id
    dest_dir.mkdir(exist_ok=True)
    src.rename(dest_dir / m.group(1))
    # A document's extraction sidecar ('<file>.txt') travels with it.
    side = src.with_name(src.name + ".txt")
    if side.is_file():
        side.rename(dest_dir / side.name)
    return f"/uploads/{conversation_id}/{m.group(1)}"


def _to_api_messages(history: list[dict[str, Any]], provider: str | None = None) -> list[dict[str, Any]]:
    """Convert stored messages into OpenAI-ready ones, expanding any attached
    images into multimodal `content` parts and dropping the non-API `images` key.

    Document attachments (`docs`) are expanded into their extracted text,
    prepended to the user's message inside [Attached file: ...] markers — the
    model reads the document as part of the question, every turn.

    For providers that require the chain-of-thought to be replayed on tool-call
    turns (DeepSeek, Xiaomi MiMo), the stored `reasoning` is emitted back as
    `reasoning_content`; for everyone else it's dropped."""
    drop = ("images", "reasoning", "model", "audio", "docs")  # non-API display fields
    keep_reasoning = needs_reasoning_replay(provider)
    out: list[dict[str, Any]] = []
    for msg in history:
        images = msg.get("images")
        base = {k: v for k, v in msg.items() if k not in drop}
        if msg.get("docs"):
            blocks = []
            for d in msg["docs"]:
                path = _upload_path(d.get("url", ""))
                if path is not None and path.exists():
                    blocks.append(docs.prompt_block(d.get("name") or path.name, path))
                else:
                    blocks.append(f"[Attached file {d.get('name', '?')!r} is no longer available]")
            base["content"] = "\n\n".join(blocks + [msg.get("content") or ""]).strip()
        if images:
            parts: list[dict[str, Any]] = []
            text = base.get("content")  # includes any expanded doc blocks
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
    images: list[str] | None = None,
    think: bool = True,
    effort: str | None = None,
    preamble: str | None = None,
    audio: str | None = None,
    docs: list[dict] | None = None,
    max_iterations: int | None = None,  # tool-loop budget; None = the default cap
    research_mode: bool = False,  # composer toggle: scope the question, then deep_research
) -> AsyncGenerator[dict[str, Any], None]:
    client = client_for(provider)
    replay_reasoning = needs_reasoning_replay(provider)
    # If this turn launches a deep_research, its finished report links back here.
    set_research_origin(conversation_id)

    # `audio` is the user's voice clip ('/uploads/...'), stored on the user row
    # for playback. The model itself only ever sees the transcript.
    # `docs` are document attachments [{"url", "name"}]; their extracted text is
    # injected at prompt-build time (_to_api_messages), not stored in content.
    if images:
        images = [_adopt_upload(conversation_id, u) for u in images]
    if audio:
        audio = _adopt_upload(conversation_id, audio)
    if docs:
        docs = [
            {**d, "url": _adopt_upload(conversation_id, d["url"])}
            for d in docs if d.get("url")
        ] or None
    user_msg_id = storage.add_message(
        conversation_id, "user", content=user_message, images=images, audio=audio,
        docs=docs,
    )
    storage.touch_conversation(conversation_id, provider, model)
    # Tell the client the stored id of this user message so the UI can offer
    # retry-from-here (which needs to rewind by id on the server).
    yield {"type": "user_message", "id": user_msg_id}

    # Static system prefix (wiki home + conversation-fixed day-level time +
    # wiki index) — stable across turns so it (and all prior turns) stay in
    # the prompt cache. The model pulls specifics via tool calls that happen
    # after the cached prefix (`read_wiki_page`, `search_wiki`, `current_time`),
    # so routine turns never bust it; only an actual wiki edit does.
    convo = storage.get_conversation(conversation_id)
    started_at = convo.get("created_at") if convo else None
    time_block = conversation_time_block(started_at)
    # Optional surface preamble (e.g. the cron job context) slots in before the
    # time block, mirroring discord_system_prompt's ordering.
    extra = ([preamble] if preamble else []) + [time_block]
    sys_content = await asyncio.to_thread(static_system_prompt, *extra)
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
    messages.extend(_to_api_messages(storage.get_messages(conversation_id), provider))
    # Research mode rides at the END of the prompt (right after the user's
    # message) where models actually obey it — inside the system prefix it gets
    # drowned out by the standing "answer with tools" instructions, and it
    # would bust the prompt cache. Not persisted: the client re-sends the flag
    # on the clarifying round-trip until deep_research actually launches.
    if research_mode:
        messages.append({"role": "system", "content": RESEARCH_MODE_PREAMBLE})

    tool_schemas = registry.schemas()
    if research_mode:
        # Structural enforcement of the preamble: with only deep_research on
        # the menu, the model can't "helpfully" search and answer the question
        # itself (deepseek-flash did exactly that when merely told not to).
        tool_schemas = [
            s for s in tool_schemas if s["function"]["name"] == "deep_research"
        ]
    extra_body = reasoning_extra_body(provider, think, effort)
    final_answer = ""
    final_answer_id: str | None = None  # stored id of the message holding it
    total_tokens = 0  # completion tokens across all model calls this turn (for tok/s)
    last_prompt_tokens = 0  # prompt tokens of the final model call (context usage)
    turn_prompt_tokens = 0  # prompt tokens summed over every call (billed total)

    iteration_cap = max_iterations or MAX_AGENT_ITERATIONS
    for _i in range(iteration_cap):
        # On the final allowed step, don't let the loop dead-end with no answer:
        # tell the model it's out of tool budget and strip the tools so it must
        # return its best prose answer from what it already gathered.
        final_pass = _i == iteration_cap - 1
        if final_pass:
            messages.append({
                "role": "user",
                "content": (
                    "(system) You've reached the tool-use limit for this turn. "
                    "Do not call any more tools — give your best final answer now "
                    "using what you already have."
                ),
            })
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=None if final_pass else (tool_schemas or None),
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
                    turn_prompt_tokens += pt
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
                model=f"{provider}/{model}",
            )
        elif content_buf or reasoning_buf:
            messages.append({"role": "assistant", "content": content_buf})
            final_answer_id = storage.add_message(
                conversation_id, "assistant", content=content_buf, reasoning=reasoning_buf,
                model=f"{provider}/{model}",
            )
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

        # No tool calls => the turn is complete. message_id lets the web UI tag
        # the finished bubble with its stored id (read-aloud needs it).
        if last_prompt_tokens:
            storage.set_prompt_tokens(conversation_id, last_prompt_tokens)
        storage.add_token_usage(turn_prompt_tokens, total_tokens)
        yield {
            "type": "done", "tokens": total_tokens or None,
            "prompt_tokens": last_prompt_tokens or None,
            "message_id": final_answer_id,
        }
        return

    storage.add_token_usage(turn_prompt_tokens, total_tokens)  # spent even on a dead-end
    yield {"type": "error", "message": f"Stopped after {iteration_cap} tool iterations."}

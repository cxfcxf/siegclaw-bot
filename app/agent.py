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
from .config import HARNESS_TZ, MAX_AGENT_ITERATIONS, SOUL_PATH, THINK_KWARG, UPLOADS_DIR
from .providers import client_for
from .skills import discover_skills, load_skill_tool, skills_index
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
from .tools.registry import Registry
from .tools.web import web_tools


def read_soul() -> str:
    if SOUL_PATH.exists():
        return SOUL_PATH.read_text(errors="replace").strip()
    return "You are a helpful assistant."


def build_registry(mcp_tools: list | None = None) -> tuple[Registry, dict]:
    """Assemble all tools. Returns (registry, skills) — skills feed the prompt."""
    registry = Registry()
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


def current_datetime_block() -> str:
    """A fresh 'now' line so the model never has to search for today's date."""
    try:
        now = datetime.now(ZoneInfo(HARNESS_TZ))
    except Exception:
        now = datetime.now().astimezone()
    stamp = now.strftime("%A, %B %-d, %Y, %-I:%M %p %Z")
    return (
        f"The current date and time is {stamp} ({HARNESS_TZ}). "
        "Treat this as the present moment — you already know the date, so do not "
        "search the web just to determine today's date or the current year. Use it "
        "directly when forming queries or reasoning about current events."
    )


def system_prompt(skills: dict, query: str) -> str:
    parts = [read_soul(), current_datetime_block(), memory.relevant_block(query)]
    index = skills_index(skills)
    if index:
        parts.append(index)
    return "\n\n".join(parts)


def reasoning_extra_body(provider: str, think: bool) -> dict[str, Any]:
    """Per-request reasoning toggle. The mechanism differs per provider:
    - llama.cpp/Ollama/LM Studio/sglang: chat_template_kwargs flag (server-side).
    - OpenRouter: reasoning.enabled (gateway-level reasoning control).
    - Others (OpenAI/Groq): no known toggle; let the model default.
    Shared by the web turn (run_turn) and the Discord turn."""
    if provider in ("llamacpp", "ollama", "lmstudio", "local"):
        return {"chat_template_kwargs": {THINK_KWARG: think}}
    if provider == "openrouter":
        return {"reasoning": {"enabled": bool(think)}}
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


def _to_api_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored messages into OpenAI-ready ones, expanding any attached
    images into multimodal `content` parts and dropping the non-API `images` key."""
    drop = ("images", "reasoning")  # non-API display fields
    out: list[dict[str, Any]] = []
    for msg in history:
        images = msg.get("images")
        if not images:
            out.append({k: v for k, v in msg.items() if k not in drop})
            continue
        parts: list[dict[str, Any]] = []
        text = msg.get("content")
        if text:
            parts.append({"type": "text", "text": text})
        for url in images:
            data_url = _image_data_url(url)
            if data_url:
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
        new = {k: v for k, v in msg.items() if k not in (*drop, "content")}
        new["content"] = parts
        out.append(new)
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
) -> AsyncGenerator[dict[str, Any], None]:
    client = client_for(provider)

    user_msg_id = storage.add_message(
        conversation_id, "user", content=user_message, images=images
    )
    storage.touch_conversation(conversation_id, provider, model)
    # Tell the client the stored id of this user message so the UI can offer
    # retry-from-here (which needs to rewind by id on the server).
    yield {"type": "user_message", "id": user_msg_id}

    # Retrieve relevant memories (may embed the query) off the event loop.
    sys_content = await asyncio.to_thread(system_prompt, skills, user_message)
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
    messages.extend(_to_api_messages(storage.get_messages(conversation_id)))

    tool_schemas = registry.schemas()
    extra_body = reasoning_extra_body(provider, think)
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
        # stored for the UI but not fed back to the model (stripped on replay).
        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content_buf or None,
                "tool_calls": tool_calls,
            }
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

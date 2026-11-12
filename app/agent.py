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
import logging
import mimetypes
import re
import time
from datetime import datetime
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, AsyncGenerator, TypeVar
from zoneinfo import ZoneInfo

import openai

from . import docs, storage, wiki
from .config import (
    UPLOADS_DIR,
    get_provider,
    is_openrouter,
    resolve_default_model,
    settings,
)
from .providers import client_for
from .research import RESEARCH_MODE_PREAMBLE, research_tools, set_origin as set_research_origin
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
from .tools.clock import clock_tools
from .tools.delegate import delegate_tools
from .tools.jobs import job_tools
from .tools.registry import Registry
from .tools.web import web_tools


log = logging.getLogger("siegclaw.agent")

T = TypeVar("T")


def build_registry(mcp_tools: list | None = None) -> Registry:
    """Assemble all tools for owner surfaces (web UI + Discord DMs). These get
    the PRIVATE wiki, read-write. They never get the public (Discord-channel)
    space: the owner's surfaces have no reason to search it, and keeping the
    binding one-way means channel-authored text can't reach an owner prompt."""
    registry = Registry()
    registry.extend(clock_tools())
    registry.extend(builtin_tools())
    registry.extend(web_tools())
    registry.extend(delegate_tools())
    registry.extend(browser_tools())
    registry.extend(wiki.wiki_tools(writable=True, space=wiki.PRIVATE))
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
        tz = ZoneInfo(settings.HARNESS_TZ)
        dt = datetime.fromtimestamp(started_at, tz) if started_at else datetime.now(tz)
    except Exception:
        dt = datetime.now().astimezone()
    stamp = dt.strftime("%A, %B %-d, %Y")
    return (
        f"This conversation started on {stamp} ({settings.HARNESS_TZ}). Treat that as the "
        "current date — you already know it, so do not search the web just to "
        "determine today's date or year. Only the date is provided here (kept "
        "stable so the prompt can be cached); for the precise time of day, to the "
        "second, call the `current_time` tool."
    )


def static_system_prompt(*extra: str, space: str = wiki.PRIVATE) -> str:
    """The system prompt: the home page of ONE wiki space (the model's own root
    page) + any caller-supplied sections (surface preamble, day-level time) +
    that space's index. It changes whenever a wiki page's name or summary does,
    so run_turn freezes the result per conversation rather than rebuilding it
    every turn.

    `space` decides which corpus the turn can see at all. Owner surfaces pass
    PRIVATE (the default); Discord channels pass PUBLIC, so neither the owner's
    home page nor even the *names* of their pages enter the prompt."""
    sections = [wiki.home_text(space), *extra, wiki.wiki_index(space)]
    return "\n\n".join(p for p in sections if p)


async def resolve_for_turn(
    conversation_id: str | None,
    provider: str,
    model: str,
    effort: str | None,
) -> tuple[str, str, str | None]:
    """Pick the (provider, model, effort) to run this turn with.

    The requested model runs as-is — a model the endpoint rejects surfaces as
    the API's own error. The one exception is a conversation stored under a
    provider that no longer exists (or with no model at all): it moves to the
    default model, persisted so later turns stay there."""
    if get_provider(provider) is not None and model:
        return provider, model, effort
    default = resolve_default_model()
    if default is None:
        return provider, model, effort
    if conversation_id:
        storage.update_conversation_model(conversation_id, default[0], default[1])
    return default


def _retryable(exc: Exception) -> bool:
    """Worth another attempt: network trouble, timeouts, rate limits and server
    errors. A 4xx (bad key, unknown model, malformed request) fails the same way
    every time, so it surfaces at once."""
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return isinstance(exc, openai.APIError)


async def with_retries(call: Callable[[], Awaitable[T]]) -> T:
    """Run a completion call, retrying the same model up to SEND_RETRIES times
    (waiting SEND_RETRY_DELAY × attempt between tries) before the last error
    propagates. There is no fallback model: a call that keeps failing is the
    caller's error to show."""
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:
            if attempt >= settings.SEND_RETRIES or not _retryable(exc):
                raise
            attempt += 1
            log.warning(
                "Completion call failed (%s: %s); retry %d/%d",
                type(exc).__name__, exc, attempt, settings.SEND_RETRIES,
            )
            await asyncio.sleep(settings.SEND_RETRY_DELAY * attempt)


def reasoning_extra_body(
    provider: str,
    think: bool,
    effort: str | None = None,
    *,
    model: str = "",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Per-request reasoning toggle + effort, plus caching hints. The mechanism
    depends on what the endpoint is:
    - OpenRouter: its unified {"reasoning": {...}} object. Claude only caches
      when asked, so Anthropic models also get a top-level `cache_control`,
      which OpenRouter advances to the end of the prompt on every call.
      `session_id` (the conversation id) pins a conversation to one upstream
      host so its cache stays warm; OPENROUTER_PROVIDERS instead restricts
      every request to the listed hosts.
    - Anything else: OpenAI's standard `reasoning_effort`, sent only when
      thinking is on with an effort chosen; servers that lack it ignore it.
    Shared by the web turn (run_turn) and the Discord turn."""
    spec = get_provider(provider)
    if spec is None:
        return {}
    if is_openrouter(spec.base_url()):
        reasoning: dict[str, Any] = {"enabled": think}
        if think and effort:
            reasoning["effort"] = effort
        body: dict[str, Any] = {"reasoning": reasoning}
        if model.lstrip("~").startswith("anthropic/"):
            body["cache_control"] = {"type": "ephemeral"}
        if session_id:
            body["session_id"] = session_id[:256]
        hosts = [h.strip() for h in settings.OPENROUTER_PROVIDERS.split(",") if h.strip()]
        if hosts:
            body["provider"] = {"order": hosts, "allow_fallbacks": False}
        return body
    return {"reasoning_effort": effort} if think and effort else {}


def _merge_reasoning_details(acc: list[dict[str, Any]], pieces: list[Any]) -> None:
    """Fold streamed `reasoning_details` deltas into whole blocks. OpenRouter
    streams each block in fragments sharing an `index`; text-like fields
    concatenate and the rest (type, format, id) take the latest value. The
    merged blocks are replayed on tool-call turns — Claude rejects a tool
    result whose preceding thinking block (and its signature) went missing."""
    for piece in pieces:
        d = piece if isinstance(piece, dict) else piece.model_dump(exclude_none=True)
        idx = d.get("index", len(acc))
        while len(acc) <= idx:
            acc.append({})
        slot = acc[idx]
        for k, v in d.items():
            if k in ("text", "summary", "data", "signature") and isinstance(v, str):
                slot[k] = slot.get(k, "") + v
            elif v is not None:
                slot[k] = v


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


def _to_api_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored messages into OpenAI-ready ones, expanding any attached
    images into multimodal `content` parts and dropping the non-API `images` key.

    Document attachments (`docs`) are expanded into their extracted text,
    prepended to the user's message inside [Attached file: ...] markers — the
    model reads the document as part of the question, every turn.

    Stored `reasoning` is display-only and dropped: earlier turns' thinking is
    never replayed, only the current turn's (see `_merge_reasoning_details`)."""
    drop = ("images", "reasoning", "model", "audio", "docs")  # non-API display fields
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

    # System prefix (wiki home + conversation-fixed day-level time + wiki
    # index), frozen per conversation: it is built once, stored, and replayed
    # byte-identical on later turns so it (and all prior turns) stay in the
    # prompt cache. A wiki write therefore does NOT reach this conversation's
    # prompt — the model already has what it wrote in its history, and reads
    # anything else via `read_wiki_page` / `search_wiki`. The prompt is rebuilt
    # from the current wiki once the conversation has sat idle past
    # SYSTEM_PROMPT_REFRESH_IDLE (the cache is gone by then, so it costs
    # nothing), or when the surface preamble isn't the one it was frozen with.
    convo = storage.get_conversation(conversation_id) or {}
    sys_content = convo.get("system_prompt")
    idle = time.time() - (convo.get("system_prompt_at") or 0)
    if (
        not sys_content
        or idle >= settings.SYSTEM_PROMPT_REFRESH_IDLE
        or (preamble or "") not in sys_content
    ):
        # Optional surface preamble (e.g. the cron job context) slots in before
        # the time block, mirroring discord_system_prompt's ordering.
        extra = ([preamble] if preamble else []) + [
            conversation_time_block(convo.get("created_at"))
        ]
        sys_content = await asyncio.to_thread(static_system_prompt, *extra)
        storage.set_system_prompt(conversation_id, sys_content)
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
    messages.extend(_to_api_messages(storage.get_messages(conversation_id)))
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
        # itself (models did exactly that when merely told not to).
        tool_schemas = [
            s for s in tool_schemas if s["function"]["name"] == "deep_research"
        ]
    extra_body = reasoning_extra_body(
        provider, think, effort, model=model, session_id=conversation_id,
    )

    def persist(role: str, **fields: Any) -> str:
        return storage.add_message(conversation_id, role, **fields)

    async for event in agent_loop(
        client, provider, model, messages, registry,
        tool_schemas=tool_schemas, extra_body=extra_body,
        max_iterations=max_iterations, persist=persist,
    ):
        et = event["type"]
        if et == "usage":
            # Billed on every exit — a dead-ended turn still spent tokens.
            storage.add_token_usage(event["prompt_tokens"], event["completion_tokens"])
            # The cache was last refreshed by this turn's final model call.
            storage.set_system_prompt(conversation_id)
            if event["last_prompt_tokens"]:
                storage.set_prompt_tokens(conversation_id, event["last_prompt_tokens"])
            continue
        if et == "done":
            # message_id lets the web UI tag the finished bubble with its
            # stored id (read-aloud needs it).
            event = {k: v for k, v in event.items() if k != "text"}
        yield event


# --------------------------------------------------------------------------- #
# The shared agent loop
# --------------------------------------------------------------------------- #

# Tools that only read. They are safe to run concurrently, and calling one again
# with identical arguments within a turn can only return what is already in the
# context, so a repeat is answered from the earlier result instead. Anything not
# listed (shell, browser, wiki writes, jobs, MCP) runs one call at a time, in
# the order the model asked.
READ_ONLY_TOOLS = frozenset({
    "web_search", "image_search", "web_scrape", "web_map", "web_crawl",
    "search_wiki", "read_wiki_page", "current_time", "delegate",
    "fetch_user_messages", "fetch_channel_history",
})

_DUPLICATE_CALL = (
    "You already made this exact tool call — its result is above. Use that "
    "result, or change the arguments."
)


def clear_old_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages as sent to the model, with stale tool results replaced by a
    stub so a long tool-heavy conversation doesn't drag every page it ever read
    into every later call.

    Walking the results oldest-first, whenever the still-uncleared ones exceed
    CONTEXT_CLEAR_TRIGGER characters, all but the newest CONTEXT_KEEP_TOOL_RESULTS
    are cleared. The boundary depends only on the results seen so far, so
    appending messages never moves it backwards: it advances in occasional
    jumps, and the prompt cache survives everywhere except right after a jump.
    The stored conversation is untouched; this is a view."""
    trigger = settings.CONTEXT_CLEAR_TRIGGER
    if trigger <= 0:
        return messages
    keep = settings.CONTEXT_KEEP_TOOL_RESULTS
    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    sizes = [len(messages[i].get("content") or "") for i in idx]
    boundary = 0  # tool results before this position are cleared
    live = 0
    for n, size in enumerate(sizes):
        live += size
        if live > trigger:
            new_boundary = max(boundary, n + 1 - keep)
            live -= sum(sizes[boundary:new_boundary])
            boundary = new_boundary
    if boundary == 0:
        return messages
    out = list(messages)
    for i in idx[:boundary]:
        m = out[i]
        out[i] = {
            **m,
            "content": (
                f"[{m.get('name') or 'tool'} result cleared to save context "
                f"({len(m.get('content') or '')} chars). Call the tool again if "
                "you need it.]"
            ),
        }
    return out


async def _run_tool_calls(
    registry: Registry, calls: list[dict[str, Any]], seen: set[tuple[str, str]],
) -> AsyncGenerator[tuple[int, str], None]:
    """Yield (index into `calls`, result) as each call finishes. Consecutive
    read-only calls run concurrently, so their results arrive in completion
    order; any other call runs alone, after everything before it."""
    async def one(i: int, tc: dict[str, Any]) -> tuple[int, str]:
        return i, await _run_one(tc)

    async def _run_one(tc: dict[str, Any]) -> str:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"] or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return f"Invalid tool arguments (not valid JSON): {exc}"
        if name in READ_ONLY_TOOLS:
            key = (name, json.dumps(args, sort_keys=True))
            if key in seen:
                log.info("Tool %s skipped (duplicate call)", name)
                return _DUPLICATE_CALL
            seen.add(key)
        return await registry.call(name, args)

    batch: list[tuple[int, dict[str, Any]]] = []
    for i, tc in enumerate([*calls, None]):  # sentinel flushes the last batch
        if tc is not None and tc["function"]["name"] in READ_ONLY_TOOLS:
            batch.append((i, tc))
            continue
        if batch:
            for done in asyncio.as_completed([one(j, b) for j, b in batch]):
                yield await done
            batch = []
        if tc is not None:
            yield await one(i, tc)


async def agent_loop(
    client: Any,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    registry: Registry,
    *,
    tool_schemas: list[dict[str, Any]] | None = None,
    extra_body: dict[str, Any] | None = None,
    max_iterations: int | None = None,
    persist: Callable[..., str | None] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """The model ⇄ tools loop every surface runs: stream a completion, run the
    tool calls it asks for, append the results, repeat until it answers
    without tools or the iteration cap is hit.

    `messages` is extended in place. `persist(role, **fields)` (optional)
    stores each assistant and tool message and returns the stored id. Yields
    the UI events (reasoning, token, tool_call, tool_result), then one `usage`
    event ({prompt_tokens, completion_tokens, last_prompt_tokens}) and finally
    `done` ({tokens, prompt_tokens, message_id, text}) or `error`."""
    if tool_schemas is None:
        tool_schemas = registry.schemas()
    final_answer = ""
    final_answer_id: str | None = None  # stored id of the message holding it
    total_tokens = 0  # completion tokens across all model calls this turn (for tok/s)
    last_prompt_tokens = 0  # prompt tokens of the final model call (context usage)
    turn_prompt_tokens = 0  # prompt tokens summed over every call (billed total)
    seen_calls: set[tuple[str, str]] = set()

    def usage() -> dict[str, Any]:
        return {
            "type": "usage", "prompt_tokens": turn_prompt_tokens,
            "completion_tokens": total_tokens, "last_prompt_tokens": last_prompt_tokens,
        }

    iteration_cap = max_iterations or settings.MAX_AGENT_ITERATIONS
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
        sent = clear_old_tool_results(messages)

        async def _create():
            return await client.chat.completions.create(
                model=model,
                messages=sent,
                tools=None if final_pass else (tool_schemas or None),
                stream=True,
                # Ask the server to emit a final usage chunk so the UI can show tok/s.
                stream_options={"include_usage": True},
                extra_body=extra_body,
            )

        try:
            stream = await with_retries(_create)
        except Exception as exc:
            yield usage()
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return

        content_buf = ""
        reasoning_buf = ""
        reasoning_details: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        finish_reason = None

        async for chunk in stream:
            # The usage chunk arrives last and carries no choices — read it first.
            u = getattr(chunk, "usage", None)
            if u:
                if getattr(u, "completion_tokens", None):
                    total_tokens += u.completion_tokens
                pt = getattr(u, "prompt_tokens", None)
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
            # - reasoning_content: DeepSeek-style servers.
            # - reasoning: flat string per chunk (some gateways).
            # - reasoning_details: structured array fallback.
            reasoning_piece = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            details = getattr(delta, "reasoning_details", None) or []
            _merge_reasoning_details(reasoning_details, details)
            if not reasoning_piece:
                for rd in details:
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
        # stored for the UI; its structured blocks are also kept on the
        # in-memory assistant message so the next iteration replays them.
        tag = f"{provider}/{model}"
        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content_buf or None,
                "tool_calls": tool_calls,
            }
            if reasoning_details:
                assistant_msg["reasoning_details"] = reasoning_details
            messages.append(assistant_msg)
            if persist:
                persist(
                    "assistant", content=content_buf or None, tool_calls=tool_calls,
                    reasoning=reasoning_buf, model=tag,
                )
        elif content_buf or reasoning_buf:
            messages.append({"role": "assistant", "content": content_buf})
            if persist:
                final_answer_id = persist(
                    "assistant", content=content_buf, reasoning=reasoning_buf, model=tag,
                )
        if content_buf:
            final_answer = content_buf

        if finish_reason == "tool_calls" or tool_calls:
            for tc in tool_calls:
                yield {
                    "type": "tool_call", "id": tc["id"], "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"] or "{}",
                }
            # Each result reaches the UI the moment its tool finishes; the
            # conversation records them in the order the model asked.
            results: dict[int, str] = {}
            async for i, result in _run_tool_calls(registry, tool_calls, seen_calls):
                results[i] = result
                tc = tool_calls[i]
                yield {
                    "type": "tool_result", "id": tc["id"],
                    "name": tc["function"]["name"], "result": result,
                }
            for i, tc in enumerate(tool_calls):
                name, result = tc["function"]["name"], results[i]
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "name": name, "content": result}
                )
                if persist:
                    persist("tool", content=result, tool_call_id=tc["id"], name=name)
            continue  # loop back for the model's next step

        yield usage()
        yield {
            "type": "done", "tokens": total_tokens or None,
            "prompt_tokens": last_prompt_tokens or None,
            "message_id": final_answer_id, "text": final_answer,
        }
        return

    yield usage()
    yield {"type": "error", "message": f"Stopped after {iteration_cap} tool iterations."}

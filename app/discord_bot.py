"""Discord adapter: runs the harness agent as a Discord bot in the same process
as the web UI.

A triggered message (mention in a channel, any DM, or a reply to the bot) builds
its conversation context. For channel mentions/replies that context is live
Discord history (Discord is the source of truth — those turns are not stored).
For DMs, the stored conversation IS the source of truth and is shared with the
web UI: a DM appends to the same conversation the web UI reads, so either side
can resume the other. DM sessions are managed with text commands:
/new /list /resume <ref> /model [provider] [model]. Either way, the handler
assembles a per-message tool registry (reusing the web/browser/skills/MCP tools
plus per-user-scoped memory and Discord-history tools), runs a non-streaming
tool-calling loop, and posts a chunked reply.

The Discord client shares uvicorn's asyncio loop, so the web UI and the bot run
in one process. See `app/main.py` lifespan for startup/shutdown.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
import uuid
from typing import Optional

import discord
from discord import app_commands

from . import memory, storage
from .agent import (
    build_registry,
    conversation_time_block,
    needs_reasoning_replay,
    reasoning_extra_body,
    resolve_for_turn,
    run_turn,
    static_system_prompt,
)
from .config import (
    DISCORD_ENABLE_SHELL,
    MAX_AGENT_ITERATIONS,
    MAX_DISCORD_LENGTH,
    UPLOADS_DIR,
    detect_providers,
    effort_for,
    get_provider,
    resolve_default_model,
)
from .discord_context import (
    YOUTUBE_RE,
    build_user_content,
    download_images,
    fetch_context,
    format_line,
    is_status_message,
    message_text,
)
from .providers import client_for
from .skills import discover_skills, load_skill_tool
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
from .tools.clock import clock_tools
from .tools.registry import Registry, Tool
from .tools.web import web_tools

log = logging.getLogger("siegclaw.discord")


DISCORD_PREAMBLE = """You are SiegClaw, operating in a Discord server with multiple users.

## Reading the conversation
- Each message gives you a timestamped transcript of recent channel messages (oldest first), then the current question.
- Timestamps are `[MM-DD HH:MM]`. "Latest", "just now", or "the rant" means the most recent matching messages — read from the bottom up.
- Always refer to people by name (e.g. "siegfried said...", "ED asked...") — never "you" or "we", since there are multiple participants. In a DM there's only one person, so addressing them as "you" is fine there.

## Questions about the conversation itself
When someone asks about things said in the channel ("what do you think of X's rant", "his response to Y", "the argument about Z"):
- Find the actual messages first. Check the transcript; if it's not there, call `fetch_channel_history` or `fetch_user_messages` before answering.
- Ground your answer in what was actually said — engage with the specific points people made, quote short fragments when useful.
- Never substitute a generic researched overview of the topic for the actual discussion. Web search is for adding outside facts to your take, not replacing it.
- If you can't find what they're referring to even after fetching history, say so plainly and ask — don't guess at a different conversation.

## Questions about the world
- For current events, prices, news, or anything you're not certain about: call `web_search`, then `web_scrape` or `browser_use` instead of saying you don't know.
- Use `search_memory` for facts, preferences, and decisions from past conversations.
- Only say you lack information if a search also fails to find it.

## Replying
- Be concise — no fluff, no padding, no filler intros like "Good question".
- When asked for your opinion, commit to an actual take. A "both sides have a point" essay is a non-answer.
- Occasionally add a brief witty remark if it fits naturally — keep it short, never force it.
- Use Discord markdown formatting when helpful.
"""


# Tools whose result never changes for identical arguments within one reply —
# repeating them just burns iterations.
_IDEMPOTENT_TOOLS = {
    "web_search", "web_scrape", "web_map", "web_crawl",
    "search_memory", "fetch_user_messages", "fetch_channel_history",
}

# Short human-readable status lines shown live in Discord while a tool runs.
_TOOL_STATUS = {
    "web_search": lambda a: f"🔍 searching: *{a.get('query', '')}*",
    "web_scrape": lambda a: f"🌐 reading: `{a.get('url', '')}`",
    "web_map": lambda a: f"🗺️ mapping: `{a.get('url', '')}`",
    "web_crawl": lambda a: f"🕷️ crawling: `{a.get('url', '')}`",
    "search_memory": lambda a: f"🧠 searching memories: *{a.get('query', '')}*",
    "remember": lambda a: "🧠 saving to memory",
    "fetch_user_messages": lambda a: f"📜 fetching messages from *{a.get('user_name', '')}*",
    "fetch_channel_history": lambda a: "📜 reading older channel history",
    "browser_use": lambda a: f"🌐 browser: {a.get('action', '')} {a.get('url', '') or a.get('query', '')}".strip(),
    "load_skill": lambda a: f"📚 loading skill: {a.get('name', '')}",
}


# --------------------------------------------------------------------------- #
# Discord-history tools (bound to the current channel)
# --------------------------------------------------------------------------- #
_TOOL_RESULT_MAX_CHARS = 8000
_SURROUNDING_MESSAGES = 3


def _truncate_front(text: str, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> str:
    """Truncate from the front, keeping the most recent (bottom) messages."""
    if len(text) <= max_chars:
        return text
    return "[... earlier messages truncated ...]\n" + text[-max_chars:]


def discord_history_tools(channel: discord.abc.Messageable, bot_user_id: int) -> list[Tool]:
    async def fetch_user_messages(user_name: str, limit: int = 50) -> str:
        limit = min(max(limit, 1), 100)
        history = []
        async for msg in channel.history(limit=300):
            if not is_status_message(msg, bot_user_id):
                history.append(msg)
        history.reverse()  # chronological

        match_idx = {
            i for i, m in enumerate(history)
            if user_name.lower() in m.author.display_name.lower() and m.content
        }
        if not match_idx:
            return f"No messages found from '{user_name}' in the last {len(history)} channel messages."

        keep_matches = sorted(match_idx)[-limit:]
        keep: set[int] = set()
        for i in keep_matches:
            keep.update(range(max(0, i - _SURROUNDING_MESSAGES), min(len(history), i + _SURROUNDING_MESSAGES + 1)))

        lines = []
        prev = None
        for i in sorted(keep):
            if prev is not None and i > prev + 1:
                lines.append("[...]")
            marker = ">> " if i in match_idx else "   "
            lines.append(marker + format_line(history[i]))
            prev = i

        header = f"Messages from {user_name} (marked >>) with surrounding conversation, oldest first:\n"
        return _truncate_front(header + "\n".join(lines))

    async def fetch_channel_history(limit: int = 150) -> str:
        limit = min(max(limit, 10), 300)
        history = []
        async for msg in channel.history(limit=limit):
            if not is_status_message(msg, bot_user_id) and (msg.content or msg.attachments):
                history.append(msg)
        history.reverse()  # chronological
        if not history:
            return "No messages found in channel history."
        lines = [format_line(m) for m in history]
        header = f"Last {len(history)} channel messages, oldest first:\n"
        return _truncate_front(header + "\n".join(lines))

    return [
        Tool(
            "fetch_user_messages",
            "Fetch a specific user's recent messages in this channel, including the surrounding "
            "conversation so you can see what they were responding to. Use when asked to summarize, "
            "review, or analyse what a particular person said and it isn't already in the provided history.",
            {
                "type": "object",
                "properties": {
                    "user_name": {"type": "string", "description": "Display name of the user (partial match is fine)"},
                    "limit": {"type": "integer", "description": "Max number of that user's messages to return (default 50, max 100)"},
                },
                "required": ["user_name"],
            },
            fetch_user_messages,
        ),
        Tool(
            "fetch_channel_history",
            "Fetch older channel messages beyond the recent history you were given. Use when the user "
            "references a conversation, argument, or topic you can't see in the provided context.",
            {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many messages to fetch (default 150, max 300)"},
                },
            },
            fetch_channel_history,
        ),
    ]


# --------------------------------------------------------------------------- #
# Registry + system prompt
# --------------------------------------------------------------------------- #
def build_discord_registry(
    channel: discord.abc.Messageable | None,
    bot_user_id: int | None,
    scope: str,
    mcp_tools: list | None,
) -> tuple[Registry, dict]:
    """Per-message registry: harness web/browser/skills/MCP tools + per-user-scoped
    memory + Discord-history tools. Shell/file tools are withheld unless enabled.

    When `channel` is None (e.g. a scheduled job with no channel context) the
    Discord-history tools are omitted."""
    registry = Registry()
    registry.extend(clock_tools())
    if DISCORD_ENABLE_SHELL:
        registry.extend(builtin_tools())
    registry.extend(web_tools())
    registry.extend(browser_tools())
    registry.extend(memory.scoped_tools(scope))
    skills = discover_skills()
    if skills:
        registry.add(load_skill_tool(skills))
    if mcp_tools:
        registry.extend(mcp_tools)
    if channel is not None and bot_user_id is not None:
        registry.extend(discord_history_tools(channel, bot_user_id))
    return registry, skills


def discord_system_prompt(skills: dict, query: str, scope: str | None) -> str:
    """Stable Discord system prefix for channel mentions: soul + multi-user
    preamble + day-level time + query-relevant memory + skills. Channel turns
    are single-shot (one user message, not stored), so time and memory are
    captured at request time and stay byte-identical across the turn's own
    tool-loop iterations — keeping the prefix cacheable. For the precise time
    the model calls `current_time`; for other memories, `search_memory`."""
    return static_system_prompt(
        skills, DISCORD_PREAMBLE,
        conversation_time_block(), memory.relevant_block(query, scope),
    )


# --------------------------------------------------------------------------- #
# Non-streaming tool loop
# --------------------------------------------------------------------------- #
async def run_discord_turn(
    provider: str,
    model: str,
    messages: list[dict],
    registry: Registry,
    *,
    think: bool = True,
    effort: str | None = None,
    status_fn=None,
    conversation_id: str | None = None,
) -> str:
    """Drive the model + tools to a final text answer for Discord (no streaming).

    When `conversation_id` is given (the DM path), every assistant message and
    tool result is also appended to that stored conversation so the shared
    history stays valid for the next turn (and shows up in the web UI)."""
    client = client_for(provider)
    extra_body = reasoning_extra_body(provider, think, effort)
    schemas = registry.schemas()
    seen_calls: set[tuple[str, str]] = set()
    replay_reasoning = needs_reasoning_replay(provider)

    async def persist(fn, *a, **kw):
        if conversation_id:
            try:
                await asyncio.to_thread(fn, conversation_id, *a, **kw)
            except Exception as e:
                log.warning("DM persist failed: %s", e)

    def _reasoning_of(m) -> str | None:
        # Non-streaming responses expose the chain-of-thought as reasoning_content.
        return getattr(m, "reasoning_content", None)

    for _ in range(MAX_AGENT_ITERATIONS):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schemas or None,
            extra_body=extra_body,
        )
        msg = resp.choices[0].message
        # Always read + persist the reasoning so DM turns (shared with the web
        # UI) show their thinking block there — Discord itself only surfaces tool
        # status lines, never reasoning. `replay_reasoning` only governs whether
        # it's ALSO reattached to the in-memory assistant message for providers
        # that 400 without the chain-of-thought replayed on tool-call turns.
        reasoning_text = _reasoning_of(msg)
        if not msg.tool_calls:
            await persist(storage.add_message, "assistant", content=msg.content or "", reasoning=reasoning_text)
            return msg.content or ""

        assistant_dump = msg.model_dump(exclude_none=True)
        if replay_reasoning and reasoning_text:
            assistant_dump["reasoning_content"] = reasoning_text
        messages.append(assistant_dump)
        await persist(
            storage.add_message, "assistant",
            content=assistant_dump.get("content"),
            tool_calls=assistant_dump.get("tool_calls"),
            reasoning=reasoning_text,
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                content = f"Invalid tool arguments (not valid JSON): {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                await persist(storage.add_message, "tool", content=content, tool_call_id=tc.id, name=name)
                continue

            call_key = (name, raw_args)
            if name in _IDEMPOTENT_TOOLS and call_key in seen_calls:
                log.info("Tool %s skipped (duplicate call)", name)
                content = "You already made this exact tool call — its result is above. Use that result, or change the arguments."
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                await persist(storage.add_message, "tool", content=content, tool_call_id=tc.id, name=name)
                continue
            seen_calls.add(call_key)

            if status_fn:
                try:
                    await status_fn(name, args)
                except Exception as e:
                    log.debug("Status update failed: %s", e)

            result = await registry.call(name, args)
            log.info("Tool %s(%s) → %d chars", name, list(args.keys()), len(result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            await persist(storage.add_message, "tool", content=result, tool_call_id=tc.id, name=name)

    # Max iterations hit — force a final answer with no tools.
    messages.append({"role": "user", "content": "You have reached the maximum number of tool calls. Based on everything gathered so far, give your best answer now."})
    resp = await client.chat.completions.create(model=model, messages=messages, extra_body=extra_body)
    final = resp.choices[0].message
    await persist(storage.add_message, "assistant", content=final.content or "", reasoning=_reasoning_of(final))
    return final.content or ""


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #
def _chunk_message(text: str) -> list[str]:
    """Split text into Discord's 2000-char limit, preferring newline boundaries."""
    chunks = []
    while len(text) > MAX_DISCORD_LENGTH:
        split_at = text.rfind("\n", 0, MAX_DISCORD_LENGTH)
        if split_at == -1:
            split_at = MAX_DISCORD_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)
    return chunks


async def _send_long_message(message: discord.Message, text: str) -> None:
    for i, chunk in enumerate(_chunk_message(text)):
        if i == 0:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)


async def send_chunked(destination: discord.abc.Messageable, text: str) -> None:
    """Send (chunked) to any messageable — a channel or DM — without needing a
    triggering message. Used by the scheduler to deliver job results."""
    for chunk in _chunk_message(text):
        await destination.send(chunk)


# --------------------------------------------------------------------------- #
# Client + event wiring
# --------------------------------------------------------------------------- #
def create_client(mcp_manager) -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    # --- Slash commands (the primary UI; replies are ephemeral) ---------------
    # Restricted to DMs (and group DMs) — these manage a user's DM conversation
    # state, which has nothing to do with @mentions in channels (those take a
    # separate, live-Discord-history path). guilds=False hides them from the
    # channel autocomplete so an @mention doesn't surface the slash menu.
    dm_only = app_commands.allowed_contexts(guilds=False, dms=True, private_channels=True)

    @tree.command(name="new", description="Start a new conversation")
    @dm_only
    async def new_cmd(interaction: discord.Interaction):
        await _respond(interaction, cmd_new(str(interaction.user.id)))

    @tree.command(name="list", description="List all conversations")
    @dm_only
    async def list_cmd(interaction: discord.Interaction):
        await _respond(interaction, cmd_list(str(interaction.user.id)))

    @tree.command(name="resume", description="Resume a conversation by its ref number")
    @dm_only
    @app_commands.describe(ref="Conversation number from /list")
    async def resume_cmd(interaction: discord.Interaction, ref: int):
        await _respond(interaction, cmd_resume(str(interaction.user.id), ref))

    @tree.command(name="model", description="Show or set the active conversation's model")
    @dm_only
    @app_commands.describe(
        provider="Pick a provider (autocomplete; omit to keep current)",
        model="Pick a model (autocomplete shows the chosen provider's models)",
    )
    @app_commands.autocomplete(provider=_provider_autocomplete, model=_model_autocomplete)
    async def model_cmd(
        interaction: discord.Interaction,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        await _respond(interaction, cmd_model(str(interaction.user.id), provider, model))

    @client.event
    async def on_ready():
        memory.init_memory()  # idempotent; ensures the backend is ready
        # Sync once per process (on_ready also fires on reconnect). Overwriting
        # stale global commands left by older bot versions.
        if not on_ready._synced:
            on_ready._synced = True
            try:
                synced = await tree.sync()
                log.info("Synced %d slash commands", len(synced))
            except Exception as e:
                log.warning("Slash command sync failed: %s", e)
        log.info("Logged in as %s (ID: %s)", client.user, client.user.id)
    on_ready._synced = False

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return

        ref_msg = None
        if message.reference is not None and message.reference.message_id:
            try:
                resolved = (
                    message.reference.resolved
                    or await message.channel.fetch_message(message.reference.message_id)
                )
                if isinstance(resolved, discord.Message):
                    ref_msg = resolved
            except Exception:
                pass

        triggered = (
            client.user in message.mentions
            or message.guild is None  # DMs don't need a mention
            or (ref_msg is not None and ref_msg.author == client.user)
        )
        if not triggered:
            return

        is_dm = message.guild is None
        user_id = str(message.author.id)

        # Resolve provider + model. DMs use the active conversation's stored
        # choice (or the default on first contact); channel mentions use default.
        dm_cid: str | None = None
        pm = resolve_default_model()
        if pm is None:
            log.warning("No provider/model available; ignoring message")
            return
        provider, model, effort = pm
        if is_dm:
            dm_cid = storage.dm_active_cid(user_id)
            if dm_cid is not None:
                convo = storage.get_conversation(dm_cid)
                if convo and convo["provider"] and convo["model"]:
                    provider, model = convo["provider"], convo["model"]
                    effort = effort_for(provider, model)

        async with message.channel.typing():
            user_text = message_text(message, strip_bot_id=client.user.id)

            # Tool status lines, posted live to Discord as one growing -# message.
            # Defined before the branch so both the DM (run_turn event stream) and
            # the channel (run_discord_turn status_fn) paths can drive it.
            _status_msg: discord.Message | None = None

            async def update_status(tool_name: str, args: dict):
                nonlocal _status_msg
                try:
                    fn = _TOOL_STATUS.get(tool_name, lambda a: f"⚙️ {tool_name}")
                    line = f"-# {fn(args)}"[:300]
                    if _status_msg is None:
                        _status_msg = await message.reply(line)
                    else:
                        content = f"{_status_msg.content}\n{line}"
                        if len(content) > 1900:  # stay under Discord's 2000-char limit
                            content = line
                        await _status_msg.edit(content=content)
                except Exception as e:
                    log.debug("Status update failed: %s", e)

            try:
                if is_dm:
                    # The stored conversation IS the DM — same one the web UI reads.
                    # Run the SAME streaming turn generator the web UI uses (run_turn),
                    # so prompt construction, storage, and reasoning capture are
                    # byte-identical to a web turn. We only consume the event stream
                    # differently: accumulate the reply text, surface tool calls as
                    # Discord status lines, and ignore reasoning events (Discord never
                    # shows thinking). run_turn still saves reasoning to the DB, so the
                    # web UI shows the thinking block when this DM session is opened
                    # there — exactly like a web-asked turn.
                    if dm_cid is None:
                        dm_cid = storage.create_conversation(provider, model, user_text[:60] or "New chat")
                        storage.dm_set_active_cid(user_id, dm_cid)
                    elif user_text:
                        convo = storage.get_conversation(dm_cid)
                        if convo and convo.get("title") == "New chat":
                            storage.rename_conversation(dm_cid, user_text[:60])

                    registry, skills = build_registry(mcp_manager.tools)
                    # Send-time fallback: if the conversation's provider isn't
                    # serving (e.g. llama.cpp was stopped), retry a few times
                    # then switch this DM to the fallback model for the session.
                    provider, model, effort = await resolve_for_turn(dm_cid, provider, model, effort)
                    image_paths = _save_discord_images(await download_images(message, ref_msg))

                    reply_text = ""
                    async for event in run_turn(
                        dm_cid, provider, model, user_text, registry, skills,
                        images=image_paths or None, think=True, effort=effort,
                    ):
                        et = event.get("type")
                        if et == "token":
                            reply_text += event.get("text", "")
                        elif et == "tool_call":
                            try:
                                await update_status(event.get("name", ""), json.loads(event.get("arguments") or "{}"))
                            except Exception:
                                pass
                        elif et == "error":
                            raise RuntimeError(event.get("message", "turn error"))
                    if not reply_text:
                        raise ValueError("Empty response")
                    turn_failed = False
                else:
                    # Channel mention: live Discord history is the source of truth
                    # (not persisted). Build the timestamped transcript + question,
                    # then drive the non-streaming tool loop to a single reply.
                    # Memory is scoped per SERVER, not per triggering user: channel
                    # transcripts carry many speakers, extraction attributes facts
                    # by display name ("John lives in NYC"), and a shared guild pool
                    # makes them retrievable no matter who pings the bot next.
                    scope = f"discord-guild:{message.guild.id}"
                    registry, skills = build_discord_registry(
                        message.channel, client.user.id, scope, mcp_manager.tools
                    )
                    system = await asyncio.to_thread(discord_system_prompt, skills, user_text, scope)

                    reply_note = ""
                    if ref_msg is not None:
                        ref_text = message_text(ref_msg)[:500]
                        ref_author = "your (SiegClaw's)" if ref_msg.author == client.user else f"{ref_msg.author.display_name}'s"
                        reply_note = f"[{message.author.display_name} is replying to {ref_author} message]: {ref_text}\n"
                    prompt, _ = await fetch_context(message.channel, client.user.id, message.id)
                    prompt = f"{prompt}\n\n{reply_note}[Current question from {message.author.display_name}]: {user_text}"
                    images = await download_images(message, ref_msg)
                    yt_video_ids = YOUTUBE_RE.findall(user_text)
                    user_content = build_user_content(prompt, images, yt_video_ids)
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ]

                    reply_text = await run_discord_turn(
                        provider, model, messages, registry,
                        think=True, effort=effort, status_fn=update_status,
                    )
                    if not reply_text:
                        raise ValueError("Empty response")
                    turn_failed = False
            except Exception as e:
                log.error("Discord turn failed: %s", e)
                reply_text = f"Sorry, I couldn't generate a response ({type(e).__name__}). Please try again."
                turn_failed = True

        await _send_long_message(message, reply_text)

        # If the turn failed after the user message was already stored, drop it
        # (and any partially-persisted tool messages). If that leaves the
        # conversation empty, remove it entirely + clear the pointer so we don't
        # leave a zero-message conversation behind.
        if is_dm and dm_cid and turn_failed:
            storage.rewind_last_user_turn(dm_cid)
            if storage.message_count(dm_cid) == 0:
                storage.delete_conversation(dm_cid)
                storage.dm_clear_active(user_id)

        # Memory extraction. DM turns run through run_turn, which already
        # schedules the debounced global extraction — so only the ephemeral
        # channel-mention path needs explicit (per-user-scoped) extraction here.
        if not is_dm:
            asyncio.create_task(_extract_memory(scope, prompt[-2000:], reply_text))

    return client


async def _extract_memory(scope: str, conversation: str, reply: str) -> None:
    try:
        await asyncio.to_thread(memory.record_turn, conversation, reply, scope)
    except Exception as e:
        log.warning("Memory extraction failed: %s", e)


def _save_discord_images(images: list[dict]) -> list[str]:
    """Save downloaded Discord image bytes to UPLOADS_DIR and return their
    '/uploads/<name>' URLs — the same shape the web UI's /api/upload produces.
    This lets a DM turn feed run_turn identically to a web turn: images get
    stored on the message row and re-expanded from disk on resume (so a DM with
    an image replays correctly when the conversation is opened in the web UI)."""
    paths = []
    for img in images:
        ext = mimetypes.guess_extension(img.get("mime_type", "image/png")) or ".png"
        name = f"{uuid.uuid4().hex}{ext}"
        (UPLOADS_DIR / name).write_bytes(img["data"])
        paths.append(f"/uploads/{name}")
    return paths


# --------------------------------------------------------------------------- #
# Conversation commands — pure logic returning a reply string.
# Shared by the slash-command handlers (interactions) and the DM text-command
# fallback, so both paths behave identically.
# --------------------------------------------------------------------------- #
def _rel_time(ts: float | None) -> str:
    if not ts:
        return ""
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


_LIST_CAP = 20


def _fmt_ctx(n: int | None) -> str | None:
    """Human-readable context window: 131072 -> 128K, 1048576 -> 1M."""
    if not n:
        return None
    if n >= 1000:
        k = n / 1000
        if k >= 1024:
            m = k / 1024
            return f"{round(m)}M" if m >= 10 else f"{m:.1f}".rstrip("0").rstrip(".") + "M"
        return f"{round(k)}K"
    return str(n)


def _default_model_line() -> str:
    """Describe the model a new conversation will start on: `provider/model`,
    its context window (when the provider reports one), and effort if set."""
    pm = resolve_default_model()
    if pm is None:
        return "No model available — check provider config."
    provider, model, effort = pm
    ctx = None
    for p in detect_providers():
        if p.id == provider:
            ctx = p.model_context.get(model)
            break
    parts = [f"`{provider}/{model}`"]
    ctx_str = _fmt_ctx(ctx)
    parts.append(f"{ctx_str} context" if ctx_str else "context window unknown")
    if effort:
        parts.append(f"effort: {effort}")
    return " · ".join(parts)


def cmd_new(user_id: str) -> str:
    # Don't create a conversation yet — just clear the active pointer. The
    # conversation is created (and titled) when the user actually sends a
    # message, matching the web UI's "new chat just resets, persists on send".
    storage.dm_clear_active(user_id)
    return (
        "Starting a fresh conversation — your next message begins it.\n"
        f"Model: {_default_model_line()}"
    )


def cmd_list(user_id: str) -> str:
    convos = storage.list_all_conversations()
    if not convos:
        return "No conversations yet. Send a message or use `/new`."
    active_cid = storage.dm_active_cid(user_id)
    shown = convos[:_LIST_CAP]
    lines = ["**Conversations:**"]
    for c in shown:
        mark = " \u2190 active" if c["id"] == active_cid else ""
        title = c["title"] or "Untitled"
        lines.append(
            f"**#{c['ref']}** \u00b7 {title} \u00b7 `{c['provider']}/{c['model']}` \u00b7 "
            f"{c['msg_count']} msgs \u00b7 {_rel_time(c['updated_at'])}{mark}"
        )
    if len(convos) > _LIST_CAP:
        lines.append(f"_(showing {_LIST_CAP} of {len(convos)}; resume older by `ref`)_")
    return "\n".join(lines)


_HISTORY_PREVIEW = 20


def _format_history_preview(cid: str) -> str:
    """A readable transcript of the last user/assistant text messages, for the
    /resume reply. Tool-only messages and raw tool results are skipped. Full
    message text is included (the reply is chunked across Discord messages)."""
    visible: list[tuple[str, str]] = []
    for m in storage.get_messages(cid):
        role, content = m.get("role"), m.get("content")
        if role == "user" and content:
            visible.append(("you", content))
        elif role == "assistant" and content:
            visible.append(("siegclaw", content))
    if not visible:
        return "_(no messages yet)_"
    total = len(visible)
    recent = visible[-_HISTORY_PREVIEW:]
    body = "\n\n".join(f"**{who}**: {text.strip()}" for who, text in recent)
    if total > len(recent):
        return f"_(showing last {len(recent)} of {total} messages)_\n\n{body}"
    return body


def cmd_resume(user_id: str, ref: int) -> str:
    convo = storage.conversation_by_ref(ref)
    if convo is None:
        return f"No conversation **#{ref}**. Use `/list` to see them."
    storage.dm_set_active_cid(user_id, convo["id"])
    return (
        f"Resumed **#{ref}** ({convo['title'] or 'Untitled'}).\n"
        f"Model: `{convo['provider']}/{convo['model']}`\n\n"
        + _format_history_preview(convo["id"])
    )


def cmd_model(user_id: str, provider: str | None = None, model: str | None = None) -> str:
    cid = storage.dm_active_cid(user_id)
    if cid is None:
        return "No active conversation — send a message or use `/new` first."
    convo = storage.get_conversation(cid)
    providers = detect_providers()
    provider_ids = {p.id for p in providers}

    if not model:
        lines = [
            f"Active **#{convo['ref']}**: `{convo['provider']}/{convo['model']}`",
            "**Available providers:**",
        ]
        for p in providers:
            n = len(p.models)
            lines.append(f"- `{p.id}` ({p.name}) \u2014 {n} model{'s' if n != 1 else ''}")
        lines.append(
            "\nSet with `/model model:<model>` (keeps current provider) "
            "or `/model provider:<p> model:<m>`."
        )
        return "\n".join(lines)

    provider_id = provider if (provider in provider_ids) else convo["provider"]
    if get_provider(provider_id) is None:
        return f"Unknown provider `{provider_id}`. Known: {', '.join(sorted(provider_ids))}."
    storage.update_conversation_model(cid, provider_id, model)
    return f"Conversation **#{convo['ref']}** model set to `{provider_id}/{model}`."


async def _provider_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Offer the detected providers (with model counts) as you type the provider
    field. Substring-match on both id and display name."""
    cur = (current or "").lower()
    out: list[app_commands.Choice[str]] = []
    for p in detect_providers():
        if cur and cur not in p.id.lower() and cur not in p.name.lower():
            continue
        n = len(p.models)
        label = f"{p.name} ({n} model{'s' if n != 1 else ''})"
        out.append(app_commands.Choice(name=label[:100], value=p.id))
    return out[:25]


async def _model_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Offer models for the provider already picked in this command (falling back
    to the active conversation's provider). Discord caps choices at 25, so the
    typed text filters the list — exactly like the web UI combobox."""
    cur = (current or "").lower()
    providers = {p.id: p for p in detect_providers()}
    pid = (getattr(interaction.namespace, "provider", None) or "").strip()
    if pid not in providers:
        cid = storage.dm_active_cid(str(interaction.user.id))
        convo = storage.get_conversation(cid) if cid else None
        pid = convo["provider"] if (convo and convo["provider"] in providers) else None
    chosen = providers.get(pid)
    models = chosen.models if chosen else [m for p in providers.values() for m in p.models]
    out: list[app_commands.Choice[str]] = []
    for m in models:
        if cur and cur not in m.lower():
            continue
        out.append(app_commands.Choice(name=m[:100], value=m))
        if len(out) >= 25:
            break
    return out


async def _respond(interaction: discord.Interaction, text: str, *, ephemeral: bool = True) -> None:
    """Reply to a slash-command interaction, chunking past Discord's 2000-char cap."""
    chunks = _chunk_message(text)
    await interaction.response.send_message(chunks[0], ephemeral=ephemeral)
    for c in chunks[1:]:
        await interaction.followup.send(c, ephemeral=ephemeral)

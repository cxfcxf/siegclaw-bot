"""Discord adapter: runs the harness agent as a Discord bot in the same process
as the web UI.

A triggered message (mention in a channel, any DM, or a reply to the bot) builds
its conversation context from live Discord history (Discord is the source of
truth — these turns are not stored in the SQLite conversation db), assembles a
per-message tool registry (reusing the harness web/browser/skills/MCP tools plus
per-user-scoped memory and Discord-history tools), runs a non-streaming
tool-calling loop, and posts a chunked reply.

The Discord client shares uvicorn's asyncio loop, so the web UI and the bot run
in one process. See `app/main.py` lifespan for startup/shutdown.
"""
from __future__ import annotations

import asyncio
import json
import logging

import discord

from . import memory
from .agent import current_datetime_block, reasoning_extra_body, read_soul
from .config import (
    DISCORD_ENABLE_SHELL,
    MAX_AGENT_ITERATIONS,
    MAX_DISCORD_LENGTH,
    default_provider_model,
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
from .skills import discover_skills, load_skill_tool, skills_index
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
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
    channel: discord.abc.Messageable, bot_user_id: int, scope: str, mcp_tools: list | None
) -> tuple[Registry, dict]:
    """Per-message registry: harness web/browser/skills/MCP tools + per-user-scoped
    memory + Discord-history tools. Shell/file tools are withheld unless enabled."""
    registry = Registry()
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
    registry.extend(discord_history_tools(channel, bot_user_id))
    return registry, skills


def discord_system_prompt(skills: dict, query: str, scope: str) -> str:
    parts = [read_soul(), DISCORD_PREAMBLE, current_datetime_block(), memory.relevant_block(query, scope)]
    index = skills_index(skills)
    if index:
        parts.append(index)
    return "\n\n".join(parts)


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
    status_fn=None,
) -> str:
    """Drive the model + tools to a final text answer for Discord (no streaming).
    Mirrors run_turn's contract but builds nothing in storage."""
    client = client_for(provider)
    extra_body = reasoning_extra_body(provider, think)
    schemas = registry.schemas()
    seen_calls: set[tuple[str, str]] = set()

    for _ in range(MAX_AGENT_ITERATIONS):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schemas or None,
            extra_body=extra_body,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""

        messages.append(msg.model_dump(exclude_none=True))

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Invalid tool arguments (not valid JSON): {e}"})
                continue

            call_key = (name, raw_args)
            if name in _IDEMPOTENT_TOOLS and call_key in seen_calls:
                log.info("Tool %s skipped (duplicate call)", name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "You already made this exact tool call — its result is above. Use that result, or change the arguments."})
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

    # Max iterations hit — force a final answer with no tools.
    messages.append({"role": "user", "content": "You have reached the maximum number of tool calls. Based on everything gathered so far, give your best answer now."})
    resp = await client.chat.completions.create(model=model, messages=messages, extra_body=extra_body)
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #
async def _send_long_message(message: discord.Message, text: str) -> None:
    chunks = []
    while len(text) > MAX_DISCORD_LENGTH:
        split_at = text.rfind("\n", 0, MAX_DISCORD_LENGTH)
        if split_at == -1:
            split_at = MAX_DISCORD_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)

    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)


# --------------------------------------------------------------------------- #
# Client + event wiring
# --------------------------------------------------------------------------- #
def create_client(mcp_manager) -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        memory.init_memory()  # idempotent; ensures the backend is ready
        log.info("Logged in as %s (ID: %s)", client.user, client.user.id)

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

        pm = default_provider_model()
        if pm is None:
            log.warning("No provider/model available; ignoring message")
            return
        provider, model = pm

        async with message.channel.typing():
            user_text = message_text(message, strip_bot_id=client.user.id)
            user_id = str(message.author.id)
            scope = f"discord:{user_id}"

            reply_note = ""
            if ref_msg is not None:
                ref_text = message_text(ref_msg)[:500]
                ref_author = "your (SiegClaw's)" if ref_msg.author == client.user else f"{ref_msg.author.display_name}'s"
                reply_note = f"[{message.author.display_name} is replying to {ref_author} message]: {ref_text}\n"

            prompt, _ = await fetch_context(message.channel, client.user.id, message.id)
            prompt = f"{prompt}\n\n{reply_note}[Current question from {message.author.display_name}]: {user_text}"

            images = await download_images(message, ref_msg)
            yt_video_ids = YOUTUBE_RE.findall(user_text)

            registry, skills = build_discord_registry(message.channel, client.user.id, scope, mcp_manager.tools)
            system = await asyncio.to_thread(discord_system_prompt, skills, user_text, scope)
            user_content = build_user_content(prompt, images, yt_video_ids)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]

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
                reply_text = await run_discord_turn(
                    provider, model, messages, registry, think=True, status_fn=update_status
                )
                if not reply_text:
                    raise ValueError("Empty response")
            except Exception as e:
                log.error("Discord turn failed: %s", e)
                reply_text = f"Sorry, I couldn't generate a response ({type(e).__name__}). Please try again."

        await _send_long_message(message, reply_text)

        # Only the tail of the transcript — re-extracting the full window every
        # mention is costly and floods memory with near-duplicate facts.
        asyncio.create_task(_extract_memory(scope, prompt[-2000:], reply_text))

    return client


async def _extract_memory(scope: str, conversation: str, reply: str) -> None:
    try:
        await asyncio.to_thread(memory.record_turn, conversation, reply, scope)
    except Exception as e:
        log.warning("Memory extraction failed: %s", e)

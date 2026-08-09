"""Discord adapter: runs the harness agent as a Discord bot in the same process
as the web UI.

A triggered message (mention in a channel, any DM, or a reply to the bot) builds
its conversation context. For channel mentions/replies that context is live
Discord history (Discord is the source of truth — those turns are not stored).
For DMs, the stored conversation IS the source of truth and is shared with the
web UI: a DM appends to the same conversation the web UI reads, so either side
can resume the other. DM sessions are managed with slash commands:
/new /resume /model (clickable pickers) and /rename (a modal form that renames
the chat and sets its group; /resume filters by group). Either way, the handler
assembles a per-message tool registry (reusing the web/browser/wiki/MCP tools
plus Discord-history tools), runs a non-streaming
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
from pathlib import Path

import discord
from discord import app_commands

from . import research, storage, stt, wiki
from . import docs as docs_mod
from .agent import (
    build_registry,
    conversation_time_block,
    fallback_after_failure,
    needs_reasoning_replay,
    reasoning_extra_body,
    resolve_for_turn,
    run_turn,
    static_system_prompt,
)
from .config import (
    DISCORD_ENABLE_SHELL,
    DISCORD_OWNER_ID,
    DISCORD_STREAM_DMS,
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
from .tools.browser import browser_tools
from .tools.builtin import builtin_tools
from .tools.clock import clock_tools
from .tools.jobs import job_tools
from .tools.registry import Registry, Tool
from .tools.web import web_tools

log = logging.getLogger("siegclaw.discord")


DISCORD_PREAMBLE = """You are SiegClaw, operating in a Discord server with multiple users.

## Reading the conversation
- Each message gives you a timestamped transcript of recent channel messages (oldest first), then the current question.
- Timestamps are `[MM-DD HH:MM]`. "Latest", "just now", or "the rant" means the most recent matching messages — read from the bottom up.
- Always refer to people by name (e.g. "Alice said...", "Bob asked...") — never "you" or "we", since there are multiple participants. In a DM there's only one person, so addressing them as "you" is fine there.

## Questions about the conversation itself
When someone asks about things said in the channel ("what do you think of X's rant", "his response to Y", "the argument about Z"):
- Find the actual messages first. Check the transcript; if it's not there, call `fetch_channel_history` or `fetch_user_messages` before answering.
- Ground your answer in what was actually said — engage with the specific points people made, quote short fragments when useful.
- Never substitute a generic researched overview of the topic for the actual discussion. Web search is for adding outside facts to your take, not replacing it.
- If you can't find what they're referring to even after fetching history, say so plainly and ask — don't guess at a different conversation.

## Questions about the world
- For current events, prices, news, or anything you're not certain about: call `web_search`, then `web_scrape` or `browser_use` instead of saying you don't know.
- Use `search_wiki` / `read_wiki_page` for facts, preferences, and decisions from past conversations.
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
    "web_search", "image_search", "web_scrape", "web_map", "web_crawl",
    "search_wiki", "read_wiki_page", "fetch_user_messages", "fetch_channel_history",
}

# Short human-readable status lines shown live in Discord while a tool runs.
_TOOL_STATUS = {
    "web_search": lambda a: f"🔍 searching: *{a.get('query', '')}*",
    "image_search": lambda a: f"🖼️ finding images: *{a.get('query', '')}*",
    "web_scrape": lambda a: f"🌐 reading: `{a.get('url', '')}`",
    "web_map": lambda a: f"🗺️ mapping: `{a.get('url', '')}`",
    "web_crawl": lambda a: f"🕷️ crawling: `{a.get('url', '')}`",
    "search_wiki": lambda a: f"🧠 searching wiki: *{a.get('query', '')}*",
    "read_wiki_page": lambda a: f"📖 reading wiki: {a.get('name', '')}",
    "write_wiki_page": lambda a: f"✍️ updating wiki: {a.get('name', '')}",
    "fetch_user_messages": lambda a: f"📜 fetching messages from *{a.get('user_name', '')}*",
    "fetch_channel_history": lambda a: "📜 reading older channel history",
    "browser_use": lambda a: f"🌐 browser: {a.get('action', '')} {a.get('url', '') or a.get('query', '')}".strip(),
    "deep_research": lambda a: f"🔬 launching deep research: *{a.get('question', '')[:120]}*",
    "schedule_job": lambda a: f"⏰ scheduling: {a.get('name', 'job')}",
    "list_scheduled_jobs": lambda a: "⏰ listing scheduled jobs",
    "cancel_scheduled_job": lambda a: "🗑️ cancelling scheduled job",
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
    mcp_tools: list | None,
    *,
    wiki_space: str = wiki.PUBLIC,
    wiki_writable: bool = True,
) -> Registry:
    """Per-message registry for NON-owner surfaces (channel mentions, cron):
    harness web/browser/MCP tools + one wiki space + Discord-history tools.
    Shell/file tools are withheld unless enabled.

    Channel mentions take the defaults: the PUBLIC wiki (`wiki-public/`), read
    AND write. The channel keeps its own shared notebook, and the owner's
    private wiki is unreachable — these tools are bound to the public directory
    at construction, so there is no page name a channel user could ask for that
    resolves into it. Owner surfaces (web UI, DMs via build_registry) get the
    private space instead.

    The scheduler passes `wiki_space=PRIVATE, wiki_writable=False`: cron
    prompts are owner-authored, and their system prompt comes from
    `static_system_prompt()`'s private default — the two must name the same
    space or the prompt would index pages the tools can't open.

    When `channel` is None (e.g. a scheduled job with no channel context) the
    Discord-history tools are omitted."""
    registry = Registry()
    registry.extend(clock_tools())
    if DISCORD_ENABLE_SHELL:
        registry.extend(builtin_tools())
    registry.extend(web_tools())
    registry.extend(browser_tools())
    registry.extend(wiki.wiki_tools(writable=wiki_writable, space=wiki_space))
    # Scheduling is an OWNER capability, so it rides with the private wiki and is
    # withheld from channels. A channel user with `schedule_job` could otherwise
    # walk around the wiki boundary entirely: jobs run on the owner's private
    # space and can be pointed at any channel ("run this prompt in one minute,
    # deliver here"). `list_scheduled_jobs` also prints every job's prompt, and
    # `cancel_scheduled_job` would let a stranger delete the owner's jobs.
    if wiki_space == wiki.PRIVATE:
        registry.extend(job_tools())
    if mcp_tools:
        registry.extend(mcp_tools)
    if channel is not None and bot_user_id is not None:
        registry.extend(discord_history_tools(channel, bot_user_id))
    return registry


def discord_system_prompt() -> str:
    """Stable Discord system prefix for channel mentions: PUBLIC wiki home +
    multi-user preamble + day-level time + public wiki index. Channel turns are
    single-shot, so the prefix stays byte-identical across the turn's own
    tool-loop iterations — keeping it cacheable. For the precise time the model
    calls `current_time`; for remembered facts, `search_wiki` /
    `read_wiki_page` — all against the public space.

    Nothing from the owner's private wiki appears here, not even page names."""
    return static_system_prompt(DISCORD_PREAMBLE, conversation_time_block(), space=wiki.PUBLIC)


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
    spent = {"prompt": 0, "completion": 0}  # summed over all calls; recorded at return

    def _track(resp) -> None:
        u = getattr(resp, "usage", None)
        if u:
            spent["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            spent["completion"] += getattr(u, "completion_tokens", 0) or 0

    async def persist(fn, *a, **kw):
        if conversation_id:
            try:
                await asyncio.to_thread(fn, conversation_id, *a, **kw)
            except Exception as e:
                log.warning("DM persist failed: %s", e)

    def _reasoning_of(m) -> str | None:
        # Non-streaming responses expose the chain-of-thought as reasoning_content.
        return getattr(m, "reasoning_content", None)

    fell_back = False  # one call-failure fallback per turn
    for _ in range(MAX_AGENT_ITERATIONS):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=schemas or None,
                extra_body=extra_body,
            )
        except Exception as e:
            # The completion call failed on a provider the preflight probe
            # called serving (e.g. llama.cpp up with no model loaded). Switch
            # to the fallback model once per turn and retry in place.
            fb = None if fell_back else fallback_after_failure(provider, model)
            if fb is None:
                raise
            fell_back = True
            log.warning(
                "Completion call on %s/%s failed (%s); retrying on %s/%s",
                provider, model, type(e).__name__, fb[0], fb[1],
            )
            provider, model, effort = fb
            client = client_for(provider)
            replay_reasoning = needs_reasoning_replay(provider)
            extra_body = reasoning_extra_body(provider, think, effort)
            if conversation_id:
                storage.update_conversation_model(conversation_id, provider, model)
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=schemas or None,
                extra_body=extra_body,
            )
        _track(resp)
        msg = resp.choices[0].message
        # Always read + persist the reasoning so DM turns (shared with the web
        # UI) show their thinking block there — Discord itself only surfaces tool
        # status lines, never reasoning. `replay_reasoning` only governs whether
        # it's ALSO reattached to the in-memory assistant message for providers
        # that 400 without the chain-of-thought replayed on tool-call turns.
        reasoning_text = _reasoning_of(msg)
        if not msg.tool_calls:
            await persist(storage.add_message, "assistant", content=msg.content or "", reasoning=reasoning_text)
            storage.add_token_usage(spent["prompt"], spent["completion"])
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
    _track(resp)
    final = resp.choices[0].message
    await persist(storage.add_message, "assistant", content=final.content or "", reasoning=_reasoning_of(final))
    storage.add_token_usage(spent["prompt"], spent["completion"])
    return final.content or ""


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #
# Minimum seconds between edits of a streaming DM reply (Discord allows roughly
# 5 edits per 5s per channel; 1s keeps headroom for the status message too).
STREAM_EDIT_SECONDS = 1.0


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
    # In a channel, reply-quote the triggering message so it's clear who is
    # being answered; in a DM it's 1-on-1, so the quote is just noise.
    quote_first = message.guild is not None
    for i, chunk in enumerate(_chunk_message(text)):
        if i == 0 and quote_first:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)


async def send_chunked(destination: discord.abc.Messageable, text: str) -> None:
    """Send (chunked) to any messageable — a channel or DM — without needing a
    triggering message. Used by the scheduler to deliver job results."""
    for chunk in _chunk_message(text):
        await destination.send(chunk)


_owner_id: int | None = None  # cached across messages; resolved once


async def resolve_owner_id(client: discord.Client) -> int | None:
    """The Discord user id treated as the owner, cached. DISCORD_OWNER_ID wins;
    otherwise ask Discord for the application owner. Returns None only if both
    fail — callers must treat that as "not the owner" and fail closed."""
    global _owner_id
    if _owner_id is not None:
        return _owner_id
    if DISCORD_OWNER_ID:
        try:
            _owner_id = int(DISCORD_OWNER_ID)
            return _owner_id
        except ValueError:
            log.error("DISCORD_OWNER_ID=%r is not a user id; ignoring it", DISCORD_OWNER_ID)
    try:
        info = await client.application_info()
        oid = getattr(info.owner, "id", None)
    except Exception as e:
        log.error("Could not resolve the Discord application owner (%s); "
                  "DMs stay closed until this succeeds", e)
        return None
    if oid is None:
        return None
    _owner_id = int(oid)
    return _owner_id


async def owner_or_user(client: discord.Client, target_id: str = "owner"):
    """Resolve a DM recipient: the bot's application owner for the 'owner'
    sentinel (or an empty id), otherwise a specific user id. Shared by the
    scheduler and background research delivery."""
    if not target_id or target_id == "owner":
        info = await client.application_info()
        oid = getattr(info.owner, "id", None)
        if oid is None:
            return None
        return client.get_user(oid) or await client.fetch_user(oid)
    tid = int(target_id)
    return client.get_user(tid) or await client.fetch_user(tid)


# --------------------------------------------------------------------------- #
# Interactive pickers (DM slash-command UI)
# --------------------------------------------------------------------------- #
class _PagedSelect(discord.ui.View):
    """An ephemeral dropdown picker. Discord caps a select menu at 25 options,
    so longer lists get ◀ ▶ page buttons (e.g. OpenRouter's model list)."""

    PAGE = 25

    def __init__(self, options: list[discord.SelectOption], placeholder: str, on_pick):
        super().__init__(timeout=180)
        self._options = options
        self._placeholder = placeholder
        self._on_pick = on_pick  # async (interaction, value) -> None
        self._page = 0
        self._build()

    def _build(self) -> None:
        self.clear_items()
        pages = (len(self._options) + self.PAGE - 1) // self.PAGE
        start = self._page * self.PAGE
        ph = self._placeholder if pages <= 1 else f"{self._placeholder} ({self._page + 1}/{pages})"
        select = discord.ui.Select(placeholder=ph[:150], options=self._options[start : start + self.PAGE])

        async def picked(interaction: discord.Interaction):
            await self._on_pick(interaction, select.values[0])

        select.callback = picked
        self.add_item(select)
        if pages > 1:
            prev = discord.ui.Button(label="◀", disabled=self._page == 0)
            nxt = discord.ui.Button(label="▶", disabled=self._page >= pages - 1)

            async def flip(interaction: discord.Interaction, delta: int):
                self._page += delta
                self._build()
                await interaction.response.edit_message(view=self)

            prev.callback = lambda i: flip(i, -1)
            nxt.callback = lambda i: flip(i, +1)
            self.add_item(prev)
            self.add_item(nxt)


def _conversation_options(user_id: str, grp: str | None = None) -> list[discord.SelectOption]:
    """grp=None -> all conversations (group shown in the description);
    grp='name' -> only that group's conversations."""
    active_cid = storage.dm_active_cid(user_id)
    opts = []
    for c in storage.list_all_conversations():
        if grp is not None and (c.get("grp") or "") != grp:
            continue
        mark = "→ " if c["id"] == active_cid else ""
        tag = f"📁{c['grp']} · " if c.get("grp") and grp is None else ""
        opts.append(discord.SelectOption(
            label=f"{mark}{c['title'] or 'Untitled'}"[:100],
            value=str(c["ref"]),
            description=f"#{c['ref']} · {tag}{c['provider']}/{c['model']} · "
                        f"{c['msg_count']} msgs · {_rel_time(c['updated_at'])}"[:100],
        ))
    return opts


def _resume_view(user_id: str, grp: str | None = None) -> _PagedSelect | None:
    """A conversation dropdown that resumes on click. The picker itself is
    ephemeral, but the confirmation is posted for real — it marks a session
    boundary and should survive in DM history."""
    opts = _conversation_options(user_id, grp)
    if not opts:
        return None

    async def on_pick(inter: discord.Interaction, value: str):
        await inter.response.edit_message(content=f"Resuming **#{value}**…", view=None)
        for chunk in _chunk_message(cmd_resume(user_id, int(value))):
            await inter.channel.send(chunk)

    return _PagedSelect(opts, "Conversation…", on_pick)


class _EditChatModal(discord.ui.Modal):
    """The /rename popup: a native Discord form pre-filled with the active
    conversation's title and group. Submit renames (and re-groups) in place."""

    def __init__(self, cid: str, cur_title: str, cur_grp: str | None):
        super().__init__(title="Edit chat")
        self.cid = cid
        self.title_in = discord.ui.TextInput(
            label="Title", default=(cur_title or "")[:100], max_length=100)
        self.grp_in = discord.ui.TextInput(
            label="Group (empty = none)", default=(cur_grp or "")[:50],
            required=False, max_length=50)
        self.add_item(self.title_in)
        self.add_item(self.grp_in)

    async def on_submit(self, inter: discord.Interaction):
        title = self.title_in.value.strip()[:100] or "Untitled"
        grp = self.grp_in.value.strip()[:50] or None
        storage.rename_conversation(self.cid, title)
        storage.set_conversation_group(self.cid, grp)
        # Posted for real (not ephemeral): the new name marks the session in DM history.
        await inter.response.send_message(f"✏️ **{title}**" + (f" · 📁 {grp}" if grp else ""))


# DM users with research mode armed (the /research toggle — the DM counterpart
# of the web composer's Deep Research pill). While armed, DM turns carry
# research_mode=True so the model scopes the question first, then hands off to
# deep_research; the flag auto-disarms when that tool actually fires. In-memory
# on purpose: it's transient UI state, same as the web toggle.
_research_armed: set[str] = set()


# --------------------------------------------------------------------------- #
# Client + event wiring
# --------------------------------------------------------------------------- #
def create_client(mcp_manager) -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    # --- Slash commands (the primary UI) ---------------------------------------
    # Restricted to DMs (and group DMs) — these manage a user's DM conversation
    # state, which has nothing to do with @mentions in channels (those take a
    # separate, live-Discord-history path). guilds=False hides them from the
    # channel autocomplete so an @mention doesn't surface the slash menu.
    # Replies that mark a session boundary (/new, /resume) are posted for real so
    # they survive in the DM history; the rest stay ephemeral (Discord drops
    # ephemeral messages on client reload).
    dm_only = app_commands.allowed_contexts(guilds=False, dms=True, private_channels=True)

    async def _is_owner(interaction: discord.Interaction) -> bool:
        """These are owner tools, not public ones: /resume lists every stored
        conversation (web UI included), /research runs against the private wiki,
        /model spends the owner's tokens. `dm_only` hides them from channels but
        anyone who can DM the bot could still invoke them, so check the caller.
        Fails closed when the owner id can't be resolved."""
        oid = await resolve_owner_id(client)
        if oid is not None and interaction.user.id == oid:
            return True
        name = interaction.command.name if interaction.command else "?"
        log.info("Ignoring /%s from non-owner %s (%s)", name, interaction.user, interaction.user.id)
        await _respond(interaction, "These commands are limited to the bot's owner.")
        return False

    owner_only = app_commands.check(_is_owner)

    @tree.error
    async def on_app_command_error(interaction: discord.Interaction, error: Exception):
        # _is_owner already replied; swallow its CheckFailure instead of logging
        # a traceback for every non-owner poke.
        if isinstance(error, app_commands.CheckFailure):
            return
        log.warning("Slash command error: %s", error)
        try:
            await _respond(interaction, f"Command failed: {type(error).__name__}")
        except Exception:
            pass

    @tree.command(name="new", description="Start a new conversation")
    @dm_only
    @owner_only
    async def new_cmd(interaction: discord.Interaction):
        await _respond(interaction, cmd_new(str(interaction.user.id)), ephemeral=False)

    @tree.command(name="research", description="Toggle deep research mode for this DM")
    @dm_only
    @owner_only
    async def research_cmd(interaction: discord.Interaction):
        # Posted for real (not ephemeral): the mode boundary should survive in
        # the DM history, like /new does.
        user_id = str(interaction.user.id)
        if user_id in _research_armed:
            _research_armed.discard(user_id)
            await _respond(interaction, "💬 Deep research mode **off** — back to normal chat.", ephemeral=False)
        else:
            _research_armed.add(user_id)
            await _respond(
                interaction,
                "🔬 Deep research mode **on** — describe what you want researched. "
                "I'll ask a couple of scoping questions, then run the full research "
                "in the background and DM you the report. `/research` again to turn it off.",
                ephemeral=False,
            )

    @tree.command(name="resume", description="Resume a conversation")
    @dm_only
    @owner_only
    async def resume_cmd(interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        convos = storage.list_all_conversations()
        if not convos:
            await _respond(interaction, "No conversations yet. Send a message or use `/new`.")
            return
        # With groups in play, filter first: All chats / one group per row.
        counts: dict[str, int] = {}
        for c in convos:
            if c.get("grp"):
                counts[c["grp"]] = counts.get(c["grp"], 0) + 1
        if counts:
            gopts = [discord.SelectOption(
                label="All chats", value="all", emoji="💬",
                description=f"{len(convos)} conversations")]
            gopts += [discord.SelectOption(
                label=g[:100], value=f"g:{g}"[:100], emoji="📁",
                description=f"{n} conversation{'s' if n != 1 else ''}")
                for g, n in sorted(counts.items())]

            async def on_group(inter: discord.Interaction, value: str):
                grp = value[2:] if value.startswith("g:") else None
                view = _resume_view(user_id, grp)
                if view is None:
                    await inter.response.edit_message(content="That group is empty now.", view=None)
                    return
                await inter.response.edit_message(content="Pick a conversation to resume:", view=view)

            await interaction.response.send_message(
                "Pick a group:", view=_PagedSelect(gopts, "Group…", on_group), ephemeral=True,
            )
            return
        view = _resume_view(user_id)
        await interaction.response.send_message(
            "Pick a conversation to resume:", view=view, ephemeral=True,
        )

    @tree.command(name="rename", description="Rename the current chat and set its group")
    @dm_only
    @owner_only
    async def rename_cmd(interaction: discord.Interaction):
        convo = _active_or_pending_conversation(str(interaction.user.id))
        await interaction.response.send_modal(
            _EditChatModal(convo["id"], convo.get("title") or "", convo.get("grp"))
        )

    @tree.command(name="model", description="Switch the active conversation's model")
    @dm_only
    @owner_only
    async def model_cmd(interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        convo = _active_or_pending_conversation(user_id)
        providers = detect_providers()

        async def on_provider(inter: discord.Interaction, pid: str):
            p = next((x for x in detect_providers() if x.id == pid), None)
            if p is None or not p.models:
                await inter.response.edit_message(
                    content=f"`{pid}` doesn't list models — set one from the web UI picker.",
                    view=None,
                )
                return

            async def on_model(inter2: discord.Interaction, m: str):
                await inter2.response.edit_message(content=cmd_model(user_id, pid, m), view=None)

            mopts = [discord.SelectOption(label=m[:100], value=m[:100]) for m in p.models]
            await inter.response.edit_message(
                content=f"Provider `{pid}` — pick a model:",
                view=_PagedSelect(mopts, "Model…", on_model),
            )

        popts = [
            discord.SelectOption(
                label=p.name[:100], value=p.id,
                description=f"{len(p.models)} model{'s' if len(p.models) != 1 else ''}"[:100],
            )
            for p in providers
        ]
        if not popts:
            await _respond(interaction, "No providers available — check provider config.")
            return
        await interaction.response.send_message(
            f"Active **#{convo['ref']}**: `{convo['provider']}/{convo['model']}`\nPick a provider:",
            view=_PagedSelect(popts, "Provider…", on_provider), ephemeral=True,
        )

    @client.event
    async def on_ready():
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
        # Resolve the DM owner once here (rather than lazily on the first DM) so
        # the id is in the startup log — it's the thing to check if DMs are
        # unexpectedly ignored, and nobody remembers their own Discord user id.
        owner_id = await resolve_owner_id(client)
        if owner_id is None:
            log.error("No DM owner resolved — ALL DMs will be ignored. "
                      "Set DISCORD_OWNER_ID to your Discord user id.")
        else:
            src = "DISCORD_OWNER_ID" if DISCORD_OWNER_ID else "Discord app owner"
            log.info("DMs restricted to owner id %s (from %s); channel mentions stay open",
                     owner_id, src)
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

        # DMs are the owner surface — they get the private wiki, read-write, and
        # `home` there IS the system prompt. Anyone sharing a server with the bot
        # can open a DM with it, so gate on the owner and drop everything else
        # silently (no reply, no model call). Fails closed if the owner id can't
        # be resolved. Channel mentions are unaffected: they run on the public
        # wiki and stay open to everyone.
        if is_dm:
            owner_id = await resolve_owner_id(client)
            if owner_id is None or message.author.id != owner_id:
                log.info("Ignoring DM from non-owner %s (%s)", message.author, user_id)
                return

        # Resolve provider + model. Everything uses the freshly resolved default
        # (local engine if it's up, else the fallback) — a conversation's stored
        # model is provenance, not a preference, so resumed DMs don't drag you
        # back to whatever ran last time (matches the web UI). The one exception:
        # an explicit /model pick, which pins the DM conversation until the bot
        # restarts or /model is run again.
        dm_cid: str | None = None
        pm = resolve_default_model()
        if pm is None:
            log.warning("No provider/model available; ignoring message")
            return
        provider, model, effort = pm
        if is_dm:
            dm_cid = storage.dm_active_cid(user_id)
            if dm_cid is not None and dm_cid in _MODEL_OVERRIDES:
                provider, model = _MODEL_OVERRIDES[dm_cid]
                effort = effort_for(provider, model)

        async with message.channel.typing():
            user_text = message_text(message, strip_bot_id=client.user.id)

            # Tool status lines, posted live to Discord as one growing -# message.
            # Defined before the branch so both the DM (run_turn event stream) and
            # the channel (run_discord_turn status_fn) paths can drive it.
            _status_msg: discord.Message | None = None

            # DM streaming: the reply is sent early and edited as tokens arrive,
            # so a long answer reads like the web UI instead of one late dump.
            stream_msg: discord.Message | None = None
            stream_edit_at = 0.0

            # Voice turns (DM only): the user's clip URL, stored on the message
            # row so the web UI can replay the recording above its transcript.
            voice_url: str | None = None

            async def update_status(tool_name: str, args: dict):
                nonlocal _status_msg
                try:
                    fn = _TOOL_STATUS.get(tool_name, lambda a: f"⚙️ {tool_name}")
                    line = f"-# {fn(args)}"[:300]
                    if _status_msg is None:
                        # Quote-reply only in channels; DMs are 1-on-1.
                        if message.guild is not None:
                            _status_msg = await message.reply(line)
                        else:
                            _status_msg = await message.channel.send(line)
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
                    # Voice message (or any audio attachment): save the clip,
                    # transcribe it locally (same faster-whisper the web mic
                    # uses), and run the turn on the transcript. The clip URL
                    # rides along so the web UI can replay it. The reply is
                    # text-only — the web UI's read-aloud button does TTS.
                    audio_att = _audio_attachment(message)
                    if audio_att is not None:
                        voice_url = await _save_dm_audio(audio_att)
                        transcript = await asyncio.to_thread(
                            stt.transcribe, str(UPLOADS_DIR / Path(voice_url).name)
                        )
                        if not transcript:
                            await message.channel.send(
                                "🎤 I couldn't make out any speech in that clip — try again?"
                            )
                            return
                        user_text = f"{user_text}\n{transcript}".strip() if user_text else transcript

                    # Document attachments (PDF/text): saved + text-extracted,
                    # then injected into the prompt by run_turn. A DM can carry
                    # a doc, a question, and even a voice clip in one message.
                    dm_docs = await _save_dm_docs(message)
                    if dm_docs and not user_text:
                        user_text = f"(user attached {', '.join(d['name'] for d in dm_docs)})"

                    if dm_cid is None:
                        title = user_text[:60] or (f"📄 {dm_docs[0]['name']}"[:60] if dm_docs else "New chat")
                        dm_cid = storage.create_conversation(provider, model, title)
                        storage.dm_set_active_cid(user_id, dm_cid)
                    elif user_text:
                        convo = storage.get_conversation(dm_cid)
                        if convo and convo.get("title") == "New chat":
                            storage.rename_conversation(dm_cid, user_text[:60])

                    registry = build_registry(mcp_manager.tools)
                    # A deep_research launched from this turn should deliver its
                    # report back here (DM), not just note the web conversation.
                    research.set_surface("dm")
                    # Send-time fallback: if the conversation's provider isn't
                    # serving (e.g. llama.cpp was stopped), retry a few times
                    # then switch this DM to the fallback model for the session.
                    provider, model, effort = await resolve_for_turn(dm_cid, provider, model, effort)
                    image_paths = _save_discord_images(await download_images(message, ref_msg))

                    reply_text = ""
                    async for event in run_turn(
                        dm_cid, provider, model, user_text, registry,
                        images=image_paths or None, think=True, effort=effort,
                        audio=voice_url, docs=dm_docs or None,
                        research_mode=user_id in _research_armed,
                    ):
                        et = event.get("type")
                        if et == "token":
                            reply_text += event.get("text", "")
                            # Optionally stream into a live message: send once
                            # there's text, then edit in place (throttled to
                            # respect rate limits). Overflow past one Discord
                            # message is handled by the final chunked delivery.
                            now = time.monotonic()
                            if DISCORD_STREAM_DMS and reply_text.strip() and now - stream_edit_at >= STREAM_EDIT_SECONDS:
                                stream_edit_at = now
                                preview = reply_text[: MAX_DISCORD_LENGTH - 2] + " ▌"
                                try:
                                    if stream_msg is None:
                                        stream_msg = await message.channel.send(preview)
                                    else:
                                        await stream_msg.edit(content=preview)
                                except Exception as e:
                                    log.debug("Stream edit failed: %s", e)
                        elif et == "tool_call":
                            if event.get("name") == "deep_research":
                                # Handoff happened — disarm, same as the web
                                # pill switching itself off on launch.
                                _research_armed.discard(user_id)
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
                    # Same send-time fallback as the DM path: fresh-probe the
                    # chosen provider (retrying transient blips) and switch to
                    # the fallback model if it's down. No conversation to
                    # persist the switch on — channel turns are stateless.
                    provider, model, effort = await resolve_for_turn(None, provider, model, effort)
                    registry = build_discord_registry(
                        message.channel, client.user.id, mcp_manager.tools
                    )
                    system = await asyncio.to_thread(discord_system_prompt)

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

        # Deliver. If a DM was streaming into a live message, finish it in
        # place: the first chunk replaces the preview (dropping the ▌ cursor),
        # overflow continues as plain messages. On failure the partial preview
        # is deleted to match the DB rollback below.
        if stream_msg is not None and turn_failed:
            try:
                await stream_msg.delete()
                await message.channel.send(reply_text)
            except Exception as e:
                log.warning("Finalizing streamed DM failed (%s); resending", e)
                await _send_long_message(message, reply_text)
        else:
            await _deliver_dm_text(message, stream_msg, reply_text)

        # If the turn failed after the user message was already stored, drop it
        # (and any partially-persisted tool messages). If that leaves the
        # conversation empty, remove it entirely + clear the pointer so we don't
        # leave a zero-message conversation behind.
        if is_dm and dm_cid and turn_failed:
            storage.rewind_last_user_turn(dm_cid)
            if storage.message_count(dm_cid) == 0:
                storage.delete_conversation(dm_cid)
                storage.dm_clear_active(user_id)

    return client


async def _deliver_dm_text(
    message: discord.Message, stream_msg: discord.Message | None, reply_text: str
) -> None:
    """Send a finished reply: finalize the live streaming preview in place if
    there is one (falling back to plain sends if the edit fails), else send
    the text chunked."""
    if stream_msg is not None:
        try:
            chunks = _chunk_message(reply_text)
            await stream_msg.edit(content=chunks[0])
            for c in chunks[1:]:
                await message.channel.send(c)
            return
        except Exception as e:
            log.warning("Finalizing streamed DM failed (%s); resending", e)
    await _send_long_message(message, reply_text)


_AUDIO_EXT = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".webm", ".flac", ".aac"}


def _audio_attachment(message: discord.Message) -> discord.Attachment | None:
    """First audio attachment on the message: a Discord voice message (ogg) or
    any uploaded audio file."""
    for att in message.attachments:
        if (att.content_type or "").startswith("audio/"):
            return att
        if Path(att.filename).suffix.lower() in _AUDIO_EXT:
            return att
    return None


async def _save_dm_docs(message: discord.Message) -> list[dict]:
    """Save document attachments (PDF/text) from a DM and return run_turn-shaped
    docs [{"url", "name"}]. Extraction is warmed here (sidecar cache) so a
    broken file surfaces as a note instead of failing mid-turn; unreadable
    attachments are skipped with their name flagged in the returned list."""
    out: list[dict] = []
    for att in message.attachments:
        if not docs_mod.is_doc(att.filename):
            continue
        if (att.content_type or "").startswith("audio/"):
            continue  # voice clips take the transcription path
        ext = Path(att.filename).suffix.lower()
        name = f"doc-{uuid.uuid4().hex}{ext}"
        dest = UPLOADS_DIR / name
        dest.write_bytes(await att.read())
        try:
            text = await asyncio.to_thread(docs_mod.text_for, dest)
        except Exception as e:
            log.warning("DM doc %s unreadable: %s", att.filename, e)
            dest.unlink(missing_ok=True)
            continue
        if not text.strip():
            dest.unlink(missing_ok=True)
            dest.with_name(dest.name + ".txt").unlink(missing_ok=True)
            continue
        out.append({"url": f"/uploads/{name}", "name": att.filename})
    return out


async def _save_dm_audio(att: discord.Attachment) -> str:
    """Save a DM voice clip to UPLOADS_DIR and return its '/uploads/<name>' URL
    (same shape as images), so the web UI can replay it when the conversation
    is opened there."""
    ext = Path(att.filename).suffix.lower() or mimetypes.guess_extension(att.content_type or "") or ".ogg"
    name = f"voice-{uuid.uuid4().hex}{ext}"
    (UPLOADS_DIR / name).write_bytes(await att.read())
    return f"/uploads/{name}"


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
    # (Exception: /model before any message creates it early — see below. If
    # that pending conversation is abandoned with another /new, drop it.)
    cid = storage.dm_active_cid(user_id)
    if cid and storage.message_count(cid) == 0:
        storage.delete_conversation(cid)
    storage.dm_clear_active(user_id)
    return (
        "Starting a fresh conversation — your next message begins it.\n"
        f"Model: {_default_model_line()}{_disarm_note(user_id)}"
    )


def _disarm_note(user_id: str) -> str:
    """Session boundaries (/new, /resume) drop research mode — a fresh or
    resumed chat always starts in normal mode. Returns the confirmation line
    to append when the mode was actually on (/model deliberately does NOT
    disarm: switching engines mid-clarify shouldn't eat the mode)."""
    if user_id in _research_armed:
        _research_armed.discard(user_id)
        return "\n💬 Deep research mode switched **off** for this chat."
    return ""


def _active_or_pending_conversation(user_id: str) -> dict:
    """The active conversation, or — right after /new (or first contact) — the
    pending one, created early so /model has something to set the model on. It
    keeps the placeholder title and gets renamed by the first message, exactly
    as if it had been created on send."""
    cid = storage.dm_active_cid(user_id)
    convo = storage.get_conversation(cid) if cid else None
    if convo is not None:
        return convo
    pm = resolve_default_model()
    provider, model = (pm[0], pm[1]) if pm else ("", "")
    cid = storage.create_conversation(provider, model, "New chat")
    storage.dm_set_active_cid(user_id, cid)
    return storage.get_conversation(cid)


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
        return f"No conversation **#{ref}**. Use `/resume` to pick from the menu."
    storage.dm_set_active_cid(user_id, convo["id"])
    return (
        f"Resumed **#{ref}** ({convo['title'] or 'Untitled'}).\n"
        f"Model: `{convo['provider']}/{convo['model']}`"
        f"{_disarm_note(user_id)}\n\n"
        + _format_history_preview(convo["id"])
    )


# Explicit /model picks, keyed by conversation id. In-memory on purpose: an
# override lasts for the bot's uptime, then everything snaps back to the
# default model (DM turns otherwise always use the freshly resolved default).
_MODEL_OVERRIDES: dict[str, tuple[str, str]] = {}


def cmd_model(user_id: str, provider: str, model: str) -> str:
    """Set the active (or pending — right after /new) conversation's model.
    Only reached from the /model picker, so provider/model are known-good."""
    convo = _active_or_pending_conversation(user_id)
    if get_provider(provider) is None:
        return f"Unknown provider `{provider}` — it may have gone away; run `/model` again."
    _MODEL_OVERRIDES[convo["id"]] = (provider, model)
    storage.update_conversation_model(convo["id"], provider, model)
    return (
        f"Conversation **#{convo['ref']}** pinned to `{provider}/{model}`"
        " (until the bot restarts or you run /model again)."
    )


async def _respond(interaction: discord.Interaction, text: str, *, ephemeral: bool = True) -> None:
    """Reply to a slash-command interaction, chunking past Discord's 2000-char cap."""
    chunks = _chunk_message(text)
    await interaction.response.send_message(chunks[0], ephemeral=ephemeral)
    for c in chunks[1:]:
        await interaction.followup.send(c, ephemeral=ephemeral)

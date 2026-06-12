import asyncio
import base64
import json
import logging
import re
from datetime import datetime

_YOUTUBE_RE = re.compile(
    r'https?://(?:www\.)?(?:youtube\.com/watch\?(?:[^&\s]*&)*v=|youtu\.be/)([a-zA-Z0-9_-]{11})[^\s]*'
)

import discord
import httpx

from browser import browse_page, browser_click, browser_snapshot, browser_type, close_browser_session, screenshot_bytes
from config import MAX_DISCORD_LENGTH, MODEL, MODEL_TEMPERATURE, SYSTEM_INSTRUCTION, TOOL_MAX_ITERS, mimo_client
from context import PT, fetch_context, format_line, is_status_message, message_text
from memory import extract_and_store_memories, init_db, search_memories
from search import web_search

log = logging.getLogger("siegclaw.handler")

_TOOL_STATUS = {
    "web_search":           lambda a: f"🔍 searching: *{a.get('query', '')}*",
    "recall_memories":      lambda a: f"🧠 searching memories: *{a.get('query', '')}*",
    "fetch_user_messages":  lambda a: f"📜 fetching messages from *{a.get('user_name', '')}*",
    "fetch_channel_history": lambda a: "📜 reading older channel history",
    "browse_page":          lambda a: f"🌐 browsing: `{a.get('url', '')}`",
    "browser_click":        lambda a: f"🖱️ clicking element `{a.get('ref', '')}`",
    "browser_type":         lambda a: "⌨️ typing into page",
    "browser_snapshot":     lambda a: "📄 reading page",
    "browser_screenshot":   lambda a: "📸 taking screenshot",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_user_messages",
            "description": "Fetch a specific user's recent messages in this channel, including the surrounding conversation so you can see what they were responding to. Use when asked to summarize, review, or analyse what a particular person said and it isn't already in the provided history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {"type": "string", "description": "Display name of the user (partial match is fine)"},
                    "limit": {"type": "integer", "description": "Max number of that user's messages to return (default 50, max 100)"},
                },
                "required": ["user_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_channel_history",
            "description": "Fetch older channel messages beyond the recent history you were given. Use when the user references a conversation, argument, or topic you can't see in the provided context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many messages to fetch (default 150, max 300)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, recent news, events, or factual lookups",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": "Search past conversation memories for facts, decisions, preferences, or details about people",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_page",
            "description": "Open a URL in a real browser and return the page content as an accessibility tree. Use this for JS-heavy pages, login-walled content, or when web_search isn't enough.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element on the current browser page using its ref ID from the snapshot",
            "parameters": {
                "type": "object",
                "properties": {"ref": {"type": "string", "description": "The ref ID from the page snapshot"}},
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text into an input field on the current browser page",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "The ref ID of the input from the snapshot"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "description": "Press Enter after typing"},
                },
                "required": ["ref", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": "Get the current state of the browser page as an accessibility tree",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take a visual screenshot of the current browser page and see it as an image",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def setup_events(client: discord.Client):

    @client.event
    async def on_ready():
        init_db()
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

        async with message.channel.typing():
            user_text = message_text(message, strip_bot_id=client.user.id)
            channel_id = str(message.channel.id)
            user_id = str(message.author.id)

            reply_note = ""
            if ref_msg is not None:
                ref_text = message_text(ref_msg)[:500]
                ref_author = "your (SiegClaw's)" if ref_msg.author == client.user else f"{ref_msg.author.display_name}'s"
                reply_note = (
                    f"[{message.author.display_name} is replying to "
                    f"{ref_author} message]: {ref_text}\n"
                )

            prompt, _ = await fetch_context(message.channel, client.user.id, message.id)
            prompt = f"{prompt}\n\n{reply_note}[Current question from {message.author.display_name}]: {user_text}"

            images = await _download_images(message, ref_msg)
            yt_video_ids = _YOUTUBE_RE.findall(user_text)
            system = _build_system()
            if user_text:
                try:
                    memories = await asyncio.to_thread(
                        search_memories, channel_id, user_text, user_id, 5
                    )
                    if memories:
                        system += "\n\n## Possibly relevant memories from past conversations\n" + "\n".join(
                            f"- {m}" for m in memories
                        )
                except Exception as e:
                    log.warning("Memory preload failed: %s", e)
            user_content = _build_user_content(prompt, images, yt_video_ids)

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
                reply_text = await _run_with_tools(system, user_content, message.channel, channel_id, user_id, client.user.id, update_status)
                if not reply_text:
                    raise ValueError("Empty response")
            except Exception as e:
                log.error("API call failed: %s", e)
                reply_text = f"Sorry, I couldn't generate a response ({type(e).__name__}). Please try again."

        await _send_long_message(message, reply_text)
        # Only the tail of the transcript — re-extracting the full window every
        # mention is costly and floods memory with near-duplicate facts.
        asyncio.create_task(
            extract_and_store_memories(channel_id, user_id, prompt[-2000:], reply_text)
        )
        asyncio.create_task(asyncio.to_thread(close_browser_session, user_id))


# Tools whose result never changes for identical arguments within one reply —
# repeating them just burns iterations.
_IDEMPOTENT_TOOLS = {"web_search", "recall_memories", "fetch_user_messages", "fetch_channel_history", "browse_page"}


async def _chat(messages: list, tools: list | None = None):
    kwargs = {"model": MODEL, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    if MODEL_TEMPERATURE is not None:
        kwargs["temperature"] = MODEL_TEMPERATURE
    return await mimo_client.chat.completions.create(**kwargs)


async def _run_with_tools(
    system: str, user_content: list, channel: discord.TextChannel, channel_id: str, user_id: str, bot_user_id: int, status_fn=None, max_iters: int = TOOL_MAX_ITERS
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    seen_calls = set()

    for _ in range(max_iters):
        response = await _chat(messages, TOOLS)
        choice = response.choices[0]

        if not choice.message.tool_calls:
            return choice.message.content or ""

        messages.append(choice.message)
        # Some OpenAI-compatible APIs reject images inside tool-role messages,
        # so screenshots are delivered in a follow-up user message instead.
        pending_images = []

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Invalid tool arguments (not valid JSON): {e}"})
                continue

            call_key = (tc.function.name, tc.function.arguments)
            if tc.function.name in _IDEMPOTENT_TOOLS and call_key in seen_calls:
                log.info("Tool %s skipped (duplicate call)", tc.function.name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "You already made this exact tool call — its result is above. Use that result, or change the arguments."})
                continue
            seen_calls.add(call_key)

            if status_fn:
                await status_fn(tc.function.name, args)

            if tc.function.name == "fetch_user_messages":
                result = await _fetch_user_messages(channel, args.get("user_name", ""), args.get("limit", 50), bot_user_id)
                log.info("Tool fetch_user_messages(%s) → %d chars", args.get("user_name"), len(result))
                tool_content = result
            elif tc.function.name == "fetch_channel_history":
                result = await _fetch_channel_history(channel, args.get("limit", 150), bot_user_id)
                log.info("Tool fetch_channel_history(%s) → %d chars", args.get("limit", 150), len(result))
                tool_content = result
            elif tc.function.name == "browser_screenshot":
                img_bytes = await asyncio.to_thread(screenshot_bytes, user_id)
                if img_bytes:
                    log.info("Tool browser_screenshot → %d bytes", len(img_bytes))
                    b64 = base64.b64encode(img_bytes).decode()
                    pending_images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    tool_content = "Screenshot taken — the image follows in the next user message."
                else:
                    tool_content = "Screenshot failed — no active browser session."
            else:
                result = await asyncio.to_thread(_dispatch_tool, tc.function.name, args, channel_id, user_id)
                log.info("Tool %s(%s) → %d chars", tc.function.name, list(args.keys()), len(result))
                tool_content = result

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})

        if pending_images:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "[Browser screenshot(s) from the tool call(s) above]"}] + pending_images,
            })

    # Max iterations hit — force a final answer with no tools
    messages.append({"role": "user", "content": "You have reached the maximum number of tool calls. Based on everything gathered so far, give your best answer now."})
    response = await _chat(messages)
    return response.choices[0].message.content or ""


def _dispatch_tool(name: str, args: dict, channel_id: str, user_id: str) -> str:
    if name == "web_search":
        result = web_search(args.get("query", ""))
        return result or "No results found."
    if name == "recall_memories":
        try:
            facts = search_memories(channel_id, args.get("query", ""), user_id)
            return "\n".join(f"- {f}" for f in facts) if facts else "No relevant memories found."
        except Exception as e:
            return f"Memory recall failed: {e}"
    if name == "browse_page":
        try:
            return browse_page(args.get("url", ""), user_id)
        except Exception as e:
            return f"Browser failed to load page: {e}"
    if name == "browser_click":
        try:
            return browser_click(args.get("ref", ""), user_id)
        except Exception as e:
            return f"Click failed: {e}"
    if name == "browser_type":
        try:
            return browser_type(args.get("ref", ""), args.get("text", ""), user_id, args.get("submit", False))
        except Exception as e:
            return f"Type failed: {e}"
    if name == "browser_snapshot":
        try:
            return browser_snapshot(user_id)
        except Exception as e:
            return f"Snapshot failed: {e}"
    return f"Unknown tool: {name}"


_TOOL_RESULT_MAX_CHARS = 8000
_SURROUNDING_MESSAGES = 3


async def _fetch_user_messages(channel: discord.TextChannel, user_name: str, limit: int, bot_user_id: int) -> str:
    """Return the user's messages plus surrounding conversation, so the model
    can see what they were responding to."""
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
    keep = set()
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


async def _fetch_channel_history(channel: discord.TextChannel, limit: int, bot_user_id: int) -> str:
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


def _truncate_front(text: str, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> str:
    """Truncate from the front, keeping the most recent (bottom) messages."""
    if len(text) <= max_chars:
        return text
    return "[... earlier messages truncated ...]\n" + text[-max_chars:]


def _build_system() -> str:
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S")
    return SYSTEM_INSTRUCTION + f"\n\n## Today's date time\n{now} PT"


def _build_user_content(prompt: str, images: list[dict], yt_video_ids: list[str] | None = None) -> list[dict]:
    text = prompt
    if yt_video_ids:
        yt_links = "\n".join(f"https://www.youtube.com/watch?v={vid}" for vid in yt_video_ids)
        text = f"{prompt}\n\nYouTube links mentioned:\n{yt_links}"
    parts = [{"type": "text", "text": text}]
    for img in images:
        b64 = base64.b64encode(img["data"]).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{b64}"}})
    return parts


def _collect_media_urls(msg: discord.Message) -> list[dict]:
    media = []
    for a in msg.attachments:
        if a.content_type and a.content_type.startswith("image/"):
            media.append({"url": a.url, "mime_type": a.content_type, "id": str(a.id)})
    for e in msg.embeds:
        if e.type in ("gifv", "image"):
            url = None
            mime = None
            if e.thumbnail and e.thumbnail.url:
                url = e.thumbnail.proxy_url or e.thumbnail.url
                mime = "image/gif"
            elif e.video and e.video.url:
                url = e.video.proxy_url or e.video.url
                mime = "video/mp4"
            if url:
                media.append({"url": url, "mime_type": mime, "id": url})
    return media


async def _download_images(message: discord.Message, ref_msg: discord.Message | None = None) -> list[dict]:
    seen_ids = set()
    all_media = []
    for msg in filter(None, [ref_msg, message]):
        for item in _collect_media_urls(msg):
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_media.append(item)

    if not all_media:
        return []

    async def fetch_one(http: httpx.AsyncClient, item: dict) -> dict | None:
        try:
            resp = await http.get(item["url"])
            if resp.status_code == 200:
                return {"mime_type": item["mime_type"], "data": resp.content}
        except Exception as e:
            log.warning("Failed to download image: %s", e)
        return None

    async with httpx.AsyncClient(timeout=10) as http:
        results = await asyncio.gather(*[fetch_one(http, item) for item in all_media])
    return [r for r in results if r is not None]


async def _send_long_message(message: discord.Message, text: str):
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

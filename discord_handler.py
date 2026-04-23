import asyncio
import base64
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import discord
import httpx

from browser import browse_page, browser_click, browser_snapshot, browser_type, close_browser_session, screenshot_bytes
from config import MAX_DISCORD_LENGTH, MODEL, SYSTEM_INSTRUCTION, openai_generate
from context import fetch_context
from finance import get_financial_data
from memory import extract_and_store_memories, init_db, search_memories
from search import web_search

log = logging.getLogger("siegclaw.handler")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, recent news, events, or factual lookups",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Get live prices for stocks, crypto, commodities, market indices, or any financial instrument",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
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
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_page",
            "description": "Open a URL in a real browser and return the page content as an accessibility tree. Use this for JS-heavy pages, login-walled content, or when web_search isn't enough to get the full page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
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
                "properties": {
                    "ref": {"type": "string", "description": "The ref ID from the page snapshot"},
                },
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
        if message.author == client.user:
            return
        if client.user not in message.mentions:
            return

        async with message.channel.typing():
            user_text = message.content.replace(f"<@{client.user.id}>", "").strip()
            channel_id = str(message.channel.id)
            user_id = str(message.author.id)

            replied_to_bot = False
            reply_context_prefix = ""
            ref_msg = None
            if message.reference is not None and message.reference.message_id:
                try:
                    ref_msg = (
                        message.reference.resolved
                        or await message.channel.fetch_message(message.reference.message_id)
                    )
                    if isinstance(ref_msg, discord.Message):
                        if ref_msg.author == client.user:
                            prompt = f"{ref_msg.author.display_name}: {ref_msg.content}\n{message.author.display_name}: {user_text}"
                            replied_to_bot = True
                        else:
                            reply_context_prefix = (
                                f"[{message.author.display_name} is replying to "
                                f"{ref_msg.author.display_name}'s message]: {ref_msg.content}\n\n"
                            )
                except Exception:
                    pass

            if not replied_to_bot:
                prompt, _ = await fetch_context(message.channel, client.user.id, message.id)
                if reply_context_prefix:
                    prompt = reply_context_prefix + prompt
                prompt = f"{prompt}\n\n[Current question from {message.author.display_name}]: {user_text}"

            images = await _download_images(message, ref_msg)
            system = _build_system()
            user_content = _build_user_content(prompt, images)

            _status_msg: discord.Message | None = None

            async def update_status(tool_name: str, args: dict):
                nonlocal _status_msg
                _TOOL_STATUS = {
                    "web_search":        lambda a: f"🔍 searching: *{a.get('query', '')}*",
                    "get_financial_data": lambda a: f"📈 fetching financial data: *{a.get('query', '')}*",
                    "recall_memories":   lambda a: f"🧠 searching memories: *{a.get('query', '')}*",
                    "browse_page":       lambda a: f"🌐 browsing: `{a.get('url', '')}`",
                    "browser_click":     lambda a: f"🖱️ clicking element `{a.get('ref', '')}`",
                    "browser_type":      lambda a: f"⌨️ typing into page",
                    "browser_snapshot":  lambda a: f"📄 reading page",
                    "browser_screenshot": lambda a: f"📸 taking screenshot",
                }
                fn = _TOOL_STATUS.get(tool_name, lambda a: f"⚙️ {tool_name}")
                line = f"-# {fn(args)}"
                if _status_msg is None:
                    _status_msg = await message.reply(line)
                else:
                    await _status_msg.edit(content=f"{_status_msg.content}\n{line}")

            try:
                reply_text = await _run_with_tools(system, user_content, channel_id, user_id, update_status)
                if not reply_text:
                    raise ValueError("Empty response")
            except Exception as e:
                log.error("API call failed: %s", e)
                reply_text = f"Sorry, I couldn't generate a response ({type(e).__name__}). Please try again."

        await _send_long_message(message, reply_text)
        asyncio.create_task(
            extract_and_store_memories(channel_id, user_id, prompt, reply_text)
        )
        asyncio.create_task(asyncio.to_thread(close_browser_session, user_id))


async def _run_with_tools(
    system: str, user_content, channel_id: str, user_id: str, status_fn=None, max_iters: int = 5
) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    for _ in range(max_iters):
        response = await asyncio.to_thread(openai_generate, MODEL, msgs, tools=TOOLS)
        choice = response.choices[0]

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return (choice.message.content or "").strip()

        msgs.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ],
        })

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if status_fn:
                await status_fn(tc.function.name, args)

            if tc.function.name == "browser_screenshot":
                img_bytes = await asyncio.to_thread(screenshot_bytes, user_id)
                if img_bytes:
                    b64 = base64.b64encode(img_bytes).decode()
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": "Screenshot taken."})
                    msgs.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
                    })
                    log.info("Tool browser_screenshot → %d bytes", len(img_bytes))
                else:
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": "Screenshot failed — no active browser session."})
            else:
                result = await asyncio.to_thread(
                    _dispatch_tool, tc.function.name, args, channel_id, user_id
                )
                log.info("Tool %s(%s) → %d chars", tc.function.name, list(args.keys()), len(result))
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Max iterations hit — final call without tools
    response = await asyncio.to_thread(openai_generate, MODEL, msgs)
    return (response.choices[0].message.content or "").strip()


def _dispatch_tool(name: str, args: dict, channel_id: str, user_id: str) -> str:
    if name == "web_search":
        result = web_search(args.get("query", ""))
        return result or "No results found."
    if name == "get_financial_data":
        try:
            result = get_financial_data(args.get("query", ""))
            return result or "No financial data found."
        except Exception as e:
            return f"Financial data unavailable: {e}"
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


def _build_system() -> str:
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%A, %B %-d, %Y")
    return SYSTEM_INSTRUCTION + f"\n\nToday's date is {today} (Pacific Time)."


def _build_user_content(prompt: str, images: list[dict]):
    if not images:
        return prompt
    content = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = base64.b64encode(img["data"]).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime_type']};base64,{b64}"},
        })
    return content


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

    images = []
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            for item in all_media:
                resp = await http.get(item["url"])
                if resp.status_code == 200:
                    images.append({"mime_type": item["mime_type"], "data": resp.content})
    except Exception as e:
        log.warning("Failed to download images: %s", e)
    return images


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

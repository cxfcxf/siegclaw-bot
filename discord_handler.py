import asyncio
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

_YOUTUBE_RE = re.compile(
    r'https?://(?:www\.)?(?:youtube\.com/watch\?(?:[^&\s]*&)*v=|youtu\.be/)([a-zA-Z0-9_-]{11})[^\s]*'
)

import discord
import httpx
from google.genai import types as genai_types

from browser import browse_page, browser_click, browser_snapshot, browser_type, close_browser_session, screenshot_bytes
from config import MAX_DISCORD_LENGTH, MODEL, SYSTEM_INSTRUCTION, TOOL_MAX_ITERS, genai_client
from context import fetch_context
from memory import extract_and_store_memories, init_db, search_memories
from search import web_search

log = logging.getLogger("siegclaw.handler")

_TOOL_STATUS = {
    "web_search":           lambda a: f"🔍 searching: *{a.get('query', '')}*",
    "recall_memories":      lambda a: f"🧠 searching memories: *{a.get('query', '')}*",
    "fetch_user_messages":  lambda a: f"📜 fetching messages from *{a.get('user_name', '')}*",
    "browse_page":        lambda a: f"🌐 browsing: `{a.get('url', '')}`",
    "browser_click":      lambda a: f"🖱️ clicking element `{a.get('ref', '')}`",
    "browser_type":       lambda a: "⌨️ typing into page",
    "browser_snapshot":   lambda a: "📄 reading page",
    "browser_screenshot": lambda a: "📸 taking screenshot",
}

def _schema(properties: dict, required: list[str] | None = None) -> genai_types.Schema:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={k: genai_types.Schema(**v) for k, v in properties.items()},
        required=required or [],
    )


TOOLS = genai_types.Tool(function_declarations=[
    genai_types.FunctionDeclaration(
        name="fetch_user_messages",
        description="Fetch recent messages from a specific user in this channel. Use when asked to summarize, review, or analyse what a particular person has said.",
        parameters=_schema(
            {"user_name": {"type": genai_types.Type.STRING, "description": "Display name of the user (partial match is fine)"},
             "limit": {"type": genai_types.Type.INTEGER, "description": "Max number of that user's messages to return (default 50, max 100)"}},
            required=["user_name"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current information, recent news, events, or factual lookups",
        parameters=_schema({"query": {"type": genai_types.Type.STRING}}, required=["query"]),
    ),
    genai_types.FunctionDeclaration(
        name="recall_memories",
        description="Search past conversation memories for facts, decisions, preferences, or details about people",
        parameters=_schema({"query": {"type": genai_types.Type.STRING}}, required=["query"]),
    ),
    genai_types.FunctionDeclaration(
        name="browse_page",
        description="Open a URL in a real browser and return the page content as an accessibility tree. Use this for JS-heavy pages, login-walled content, or when web_search isn't enough.",
        parameters=_schema({"url": {"type": genai_types.Type.STRING}}, required=["url"]),
    ),
    genai_types.FunctionDeclaration(
        name="browser_click",
        description="Click an element on the current browser page using its ref ID from the snapshot",
        parameters=_schema({"ref": {"type": genai_types.Type.STRING, "description": "The ref ID from the page snapshot"}}, required=["ref"]),
    ),
    genai_types.FunctionDeclaration(
        name="browser_type",
        description="Type text into an input field on the current browser page",
        parameters=_schema(
            {"ref": {"type": genai_types.Type.STRING, "description": "The ref ID of the input from the snapshot"},
             "text": {"type": genai_types.Type.STRING},
             "submit": {"type": genai_types.Type.BOOLEAN, "description": "Press Enter after typing"}},
            required=["ref", "text"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="browser_snapshot",
        description="Get the current state of the browser page as an accessibility tree",
        parameters=_schema({}),
    ),
    genai_types.FunctionDeclaration(
        name="browser_screenshot",
        description="Take a visual screenshot of the current browser page and see it as an image",
        parameters=_schema({}),
    ),
])


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
            yt_video_ids = _YOUTUBE_RE.findall(user_text)
            system = _build_system()
            user_content = _build_user_content(prompt, images, yt_video_ids)

            _status_msg: discord.Message | None = None

            async def update_status(tool_name: str, args: dict):
                nonlocal _status_msg
                fn = _TOOL_STATUS.get(tool_name, lambda a: f"⚙️ {tool_name}")
                line = f"-# {fn(args)}"
                if _status_msg is None:
                    _status_msg = await message.reply(line)
                else:
                    await _status_msg.edit(content=f"{_status_msg.content}\n{line}")

            try:
                reply_text = await _run_with_tools(system, user_content, message.channel, channel_id, user_id, update_status)
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
    system: str, user_content: genai_types.Content, channel: discord.TextChannel, channel_id: str, user_id: str, status_fn=None, max_iters: int = TOOL_MAX_ITERS
) -> str:
    contents = [user_content]
    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        tools=[TOOLS],
        thinking_config=genai_types.ThinkingConfig(thinking_budget=-1),
    )

    for _ in range(max_iters):
        response = await genai_client.aio.models.generate_content(
            model=MODEL, contents=contents, config=config,
        )

        if not response.function_calls:
            return response.text or ""

        contents.append(response.candidates[0].content)

        fn_response_parts = []
        for fc in response.function_calls:
            args = dict(fc.args)
            if status_fn:
                await status_fn(fc.name, args)

            if fc.name == "fetch_user_messages":
                result = await _fetch_user_messages(channel, args.get("user_name", ""), args.get("limit", 50))
                log.info("Tool fetch_user_messages(%s) → %d chars", args.get("user_name"), len(result))
            elif fc.name == "browser_screenshot":
                img_bytes = await asyncio.to_thread(screenshot_bytes, user_id)
                if img_bytes:
                    log.info("Tool browser_screenshot → %d bytes", len(img_bytes))
                    fn_response_parts.append(genai_types.Part(
                        function_response=genai_types.FunctionResponse(name=fc.name, response={"result": "Screenshot taken."})
                    ))
                    fn_response_parts.append(genai_types.Part(
                        inline_data=genai_types.Blob(mime_type="image/png", data=img_bytes)
                    ))
                    continue
                else:
                    result = "Screenshot failed — no active browser session."
            else:
                result = await asyncio.to_thread(_dispatch_tool, fc.name, args, channel_id, user_id)
                log.info("Tool %s(%s) → %d chars", fc.name, list(args.keys()), len(result))

            fn_response_parts.append(genai_types.Part(
                function_response=genai_types.FunctionResponse(name=fc.name, response={"result": result})
            ))

        contents.append(genai_types.Content(role="user", parts=fn_response_parts))

    # Max iterations hit — force a final answer with no tools
    contents.append(genai_types.Content(role="user", parts=[
        genai_types.Part(text="You have reached the maximum number of tool calls. Based on everything gathered so far, give your best answer now.")
    ]))
    final_config = genai_types.GenerateContentConfig(
        system_instruction=config.system_instruction,
        thinking_config=config.thinking_config,
    )
    response = await genai_client.aio.models.generate_content(model=MODEL, contents=contents, config=final_config)
    return response.text or ""


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


async def _fetch_user_messages(channel: discord.TextChannel, user_name: str, limit: int) -> str:
    limit = min(max(limit, 1), 50)
    matches = []
    async for msg in channel.history(limit=100):
        if user_name.lower() in msg.author.display_name.lower() and msg.content:
            matches.append(f"{msg.author.display_name}: {msg.content}")
            if len(matches) >= limit:
                break
    if not matches:
        return f"No messages found from '{user_name}' in the last 100 channel messages."
    matches.reverse()
    return "\n".join(matches)


def _build_system() -> str:
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S")
    return SYSTEM_INSTRUCTION + f"\n\n## Today's date time\n{now} PT"


def _build_user_content(prompt: str, images: list[dict], yt_video_ids: list[str] | None = None) -> genai_types.Content:
    parts = [genai_types.Part(text=prompt)]
    for video_id in (yt_video_ids or []):
        parts.append(genai_types.Part(file_data=genai_types.FileData(
            mime_type="video/*", file_uri=f"https://www.youtube.com/watch?v={video_id}",
        )))
    for img in images:
        parts.append(genai_types.Part(inline_data=genai_types.Blob(
            mime_type=img["mime_type"], data=img["data"],
        )))
    return genai_types.Content(role="user", parts=parts)


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

"""Discord conversation context: build a timestamped transcript from live channel
history, and download image/media attachments for multimodal turns.

Discord is the source of truth for Discord conversations — these turns are NOT
stored in the SQLite conversation db. Each mention/DM rebuilds context from the
channel's recent history with a hybrid time/count window. Ported from the
standalone siegclaw-bot (`context.py` + the media helpers in `discord_handler.py`).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import discord
import httpx

from .config import (
    CONTEXT_ACTIVITY_THRESHOLD,
    CONTEXT_MAX_CHARS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MESSAGE_COUNT,
    CONTEXT_TIME_WINDOW_HOURS,
    HARNESS_TZ,
)

log = logging.getLogger("siegclaw.context")

try:
    TZ = ZoneInfo(HARNESS_TZ)
except Exception:
    TZ = ZoneInfo("America/Los_Angeles")

YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?(?:[^&\s]*&)*v=|youtu\.be/)([a-zA-Z0-9_-]{11})[^\s]*"
)


def message_text(msg: discord.Message, strip_bot_id: int | None = None) -> str:
    """Message content with raw mention tokens resolved to readable @names.

    If strip_bot_id is given, that user's mention is removed entirely
    (used for the trigger message, where the bot @mention is just noise).
    """
    text = msg.content
    if strip_bot_id is not None:
        text = text.replace(f"<@{strip_bot_id}>", "").replace(f"<@!{strip_bot_id}>", "")
    for m in msg.mentions:
        text = text.replace(f"<@{m.id}>", f"@{m.display_name}")
        text = text.replace(f"<@!{m.id}>", f"@{m.display_name}")
    for r in msg.role_mentions:
        text = text.replace(f"<@&{r.id}>", f"@{r.name}")
    return text.strip()


def is_status_message(msg: discord.Message, bot_user_id: int) -> bool:
    """Bot tool-status lines ('-# 🔍 searching...') — UI noise, not conversation."""
    return msg.author.id == bot_user_id and msg.content.startswith("-#")


def format_line(msg: discord.Message) -> str:
    ts = msg.created_at.astimezone(TZ).strftime("%m-%d %H:%M")
    text = message_text(msg)
    if not text and msg.attachments:
        text = "[attachment]"
    return f"[{ts}] {msg.author.display_name}: {text}"


async def fetch_context(
    channel: discord.abc.Messageable,
    bot_user_id: int,
    trigger_message_id: int,
) -> tuple[str, list[discord.Message]]:
    """Fetch recent messages with hybrid time/count windowing.

    Returns (formatted_prompt, raw_messages).
    """
    time_window = timedelta(hours=CONTEXT_TIME_WINDOW_HOURS)
    cutoff = datetime.now(timezone.utc) - time_window

    fetched = []  # newest first
    async for msg in channel.history(limit=CONTEXT_MAX_MESSAGES):
        fetched.append(msg)

    recent = fetched[:CONTEXT_MESSAGE_COUNT]

    # Busy channel: the count window is dense, so expand to the full time
    # window. Always newest-first, so the most recent conversation is kept
    # even when the window holds more messages than the limit.
    if len(recent) >= CONTEXT_ACTIVITY_THRESHOLD and recent and recent[-1].created_at > cutoff:
        messages = [m for m in fetched if m.created_at >= cutoff]
        log.info(
            "Active channel %s: using %d messages (%dh window)",
            channel.id, len(messages), CONTEXT_TIME_WINDOW_HOURS,
        )
    else:
        messages = recent
        log.debug("Channel %s: using %d messages (count-based)", channel.id, len(messages))

    messages.reverse()  # chronological, oldest first
    return _format_messages(messages, bot_user_id, trigger_message_id), messages


def _format_messages(
    messages: list[discord.Message],
    bot_user_id: int,
    trigger_message_id: int,
) -> str:
    lines = []
    for msg in messages:
        if msg.id == trigger_message_id:
            continue  # appended separately as the current question
        if is_status_message(msg, bot_user_id):
            continue
        lines.append(format_line(msg))
    text = "\n".join(lines)
    if len(text) > CONTEXT_MAX_CHARS:
        text = "[... earlier messages truncated ...]\n" + text[-CONTEXT_MAX_CHARS:]
    return text


# --------------------------------------------------------------------------- #
# Media (images / gifs / video thumbnails)
# --------------------------------------------------------------------------- #
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


async def download_images(message: discord.Message, ref_msg: discord.Message | None = None) -> list[dict]:
    """Download images/media from the trigger message and any replied-to message.
    Returns [{mime_type, data(bytes)}]."""
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


def build_user_content(prompt: str, images: list[dict], yt_video_ids: list[str] | None = None) -> list[dict]:
    """Assemble the OpenAI multimodal `content` list: transcript text (+ any
    YouTube links) followed by image parts."""
    text = prompt
    if yt_video_ids:
        yt_links = "\n".join(f"https://www.youtube.com/watch?v={vid}" for vid in yt_video_ids)
        text = f"{prompt}\n\nYouTube links mentioned:\n{yt_links}"
    parts: list[dict] = [{"type": "text", "text": text}]
    for img in images:
        b64 = base64.b64encode(img["data"]).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{img['mime_type']};base64,{b64}"}})
    return parts

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import discord

from config import (
    CONTEXT_ACTIVITY_THRESHOLD,
    CONTEXT_MAX_CHARS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MESSAGE_COUNT,
    CONTEXT_TIME_WINDOW_HOURS,
)

log = logging.getLogger("siegclaw.context")

PT = ZoneInfo("America/Los_Angeles")


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
    ts = msg.created_at.astimezone(PT).strftime("%m-%d %H:%M")
    text = message_text(msg)
    if not text and msg.attachments:
        text = "[attachment]"
    return f"[{ts}] {msg.author.display_name}: {text}"


async def fetch_context(
    channel: discord.TextChannel,
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
    if len(recent) >= CONTEXT_ACTIVITY_THRESHOLD and recent[-1].created_at > cutoff:
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

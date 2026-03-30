import logging
from datetime import datetime, timedelta, timezone

import discord

from config import (
    CONTEXT_ACTIVITY_THRESHOLD,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MESSAGE_COUNT,
    CONTEXT_TIME_WINDOW_HOURS,
)

log = logging.getLogger("siegclaw.context")


async def fetch_context(
    channel: discord.TextChannel,
    bot_user_id: int,
    trigger_message_id: int,
) -> tuple[str, list[discord.Message]]:
    """Fetch recent messages with hybrid time/count windowing.

    Returns (formatted_prompt, raw_messages) so callers can reuse
    the message list without a second Discord API call.
    """
    messages = []
    async for msg in channel.history(limit=CONTEXT_MESSAGE_COUNT):
        messages.append(msg)

    if len(messages) < 2:
        return _format_messages(messages, bot_user_id, trigger_message_id), messages

    messages.reverse()

    oldest = messages[0].created_at
    newest = messages[-1].created_at
    span = newest - oldest
    time_window = timedelta(hours=CONTEXT_TIME_WINDOW_HOURS)

    if len(messages) >= CONTEXT_ACTIVITY_THRESHOLD and span < time_window:
        after = datetime.now(timezone.utc) - time_window
        messages = []
        async for msg in channel.history(limit=CONTEXT_MAX_MESSAGES, after=after):
            messages.append(msg)
        messages.reverse()
        log.info(
            "Active channel %s: fetched %d messages (24h window)",
            channel.id, len(messages),
        )
    else:
        log.debug(
            "Channel %s: using %d messages (count-based)",
            channel.id, len(messages),
        )

    return _format_messages(messages, bot_user_id, trigger_message_id), messages


def _format_messages(
    messages: list[discord.Message],
    bot_user_id: int,
    trigger_message_id: int,
) -> str:
    lines = []
    for msg in messages:
        content = msg.content
        if msg.id == trigger_message_id:
            content = content.replace(f"<@{bot_user_id}>", "").strip()
        lines.append(f"{msg.author.display_name}: {content}")
    return "\n".join(lines)

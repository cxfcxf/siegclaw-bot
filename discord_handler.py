import asyncio
import logging

import discord
import httpx
from google.genai import types

from config import (
    MAX_DISCORD_LENGTH,
    MODEL,
    SYSTEM_INSTRUCTION,
    gemini_client,
)
from context import fetch_context
from finance import get_financial_data
from memory import extract_and_store_memories, init_db, search_memories
from router import classify_query
from search import web_search

log = logging.getLogger("siegclaw.handler")


def setup_events(client: discord.Client):
    """Register Discord event handlers."""

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
            user_text = message.content.replace(
                f"<@{client.user.id}>", ""
            ).strip()
            channel_id = str(message.channel.id)
            user_id = str(message.author.id)

            prompt, raw_messages = await fetch_context(
                message.channel, client.user.id, message.id
            )

            recent_lines = [
                f"{m.author.display_name}: {m.content[:100]}"
                for m in raw_messages[-5:]
                if m.id != message.id
            ]
            recent_context = "\n".join(recent_lines[-4:])

            route = await classify_query(user_text, recent_context)

            # For search/finance, skip the chat history — it's just noise
            if route in ("search", "finance"):
                prompt = user_text
            else:
                # Clearly separate history from the current question
                prompt = f"{prompt}\n\n[Current question from {message.author.display_name}]: {user_text}"

            memories = []
            search_results = None
            finance_data = None

            errors = []

            if route == "finance":
                finance_data, err = await _safe_named(asyncio.to_thread(get_financial_data, user_text))
                if err:
                    errors.append("live price data")
                if not finance_data:
                    search_results, err = await _safe_named(asyncio.to_thread(web_search, user_text))
                    if err:
                        errors.append("web search")
            elif route == "search":
                search_results, err = await _safe_named(asyncio.to_thread(web_search, user_text))
                if err:
                    errors.append("web search")
            elif route == "memory":
                memories, err = await _safe_named(
                    asyncio.to_thread(search_memories, channel_id, user_text, user_id)
                )
                memories = memories or []
                if err:
                    errors.append("memory recall")
            elif route == "both":
                (memories, merr), (search_results, serr) = await asyncio.gather(
                    _safe_named(asyncio.to_thread(search_memories, channel_id, user_text, user_id)),
                    _safe_named(asyncio.to_thread(web_search, user_text)),
                )
                memories = memories or []
                if merr:
                    errors.append("memory recall")
                if serr:
                    errors.append("web search")

            system = SYSTEM_INSTRUCTION
            if memories:
                memory_text = "\n".join(f"- {m}" for m in memories)
                system += (
                    "\n\nYou have the following memories from past conversations. "
                    "Use them if relevant, but don't force them into the conversation:\n"
                    + memory_text
                )
            if finance_data:
                system += (
                    "\n\nHere is live financial data. Present it clearly and "
                    "mention that prices are real-time:\n\n" + finance_data
                )
            if search_results:
                system += (
                    "\n\nHere are recent web search results relevant to the "
                    "user's query. Use this information to provide an accurate, "
                    "up-to-date answer. Cite sources when appropriate:\n\n"
                    + search_results
                )

            images = await _download_images(message)

            parts = [types.Part.from_text(text=prompt)]
            for img in images:
                parts.append(
                    types.Part.from_bytes(data=img["data"], mime_type=img["mime_type"])
                )

            try:
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=MODEL,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                    ),
                )
                reply_text = response.text
                if errors:
                    failed = " and ".join(errors)
                    reply_text += f"\n\n⚠️ *Note: {failed} unavailable, response may be incomplete.*"
            except Exception as e:
                log.error("Gemini API call failed: %s", e)
                reply_text = f"Sorry, I couldn't generate a response (Gemini API error: {type(e).__name__}). Please try again."

        await _send_long_message(message, reply_text)

        asyncio.create_task(
            extract_and_store_memories(channel_id, user_id, prompt, reply_text)
        )


async def _safe_named(coro) -> tuple:
    """Run a coroutine, returning (result, error_or_None)."""
    try:
        return await coro, None
    except Exception as e:
        log.warning("Non-critical operation failed: %s", e)
        return None, e


async def _download_images(message: discord.Message) -> list[dict]:
    """Download image attachments from a Discord message."""
    images = []
    image_attachments = [
        a for a in message.attachments
        if a.content_type and a.content_type.startswith("image/")
    ]
    if not image_attachments:
        return images

    try:
        async with httpx.AsyncClient() as http:
            for attachment in image_attachments:
                resp = await http.get(attachment.url)
                if resp.status_code == 200:
                    images.append({
                        "mime_type": attachment.content_type,
                        "data": resp.content,
                    })
    except Exception as e:
        log.warning("Failed to download images: %s", e)
    return images



async def _send_long_message(message: discord.Message, text: str):
    """Send a message, splitting into chunks if it exceeds Discord's limit."""
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

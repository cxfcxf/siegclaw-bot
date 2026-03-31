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
from router import classify_query, is_response_certain
from search import web_search

log = logging.getLogger("siegclaw.handler")


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
            user_text = message.content.replace(
                f"<@{client.user.id}>", ""
            ).strip()
            channel_id = str(message.channel.id)
            user_id = str(message.author.id)

            # If replying to the bot, use focused thread context instead of full history
            replied_to_bot = False
            if message.reference is not None and message.reference.message_id:
                try:
                    ref_msg = (
                        message.reference.resolved
                        or await message.channel.fetch_message(message.reference.message_id)
                    )
                    if isinstance(ref_msg, discord.Message) and ref_msg.author == client.user:
                        prompt = f"{ref_msg.author.display_name}: {ref_msg.content}\n{message.author.display_name}: {user_text}"
                        raw_messages = [ref_msg, message]
                        replied_to_bot = True
                except Exception:
                    pass

            if not replied_to_bot:
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

            if route in ("search", "finance") and not replied_to_bot:
                prompt = user_text
            else:
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

            system = _build_system(memories, finance_data, search_results)
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
                    config=types.GenerateContentConfig(system_instruction=system),
                )
                reply_text = response.text

                # Fallback: if DIRECT route and response is uncertain, search and retry
                if route == "direct" and not await is_response_certain(reply_text):
                    log.info("Uncertain response detected, running fallback search for '%s'", user_text[:60])
                    search_results, _ = await _safe_named(asyncio.to_thread(web_search, user_text))
                    if search_results:
                        system = _build_system(memories, finance_data, search_results)
                        retry = await asyncio.to_thread(
                            gemini_client.models.generate_content,
                            model=MODEL,
                            contents=parts,
                            config=types.GenerateContentConfig(system_instruction=system),
                        )
                        reply_text = retry.text

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


def _build_system(memories: list, finance_data: str | None, search_results: str | None) -> str:
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
    return system


async def _safe_named(coro) -> tuple:
    try:
        return await coro, None
    except Exception as e:
        log.warning("Non-critical operation failed: %s", e)
        return None, e


async def _download_images(message: discord.Message) -> list[dict]:
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

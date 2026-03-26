import asyncio
import json
import os
import sqlite3

import discord
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_DISCORD_LENGTH = 2000
CONTEXT_MESSAGE_COUNT = int(os.getenv("CONTEXT_MESSAGE_COUNT", "15"))
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memories.db")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
SYSTEM_INSTRUCTION = (
    "You are SiegClaw, a helpful AI assistant in a Discord chat. "
    "Keep responses concise and conversational. "
    "Use Discord markdown formatting when helpful. "
    "If a conversation has multiple participants, be aware of who said what."
)

MEMORY_EXTRACTION_PROMPT = """\
Analyze the following conversation and extract important facts worth remembering \
for future conversations. Focus on:
- People's names, roles, preferences, birthdays
- Decisions made by the group
- Important events, dates, plans
- Personal details people share
- Anything that would be useful to recall later

Return ONLY a JSON array of short fact strings. If nothing is worth remembering, return [].
Example: ["John's birthday is March 5", "The team decided to use React for the frontend"]

Conversation:
{conversation}

Bot's reply:
{bot_reply}"""

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


# --- Memory functions ---

def init_db():
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def search_memories(channel_id, query, limit=10):
    conn = sqlite3.connect(MEMORY_DB_PATH)
    # Extract words longer than 2 chars as keywords
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    if not keywords:
        conn.close()
        return []

    # Search for facts containing any keyword
    conditions = " OR ".join(["LOWER(fact) LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords]
    params.append(channel_id)

    cursor = conn.execute(
        f"SELECT DISTINCT fact FROM memories WHERE ({conditions}) AND channel_id = ? ORDER BY created_at DESC LIMIT ?",
        params + [limit],
    )
    facts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return facts


def store_memories(channel_id, facts):
    if not facts:
        return
    conn = sqlite3.connect(MEMORY_DB_PATH)
    for fact in facts:
        # Avoid exact duplicates
        existing = conn.execute(
            "SELECT 1 FROM memories WHERE fact = ? AND channel_id = ?",
            (fact, channel_id),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO memories (fact, channel_id) VALUES (?, ?)",
                (fact, channel_id),
            )
    conn.commit()
    conn.close()


async def extract_and_store_memories(channel_id, conversation, bot_reply):
    try:
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            conversation=conversation, bot_reply=bot_reply
        )
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You extract facts from conversations. Return only valid JSON arrays.",
                response_mime_type="application/json",
            ),
        )
        facts = json.loads(response.text)
        if isinstance(facts, list) and facts:
            store_memories(channel_id, [f for f in facts if isinstance(f, str)])
    except Exception as e:
        print(f"[MEMORY] Extraction error: {e}")


# --- Discord events ---

@client.event
async def on_ready():
    init_db()
    print(f"Logged in as {client.user} (ID: {client.user.id})")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user not in message.mentions:
        return

    async with message.channel.typing():
        # Collect context from recent messages
        context_messages = []
        async for msg in message.channel.history(limit=CONTEXT_MESSAGE_COUNT):
            author_name = msg.author.display_name
            content = msg.content
            if msg.id == message.id:
                content = content.replace(f"<@{client.user.id}>", "").strip()
            context_messages.append(f"{author_name}: {content}")

        context_messages.reverse()
        prompt = "\n".join(context_messages)

        # Search for relevant memories
        user_text = message.content.replace(f"<@{client.user.id}>", "").strip()
        memories = search_memories(str(message.channel.id), user_text)

        # Build system instruction with memories
        system = SYSTEM_INSTRUCTION
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            system += (
                f"\n\nYou have the following memories from past conversations "
                f"in this channel. Use them if relevant, but don't force them "
                f"into the conversation:\n{memory_text}"
            )

        # Download images from the triggering message
        images = []
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                async with httpx.AsyncClient() as http:
                    resp = await http.get(attachment.url)
                    if resp.status_code == 200:
                        images.append({
                            "mime_type": attachment.content_type,
                            "data": resp.content,
                        })

        # Build Gemini request with text + images
        parts = [types.Part.from_text(text=prompt)]
        for img in images:
            parts.append(types.Part.from_bytes(data=img["data"], mime_type=img["mime_type"]))

        try:
            response = gemini_client.models.generate_content(
                model=MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            reply_text = response.text
        except Exception as e:
            reply_text = f"Sorry, I hit an error: {e}"

    await send_long_message(message, reply_text)

    # Extract memories in the background (don't slow down the reply)
    asyncio.create_task(
        extract_and_store_memories(str(message.channel.id), prompt, reply_text)
    )


async def send_long_message(message, text):
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


if __name__ == "__main__":
    if not DISCORD_TOKEN or not GOOGLE_API_KEY:
        print("ERROR: Set DISCORD_BOT_TOKEN and GOOGLE_API_KEY in .env")
        exit(1)
    client.run(DISCORD_TOKEN)

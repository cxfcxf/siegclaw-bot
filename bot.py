import os

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

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
SYSTEM_INSTRUCTION = (
    "You are SiegClaw, a helpful AI assistant in a Discord chat. "
    "Keep responses concise and conversational. "
    "Use Discord markdown formatting when helpful. "
    "If a conversation has multiple participants, be aware of who said what."
)

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
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
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            reply_text = response.text
        except Exception as e:
            reply_text = f"Sorry, I hit an error: {e}"

    await send_long_message(message, reply_text)


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

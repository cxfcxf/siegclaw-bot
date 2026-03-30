import discord

from config import DISCORD_TOKEN, GOOGLE_API_KEY, TAVILY_API_KEY, log
from discord_handler import setup_events

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

setup_events(client)

if __name__ == "__main__":
    missing = []
    if not DISCORD_TOKEN:
        missing.append("DISCORD_BOT_TOKEN")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        log.error("Missing env vars: %s", ", ".join(missing))
        exit(1)
    client.run(DISCORD_TOKEN, log_handler=None)

import discord

from config import DISCORD_TOKEN, log
from discord_handler import setup_events

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

setup_events(client)

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("Missing env var: DISCORD_BOT_TOKEN")
        exit(1)
    client.run(DISCORD_TOKEN, log_handler=None)

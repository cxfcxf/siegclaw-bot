import asyncio

import discord
from aiohttp import web

from config import DISCORD_TOKEN, WEBHOOK_PORT, log
from discord_handler import setup_events
from webhook import create_webhook_app


async def main():
    if not DISCORD_TOKEN:
        log.error("Missing env var: DISCORD_BOT_TOKEN")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    setup_events(client)

    webhook_app = create_webhook_app(client)
    runner = web.AppRunner(webhook_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WEBHOOK_PORT)
    await site.start()
    log.info("Webhook server listening on port %d", WEBHOOK_PORT)

    async with client:
        await client.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())

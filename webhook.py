import logging

import discord
from aiohttp import web

from config import WEBHOOK_CHANNEL_ID

log = logging.getLogger("siegclaw.webhook")


def create_webhook_app(client: discord.Client) -> web.Application:
    app = web.Application()

    async def notify(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON")

        content = data.get("content", "").strip()
        if not content:
            return web.Response(status=400, text="Missing content")

        channel_id = int(data.get("channel_id") or WEBHOOK_CHANNEL_ID)
        if not channel_id:
            return web.Response(status=400, text="No channel_id configured")

        channel = client.get_channel(channel_id)
        if channel is None:
            return web.Response(status=404, text=f"Channel {channel_id} not found")

        chunks = []
        while len(content) > 2000:
            split_at = content.rfind("\n", 0, 2000) or 2000
            chunks.append(content[:split_at])
            content = content[split_at:].lstrip("\n")
        chunks.append(content)

        for chunk in chunks:
            await channel.send(chunk)
        log.info("Webhook: posted %d chars (%d chunks) to channel %s", sum(len(c) for c in chunks), len(chunks), channel_id)
        return web.Response(text="ok")

    app.router.add_post("/notify", notify)
    return app

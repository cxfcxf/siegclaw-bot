# SiegClaw Bot

A Discord AI assistant powered by Gemini with web search, browser automation, YouTube understanding, and persistent vector memory.

## Features

- **Tool-calling** — Gemini autonomously decides when to search, browse, recall memories, or fetch user history
- **Web search** — real-time search via local Firecrawl instance
- **Browser automation** — opens real browser pages (via Camofox), supports click, type, snapshot, and screenshot
- **YouTube video understanding** — paste a YouTube URL and Gemini natively processes the video (no transcript scraping needed)
- **Vector memory** — extracts and stores facts from conversations using LanceDB + Gemini embeddings
- **Image support** — attach images or reply to image messages for multimodal responses
- **User message lookup** — fetch and summarise what a specific person said in recent channel history
- **Reply-aware context** — replying to the bot uses focused thread context instead of full channel history
- **Hybrid context window** — adapts between count-based and time-based message fetching for active channels
- **Dynamic thinking** — Gemini self-allocates thinking budget per query complexity
- **Date-aware** — current PT timestamp injected into every prompt
- **Webhook endpoint** — `POST /notify` lets external services push messages into a Discord channel
- **Personality via SOUL.md** — edit `SOUL.md` to change the bot's behaviour without touching code

## Setup

### Requirements

- Docker (recommended) or Python 3.12+
- Google AI Studio API key (paid tier recommended — free tier is 20 req/day)
- Local [Firecrawl](https://github.com/mendableai/firecrawl) instance
- Local [Camofox](https://github.com/siegfried/camofox) instance (optional, for browser tools)

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | — | Discord bot token |
| `GOOGLE_API_KEY` | Yes | — | Google AI Studio API key |
| `FIRECRAWL_API_URL` | No | `http://localhost:3002` | Firecrawl instance URL |
| `CAMOFOX_URL` | No | `http://localhost:9377` | Camofox browser instance URL |
| `WEBHOOK_CHANNEL_ID` | No | — | Default Discord channel ID for webhook messages |
| `LANCEDB_PATH` | No | `data/lancedb` | LanceDB directory path |
| `MODEL` | No | `gemini-3-flash-preview` | Gemini model to use |
| `EMBEDDING_MODEL` | No | `gemini-embedding-2` | Embedding model |
| `MEMORY_DECAY_DAYS` | No | `90` | Days before memories score lower in search |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `WEBHOOK_PORT` | No | `8643` | Port for the incoming webhook server |

### Docker (recommended)

```bash
docker build -t siegclaw-bot .

docker run -d \
  --name siegclaw-bot \
  --env-file .env \
  -v /path/to/data:/app/data \
  --restart unless-stopped \
  siegclaw-bot
```

### Python

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
python bot.py
```

## Architecture

```
User @mentions bot (or replies to bot message)
        │
        ├── Reply to bot? → focused 2-message thread context
        │   else          → fetch_context() hybrid window
        │
        ├── Download attached images (parallel)
        │
        ├── YouTube URLs in message? → attached as native file_data parts
        │
        └── _run_with_tools() — Gemini tool-calling loop (max 5 iterations)
                │
                ├── web_search        → Firecrawl
                ├── browse_page       → Camofox (real browser)
                ├── browser_click/type/snapshot/screenshot
                ├── recall_memories   → LanceDB vector search
                └── fetch_user_messages → channel history filtered by author
                │
                └── Background: extract facts → embed → store in LanceDB
```

## Tools

| Tool | Description |
|---|---|
| `web_search` | Search the web via Firecrawl |
| `browse_page` | Open a URL in a real browser |
| `browser_click` | Click an element by ARIA ref |
| `browser_type` | Type into an input field |
| `browser_snapshot` | Get current page as accessibility tree |
| `browser_screenshot` | Take a visual screenshot |
| `recall_memories` | Search stored facts from past conversations |
| `fetch_user_messages` | Fetch recent messages from a specific user in the channel |

## Webhook

The bot exposes a `POST /notify` endpoint (default port 8643) for external services to push messages into Discord.

```
POST http://localhost:8643/notify
Content-Type: application/json

{
  "content": "your message here",
  "channel_id": "optional — overrides WEBHOOK_CHANNEL_ID"
}
```

From another Docker container:

```python
import urllib.request, json

urllib.request.urlopen(urllib.request.Request(
    'http://host.docker.internal:8643/notify',
    data=json.dumps({'content': 'your message here'}).encode(),
    headers={'Content-Type': 'application/json'}
))
```

## File Structure

```
bot.py              — entrypoint, runs Discord client + webhook server
config.py           — env vars, API client, prompts
SOUL.md             — system prompt / bot personality
context.py          — hybrid context window logic
discord_handler.py  — Discord events, tool loop, message handling
webhook.py          — aiohttp webhook server
search.py           — Firecrawl web search
memory.py           — LanceDB vector memory
browser.py          — Camofox browser automation
```

## Memory System

Facts are automatically extracted from every conversation and stored as vector embeddings in LanceDB using `gemini-embedding-2` (768 dimensions). Memories are:

- **Per-channel and per-user** — channel memories stay local, user preferences follow them across channels
- **Decay-weighted** — older memories score lower in search results (configurable via `MEMORY_DECAY_DAYS`)
- **Deduplicated** — near-identical facts (≥0.95 cosine similarity) are skipped on insert

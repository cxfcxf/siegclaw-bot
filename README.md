# SiegClaw Bot

A Discord AI assistant powered by Gemini with real-time web search, financial data, and persistent vector memory.

## Features

- **Smart query routing** — classifies every message as search, finance, memory, or direct using Gemini with thinking
- **Certainty fallback** — if bot is uncertain on a direct answer, automatically triggers a web search and retries
- **Reply-aware context** — replying to the bot uses focused thread context instead of full channel history
- **Web search** — Tavily (advanced depth) for real-time news, weather, and current events
- **Financial data** — live prices via CoinGecko (crypto) and Yahoo Finance (stocks, indices, commodities, forex)
- **Vector memory** — extracts and stores facts from conversations using SQLite + numpy cosine similarity
- **Multi-turn awareness** — router uses recent conversation context for follow-up questions
- **Image support** — pass images alongside your message for multimodal responses
- **Hybrid context window** — adapts between count-based and time-based message fetching
- **Date-aware** — today's date injected into every prompt so relative dates ("this Friday") are always correct
- **Webhook endpoint** — `POST /notify` lets external services (e.g. hermes-agent cron jobs) push messages to a Discord channel

## Setup

### Requirements

- Docker (recommended) or Python 3.12+
- Google API key (Gemini)
- Tavily API key

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_BOT_TOKEN` | Yes | — | Discord bot token |
| `GOOGLE_API_KEY` | Yes | — | Google API key for Gemini + embeddings |
| `TAVILY_API_KEY` | Yes | — | Tavily search API key |
| `GEMINI_MODEL` | No | `gemini-3.1-flash-lite-preview` | Gemini model to use |
| `CONTEXT_MESSAGE_COUNT` | No | `30` | Baseline messages to fetch |
| `CONTEXT_TIME_WINDOW_HOURS` | No | `24` | Time window for active channels |
| `CONTEXT_ACTIVITY_THRESHOLD` | No | `20` | Messages needed to trigger time-based mode |
| `CONTEXT_MAX_MESSAGES` | No | `50` | Hard cap on messages fetched |
| `LANCEDB_PATH` | No | `data/lancedb` | LanceDB directory path |
| `MEMORY_DECAY_DAYS` | No | `90` | Days before memories score lower in search |
| `TAVILY_MAX_RESULTS` | No | `5` | Number of search results per query |
| `ROUTER_ENABLED` | No | `true` | Toggle query router (false = always search + memory) |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WEBHOOK_PORT` | No | `8643` | Port for the incoming webhook server |
| `WEBHOOK_CHANNEL_ID` | No | — | Default Discord channel ID for webhook messages |

### Docker (recommended)

```bash
# Build on the target machine
docker build -t siegclaw-bot .

# Run
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
User @mentions bot (or replies to bot's message)
        │
        ├── Reply? → focused thread context
        │   else  → fetch_context() hybrid window
        │
        ├── classify_query()    Gemini router → SEARCH / FINANCE / MEMORY / DIRECT
        │
        ├── SEARCH   → Tavily web search
        ├── FINANCE  → CoinGecko + Yahoo Finance (fallback to Tavily)
        ├── MEMORY   → SQLite cosine similarity search
        └── DIRECT   → last 10 messages as context
        │
        └── Gemini generates response (augmented prompt + today's date)
            │
            ├── DIRECT: is_response_certain() → UNCERTAIN → fallback search → retry
            │
            └── Background: extract facts → embed → store in SQLite
```

## Webhook

The bot exposes a `POST /notify` endpoint (default port 8643) so external services can push messages into Discord.

**Request:**
```
POST http://localhost:8643/notify
Content-Type: application/json

{
  "content": "your message here",
  "channel_id": "optional — overrides WEBHOOK_CHANNEL_ID"
}
```

Messages longer than 2000 characters are automatically split into multiple Discord messages.

**From another Docker container (e.g. hermes-agent):**
```python
import urllib.request, json

r = urllib.request.urlopen(urllib.request.Request(
    'http://host.docker.internal:8643/notify',
    data=json.dumps({'content': 'your message here'}).encode(),
    headers={'Content-Type': 'application/json'}
))
```

Remember to expose the port in your `docker run` command: `-p 127.0.0.1:8643:8643`

## File Structure

```
bot.py              — entrypoint, runs Discord client + webhook server
config.py           — env vars, clients, prompts
context.py          — hybrid context window logic
discord_handler.py  — Discord event handling
webhook.py          — aiohttp webhook server
search.py           — web search
finance.py          — CoinGecko + Yahoo Finance
memory.py           — LanceDB vector memory
browser.py          — browser automation
```

## Memory System

Facts are automatically extracted from every conversation by Gemini and stored as embeddings in SQLite. Memories are:

- **Per-channel and per-user** — channel memories stay local, user preferences follow them across channels
- **Decay-weighted** — older memories score lower in search results (configurable via `MEMORY_DECAY_DAYS`)
- **Deduplicated** — near-identical facts (≥0.95 cosine similarity) are skipped on insert

## Finance Support

Supported assets out of the box:

- **Crypto** — BTC, ETH, SOL, XRP, ADA, DOGE, and more
- **Indices** — Dow Jones, S&P 500, Nasdaq, Russell 2000, FTSE, Nikkei, Hang Seng
- **Commodities** — Gold, Silver, Oil, Natural Gas, Copper, Wheat, Corn
- **Forex** — Major USD pairs (JPY, EUR, GBP, CNY, MYR, SGD, THB, IDR, PHP)
- **Stocks** — Any uppercase ticker symbol (e.g. AAPL, TSLA, MGNI, $PLTR)

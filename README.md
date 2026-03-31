# SiegClaw Bot

A Discord AI assistant powered by Gemini with real-time web search, financial data, and persistent vector memory.

## Features

- **Smart query routing** — classifies every message as search, finance, memory, or direct using Gemini with thinking
- **Web search** — Tavily for real-time news, weather, and current events
- **Financial data** — live prices via CoinGecko (crypto) and Yahoo Finance (stocks, indices, commodities, forex)
- **Vector memory** — extracts and stores facts from conversations using SQLite + numpy cosine similarity
- **Multi-turn awareness** — router uses recent conversation context for follow-up questions
- **Image support** — pass images alongside your message for multimodal responses
- **Hybrid context window** — adapts between count-based and time-based message fetching

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
| `MEMORY_DB_PATH` | No | `data/memories.db` | SQLite database path |
| `MEMORY_DECAY_DAYS` | No | `90` | Days before memories score lower in search |
| `TAVILY_MAX_RESULTS` | No | `5` | Number of search results per query |
| `ROUTER_ENABLED` | No | `true` | Toggle query router (false = always search + memory) |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

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
User @mentions bot
        │
        ├── fetch_context()     Discord message history (hybrid window)
        ├── classify_query()    Gemini router → SEARCH / FINANCE / MEMORY / DIRECT
        │
        ├── SEARCH   → Tavily web search
        ├── FINANCE  → CoinGecko + Yahoo Finance (fallback to Tavily)
        ├── MEMORY   → SQLite cosine similarity search
        └── DIRECT   → Gemini only (chat history as context)
        │
        └── Gemini generates response (augmented prompt)
            │
            └── Background: extract facts → embed → store in SQLite
```

## File Structure

```
bot.py              — entrypoint
config.py           — env vars, clients, prompts
context.py          — hybrid context window logic
discord_handler.py  — Discord event handling
router.py           — query classification
search.py           — Tavily web search
finance.py          — CoinGecko + Yahoo Finance
memory.py           — SQLite vector memory
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

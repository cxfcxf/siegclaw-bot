# SiegClaw

Self-hosted agentic assistant. **One Python process, two surfaces**: a
single-user **web UI** (vanilla HTML/CSS/JS over FastAPI + SSE) and a
multi-user **Discord bot** (DMs, @mentions, replies) — the Discord client
connects on startup iff `DISCORD_BOT_TOKEN` is set. Both surfaces share the
same agent core: a streaming tool-call loop against any OpenAI-compatible
provider (cloud or local), with a `soul.md` system prompt, web/browser/file
tools, MCP servers, skills, and durable mem0 memory.

## Quick start

Docker — full stack (bot + memory side containers), web UI on <http://localhost:8800>:

```bash
docker compose up --build -d
```

Four services: `siegclaw-bot` (web UI + Discord), `mem0` (tiny mem0 REST
service, `mem0svc/`), `pgvector` (vector storage), `embed` (fastembed/ONNX
embeddings — no torch, no GPU). Point `MEM0_LLM_BASE_URL` in
`docker-compose.yml` at your llama.cpp; it does the memory extraction.
*OrbStack note: if a build fails on DNS, add `--build-arg HTTP_PROXY=""`.*

Local — web UI only (plus the bot if the token is set), on <http://localhost:8080>:

```bash
pip install -e .
cp .env.example .env      # provider keys, MEM0_API_URL, DISCORD_BOT_TOKEN, …
python -m uvicorn app.main:app --port 8080
```

Providers appear automatically when usable: a cloud provider when its key is
set (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, …), local
llama.cpp when its `/v1/models` endpoint responds (no key). If a provider
doesn't list models, type a model id — the model field is a free-text combobox.

## Code map

```
app/
  main.py            FastAPI: /api/* + SSE chat + static UI; starts the Discord bot
  agent.py           streaming tool-call loop (web + Discord DM); frozen system prompt
  config.py          provider registry + env detection, liveness probe, default→fallback model resolution
  providers.py       OpenAI-compatible (async) client factory
  discord_bot.py     Discord client + on_message; DM slash commands; channel/cron use the non-streaming loop
  discord_context.py Discord history window + image/YouTube helpers
  scheduler.py       cron job runner (headless agent turn → Discord delivery)
  cronutil.py        cron-expression parsing / next-run (in HARNESS_TZ)
  memory.py          pluggable memory: mem0 REST service / in-process mem0 / simple SQLite
  storage.py         SQLite conversations (web UI + Discord DMs share one pool)
  skills.py          SKILL.md discovery + load_skill tool
  mcp_client.py      connects mcp.json servers (stdio or HTTP)
  tools/             registry, builtin (fs/bash), web (Firecrawl), browser (CamoFox), clock
web/                 vanilla HTML/CSS/JS frontend
mem0svc/             tiny custom mem0 REST service container
embed/               fastembed (ONNX) embeddings service container
soul.md              system prompt (editable from the UI)
mcp.json             MCP server definitions
skills/              your skills (skills/<name>/SKILL.md)
data/                SQLite DBs + uploads (created at runtime)
```

## How it works

### Agent core

- `agent.py:run_turn` is the streaming loop used by the web UI **and** Discord
  DMs — prompt construction, storage, and reasoning capture are identical.
  Channel @mentions and cron jobs use the non-streaming
  `discord_bot.py:run_discord_turn`.
- **Cache-friendly system prompt**: the prefix is byte-identical across turns
  so the provider's prompt cache never invalidates. The date is frozen at day
  precision at conversation start (`HARNESS_TZ`), and relevant memory is
  snapshotted once per conversation. Anything fresher comes from tools
  (`current_time`, `search_memory`).
- **Thinking is provider-aware**: llama.cpp via `chat_template_kwargs`
  (`THINK_KWARG`), DeepSeek/MiMo via `thinking.type` (+ `reasoning_effort`),
  OpenRouter via `reasoning.enabled`. Reasoning is read back from whichever
  field the provider emits (`reasoning_content` / `reasoning` /
  `reasoning_details`) and shown in the collapsible process trace.

### Models & fallback

- Every **new** conversation (web, DM, channel) starts on
  `DEFAULT_PROVIDER`/`DEFAULT_MODEL`; if that provider isn't reachable it
  starts on `FALLBACK_PROVIDER`/`FALLBACK_MODEL` (+ `FALLBACK_EFFORT`).
- Model is **per conversation**: switch it (web picker or `/model`) and it
  sticks; new conversations return to the default.
- **Send-time fallback** (`agent.py:resolve_for_turn`): if a conversation's
  provider stops serving mid-life, the turn retries then permanently switches
  that conversation to the fallback (persisted — no flapping); new
  conversations pick up the original once it's back. Liveness for keyless
  local engines is a real HTTP `/models` probe, not a TCP connect.

### Discord

- Responds to DMs, @mentions, and replies to its own messages. Channel context
  is built from **live Discord history** (hybrid time/count window — Discord is
  the source of truth, not SQLite); people are referred to by display name.
  Extra channel tools: `fetch_user_messages`, `fetch_channel_history`.
- Posts live `-#` tool-status lines, chunks replies over the 2000-char limit,
  reads image attachments, understands pasted YouTube links. Replies
  quote-reply in channels but send plain messages in DMs (1-on-1, no need).
  Optional **DM streaming** (`DISCORD_STREAM_DMS=true`, off by default): the
  reply is sent early and edited in place as tokens arrive (~1 edit/s — the
  most Discord's API allows); off, the reply lands as one complete message.
- **DM slash commands** (hidden from channel autocomplete): `/new`, `/list`,
  `/resume <ref>`, `/model` (provider → model autocomplete). `/new` and
  `/resume` confirmations are posted non-ephemerally so session boundaries
  stay visible in DM history. DMs and the web UI share one conversation pool
  (`/resume` works across surfaces). Channel @mentions are stateless and
  always use the default model.
- The shell/file tools are **withheld from Discord** (multi-user) unless
  `DISCORD_ENABLE_SHELL=true`.

### Memory ([mem0](https://github.com/mem0ai/mem0))

- Facts are **auto-extracted** in the background, debounced
  (`MEMORY_DEBOUNCE_SECONDS`, default 300s of quiet), including facts stated
  in passing ("for people like me around 42…" → "User is around 42").
  Contradictions are reconciled (add/update/delete); a heuristic filter drops
  conversation-log junk.
- **Scoping**: the web UI and Discord DMs share one default scope (both are
  the owner's). Channel memory is scoped **per server**
  (`discord-guild:<id>`), with facts attributed by display name ("John lives
  in NYC") so anyone's mention can recall them. Tune extraction with
  `MEM0_CUSTOM_INSTRUCTIONS` on the mem0 service.
- View / filter / add / delete facts via the **memory** button in the sidebar.
  If `MEM0_API_URL` is unset, memory falls back to a simple all-facts SQLite
  store.

### Scheduled jobs

Defined in the web UI (**cron** button) — name + prompt + 5-field cron
(evaluated in `HARNESS_TZ`) + Discord destination (channel or DM) — **or
conversationally**: the agent has `schedule_job` / `list_scheduled_jobs` /
`cancel_scheduled_job` tools, so "remind me tomorrow at 9am…" or "every
morning send me AI news" works from any surface. Recurring jobs use cron;
one-time jobs use an `at` timestamp and disarm after running. Each run is a
headless agent turn (default model, research tools, no shell) posted to the
target — a DM to the bot owner by default. Enable/disable/edit/delete/Run-now
in the dialog; jobs persist in SQLite and share the bot's process.

### Tools, skills, MCP

- **Builtin**: `read_file`, `write_file`, `edit_file`, `list_dir`, `bash`
  (workspace = `WORKSPACE_DIR`), `current_time`, `search_memory`.
- **Web**: Firecrawl (`web_search`, `web_scrape`, `web_map`, `web_crawl`) via
  `FIRECRAWL_API_URL`; `browser_use` drives a stealth CamoFox browser
  (`CAMOFOX_URL`) for JS-rendered or bot-blocked pages.
- **Skills**: Claude-style `skills/<name>/SKILL.md`; the index is always shown
  to the model, full instructions load on demand via `load_skill`.
- **MCP**: declare servers in `mcp.json`; tools are exposed as
  `mcp__<server>__<tool>`.

### Web UI

Streaming SSE chat with: a provider/model **chip in the composer** that opens
a popover (provider select + searchable model combobox, last choice
remembered); a **thinking toggle** + effort select; **image upload** (attach
or paste — needs a vision model); a collapsible **process trace** (reasoning +
nested tool calls, live activity light and timer); **per-response metrics**
(wall time, thinking time, tok/s); a **context meter** under the composer
(used vs. the model's max context, from the provider's `/models`); **stop**
mid-turn; message **copy / retry / edit-and-resend**; conversation history
grouped Today / Yesterday / Previous 7 days then exact dates; full-page
**chat search** (instant title matches + **full-text search over message
bodies** via SQLite FTS5, with highlighted snippets); dialogs for **soul.md**,
**memory**, **skills**, and **cron**.

## Configuration reference

All via `.env` (see `.env.example`) unless noted.

| Variable | What it does |
| --- | --- |
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` … | Enable a cloud provider |
| `DEFAULT_PROVIDER` / `DEFAULT_MODEL` | Model for every new conversation (blank model = whatever a local engine serves) |
| `FALLBACK_PROVIDER` / `FALLBACK_MODEL` / `FALLBACK_EFFORT` | Used when the default provider is down |
| `SEND_FALLBACK_RETRIES` / `SEND_FALLBACK_RETRY_DELAY` / `PROVIDER_LIVENESS_CACHE_TTL` | Send-time fallback tuning |
| `DISCORD_BOT_TOKEN` | Run the Discord bot (omit for web UI only) |
| `DISCORD_ENABLE_SHELL` | Allow shell/file tools from Discord (default off) |
| `DISCORD_STREAM_DMS` | Edit-in-place streaming for DM replies (default off) |
| `HARNESS_TZ` | IANA timezone for the frozen prompt date and cron |
| `WORKSPACE_DIR` | Working directory for the bash/file tools |
| `MEM0_API_URL` | mem0 service URL (unset → simple SQLite memory) |
| `MEMORY_DEBOUNCE_SECONDS` | Quiet time before background fact extraction (default 300) |
| `MEM0_CUSTOM_INSTRUCTIONS` / `MEM0_LLM_BASE_URL` | Extraction prompt / LLM for the mem0 service (docker-compose env) |
| `FIRECRAWL_API_URL` | Firecrawl backend for web tools |
| `CAMOFOX_URL` | Stealth-browser backend for `browser_use` |
| `THINK_KWARG` | llama.cpp chat-template kwarg for the thinking toggle |

## Notes

- The web UI is **single-user and local**; `bash`/file tools run arbitrary
  commands in `WORKSPACE_DIR`. Don't expose it to untrusted networks.
- Tool-calling reliability depends on the model — strong cloud models and
  tool-tuned local models work best.

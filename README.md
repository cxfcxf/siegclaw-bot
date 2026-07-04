# SiegClaw

Self-hosted agentic assistant. **One Python process, two surfaces**: a
single-user **web UI** (vanilla HTML/CSS/JS over FastAPI + SSE) and a
multi-user **Discord bot** (DMs, @mentions, replies) — the Discord client
connects on startup iff `DISCORD_BOT_TOKEN` is set. Both surfaces share the
same agent core: a streaming tool-call loop against any OpenAI-compatible
provider (cloud or local), with web/browser/file tools, MCP servers, and an
**LLM-Wiki** — the model's entire durable knowledge (system prompt, memory,
lessons) as markdown pages it reads and rewrites itself.

## Quick start

Docker — single container, web UI on <http://localhost:8800>:

```bash
docker compose up --build -d
```

*OrbStack note: if a build fails on DNS, add `--build-arg HTTP_PROXY=""`.*

Local — web UI only (plus the bot if the token is set), on <http://localhost:8080>:

```bash
pip install -e .
cp .env.example .env      # provider keys, DISCORD_BOT_TOKEN, …
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
  agent.py           streaming tool-call loop (web + Discord DM); wiki-based system prompt
  config.py          provider registry + env detection, liveness probe, default→fallback model resolution
  providers.py       OpenAI-compatible (async) client factory
  discord_bot.py     Discord client + on_message; DM slash commands; channel/cron use the non-streaming loop
  discord_context.py Discord history window + image/YouTube helpers
  scheduler.py       cron job runner (headless agent turn → Discord delivery)
  cronutil.py        cron-expression parsing / next-run (in HARNESS_TZ)
  wiki.py            the LLM-Wiki: page storage, prompt index, read/search/write tools
  storage.py         SQLite conversations (web UI + Discord DMs share one pool)
  mcp_client.py      connects mcp.json servers (stdio or HTTP)
  tools/             registry, builtin (fs/bash), web (Firecrawl), browser (CamoFox), clock
web/                 vanilla HTML/CSS/JS frontend
wiki/                the LLM-Wiki. Only home.md (the stock system prompt) is committed;
                     every other page is personal runtime data the model writes for its
                     owner (gitignored)
mcp.json             MCP server definitions
data/                SQLite DBs + uploads (created at runtime; media lives in
                     uploads/<conversation-id>/ — a chat owns its files, so
                     deleting the chat deletes them)
```

## How it works

### Agent core

- `agent.py:run_turn` is the streaming loop used by the web UI **and** Discord
  DMs — prompt construction, storage, and reasoning capture are identical.
  Channel @mentions and cron jobs use the non-streaming
  `discord_bot.py:run_discord_turn`.
- **Cache-friendly system prompt**: the prefix (wiki home page + day-frozen
  date + wiki index) is byte-identical across turns so the provider's prompt
  cache never invalidates. Anything fresher comes from tools (`current_time`,
  `read_wiki_page`, `search_wiki`); only an actual wiki edit busts the cache —
  the price of learning, paid once per edit.
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
- **Voice messages in DMs**: send a Discord voice message (or any audio file)
  and it's transcribed locally by the same in-process faster-whisper the web
  mic uses, then answered as a normal text reply. The recording is stored on
  the conversation, so opening the chat in the web UI shows your clip above
  its transcript (and the reply can be read aloud from there on demand).
- **DM slash commands** (hidden from channel autocomplete): `/new`, `/resume`,
  `/model`, and `/rename` — all argument-free. `/resume` opens a dropdown of
  conversations (with a group filter first when groups exist), `/model` a
  provider dropdown then that provider's models (◀ ▶ paged past Discord's
  25-option cap), and `/rename` pops a native **modal form** to rename the
  active chat and set its group. `/model` right after `/new` works: it creates
  the pending conversation and sets its model.
  `/new` and `/resume` confirmations are posted non-ephemerally so session
  boundaries stay visible in DM history. DMs and the web UI share one
  conversation pool (`/resume` works across surfaces). Channel @mentions are
  stateless and always use the default model.
- The shell/file tools are **withheld from Discord** (multi-user) unless
  `DISCORD_ENABLE_SHELL=true`.

### LLM-Wiki (memory + system prompt + skills, unified)

Everything the assistant durably knows is a directory of markdown pages
(`wiki/`, one file per page with a one-line `summary` in frontmatter) that
**the model curates itself** — no extraction pipeline, no vector store, no
side containers. The model is the memory system:

- **`home.md` is the system prompt** (the old soul.md): identity, principles,
  and the standing instruction to read the wiki before answering and write
  back what it learns.
- **Every other page** appears in every system prompt as an index line
  (name + summary). The model calls `read_wiki_page` when a summary is
  relevant, `search_wiki` (keyword search) when unsure which page, and
  `write_wiki_page` to save durable facts, corrections, and procedures that
  worked. A write replaces the whole page, so it's told to read-then-fold-in.
- **Write access is owner-only**: web UI and Discord DM turns can write;
  channel mentions and cron jobs get read/search only (the wiki feeds every
  future system prompt — channel users must not inject into it).
- Pages are plain files: browse/edit/delete them in the **wiki** tab in the
  sidebar, or with any editor (`wiki/` is a directory mount — edits apply on
  the next turn, no restart).
- **Only `home.md` is committed** — it's the stock system prompt a fresh
  install starts from (created automatically if missing). Everything else in
  `wiki/` is personal: pages the model accumulates about *its* owner. Those
  are gitignored, like `data/`.

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

Every run is also **saved as a resumable conversation** — titled
`<date time> — <job>`, grouped under a
collapsible `Cron: <job>` sidebar folder, full tool trail included — so you
can open it and ask follow-up questions about the result. DM-targeted jobs
additionally switch the recipient's **active DM session** to the new run:
just replying in the DM asks follow-ups against the briefing (`/resume`
switches back). Each job keeps its newest `CRON_KEEP_RUNS` runs (default 30;
older ones are auto-deleted — move a run out of its group to keep it).

### Tools & MCP

- **Builtin**: `read_file`, `write_file`, `edit_file`, `list_dir`, `bash`
  (workspace = `WORKSPACE_DIR`), `current_time`.
- **Wiki**: `read_wiki_page`, `search_wiki`, `write_wiki_page` (write is
  owner-surfaces-only — see the LLM-Wiki section).
- **Web**: Firecrawl (`web_search`, `web_scrape`, `web_map`, `web_crawl`) via
  `FIRECRAWL_API_URL`; `image_search` via `IMAGE_SEARCH_URL` (the searchmw
  middleware's `/images` — Brave image API + Tavily with 429 failover) returns
  direct image URLs the model embeds in replies (Discord auto-embeds bare
  URLs; the web UI renders markdown images); `browser_use` drives a stealth
  CamoFox browser (`CAMOFOX_URL`) for JS-rendered or bot-blocked pages.
- **MCP**: declare servers in `mcp.json`; tools are exposed as
  `mcp__<server>__<tool>`.

### Web UI

Streaming SSE chat with: a provider/model **chip in the composer** that opens
a popover (provider select + searchable model combobox, last choice
remembered); a **thinking toggle** + effort select; **image upload** (attach
or paste — needs a vision model); **voice input** (mic button — records in
the browser, transcribed by in-process faster-whisper, fully local, no cloud;
`STT_MODEL` picks the size; mic access needs HTTPS or localhost); a
**read-aloud button** on every reply (hover toolbar — TTS is strictly on
demand: the first click synthesizes via edge-tts, picking an English or
Chinese voice from the reply's language, and persists the clip on the
message; after that it's play/pause); a collapsible **process trace** (reasoning +
nested tool calls, live activity light and timer); **per-response metrics**
(wall time, thinking time, tok/s); a **context meter** under the composer
(used vs. the model's max context, from the provider's `/models`); **stop**
mid-turn; message **copy / retry / edit-and-resend**; conversation history
grouped Today / Yesterday / Previous 7 days then exact dates, with **rename**
and **user-defined groups** (hover a chat or group header for its **⋮ menu**:
rename / move to group / delete, all destructive actions behind an in-app
confirm dialog; groups render as collapsible sections and sync with Discord's
`/rename` and `/resume` filter). Groups are first-class: create one empty
("+ new group"), it persists with no chats in it; assigning opens a dropdown
of all groups with the input as filter (pick, create, or remove in one
click); renaming a group onto an existing name **merges** the two; deleting a
group only removes the folder — its chats move back to the date-sectioned
root list, nothing is deleted; full-page **chat search** (instant title
matches + **full-text search over message bodies** via SQLite FTS5, with
highlighted snippets); sidebar tabs for the **wiki** (browse/edit/delete the
LLM-Wiki pages, including `home` = the system prompt) and **cron** (scheduled
jobs); a **theme switch** (system / light / dark, top-right corner); and a
**per-reply model tag** on every assistant message (`▸ MODEL ·
provider/model` — chats always *start* on the freshly resolved default model;
resuming never drags you back to whatever the chat used last, on web or
Discord, so the tag is how you compare models across a conversation).

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
| `STT_MODEL` | faster-whisper model for the mic button + DM voice messages (tiny/base/small/medium, default base) |
| `TTS_VOICE` / `TTS_VOICE_ZH` / `TTS_MAX_CHARS` | edge-tts voices for the read-aloud button — default and Chinese, picked per reply by language — and clip length cap |
| `HARNESS_TZ` | IANA timezone for the frozen prompt date and cron |
| `CRON_KEEP_RUNS` | Newest cron-run conversations kept per job (default 30) |
| `WORKSPACE_DIR` | Working directory for the bash/file tools |
| `WIKI_DIR` | LLM-Wiki pages directory (default `./wiki`) |
| `FIRECRAWL_API_URL` | Firecrawl backend for web tools |
| `IMAGE_SEARCH_URL` | searchmw middleware for `image_search` (its `/images` endpoint) |
| `CAMOFOX_URL` | Stealth-browser backend for `browser_use` |
| `THINK_KWARG` | llama.cpp chat-template kwarg for the thinking toggle |

## Notes

- The web UI is **single-user and local**; `bash`/file tools run arbitrary
  commands in `WORKSPACE_DIR`. Don't expose it to untrusted networks.
- Tool-calling reliability depends on the model — strong cloud models and
  tool-tuned local models work best.

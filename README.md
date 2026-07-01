# SiegClaw

A self-hosted agentic assistant that runs **two surfaces from one process**: a
single-user **web UI** and a multi-user **Discord bot** (DMs, @mentions, replies).
Multi-provider (any OpenAI-compatible API, cloud or local), with a `soul.md` system
prompt, real tools (web search + scrape, stealth browser, optional files/bash),
**MCP** servers, **skills**, and durable **mem0** memory.

It's a reverse-merge of the `chat-harness` agent (web UI + tool loop) with the
original `siegclaw-bot` Discord bot — the harness is the agent, Discord is an
adapter on top. The Discord client connects on startup whenever `DISCORD_BOT_TOKEN`
is set; without it you get the web UI alone.

## Features

- **Discord bot (same process as the web UI).** Replies to DMs, @mentions in a
  channel, and replies to its own messages. Builds context from **live Discord
  history** (a hybrid time/count window — Discord is the source of truth, not the
  SQLite store), refers to people by name, posts live `-#` tool-status lines, chunks
  replies over Discord's 2000-char limit, reads image attachments, and understands
  pasted YouTube links. Extra Discord-only tools `fetch_user_messages` /
  `fetch_channel_history` read older history on demand. Channel memory is scoped
  **per server** (one shared pool per guild, facts attributed by display name —
  "John lives in NYC" — so anyone's mention can recall them); DM turns share the
  web UI's default scope (the DM surface is the owner's). The shell/file tools
  are withheld from Discord unless
  `DISCORD_ENABLE_SHELL=true`. **DM slash commands** (DM-only, hidden from channel
  autocomplete): `/new` (start a fresh conversation — reports the model + context
  window it'll use), `/list`, `/resume <ref>`, and `/model` (switch the active DM
  conversation's model, with **autocomplete**: pick a provider, then pick from that
  provider's models — same typeahead filter as the web UI). Channel @mentions are
  stateless and always use the default model.
- **One default model, shared by every surface.** A configurable **default →
  fallback** order (`config.py:resolve_default_model`): every *new* conversation —
  web UI, Discord DM, Discord channel mention — starts on `DEFAULT_PROVIDER`/`DEFAULT_MODEL`
  (blank model = whatever a local engine serves), and if that provider isn't reachable
  right now it falls back to `FALLBACK_PROVIDER`/`FALLBACK_MODEL` (with `FALLBACK_EFFORT`
  for reasoning providers). Model is **per conversation**: switch it (web picker or
  `/model`) and it sticks to that conversation; a new conversation returns to the default.
  **Send-time fallback** (`agent.py:resolve_for_turn`): if a conversation's provider
  isn't serving or its model isn't valid for it (e.g. you stop the local engine, or a
  stale model id is left on a switched provider), the turn retries a few times then
  switches that conversation to the fallback model for the rest of the session —
  persisted, so it won't flap back if the original returns; a *new* conversation picks
  the original again once it's back. Whether a keyless local engine is "serving" is a
  real HTTP `/models` probe (a TCP connect can't see through a port-forward, or a
  server still listening with no model loaded). Knobs: `SEND_FALLBACK_RETRIES`,
  `SEND_FALLBACK_RETRY_DELAY`, `PROVIDER_LIVENESS_CACHE_TTL`.
- **Scheduled jobs (cron → Discord).** Define timed jobs in the web UI (**cron**
  button in the sidebar footer): a name, a prompt, a 5-field **cron** expression
  (evaluated in `HARNESS_TZ`), and a Discord destination — a server channel the bot
  can post to, or a DM (by user ID). At each scheduled time the job runs a headless
  agent turn (default model + full research tools — web/browser/skills/MCP/memory, no
  shell) and posts the result to its target. Enable/disable, edit, delete, or **Run
  now** from the same dialog; last-run status and the next fire time are shown inline.
  Jobs persist in SQLite and survive restarts. The scheduler shares the bot's process
  and loop.
- **Multi-provider, switchable in the UI.** Any OpenAI-compatible endpoint:
  OpenAI, OpenRouter, DeepSeek, Xiaomi MiMo, and local llama.cpp. Providers are
  auto-detected from `.env` — set a key (or run a local engine) and it appears in
  the dropdown.
- **Tools.** `read_file`, `write_file`, `edit_file`, `list_dir`, `bash`, plus
  the full Firecrawl surface (`web_search`, `web_scrape`, `web_map`,
  `web_crawl`) and `browser_use` — a stealth (camoufox) browser that renders JS
  and bypasses bot detection, for pages Firecrawl can't.
- **Image upload (vision)** — attach images (📎 button, or paste into the box) and
  the model can read them. Uploads are saved under `data/uploads/` and sent to the
  model as base64 in the OpenAI-compatible multimodal format. Requires a
  vision-capable model (e.g. a llama.cpp server started with an `mmproj` projector).
- **Thinking toggle** — a compact lightbulb in the composer (per request) turns
  model reasoning on/off. The mechanism is provider-aware: `chat_template_kwargs`
  (`THINK_KWARG`) for llama.cpp, `thinking.type` (+ `reasoning_effort` for
  DeepSeek) for DeepSeek/MiMo, and `reasoning.enabled` for OpenRouter. Reasoning is read back from whichever field
  the provider emits (`reasoning_content`, `reasoning`, or structured
  `reasoning_details`) and shown inside the collapsible **process trace** (below).
- **Process trace** — thinking, tool calls/results, and memory activity for a turn
  are collected into a single collapsible "process" block (collapsed by default)
  with a **live activity light** that pulses while the turn is working, and a **live
  timer** in its header that counts the whole turn (thinking + tools + streaming) and
  settles on the total. Tool calls **nest under the reasoning step that triggered
  them**, so collapsing a Thinking block hides its tools too. Reasoning is saved with
  the conversation.
- **Per-response metrics** — each answer carries a small chip (after the copy icon)
  showing total wall time, a live timer that counts from 0 while the turn runs,
  the time spent thinking, and **tok/s** (from the server's streamed usage when
  available).
- **Context meter** — a compact `X% · YK ctx` line under the composer shows how
  full the current conversation is relative to the model's max context. Max context
  comes from the provider's `/models` endpoint (llama.cpp `meta.n_ctx`, OpenRouter
  `context_length`); used comes from each turn's `prompt_tokens`, persisted per
  conversation so it shows on reload. Turns clay at ≥80%.
- **Searchable model picker** — the model field is a combobox: type to filter
  (handy for OpenRouter's hundreds of models), arrow-key/enter to pick, or just
  type a custom model id. The last-used provider and per-provider model are
  remembered across reloads.
- **Cache-friendly system prompt (date + memory frozen)** — the system prefix
  is kept byte-identical across every turn so the prompt cache never invalidates:
  the date is stamped at **day** precision (e.g. "June 19, 2026", frozen at the
  conversation's start; set `HARNESS_TZ` to any IANA zone), and memory is
  snapshotted once per conversation into the prefix. For the things that change
  within a turn, the model calls tools instead — `current_time` for the precise
  time to the second, `search_memory` for fresher/differently-relevant facts.
- **Stop** — the send button becomes a **stop** control while a turn is streaming;
  click it (or press Enter) to abort immediately, anywhere in the turn.
- **Message actions** — hover a message for **copy**, **retry** (on your messages —
  rewinds and re-runs from that point), and **edit & resend** (only on your most
  recent message — rewinds and drops the text + images back into the composer to
  change and send again).
- **Scroll-jump** — a floating button that jumps to the first message when you're at
  the bottom, and flips to jump back to the newest message once you scroll up.
- **soul.md** — the system prompt, editable from the UI (the **soul** button in the
  sidebar footer).
- **Memory** ([mem0](https://github.com/mem0ai/mem0)) — durable facts the
  assistant keeps across chats. mem0 **auto-extracts** facts (no need to say
  "remember") and reconciles contradictions (add/update/delete). To keep the
  system prompt cacheable, the **semantically relevant facts are snapshotted once
  at the conversation's start** into the prefix and reused for that conversation;
  the model pulls fresher or differently-relevant ones on demand via the
  `search_memory` tool. Extraction runs **debounced in the background**
  (`MEMORY_DEBOUNCE_SECONDS`, default 300s of chat quiet), catches facts stated
  **in passing** ("for people like me around 42…" → "User is around 42"), and in
  multi-speaker channel transcripts **attributes facts by display name** ("John
  lives in NYC") — tune via `MEM0_CUSTOM_INSTRUCTIONS` on the mem0 service. A
  heuristic filter keeps out conversation-log junk like "User was told…". Memory
  runs as a small **self-hosted stack of side containers** so the bot image stays
  lean (no torch): a tiny custom **mem0 service** (`mem0svc/`) + **pgvector**
  (Postgres) for storage + a featherweight **fastembed** embeddings service
  (`embed/`, ONNX Runtime — no torch, no GPU). Extraction runs on your llama.cpp;
  the bot talks to mem0 over HTTP via `MEM0_API_URL`. The web UI and Discord DMs
  share one default scope (both are the owner's); **channel memory is scoped per
  server**. View, **filter**, add, or delete facts on the card board behind the
  **memory** button in the sidebar footer. If `MEM0_API_URL` is unset it falls
  back to a simple all-facts SQLite store.
- **History / resume** — every conversation is saved; pick one from the sidebar to
  resume it with full context. Grouped by Today / Yesterday / Previous 7 days, then
  by exact date. Stored in `data/conversations.db`.
- **Chat search** — a full-page search view (the **Search chats** row under
  New conversation, or the magnifier in the collapsed rail's grouped pill):
  type to filter every conversation by title, arrow-keys + Enter (or click) to
  open, Esc to go back to the chat.
- **Skills** — Claude-style `skills/<name>/SKILL.md` folders. The index is shown
  to the model; full instructions load on demand via the `load_skill` tool.
- **MCP** — declare servers in `mcp.json` (stdio or HTTP); their tools are exposed
  to the agent, namespaced `mcp__<server>__<tool>`.
- **Streaming** chat over SSE with live tool-call/result rendering and SQLite
  conversation history. The provider/model picker is a compact chip in the
  composer bar (next to send, Claude/Gemini-style) that opens a popover with the
  provider select and the searchable model combobox; the live timer signals activity, and a small floating pill surfaces only
  one-off notices (errors, "stopped"). Assistant markdown renders GFM tables,
  including tab-separated rows that local models sometimes emit raw.

## Setup

Local (web UI; add `DISCORD_BOT_TOKEN` to `.env` to also run the bot):

```bash
pip install -e .          # fastapi uvicorn openai httpx python-dotenv pyyaml mcp discord.py
cp .env.example .env      # edit: provider keys, Firecrawl/Camofox URLs, MEM0_API_URL, DISCORD_BOT_TOKEN
python -m uvicorn app.main:app --port 8080
```

Open <http://localhost:8080>.

Docker (bot + mem0 + pgvector + embed):

```bash
docker compose up --build -d   # web UI on http://localhost:8800; bot connects if DISCORD_BOT_TOKEN is set
```

Four services: `siegclaw-bot` (web UI + Discord), `mem0` (tiny mem0 REST service),
`pgvector` (vector storage), `embed` (fastembed embeddings). The `mem0` service is
wired to use your llama.cpp for extraction and the `embed` service for embeddings.
Point `MEM0_LLM_BASE_URL` (in `docker-compose.yml`) at your llama.cpp. On OrbStack,
build with `--build-arg HTTP_PROXY=""` if DNS fails; the containers also disable the
injected proxy so internal/LAN calls work.

### Configuring providers

Edit `.env`. A provider shows up when usable:

- **Cloud** (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`): set the key.
- **Local** (llama.cpp): just run the engine; it's detected if its
  `/v1/models` endpoint responds (no key).

If a provider doesn't list models, just type a model id into the model field
(it's a free-text combobox).

## Layout

```
app/
  config.py      provider registry + env detection (HTTP liveness probe, day-long model cache, send-time fallback)
  providers.py   OpenAI-compatible (async) client factory
  agent.py       streaming tool-call loop (web + Discord DM); frozen system prompt
  discord_bot.py Discord client, on_message; DM runs the shared loop, channel/cron use the non-streaming one
  discord_context.py Discord history window + image/YouTube helpers
  scheduler.py   cron job runner (headless agent turn → Discord delivery)
  cronutil.py    cron-expression parsing/next-run (in HARNESS_TZ)
  tools/         registry, builtin (fs/bash), web (Firecrawl), browser (CamoFox), clock (current_time)
  mcp_client.py  connects mcp.json servers
  skills.py      SKILL.md discovery + load_skill tool
  memory.py      pluggable memory: mem0 REST service / in-process mem0 / simple
  storage.py     SQLite conversations (web UI + Discord DM share one pool)
  main.py        FastAPI: /api/* + SSE chat + static UI; starts the Discord bot
embed/           featherweight fastembed (ONNX) embeddings service container
mem0svc/         tiny custom mem0 REST service container (pgvector + llama.cpp + embed)
web/             vanilla HTML/CSS/JS frontend
soul.md          system prompt
mcp.json         MCP server definitions
skills/          your skills (SKILL.md folders)
```

## Notes

- The web UI is single-user and local. The `bash`/file tools run arbitrary
  commands in `WORKSPACE_DIR` — intentional for the web UI, but **withheld from
  Discord** (multi-user) unless `DISCORD_ENABLE_SHELL=true`. Don't expose the web UI
  to untrusted networks.
- Tool-calling reliability depends on the model. Strong cloud models and
  tool-tuned local models work best; weak local models may not call tools well.
- **CamoFox** (`CAMOFOX_URL`) is the stealth-browser backend, driven directly by
  the `browser_use` tool for JS-rendered or bot-blocked pages. Firecrawl
  (`FIRECRAWL_API_URL`) handles plain search/scrape/map/crawl.

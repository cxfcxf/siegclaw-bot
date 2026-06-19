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
  `fetch_channel_history` read older history on demand. Memory is scoped **per
  Discord user**. The shell/file tools are withheld from Discord unless
  `DISCORD_ENABLE_SHELL=true`. Set `DISCORD_PROVIDER`/`DISCORD_MODEL` to choose the
  model (defaults to the first detected provider).
- **Multi-provider, switchable in the UI.** Any OpenAI-compatible endpoint:
  OpenAI, OpenRouter, Groq, and local engines (Ollama, LM Studio, vLLM,
  llama.cpp). Providers are auto-detected from `.env` — set a key (or run a local
  engine) and it appears in the dropdown.
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
  (`THINK_KWARG`) for local engines (llama.cpp/Ollama/LM Studio), and
  `reasoning.enabled` for OpenRouter. Reasoning is read back from whichever field
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
- **Current date/time in context** — every turn's system prompt is stamped with the
  current date and time (US Pacific by default, DST-aware; set `HARNESS_TZ` to any
  IANA zone), so the model never wastes a turn searching for "what year is it".
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
  "remember"), retrieves only the **semantically relevant** ones into the prompt,
  and reconciles contradictions (add/update/delete). A heuristic filter keeps out
  conversation-log junk like "User was told…". mem0 runs as its **own container**
  (a REST service the bot talks to over HTTP via `MEM0_API_URL`) so this image stays
  lean — point its extraction LLM at your llama.cpp. The web UI shares one default
  scope; Discord scopes memory per user. View/add/delete via the **memory** button
  in the sidebar footer. If `MEM0_API_URL` is unset it falls back to a simple
  all-facts SQLite store (or the in-process mem0 library if you install its deps).
- **History / resume** — every conversation is saved; pick one from the sidebar to
  resume it with full context. Stored in `data/conversations.db`.
- **Skills** — Claude-style `skills/<name>/SKILL.md` folders. The index is shown
  to the model; full instructions load on demand via the `load_skill` tool.
- **MCP** — declare servers in `mcp.json` (stdio or HTTP); their tools are exposed
  to the agent, namespaced `mcp__<server>__<tool>`.
- **Streaming** chat over SSE with live tool-call/result rendering and SQLite
  conversation history. Provider and model selectors live in the sidebar (no top
  bar); the live timer signals activity, and a small floating pill surfaces only
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

Docker (bot + mem0 service):

```bash
docker compose up --build      # web UI on http://localhost:8800; bot connects if DISCORD_BOT_TOKEN is set
```

The `mem0` service config (extraction LLM → llama.cpp, embedder, vector store) and
its image/tag should be confirmed against mem0's current self-hosting docs.

### Configuring providers

Edit `.env`. A provider shows up when usable:

- **Cloud** (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`): set the key.
- **Local** (Ollama, LM Studio): just run the engine; it's detected if its
  `/v1/models` endpoint responds (no key).
- **Custom** OpenAI-compatible endpoint: set `LOCAL_OPENAI_BASE_URL` (+
  `LOCAL_OPENAI_API_KEY`).

If a provider doesn't list models, just type a model id into the model field
(it's a free-text combobox).

## Layout

```
app/
  config.py      provider registry + env detection
  providers.py   OpenAI-compatible (async) client factory
  agent.py       streaming tool-call loop (web) + shared reasoning helper
  discord_bot.py Discord client, on_message, non-streaming tool loop, registry
  discord_context.py Discord history window + image/YouTube helpers
  tools/         registry, builtin (fs/bash), web (Firecrawl), browser (CamoFox)
  mcp_client.py  connects mcp.json servers
  skills.py      SKILL.md discovery + load_skill tool
  memory.py      pluggable memory: mem0 REST service / in-process mem0 / simple
  storage.py     SQLite conversations (web UI only)
  main.py        FastAPI: /api/* + SSE chat + static UI; starts the Discord bot
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

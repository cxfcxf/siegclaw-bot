"""FastAPI application: web UI, provider listing, SSE chat, conversation CRUD."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Make our own loggers (siegclaw.*) visible on stdout alongside uvicorn's, and
# surface discord.py warnings/errors. Configured here so it survives uvicorn's
# own logging setup (uvicorn only owns the "uvicorn.*" loggers).
_sc = logging.getLogger("siegclaw")
if not _sc.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    _sc.addHandler(_h)
    _sc.setLevel(logging.INFO)
    _sc.propagate = False
logging.getLogger("discord").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import storage, stt, tts, wiki
from .agent import build_registry, resolve_for_turn, run_turn
from .config import (
    DISCORD_BOT_TOKEN,
    UPLOADS_DIR,
    detect_providers,
    resolve_default_model,
)
from .cronutil import describe as cron_describe, is_valid as cron_is_valid, next_run_after
from .mcp_client import MCPManager
from .scheduler import Scheduler

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

mcp_manager = MCPManager()


# Holds the live Discord client (or None) so the scheduler and /api/discord/*
# endpoints can reach it. Set in lifespan.
_runtime: dict = {"discord_client": None, "scheduler": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    wiki.ensure_wiki()
    status = await mcp_manager.start()
    if status:
        print("[mcp] " + " | ".join(status))

    # If a Discord token is configured, run the bot in this same process/loop
    # alongside the web UI. The web UI works fine without one.
    discord_client = None
    discord_task = None
    if DISCORD_BOT_TOKEN:
        from .discord_bot import create_client

        discord_client = create_client(mcp_manager)
        discord_task = asyncio.create_task(discord_client.start(DISCORD_BOT_TOKEN))
        _runtime["discord_client"] = discord_client
        print("[discord] starting bot")

    # Scheduler runs regardless of Discord (jobs can be managed in the web UI),
    # but delivery requires a connected bot.
    scheduler = Scheduler(lambda: _runtime["discord_client"], mcp_manager)
    scheduler.start()
    _runtime["scheduler"] = scheduler
    print("[scheduler] started")

    yield

    await scheduler.stop()
    if discord_client is not None:
        await discord_client.close()
    if discord_task is not None:
        discord_task.cancel()
    await mcp_manager.stop()


app = FastAPI(title="SiegClaw", lifespan=lifespan)


@app.middleware("http")
async def bust_asset_cache(request, call_next):
    """Local frontend assets are under active development — never let the browser
    serve a stale copy. HTML, CSS, and JS all bypass cache."""
    response = await call_next(request)
    if request.url.path.endswith((".html", ".css", ".js")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# --- Models ----------------------------------------------------------------
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    provider: str
    model: str
    message: str
    images: list[str] | None = None  # '/uploads/<file>' refs from /api/upload
    think: bool = True               # toggle model reasoning per request
    effort: str | None = None        # reasoning effort for providers that support it


class NewConversation(BaseModel):
    provider: str
    model: str
    title: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None   # None = unchanged
    group: str | None = None   # None = unchanged; "" = remove from its group


class WikiPageUpdate(BaseModel):
    summary: str = ""
    content: str


# --- Providers -------------------------------------------------------------
@app.get("/api/providers")
def api_providers():
    providers = detect_providers()
    default = resolve_default_model({p.id: p for p in providers})
    return {
        "providers": [
            {
                "id": p.id,
                "name": p.name,
                "base_url": p.base_url,
                "models": p.models,
                "model_context": p.model_context,
                "effort_levels": p.effort_levels,
            }
            for p in providers
        ],
        # The model a new conversation starts on, shared with Discord.
        "default": (
            {"provider": default[0], "model": default[1], "effort": default[2]}
            if default else None
        ),
    }


# --- Conversations ---------------------------------------------------------
@app.get("/api/conversations")
def api_list_conversations():
    # groups ride along so the sidebar renders empty groups too (first-class:
    # a group can exist with no conversations in it).
    return {"conversations": storage.list_conversations(), "groups": storage.list_groups()}


class GroupCreate(BaseModel):
    name: str


@app.post("/api/groups")
def api_create_group(body: GroupCreate):
    name = body.name.strip()[:50]
    if not name:
        raise HTTPException(400, "group name cannot be empty")
    storage.create_group(name)
    return {"ok": True, "name": name}


@app.patch("/api/groups/{name}")
def api_rename_group(name: str, body: GroupCreate):
    """Rename a group (renaming onto an existing name merges the two)."""
    new = body.name.strip()[:50]
    if not new:
        raise HTTPException(400, "group name cannot be empty")
    storage.rename_group(name, new)
    return {"ok": True, "name": new}


@app.delete("/api/groups/{name}")
def api_delete_group(name: str):
    """Deletes the group only — its conversations are ungrouped, not deleted."""
    moved = storage.delete_group(name)
    return {"ok": True, "conversations_ungrouped": moved}


@app.post("/api/conversations")
def api_create_conversation(body: NewConversation):
    cid = storage.create_conversation(body.provider, body.model, body.title or "New chat")
    return {"id": cid}


@app.get("/api/conversations/{cid}")
def api_get_conversation(cid: str):
    convo = storage.get_conversation(cid)
    if not convo:
        raise HTTPException(404, "conversation not found")
    return {"conversation": convo, "messages": storage.get_messages(cid)}


@app.patch("/api/conversations/{cid}")
def api_update_conversation(cid: str, body: ConversationUpdate):
    if not storage.get_conversation(cid):
        raise HTTPException(404, "conversation not found")
    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(400, "title cannot be empty")
        storage.rename_conversation(cid, title[:100])
    if body.group is not None:
        storage.set_conversation_group(cid, body.group.strip()[:50] or None)
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
def api_delete_conversation(cid: str):
    storage.delete_conversation(cid)
    return {"ok": True}


@app.get("/api/search")
def api_search(q: str = ""):
    """Full-text search over message bodies (FTS5): per-conversation best
    matches, with \\x01…\\x02-delimited highlights in each snippet."""
    q = q.strip()
    return {"results": storage.search_messages(q) if q else []}


@app.delete("/api/conversations/{cid}/rewind/{message_id}")
def api_rewind(cid: str, message_id: str):
    """Delete the given message and everything after it. Used by retry."""
    storage.rewind_from(cid, message_id)
    return {"ok": True}


# --- LLM-Wiki ----------------------------------------------------------------
@app.get("/api/wiki")
def api_wiki_list():
    return {"pages": wiki.list_pages()}


@app.get("/api/wiki/{name}")
def api_wiki_get(name: str):
    page = wiki.read_page(name)
    if page is None:
        raise HTTPException(404, "no such wiki page")
    summary, body = page
    return {"name": name, "summary": summary, "content": body}


@app.put("/api/wiki/{name}")
def api_wiki_put(name: str, body: WikiPageUpdate):
    try:
        slug = wiki.write_page(name, body.summary, body.content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "name": slug}


@app.delete("/api/wiki/{name}")
def api_wiki_delete(name: str):
    if name == wiki.HOME:
        raise HTTPException(400, "the home page cannot be deleted")
    return {"ok": wiki.delete_page(name)}


# --- Scheduled jobs --------------------------------------------------------
class JobCreate(BaseModel):
    name: str
    prompt: str
    cron: str
    target_type: str            # 'channel' | 'dm'
    target_id: str
    enabled: bool = True


class JobUpdate(BaseModel):
    name: str | None = None
    prompt: str | None = None
    cron: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    enabled: bool | None = None


def _job_view(job: dict) -> dict:
    """Augment a stored job with a human cron hint for the UI. One-shot jobs
    (created by the schedule_job agent tool with `at`) have no cron."""
    return {**job, "cron_desc": cron_describe(job["cron"]) if job.get("cron") else "once"}


def _validate_job(cron: str | None, target_type: str, target_id: str) -> None:
    """cron=None means "not being changed" (updates to one-shot jobs pass None
    so toggling enabled doesn't trip cron validation)."""
    if cron is not None and not cron_is_valid(cron):
        raise HTTPException(400, f"Invalid cron expression: {cron!r}")
    if target_type not in ("channel", "dm"):
        raise HTTPException(400, "target_type must be 'channel' or 'dm'")
    if target_type == "channel" and not target_id:
        raise HTTPException(400, "a channel target requires a channel id")


def _normalize_target_id(target_type: str, target_id: str) -> str:
    """A DM target defaults to the bot owner ('owner' sentinel) — no id needed."""
    tid = (target_id or "").strip()
    if target_type == "dm" and not tid:
        return "owner"
    return tid


@app.get("/api/jobs")
def api_list_jobs():
    return {"jobs": [_job_view(j) for j in storage.list_jobs()]}


@app.post("/api/jobs")
def api_create_job(body: JobCreate):
    target_id = _normalize_target_id(body.target_type, body.target_id)
    _validate_job(body.cron, body.target_type, target_id)
    jid = storage.create_job(
        body.name.strip() or "Job", body.prompt, body.cron,
        body.target_type, target_id,
        next_run=next_run_after(body.cron), enabled=body.enabled,
    )
    return {"job": _job_view(storage.get_job(jid))}


@app.put("/api/jobs/{jid}")
def api_update_job(jid: str, body: JobUpdate):
    job = storage.get_job(jid)
    if job is None:
        raise HTTPException(404, "No such job")
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    target_type = fields.get("target_type", job["target_type"])
    if "target_type" in fields or "target_id" in fields:
        fields["target_id"] = _normalize_target_id(
            target_type, fields.get("target_id", job["target_id"])
        )
    _validate_job(fields.get("cron"), target_type, fields.get("target_id", job["target_id"]))
    # Re-arm the schedule when the cron changes.
    if "cron" in fields:
        fields["next_run"] = next_run_after(fields["cron"])
    storage.update_job(jid, **fields)
    return {"job": _job_view(storage.get_job(jid))}


@app.delete("/api/jobs/{jid}")
def api_delete_job(jid: str):
    storage.delete_job(jid)
    return {"ok": True}


@app.post("/api/jobs/{jid}/run")
async def api_run_job(jid: str):
    scheduler = _runtime.get("scheduler")
    if scheduler is None:
        raise HTTPException(503, "Scheduler not running")
    job = await scheduler.run_now(jid)
    if job is None:
        raise HTTPException(404, "No such job")
    return {"job": _job_view(job)}


@app.get("/api/discord/channels")
async def api_discord_channels():
    """Channels the bot can post to + the bot owner (for the job target picker).
    Empty/None when the bot isn't connected."""
    client = _runtime.get("discord_client")
    if client is None or not client.is_ready():
        return {"connected": False, "channels": [], "owner": None}
    chans = []
    for guild in client.guilds:
        me = guild.me
        for ch in guild.text_channels:
            if me and ch.permissions_for(me).send_messages:
                chans.append({
                    "id": str(ch.id),
                    "label": f"{guild.name} / #{ch.name}",
                })
    owner = None
    try:
        info = await client.application_info()
        if info.owner is not None:
            owner = {"id": str(info.owner.id), "name": str(info.owner)}
    except Exception:
        pass
    return {"connected": True, "channels": chans, "owner": owner}


# --- Image upload ----------------------------------------------------------
@app.post("/api/upload")
async def api_upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(400, f"Unsupported image type '{ext}'.")
    name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / name
    dest.write_bytes(await file.read())
    return {"url": f"/uploads/{name}"}


# --- Speech-to-text (composer mic button) -----------------------------------
@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile):
    suffix = Path(file.filename or "voice.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        text = await asyncio.to_thread(stt.transcribe, path)
    except Exception as exc:
        raise HTTPException(500, f"transcription failed: {type(exc).__name__}: {exc}")
    finally:
        os.unlink(path)
    return {"text": text}


# --- Text-to-speech (read-aloud button on assistant replies) -----------------
@app.post("/api/tts/{message_id}")
async def api_tts(message_id: str):
    """Synthesize (once) and return the TTS reading of a stored assistant
    message. The clip is persisted on the row, so replaying the conversation
    reuses it instead of re-synthesizing."""
    msg = storage.get_message(message_id)
    if not msg or msg.get("role") != "assistant" or not msg.get("content"):
        raise HTTPException(404, "message not found")
    if msg.get("audio"):
        return {"url": msg["audio"]}
    url = await tts.synthesize(msg["content"], msg["conversation_id"])
    if not url:
        raise HTTPException(502, "TTS synthesis failed")
    storage.set_message_audio(message_id, url)
    return {"url": url}


# --- Chat (SSE) ------------------------------------------------------------
@app.post("/api/chat")
async def api_chat(body: ChatRequest):
    cid = body.conversation_id
    if not cid or not storage.get_conversation(cid):
        title = (body.message[:60] or ("📷 Image" if body.images else "New chat"))
        cid = storage.create_conversation(body.provider, body.model, title)

    registry = build_registry(mcp_manager.tools)

    async def event_stream():
        # First event carries the (possibly new) conversation id.
        yield _sse({"type": "conversation", "id": cid})
        # Send-time fallback: if the chosen provider isn't serving, retry a few
        # times then switch this conversation to the fallback model (persisted,
        # so it sticks for the session). Tell the client when it changed.
        provider, model, effort = await resolve_for_turn(cid, body.provider, body.model, body.effort)
        if provider != body.provider or model != body.model:
            yield _sse({"type": "model", "provider": provider, "model": model, "effort": effort})
        async for event in run_turn(
            cid, provider, model, body.message, registry,
            images=body.images, think=body.think, effort=effort,
        ):
            yield _sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# --- Static mounts (after routes so /api/* wins; /uploads before catch-all) -
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

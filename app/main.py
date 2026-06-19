"""FastAPI application: web UI, provider listing, SSE chat, conversation CRUD."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
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

from . import memory, storage
from .agent import build_registry, read_soul, run_turn
from .config import (
    DISCORD_BOT_TOKEN,
    SOUL_PATH,
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
    memory.init_memory()
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


class SoulUpdate(BaseModel):
    content: str


# --- Providers -------------------------------------------------------------
@app.get("/api/providers")
def api_providers():
    providers = detect_providers()
    default = resolve_default_model()
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
    return {"conversations": storage.list_conversations()}


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


@app.delete("/api/conversations/{cid}")
def api_delete_conversation(cid: str):
    storage.delete_conversation(cid)
    return {"ok": True}


@app.delete("/api/conversations/{cid}/rewind/{message_id}")
def api_rewind(cid: str, message_id: str):
    """Delete the given message and everything after it. Used by retry."""
    storage.rewind_from(cid, message_id)
    return {"ok": True}


# --- Soul ------------------------------------------------------------------
@app.get("/api/soul")
def api_get_soul():
    return {"content": read_soul()}


@app.put("/api/soul")
def api_put_soul(body: SoulUpdate):
    SOUL_PATH.write_text(body.content)
    return {"ok": True}


# --- Skills ----------------------------------------------------------------
@app.get("/api/skills")
def api_skills():
    _, skills = build_registry(mcp_manager.tools)
    return {"skills": [{"name": s.name, "description": s.description} for s in skills.values()]}


# --- Memory ----------------------------------------------------------------
class MemoryCreate(BaseModel):
    content: str


@app.get("/api/memories")
def api_list_memories():
    return {"memories": memory.list_memories()}


@app.post("/api/memories")
def api_add_memory(body: MemoryCreate):
    return {"id": memory.add_memory(body.content)}


@app.delete("/api/memories/{mem_id}")
def api_delete_memory(mem_id: str):
    return {"ok": memory.delete_memory(mem_id)}


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
    """Augment a stored job with a human cron hint for the UI."""
    return {**job, "cron_desc": cron_describe(job["cron"]) if job.get("cron") else ""}


def _validate_job(cron: str, target_type: str, target_id: str) -> None:
    if not cron_is_valid(cron):
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
    cron = fields.get("cron", job["cron"])
    target_type = fields.get("target_type", job["target_type"])
    if "target_type" in fields or "target_id" in fields:
        fields["target_id"] = _normalize_target_id(
            target_type, fields.get("target_id", job["target_id"])
        )
    _validate_job(cron, target_type, fields.get("target_id", job["target_id"]))
    # Re-arm the schedule when the cron changes.
    if "cron" in fields:
        fields["next_run"] = next_run_after(cron)
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


# --- Chat (SSE) ------------------------------------------------------------
@app.post("/api/chat")
async def api_chat(body: ChatRequest):
    cid = body.conversation_id
    if not cid or not storage.get_conversation(cid):
        title = (body.message[:60] or ("📷 Image" if body.images else "New chat"))
        cid = storage.create_conversation(body.provider, body.model, title)

    registry, skills = build_registry(mcp_manager.tools)

    async def event_stream():
        # First event carries the (possibly new) conversation id.
        yield _sse({"type": "conversation", "id": cid})
        async for event in run_turn(
            cid, body.provider, body.model, body.message, registry, skills,
            images=body.images, think=body.think, effort=body.effort,
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

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
from .mcp_client import MCPManager

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

mcp_manager = MCPManager()


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
        print("[discord] starting bot")

    yield

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

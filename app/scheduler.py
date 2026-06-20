"""Scheduled jobs: run a prompt on a cron schedule and post the result to Discord.

The scheduler runs as an asyncio task in the same loop as the web UI and the
Discord bot (started from main.lifespan). Each tick it picks up due jobs, runs
each as a headless agent turn (default model + full research tools, no shell),
and delivers the answer to the job's Discord target (a channel or a DM).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

import discord

from . import memory, storage
from .agent import assemble_system_prompt, conversation_time_block, read_soul
from .config import resolve_default_model
from .cronutil import next_run_after
from .discord_bot import build_discord_registry, run_discord_turn, send_chunked

log = logging.getLogger("siegclaw.scheduler")

# Memory scope shared by all scheduled jobs (kept apart from web/Discord users).
CRON_SCOPE = "cron"

CRON_PREAMBLE = (
    "You are running as a scheduled (cron) job — there is no human in the loop to "
    "answer follow-up questions. Carry out the task fully using your tools, then "
    "produce a single, self-contained result suitable for posting to Discord: lead "
    "with the answer, keep it concise, and use Markdown. Do not ask questions."
)


def _cron_system_prompt(skills: dict, prompt: str) -> str:
    return assemble_system_prompt(
        skills, read_soul(), CRON_PREAMBLE,
        conversation_time_block(), memory.relevant_block(prompt, CRON_SCOPE),
    )


class Scheduler:
    def __init__(
        self,
        client_provider: Callable[[], discord.Client | None],
        mcp_manager,
        tick_seconds: float = 30.0,
    ):
        self._client_provider = client_provider
        self._mcp = mcp_manager
        self._tick = tick_seconds
        self._task: asyncio.Task | None = None
        self._running: set[str] = set()  # job ids currently executing (no overlap)

    # --- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                now = time.time()
                for job in storage.due_jobs(now):
                    if job["id"] in self._running:
                        continue
                    # Advance the schedule up front so a long run can't be picked
                    # up again on the next tick.
                    try:
                        storage.update_job(job["id"], next_run=next_run_after(job["cron"], time.time()))
                    except Exception as e:
                        log.warning("job %s: bad cron %r: %s", job["id"], job["cron"], e)
                        storage.update_job(job["id"], next_run=None)
                        continue
                    asyncio.create_task(self._run_and_record(job))
            except Exception as e:
                log.warning("scheduler tick error: %s", e)
            await asyncio.sleep(self._tick)

    # --- execution ---------------------------------------------------------- #
    async def run_now(self, job_id: str) -> dict[str, Any] | None:
        """Run a job immediately (ignores the schedule). Returns the updated job."""
        job = storage.get_job(job_id)
        if job is None:
            return None
        await self._run_and_record(job)
        return storage.get_job(job_id)

    async def _run_and_record(self, job: dict[str, Any]) -> None:
        jid = job["id"]
        if jid in self._running:
            return
        self._running.add(jid)
        try:
            status, text = await self._execute(job)
        finally:
            self._running.discard(jid)
        storage.update_job(jid, last_run=time.time(), last_status=status, last_result=text[:2000])

    async def _execute(self, job: dict[str, Any]) -> tuple[str, str]:
        """Run the prompt, deliver to Discord. Returns (status, stored_text)."""
        pm = resolve_default_model()
        if pm is None:
            return "error", "No model available — check provider config."
        provider, model, effort = pm

        registry, skills = build_discord_registry(None, None, CRON_SCOPE, self._mcp.tools)
        system = await asyncio.to_thread(_cron_system_prompt, skills, job["prompt"])
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": job["prompt"]},
        ]
        try:
            result = await run_discord_turn(provider, model, messages, registry, think=True, effort=effort)
        except Exception as e:
            log.warning("job %s failed: %s", job["id"], e)
            return "error", f"Job failed: {type(e).__name__}: {e}"

        result = (result or "").strip() or "(no output)"
        ok, err = await self._deliver(job, result)
        if not ok:
            return "error", f"Ran, but delivery failed: {err}\n\n{result}"
        return "ok", result

    async def _deliver(self, job: dict[str, Any], text: str) -> tuple[bool, str | None]:
        client = self._client_provider()
        if client is None or not client.is_ready():
            return False, "Discord client not connected"
        body = f"**⏰ {job['name']}**\n{text}"
        try:
            if job["target_type"] == "dm":
                user = await self._owner_or_user(client, job["target_id"])
                if user is None:
                    return False, "could not resolve DM recipient (bot owner)"
                dest: discord.abc.Messageable = user.dm_channel or await user.create_dm()
            else:
                tid = int(job["target_id"])
                dest = client.get_channel(tid) or await client.fetch_channel(tid)
            await send_chunked(dest, body)
            return True, None
        except Exception as e:
            log.warning("job %s delivery failed: %s", job["id"], e)
            return False, f"{type(e).__name__}: {e}"

    async def _owner_or_user(self, client: discord.Client, target_id: str):
        """Resolve a DM recipient: the bot's application owner for the 'owner'
        sentinel (or an empty id), otherwise a specific user id."""
        if not target_id or target_id == "owner":
            info = await client.application_info()
            owner = info.owner
            oid = getattr(owner, "id", None)
            if oid is None:
                return None
            return client.get_user(oid) or await client.fetch_user(oid)
        tid = int(target_id)
        return client.get_user(tid) or await client.fetch_user(tid)

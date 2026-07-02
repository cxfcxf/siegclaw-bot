"""Agent tools over the scheduled-jobs store — the same jobs the web UI's cron
dialog manages — so reminders and recurring tasks can be set conversationally
("remind me tomorrow at 9am…"). The scheduler delivers results over Discord
(a DM to the bot owner by default). A job with a cron expression recurs; a job
created with `at` has no cron and is disarmed by the scheduler after one run.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .. import storage
from ..config import HARNESS_TZ
from ..cronutil import describe, is_valid, next_run_after
from .registry import Tool


def _fmt(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, ZoneInfo(HARNESS_TZ)).strftime("%Y-%m-%d %H:%M")


def _schedule_job(name: str, prompt: str, cron: str = "", at: str = "", target_channel_id: str = "") -> str:
    cron, at = (cron or "").strip(), (at or "").strip()
    if bool(cron) == bool(at):
        return "Error: provide exactly one of `cron` (recurring) or `at` (one-time)."
    if cron:
        if not is_valid(cron):
            return f"Error: invalid cron expression {cron!r} — need 5 fields (min hour dom month dow)."
        next_run = next_run_after(cron)
    else:
        try:
            dt = datetime.strptime(at, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(HARNESS_TZ))
        except ValueError:
            return f"Error: `at` must be 'YYYY-MM-DD HH:MM' (in {HARNESS_TZ}), got {at!r}."
        next_run = dt.timestamp()
        if next_run <= datetime.now(ZoneInfo(HARNESS_TZ)).timestamp():
            return f"Error: {at} ({HARNESS_TZ}) is in the past."
    tid = (target_channel_id or "").strip()
    target_type, target_id = ("channel", tid) if tid else ("dm", "owner")
    jid = storage.create_job(name.strip() or "Job", prompt, cron, target_type, target_id, next_run=next_run)
    when = describe(cron) if cron else "once"
    dest = f"channel {target_id}" if tid else "a DM to the owner"
    return (
        f"Scheduled '{name}' (id {jid}): runs {when}, next at {_fmt(next_run)} {HARNESS_TZ}; "
        f"result goes to {dest}."
    )


def _list_jobs() -> str:
    jobs = storage.list_jobs()
    if not jobs:
        return "No scheduled jobs."
    lines = []
    for j in jobs:
        when = describe(j["cron"]) if j["cron"] else "once"
        state = "enabled" if j["enabled"] else "disabled"
        lines.append(
            f"- {j['name']} (id {j['id']}): {when}, {state}, next run {_fmt(j['next_run'])}, "
            f"last run {_fmt(j['last_run'])} ({j['last_status'] or 'never'}) — prompt: {j['prompt'][:120]}"
        )
    return f"Scheduled jobs (times in {HARNESS_TZ}):\n" + "\n".join(lines)


def _cancel_job(job_id: str) -> str:
    job = storage.get_job((job_id or "").strip())
    if job is None:
        return f"Error: no job with id {job_id!r} — use list_scheduled_jobs to see ids."
    storage.delete_job(job["id"])
    return f"Deleted job '{job['name']}' (id {job['id']})."


def job_tools() -> list[Tool]:
    return [
        Tool(
            "schedule_job",
            "Schedule a prompt to run later and deliver the result over Discord — use for reminders "
            "('remind me…') and recurring tasks ('every morning…'). Give `cron` (5-field, evaluated in "
            f"{HARNESS_TZ}) for recurring jobs, OR `at` ('YYYY-MM-DD HH:MM' {HARNESS_TZ}) to run once. "
            "Call current_time first to resolve relative times like 'tomorrow'. The prompt runs later as "
            "a fresh agent turn with research tools and no conversation context, so make it self-contained "
            "(a plain reminder: 'Remind the user to …'). Delivery defaults to a Discord DM to the owner.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short human-readable job name"},
                    "prompt": {"type": "string", "description": "Self-contained prompt to run at the scheduled time"},
                    "cron": {"type": "string", "description": "5-field cron for recurring jobs, e.g. '0 9 * * *' (omit for one-time)"},
                    "at": {"type": "string", "description": "One-time run: 'YYYY-MM-DD HH:MM' local time (omit for recurring)"},
                    "target_channel_id": {"type": "string", "description": "Discord channel id to post to instead of DMing the owner (optional)"},
                },
                "required": ["name", "prompt"],
            },
            _schedule_job,
        ),
        Tool(
            "list_scheduled_jobs",
            "List all scheduled jobs (reminders and recurring tasks) with ids, schedules, and status.",
            {"type": "object", "properties": {}},
            _list_jobs,
        ),
        Tool(
            "cancel_scheduled_job",
            "Delete a scheduled job by id (get ids from list_scheduled_jobs).",
            {
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Job id"}},
                "required": ["job_id"],
            },
            _cancel_job,
        ),
    ]

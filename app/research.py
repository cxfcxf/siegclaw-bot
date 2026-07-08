"""Deep research: a background multi-step research run producing a cited report.

Triggered by the `deep_research` tool (owner surfaces — web UI and Discord DMs):
the tool returns immediately and the run continues in the background as its own
conversation (sidebar group "Research"), driving the normal agent loop with a
research preamble and a bigger tool budget — fan out searches, scrape sources,
synthesize. When it finishes, the report is DM'd to the bot owner on Discord
(the run takes minutes; the DM is the "it's done" ping) and lives in the
conversation for the web UI, where it can be resumed for follow-up questions.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from . import storage, wiki
from .config import HARNESS_TZ, resolve_default_model
from .tools.browser import browser_tools
from .tools.clock import clock_tools
from .tools.registry import Registry, Tool
from .tools.web import web_tools

log = logging.getLogger("siegclaw.research")

# Tool-loop budget for a research run — searches, scrapes, and synthesis steps
# all draw from it. Roomier than a chat turn's cap by design.
RESEARCH_MAX_ITERATIONS = int(os.getenv("RESEARCH_MAX_ITERATIONS", "40"))

RESEARCH_GROUP = "Research"

RESEARCH_PREAMBLE = """You are running a DEEP RESEARCH task in the background — no human is in the loop, so never ask questions; make sensible assumptions and state them in the report.

Method (be systematic, this is a report, not a chat reply):
1. Break the question into 4-8 distinct search angles: sub-questions, synonyms, opposing views, recent developments. Search them all (`web_search`), not just one phrasing.
2. `web_scrape` the 5-10 most promising sources IN FULL — snippets are not evidence. Prefer primary sources (official docs, papers, filings, first-party announcements) over aggregators.
3. Cross-check: when sources disagree or a claim is surprising, search again specifically to confirm or refute it. Note disagreements rather than silently picking a side.
4. Then write the report.

Report format (GitHub-flavored Markdown):
- Start with a **TL;DR** of 3-5 sentences that directly answers the question.
- Then structured sections with meaningful headings; use tables for comparisons.
- Cite as you go with bracketed numbers [1], [2] tied to a final **Sources** section listing each URL.
- Distinguish established facts from analysis/speculation. Include concrete numbers and dates where they matter.
- Depth over padding: cover what the evidence supports, cut boilerplate.
"""


# Set from main.lifespan; returns the live Discord client (or None). Kept as an
# injected getter so this module never imports the runtime.
_discord_getter: Callable[[], object | None] = lambda: None


def configure(discord_getter: Callable[[], object | None]) -> None:
    global _discord_getter
    _discord_getter = discord_getter


def _research_registry() -> Registry:
    """Research runs with the reading tools only: web + browser + clock +
    read-only wiki (owner context can shape a report; a background job should
    never write memory or shell out)."""
    registry = Registry()
    registry.extend(clock_tools())
    registry.extend(web_tools())
    registry.extend(browser_tools())
    registry.extend(wiki.wiki_tools(writable=False))
    return registry


def start(question: str) -> dict | None:
    """Create the research conversation and launch the background run.
    Returns {"ref", "title"} for the tool's reply, or None if no model."""
    pm = resolve_default_model()
    if pm is None:
        return None
    provider, model, effort = pm
    now = datetime.now(ZoneInfo(HARNESS_TZ))
    title = f"🔬 {question.strip()}"[:80]
    cid = storage.create_conversation(provider, model, title=title)
    storage.set_conversation_group(cid, RESEARCH_GROUP)
    convo = storage.get_conversation(cid)
    asyncio.create_task(_run(cid, provider, model, effort, question, title))
    log.info("research started: %s (#%s, %s/%s)", title, convo["ref"], provider, model)
    return {"ref": convo["ref"], "title": title, "started": now.strftime("%H:%M")}


async def _run(cid: str, provider: str, model: str, effort: str | None,
               question: str, title: str) -> None:
    from .agent import run_turn  # deferred: agent imports this module's tools

    report = ""
    error: str | None = None
    try:
        async for ev in run_turn(
            cid, provider, model, question, _research_registry(),
            think=True, effort=effort, preamble=RESEARCH_PREAMBLE,
            max_iterations=RESEARCH_MAX_ITERATIONS,
        ):
            et = ev.get("type")
            if et == "token":
                report += ev.get("text", "")
            elif et == "tool_call":
                report = ""  # pre-tool narration isn't the report
            elif et == "error":
                error = ev.get("message")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    if error:
        log.warning("research %s failed: %s", cid, error)
        storage.add_message(cid, "assistant", content=f"(research run failed: {error})")
        await _dm_owner(f"🔬 **{title}** failed: {error}", cid)
        return
    log.info("research done: %s (%d chars)", title, len(report))
    await _dm_owner(f"🔬 **{title}** — report ready:\n\n{report.strip()}", cid)


async def _dm_owner(text: str, cid: str) -> None:
    """Deliver to the owner's DM; quietly skip if Discord isn't connected (the
    report still lives in the conversation for the web UI)."""
    from .discord_bot import owner_or_user, send_chunked  # deferred: import cycle

    client = _discord_getter()
    if client is None or not client.is_ready():
        return
    try:
        user = await owner_or_user(client)
        if user is None:
            return
        dest = user.dm_channel or await user.create_dm()
        await send_chunked(dest, text)
        # Make the report the active DM conversation so a plain DM asks
        # follow-ups against it (same convention as delivered cron briefings).
        storage.dm_set_active_cid(str(user.id), cid)
    except Exception as e:
        log.warning("research DM delivery failed: %s", e)


def research_tools() -> list[Tool]:
    async def deep_research(question: str) -> str:
        info = start(question)
        if info is None:
            return "Cannot start research: no model/provider is available."
        return (
            f"Deep research started in background conversation #{info['ref']} "
            f"({info['title']}). It will run for several minutes (multiple search "
            "angles, full-source reading, synthesis) and the finished cited report "
            "will be DM'd to the owner on Discord and saved in that conversation. "
            "Tell the user it's underway and where the report will arrive — do NOT "
            "wait for it or attempt to answer the research question yourself now."
        )

    return [
        Tool(
            "deep_research",
            "Launch a background deep-research run: fans out web searches across "
            "multiple angles, reads sources in full, and synthesizes a structured, "
            "cited report (takes minutes; delivered by Discord DM and saved as a "
            "conversation). Use ONLY when the user explicitly asks for deep/thorough "
            "research or a report — for ordinary questions, search and answer "
            "directly yourself.",
            {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question, self-contained with all relevant context/constraints the user gave",
                    },
                },
                "required": ["question"],
            },
            deep_research,
        ),
    ]

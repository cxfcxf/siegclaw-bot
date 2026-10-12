"""Sub-agent tool: hand a self-contained reading task to a helper agent.

`delegate` runs the shared agent loop in a fresh context with only the web and
clock tools, then returns the helper's short final answer. Everything it read
on the way — search results, full scraped pages — stays in the helper's
context and is discarded, so the calling conversation grows by a few
paragraphs instead of every page. It runs on the default model and effort, and
is read-only, so several delegations in one step run concurrently.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .. import storage
from ..config import resolve_default_model, settings
from ..providers import client_for
from .clock import clock_tools
from .registry import Registry, Tool
from .web import web_tools

SUBAGENT_PROMPT = """You are a research helper working for another AI agent. It handed you one task; nobody else reads your reply, and no one can answer questions, so make sensible assumptions.

Use the web tools to do the task properly — search, then read the actual pages; snippets are not evidence. Then reply with ONLY what the task asks for: the findings, concise (aim for under 400 words), each key fact with its source URL. Say plainly what you could not find or verify. No preamble, no offers of further help.

Today is {today}."""


def delegate_tools() -> list[Tool]:
    async def delegate(task: str) -> str:
        # Deferred: agent imports this module to build its registries.
        from ..agent import agent_loop, reasoning_extra_body

        default = resolve_default_model()
        if default is None:
            return "Error: no model is configured."
        provider, model, effort = default

        registry = Registry()
        registry.extend(clock_tools())
        registry.extend(web_tools())
        today = datetime.now(ZoneInfo(settings.HARNESS_TZ)).strftime("%A, %B %-d, %Y")
        messages = [
            {"role": "system", "content": SUBAGENT_PROMPT.format(today=today)},
            {"role": "user", "content": task},
        ]
        async for event in agent_loop(
            client_for(provider), provider, model, messages, registry,
            extra_body=reasoning_extra_body(provider, True, effort, model=model),
            max_iterations=settings.SUBAGENT_MAX_ITERATIONS,
        ):
            et = event["type"]
            if et == "usage":
                storage.add_token_usage(event["prompt_tokens"], event["completion_tokens"])
            elif et == "error":
                return f"Error: the helper failed: {event['message']}"
            elif et == "done":
                return event["text"].strip() or "(the helper returned no answer)"
        return "(the helper returned no answer)"

    return [
        Tool(
            "delegate",
            "Hand a self-contained reading or research task to a helper agent that "
            "searches and reads pages in its own context and returns only a short, "
            "sourced answer. Prefer this over scraping several pages yourself when "
            "you need facts from them rather than the pages themselves — your own "
            "context stays small. Write `task` so it stands alone: what to find, "
            "any URLs to start from, and what the answer should contain. Several "
            "delegate calls in one step run in parallel.",
            {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
            delegate,
        ),
    ]

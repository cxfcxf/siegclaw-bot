"""Harness clock tool: exact current time on demand.

The system prompt only carries the conversation's start DATE (day precision,
frozen once) so the prompt cache stays valid across turns. When the model needs
the precise time of day — to the second — it calls `current_time`, which reads
the clock live. A tool call happens AFTER the cached prefix, so requesting the
time never invalidates the cache.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import HARNESS_TZ
from .registry import Tool


def _now_stamp() -> str:
    try:
        dt = datetime.now(ZoneInfo(HARNESS_TZ))
    except Exception:
        dt = datetime.now().astimezone()
    return dt.strftime("%A, %B %-d, %Y, %-I:%M:%S %p %Z")


def clock_tools() -> list[Tool]:
    def current_time() -> str:
        return f"The current date and time is {_now_stamp()} ({HARNESS_TZ})."

    return [
        Tool(
            "current_time",
            "Get the exact current date and time, to the second, right now. The "
            "system prompt only carries this conversation's start date (day "
            "precision, frozen for caching), so call this whenever you need the "
            "precise time of day, the elapsed time since the conversation started, "
            "or to settle a time-sensitive question. Takes no arguments.",
            {"type": "object", "properties": {}},
            current_time,
        )
    ]

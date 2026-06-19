"""Cron-expression helpers for scheduled jobs.

Cron is evaluated in HARNESS_TZ (the same timezone stamped into prompts), so a
job set to "0 9 * * *" fires at 9am local time, DST-aware.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from croniter import croniter

from .config import HARNESS_TZ


def is_valid(expr: str) -> bool:
    """True if `expr` is a parseable 5-field cron expression."""
    return bool(expr) and croniter.is_valid(expr)


def next_run_after(expr: str, after_ts: float | None = None) -> float:
    """Unix timestamp of the next time `expr` fires after `after_ts` (or now)."""
    tz = ZoneInfo(HARNESS_TZ)
    base = datetime.fromtimestamp(after_ts, tz) if after_ts else datetime.now(tz)
    return croniter(expr, base).get_next(datetime).timestamp()


def describe(expr: str) -> str:
    """A short human hint for an expression (best-effort; falls back to the raw
    expression). Keeps the UI readable without a full cron-to-English library."""
    common = {
        "* * * * *": "every minute",
        "0 * * * *": "hourly",
        "0 0 * * *": "daily at midnight",
        "0 9 * * *": "daily at 9:00",
        "0 9 * * 1-5": "weekdays at 9:00",
        "0 0 * * 0": "weekly (Sunday midnight)",
    }
    return common.get(expr.strip(), expr.strip())

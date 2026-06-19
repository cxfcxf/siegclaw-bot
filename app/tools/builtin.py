"""Builtin filesystem and shell tools.

These operate inside WORKSPACE_DIR so the agent can write and run code to
analyze things. Paths are resolved relative to the workspace; absolute paths are
allowed (single-user local environment) but default to the workspace.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import BASH_TIMEOUT, WORKSPACE_DIR
from .registry import Tool

MAX_OUTPUT = 30_000  # chars returned to the model


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE_DIR / p
    return p


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... [truncated, {len(text) - MAX_OUTPUT} more chars]"
    return text


def read_file(path: str, max_lines: int | None = None) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"Error: file not found: {p}"
    if p.is_dir():
        return f"Error: {p} is a directory (use list_dir)."
    try:
        text = p.read_text(errors="replace")
    except Exception as exc:
        return f"Error reading {p}: {exc}"
    if max_lines is not None:
        lines = text.splitlines()
        text = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            text += f"\n... [{len(lines) - max_lines} more lines]"
    return _truncate(text) or "(empty file)"


def write_file(path: str, content: str) -> str:
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception as exc:
        return f"Error writing {p}: {exc}"
    return f"Wrote {len(content)} chars to {p}"


def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"Error: file not found: {p}"
    text = p.read_text(errors="replace")
    count = text.count(old)
    if count == 0:
        return f"Error: `old` string not found in {p}."
    if count > 1 and not replace_all:
        return f"Error: `old` appears {count} times; pass replace_all=true or add context."
    text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    p.write_text(text)
    return f"Edited {p} ({'all ' + str(count) if replace_all else '1'} occurrence(s) replaced)."


def list_dir(path: str = ".") -> str:
    p = _resolve(path)
    if not p.exists():
        return f"Error: path not found: {p}"
    if p.is_file():
        return f"{p} (file, {p.stat().st_size} bytes)"
    entries = []
    for child in sorted(p.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        size = "" if child.is_dir() else f" {child.stat().st_size}b"
        entries.append(f"  [{kind}] {child.name}{size}")
    listing = "\n".join(entries) or "  (empty)"
    return f"{p}:\n{listing}"


def bash(command: str, timeout: int | None = None) -> str:
    timeout = timeout or BASH_TIMEOUT
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s."
    out = proc.stdout or ""
    err = proc.stderr or ""
    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    if proc.returncode != 0:
        parts.append(f"[exit code: {proc.returncode}]")
    return _truncate("\n".join(parts)) or "(no output)"


def builtin_tools() -> list[Tool]:
    return [
        Tool(
            "read_file",
            "Read a text file. Paths are relative to the workspace unless absolute. "
            "Prefer this over cat/head/tail in bash.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                    "max_lines": {"type": "integer", "description": "Optional cap on lines returned."},
                },
                "required": ["path"],
            },
            read_file,
        ),
        Tool(
            "write_file",
            "Write (create or overwrite) a text file with the given content. "
            "Use for new files or full rewrites; for changes to existing files prefer edit_file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            write_file,
        ),
        Tool(
            "edit_file",
            "Replace an exact substring in a file (old → new). Fails if `old` appears more "
            "than once unless replace_all=true. Read the file first with read_file; use this "
            "for targeted changes instead of rewriting the whole file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
                },
                "required": ["path", "old", "new"],
            },
            edit_file,
        ),
        Tool(
            "list_dir",
            "List the contents of a directory (defaults to the workspace root). "
            "Prefer this over `ls` in bash.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            list_dir,
        ),
        Tool(
            "bash",
            "Run a shell command in the workspace directory and return stdout/stderr. "
            "Use for what read_file/write_file/edit_file/list_dir can't do — running code, "
            "git, installs, etc. Use `rg` for content search (no dedicated search tool). "
            "One logical action per call — don't batch unrelated commands. After changing "
            "code, verify via lint/typecheck/tests if any exist. For destructive operations "
            "(rm -rf, force-push, DROP TABLE, deleting files you didn't create), describe "
            "what you're about to do first. Don't fabricate outputs — if you didn't run a "
            "tool, don't claim you did.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Optional timeout in seconds."},
                },
                "required": ["command"],
            },
            bash,
        ),
    ]

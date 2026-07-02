"""LLM-Wiki: the assistant's entire durable knowledge as one self-curated
corpus of markdown pages ("system prompt learning" — the model maintains a
plain-text notebook instead of a vector store).

Layout: WIKI_DIR/<slug>.md with YAML frontmatter carrying a one-line
`summary`. Two tiers of context:
- `home` is the root page — its full body IS the system prompt (identity,
  operating principles, standing instructions).
- Every other page appears in the system prompt only as an index line
  (name + summary); the model reads pages on demand with `read_wiki_page`
  and rewrites them with `write_wiki_page` as it learns.

There is no extraction pipeline and no embedding store: the model is the
curator. Retrieval = the always-present index + `search_wiki` keyword search.
Pages are plain files, so everything the assistant "knows" is readable and
editable by the owner (web UI wiki tab, or any text editor).
"""
from __future__ import annotations

import re
from typing import Any

import yaml

from .config import WIKI_DIR
from .tools.registry import Tool

HOME = "home"

DEFAULT_HOME = """You are SiegClaw, a capable, candid personal assistant self-hosted by your owner.

You have a persistent wiki — markdown pages listed at the end of this prompt. It is your
only memory across conversations: read pages with `read_wiki_page` before answering
questions about your owner or past decisions, and keep it current with `write_wiki_page`
(when available) as you learn.

Operating principles:
- Be direct and concise. Lead with the answer; keep preamble minimal.
- Use your tools instead of guessing: search and scrape the web for current information,
  run code to compute, read the wiki for the past.
- If a tool fails, say so plainly and adapt; don't pretend it succeeded.
"""


# --------------------------------------------------------------------------- #
# Page storage
# --------------------------------------------------------------------------- #
def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")


def _page_path(slug: str):
    return WIKI_DIR / f"{slug}.md"


def _parse(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from the markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].strip()
    return {}, text.strip()


def ensure_wiki() -> None:
    """Create the wiki dir and a default home page on first run."""
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    if not _page_path(HOME).exists():
        write_page(HOME, "Root page — identity, principles, standing instructions.", DEFAULT_HOME)


def read_page(name: str) -> tuple[str, str] | None:
    """Returns (summary, body) or None."""
    path = _page_path(_slugify(name))
    if not path.exists():
        return None
    meta, body = _parse(path.read_text(errors="replace"))
    return str(meta.get("summary") or "").strip(), body


def write_page(name: str, summary: str, body: str) -> str:
    """Create or replace a page. Returns the slug."""
    slug = _slugify(name)
    if not slug:
        raise ValueError("page name must contain letters or digits")
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    meta = yaml.safe_dump(
        {"summary": " ".join((summary or "").split())},
        allow_unicode=True, sort_keys=False, default_flow_style=False,
    ).strip()
    _page_path(slug).write_text(f"---\n{meta}\n---\n\n{body.strip()}\n")
    return slug


def delete_page(name: str) -> bool:
    slug = _slugify(name)
    if slug == HOME:
        return False  # the root page is never deletable
    path = _page_path(slug)
    if not path.exists():
        return False
    path.unlink()
    return True


def list_pages() -> list[dict[str, Any]]:
    """All pages (home first), with summary and metadata for the UI/index."""
    ensure_wiki()
    pages = []
    for path in sorted(WIKI_DIR.glob("*.md")):
        meta, body = _parse(path.read_text(errors="replace"))
        pages.append({
            "name": path.stem,
            "summary": str(meta.get("summary") or "").strip(),
            "chars": len(body),
            "updated_at": path.stat().st_mtime,
        })
    pages.sort(key=lambda p: (p["name"] != HOME, p["name"]))
    return pages


def home_text() -> str:
    ensure_wiki()
    page = read_page(HOME)
    return page[1] if page and page[1] else DEFAULT_HOME


# --------------------------------------------------------------------------- #
# Prompt block + search
# --------------------------------------------------------------------------- #
def wiki_index() -> str:
    """The index block injected into every system prompt: every page except
    home (whose body is already the prompt), one line each."""
    lines = [
        "# Your wiki (persistent memory — survives across all conversations)",
        "Read a page with `read_wiki_page` when its summary is relevant; find "
        "pages by content with `search_wiki`. Pages:",
        "",
    ]
    others = [p for p in list_pages() if p["name"] != HOME]
    if not others:
        lines.append("(no pages yet)")
    else:
        lines += [f"- **{p['name']}**: {p['summary'] or '(no summary)'}" for p in others]
    return "\n".join(lines)


def search(query: str, max_pages: int = 5, max_lines: int = 4) -> str:
    """Case-insensitive keyword search over all pages. Returns matched pages
    with their matching lines — enough context to decide which page to read."""
    words = [w for w in re.findall(r"\w+", (query or "").lower()) if w]
    if not words:
        return "Error: empty query."
    hits: list[tuple[int, str, list[str]]] = []
    for p in list_pages():
        page = read_page(p["name"])
        if page is None:
            continue
        summary, body = page
        haystack_lines = [l for l in (summary + "\n" + body).splitlines() if l.strip()]
        matched = [l.strip() for l in haystack_lines if any(w in l.lower() for w in words)]
        if matched:
            hits.append((len(matched), p["name"], matched[:max_lines]))
    if not hits:
        return f"No wiki pages match '{query}'."
    hits.sort(key=lambda h: -h[0])
    out = []
    for _n, name, lines in hits[:max_pages]:
        out.append(f"## {name}\n" + "\n".join(f"> {l}" for l in lines))
    out.append("\nUse `read_wiki_page` to read a full page.")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def wiki_tools(writable: bool) -> list[Tool]:
    """read + search everywhere; write only on owner surfaces (web UI, DMs) —
    the wiki feeds every future system prompt, so channel users and cron jobs
    must not be able to write into it."""
    def read_wiki_page(name: str) -> str:
        page = read_page(name)
        if page is None:
            names = ", ".join(p["name"] for p in list_pages()) or "(none)"
            return f"Error: no wiki page named '{name}'. Pages: {names}"
        summary, body = page
        return f"# Wiki page: {_slugify(name)}\nSummary: {summary}\n\n{body}"

    def search_wiki(query: str) -> str:
        return search(query)

    tools = [
        Tool(
            "read_wiki_page",
            "Read the full content of a wiki page listed in the system prompt index. "
            "Read the relevant page BEFORE answering questions about your owner, past "
            "decisions, or anything the index says you know.",
            {"type": "object",
             "properties": {"name": {"type": "string", "description": "The page name from the index."}},
             "required": ["name"]},
            read_wiki_page,
        ),
        Tool(
            "search_wiki",
            "Keyword-search all wiki pages (your persistent memory). Use it when you're "
            "not sure which page holds a fact, preference, or past decision.",
            {"type": "object",
             "properties": {"query": {"type": "string", "description": "A few keywords, not a sentence."}},
             "required": ["query"]},
            search_wiki,
        ),
    ]
    if not writable:
        return tools

    def write_wiki_page(name: str, summary: str, content: str) -> str:
        name_slug = _slugify(name)
        if not name_slug:
            return "Error: page name must contain letters or digits."
        if not (summary or "").strip() or not (content or "").strip():
            return "Error: both summary and content are required."
        existed = _page_path(name_slug).exists()
        write_page(name_slug, summary, content)
        verb = "Updated" if existed else "Created"
        return f"{verb} wiki page '{name_slug}' ({len(content)} chars). It's in the index from the next message."

    tools.append(Tool(
        "write_wiki_page",
        "Create or update a wiki page — your persistent memory. `content` REPLACES the "
        "whole page: to update, `read_wiki_page` first and rewrite it with the change "
        "folded in (keep everything still true, drop what's outdated). Save durable "
        "facts about your owner, confirmed lessons, and procedures that worked — never "
        "speculation or conversation summaries. Prefer updating an existing page over "
        "creating a near-duplicate. The `home` page is your own system prompt: edit it "
        "only when explicitly asked to.",
        {"type": "object",
         "properties": {
             "name": {"type": "string", "description": "Short kebab-case page name, e.g. 'owner' or 'wh40k-lore'."},
             "summary": {"type": "string", "description": "One line saying what's on the page (shown in the always-visible index)."},
             "content": {"type": "string", "description": "The full page body in markdown."},
         },
         "required": ["name", "summary", "content"]},
        write_wiki_page,
    ))
    return tools

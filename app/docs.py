"""Document attachments: text extraction for files the user attaches to a chat
(PDFs and plain-text formats), so the model can read them.

A document is stored like any upload (uploads/<cid>/<uuid>.pdf) and referenced
from the message row's `docs` field as {"url", "name"} — url for serving the
original file, name for display (the stored filename is a uuid). Extraction
happens once, at upload time, into a sidecar '<file>.txt' next to the original;
every later turn of the conversation reads the sidecar instead of re-parsing
the PDF. The extracted text is injected into the user message at prompt-build
time (agent._to_api_messages) wrapped in [Attached file: ...] markers — with a
1M-context model, whole-document Q&A is just a long prompt.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import settings

log = logging.getLogger("siegclaw.docs")

# Formats we can turn into text. Everything but .pdf is read as-is.
DOC_EXT = {
    ".pdf", ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json",
    ".jsonl", ".log", ".yaml", ".yml", ".toml", ".ini", ".xml", ".html",
    ".py", ".js", ".ts", ".sh", ".sql", ".c", ".h", ".cpp", ".go", ".rs",
}


def is_doc(filename: str) -> bool:
    return Path(filename).suffix.lower() in DOC_EXT


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".txt")


def _extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # deferred: only doc uploads pay the import

        reader = PdfReader(str(path))
        return "\n\n".join(filter(None, (p.extract_text() for p in reader.pages)))
    return path.read_text(encoding="utf-8", errors="replace")


def text_for(path: Path) -> str:
    """The document's extracted text, from the sidecar cache when present.
    Raises on extraction failure — the uploader surfaces that to the user
    (a document that can't be read shouldn't attach silently)."""
    side = _sidecar(path)
    if side.is_file():
        return side.read_text(encoding="utf-8")
    text = _extract(path)
    side.write_text(text, encoding="utf-8")
    return text


def prompt_block(name: str, path: Path) -> str:
    """The document rendered for injection into a user message."""
    try:
        text = text_for(path)
    except Exception as e:
        log.warning("doc extraction failed for %s: %s", path, e)
        return f"[Attached file {name!r} could not be read: {type(e).__name__}]"
    clipped = ""
    cap = settings.DOC_MAX_CHARS
    if len(text) > cap:
        text = text[:cap]
        clipped = f"\n[... truncated at {cap} characters ...]"
    return f"[Attached file: {name}]\n{text}{clipped}\n[End of file: {name}]"

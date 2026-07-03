"""Text-to-speech for voice replies, via edge-tts (Microsoft Edge's online
neural voices — free, no API key, needs outbound network).

Used when a turn started from a voice message: the final assistant reply is
rendered to an mp3 in UPLOADS_DIR so both Discord (file attachment with an
inline player) and the web UI (<audio> element) can play it back. The default
voice is multilingual, so mixed 中文/English replies read naturally; override
with TTS_VOICE (see `edge-tts --list-voices` for options).
"""
from __future__ import annotations

import logging
import os
import re
import uuid

from .config import UPLOADS_DIR

log = logging.getLogger("siegclaw.tts")

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaMultilingualNeural")
# Keep clips a few minutes at most — nobody listens to a 20-minute reply, and
# huge inputs make edge-tts slow.
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "4000"))

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE_MD = re.compile(r"[*_`#>|]+")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")


def _speakable(markdown: str) -> str:
    """Strip Markdown down to prose worth reading aloud: code blocks, images,
    bare URLs and formatting marks go; link text stays."""
    text = _CODE_BLOCK.sub(" (code omitted) ", markdown)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _INLINE_MD.sub("", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:TTS_MAX_CHARS]


async def synthesize(markdown: str) -> str | None:
    """Render text to an mp3 under UPLOADS_DIR; returns its '/uploads/<name>'
    URL, or None if there was nothing speakable or synthesis failed (a missing
    voice clip should never fail the turn — the text reply already exists)."""
    text = _speakable(markdown)
    if not text:
        return None
    try:
        import edge_tts  # deferred: only voice turns pay the import

        name = f"tts-{uuid.uuid4().hex}.mp3"
        await edge_tts.Communicate(text, TTS_VOICE).save(str(UPLOADS_DIR / name))
        return f"/uploads/{name}"
    except Exception as e:
        log.warning("TTS synthesis failed: %s", e)
        return None

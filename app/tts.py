"""Text-to-speech via edge-tts (Microsoft Edge's online neural voices — free,
no API key, needs outbound network).

On-demand only: the web UI's read-aloud button on an assistant reply calls
/api/tts/<message_id>, which renders the reply to an mp3 in UPLOADS_DIR once
and persists the URL on the message row. Nothing is synthesized automatically.
The voice is picked per reply from its language: predominantly-中文 text gets
a native Chinese voice (TTS_VOICE_ZH), everything else the multilingual
default (TTS_VOICE — see `edge-tts --list-voices` for options).
"""
from __future__ import annotations

import logging
import os
import re
import uuid

from .config import UPLOADS_DIR

log = logging.getLogger("siegclaw.tts")

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaMultilingualNeural")
TTS_VOICE_ZH = os.getenv("TTS_VOICE_ZH", "zh-CN-XiaoxiaoNeural")
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


_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN = re.compile(r"[A-Za-z]")


def _voice_for(text: str) -> str:
    """Pick the voice per reply: a native Chinese voice when the text is
    predominantly 中文, the (multilingual) default otherwise. One hanzi carries
    about a word while English spells ~5 letters per word, so CJK chars are
    weighted 5x — a mostly-Chinese reply with sprinkled English terms still
    reads natively, while an English reply quoting a few 中文 words doesn't
    flip. Both voices can pronounce the other language for the leftovers."""
    cjk = len(_CJK.findall(text))
    if not cjk:
        return TTS_VOICE
    latin = len(_LATIN.findall(text))
    return TTS_VOICE_ZH if cjk * 5 >= latin else TTS_VOICE


async def synthesize(markdown: str, conversation_id: str) -> str | None:
    """Render text to an mp3 in the conversation's uploads directory; returns
    its '/uploads/<cid>/<name>' URL, or None if there was nothing speakable or
    synthesis failed (the caller's text already exists — a missing clip should
    never become an error)."""
    text = _speakable(markdown)
    if not text:
        return None
    try:
        import edge_tts  # deferred: only read-aloud requests pay the import

        dest_dir = UPLOADS_DIR / conversation_id
        dest_dir.mkdir(exist_ok=True)
        name = f"tts-{uuid.uuid4().hex}.mp3"
        await edge_tts.Communicate(text, _voice_for(text)).save(str(dest_dir / name))
        return f"/uploads/{conversation_id}/{name}"
    except Exception as e:
        log.warning("TTS synthesis failed: %s", e)
        return None

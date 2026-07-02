"""Local speech-to-text for the web UI mic button.

faster-whisper loaded in-process (no side service, no cloud): lazily on the
first transcription, then a singleton for the life of the process. Model
files download once into DATA_DIR/stt-models — bind-mounted, so they survive
rebuilds. CPU int8 inference; `base` transcribes voice-input-length clips in
a couple of seconds, `small`/`medium` (STT_MODEL) trade speed for accuracy.
Whisper is multilingual — mixed 中文/English input works.
"""
from __future__ import annotations

import os
import threading

from .config import DATA_DIR

STT_MODEL = os.getenv("STT_MODEL", "base")

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred

            _model = WhisperModel(
                STT_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=str(DATA_DIR / "stt-models"),
            )
    return _model


def transcribe(path: str) -> str:
    """Transcribe an audio file (webm/ogg/mp4/wav/…) to text."""
    segments, _info = _get_model().transcribe(path, vad_filter=True)
    return "".join(s.text for s in segments).strip()

"""Quality checks for generated speech: does it sound like the voice, and does it say the right words?"""
from __future__ import annotations

import difflib
import re

import numpy as np

from . import profiles
from .enroll import embed

_whisper: dict[str, object] = {}


def similarity(voice: str, wav: np.ndarray, sr: int) -> float:
    """Cosine similarity between the speech and the voice's whole recording (≈0.8+ is a close match;
    unrelated voices usually score well below 0.7)."""
    return float(embed(wav, sr) @ profiles.embedding(voice))


def transcribe(wav: np.ndarray, sr: int, language: str = "en", hint: str = "") -> str | None:
    """What a speech recogniser hears (faster-whisper, local). None if faster-whisper is not installed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    import librosa

    size = "small.en" if language == "en" else "small"
    if size not in _whisper:
        _whisper[size] = WhisperModel(size, device="cpu", compute_type="int8")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    segments, _ = _whisper[size].transcribe(
        w16, language=None if size == "small.en" else language,
        initial_prompt=re.sub(r"\s*—\s*", ", ", hint)[:400] or None,
    )
    return " ".join(s.text.strip() for s in segments)


def _words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s']", " ", text.lower()).split() if w]


def word_accuracy(expected: str, heard: str) -> float:
    """Share of the script's words found in order in what was heard (1.0 = every word)."""
    a, b = _words(expected), _words(heard)
    if not a:
        return 1.0
    matched = sum(block.size for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks())
    return matched / len(a)

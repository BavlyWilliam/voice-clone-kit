"""Voice profiles on disk: voices/<name>/ holds everything needed to speak in that voice.

    voices/<name>/profile.json     what was recorded, which clip is used, scores, consent
    voices/<name>/reference.wav    the ~10 s clip the model imitates (best match to the whole recording)
    voices/<name>/reference-2.wav  next-best clips, to try if the first does not sound right
    voices/<name>/styles/<s>.wav   optional clips in a particular delivery (excited, calm, whisper…)
    voices/<name>/source.wav       the cleaned full recording
    voices/<name>/embedding.npy    speaker fingerprint of the whole recording, for similarity scores

Set VOICECLONE_HOME to keep voices somewhere else (default: the `voices` folder next to this package).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOICES = Path(os.environ.get("VOICECLONE_HOME", ROOT / "voices"))


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("Give the voice a name made of letters or numbers, e.g. 'bavly'.")
    return s


def folder(name: str) -> Path:
    return VOICES / slug(name)


def exists(name: str) -> bool:
    return (folder(name) / "profile.json").exists()


def read(name: str) -> dict:
    path = folder(name) / "profile.json"
    if not path.exists():
        known = ", ".join(list_names()) or "none yet"
        raise FileNotFoundError(f"No voice called '{name}'. Known voices: {known}. Create one with `enroll`.")
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, profile: dict) -> None:
    path = folder(name) / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def list_names() -> list[str]:
    if not VOICES.exists():
        return []
    return sorted(p.parent.name for p in VOICES.glob("*/profile.json"))


def reference(name: str, style: str | None = None, alternate: int | None = None) -> Path:
    """Path of the clip to imitate: a style clip, an alternate (2, 3…), or the main reference."""
    profile = read(name)
    base = folder(name)
    if style:
        styles = profile.get("styles", {})
        if style not in styles:
            raise ValueError(f"Voice '{name}' has no style '{style}'. Styles: {', '.join(styles) or 'none'}. "
                             f"Add one with `enroll {name} <recording> --style {style}`.")
        return base / styles[style]["file"]
    if alternate and alternate > 1:
        alts = profile.get("alternates", [])
        if alternate - 2 >= len(alts):
            raise ValueError(f"Voice '{name}' has {len(alts) + 1} reference clip(s).")
        return base / alts[alternate - 2]["file"]
    return base / profile["reference"]["file"]


def embedding(name: str) -> np.ndarray:
    return np.load(folder(name) / "embedding.npy")

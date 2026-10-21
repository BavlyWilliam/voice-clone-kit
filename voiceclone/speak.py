"""Say any text in an enrolled voice.

Text is split into paragraphs (blank lines) and sentences; each sentence is generated separately (long ones are
split again at commas), checked, retried with a new seed if the take looks wrong, then everything is joined with
natural pauses, levelled and saved. Same text + same seed = the same audio, so a good take can be reproduced.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import audio as A
from . import check, model, profiles, tones

OUTPUT = profiles.ROOT / "output"
ENDS = ".!?…؟。！？"
PAUSE = {"sentence": 0.32, "dash": 0.18, "comma": 0.14, "paragraph": 0.70}
NO_SPACES = {"zh", "ja"}


def paragraphs(text: str) -> list[str]:
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def sentences(paragraph: str, max_words: int = 32) -> list[tuple[str, str]]:
    """(sentence, pause kind after it). Splits at sentence ends and em dashes; long sentences at commas."""
    parts = [p.strip() for p in re.split(rf"(?<=[{ENDS}])\s+|(?<=—)\s*", paragraph) if p.strip()]
    out: list[tuple[str, str]] = []
    for p in parts:
        kind = "dash" if p.endswith("—") else "sentence"
        chunks = _split_long(p, max_words)
        out += [(c, "comma") for c in chunks[:-1]] + [(chunks[-1], kind)]
    return out


def _split_long(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    commas = [m.end() for m in re.finditer(r"[,;:]\s", text)]
    if not commas:
        return [text]
    mid = len(text) / 2
    cut = min(commas, key=lambda c: abs(c - mid))
    return _split_long(text[:cut].strip(), max_words) + _split_long(text[cut:].strip(), max_words)


def _units(text: str, language: str) -> float:
    return len(re.sub(r"\s", "", text)) / 3 if language in NO_SPACES else max(1, len(text.split()))


def _plausible(seconds: float, text: str, language: str, pause: float) -> bool:
    n = _units(text, language)
    return 0.15 * n <= seconds <= 0.75 * n + 1.5 and pause < 1.3


def _generate(m, engine: str, text: str, language: str, knobs: dict, seed: int, retries: int, log) -> np.ndarray:
    best = None
    for attempt in range(retries):
        torch.manual_seed(seed + 1000 * attempt)
        try:
            if engine == "english":
                wav = m.generate(text, **knobs)
            else:
                wav = m.generate(text, language_id=language, **knobs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log("  GPU memory was full; retrying")
            continue
        a = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        best = a
        if _plausible(len(a) / m.sr, text, language, A.longest_pause(a, m.sr)):
            return a
        log(f"  take {attempt + 1} of \"{text[:40]}...\" looked wrong ({len(a) / m.sr:.1f} s); trying another")
    if best is None:
        raise RuntimeError("The GPU ran out of memory repeatedly. Close other GPU-heavy apps and try again.")
    return best


def speak(voice: str, text: str, tone: str = "neutral", language: str = "en", style: str | None = None,
          alternate: int | None = None, seed: int = 7, speed: float = 1.0, out: str | Path | None = None,
          split: bool = False, verify: bool = True, retries: int = 3, exaggeration: float | None = None,
          cfg_weight: float | None = None, temperature: float | None = None, log=A.say) -> list[dict]:
    """Generate speech. Returns one report per output file: path, text, seconds, similarity, word accuracy."""
    voice = profiles.slug(voice)
    ref = profiles.reference(voice, style=style, alternate=alternate)
    language = language.lower()
    engine = "english" if language == "en" else "multilingual"
    if engine == "multilingual" and language not in model.languages():
        raise ValueError(f"Language '{language}' is not supported. Supported: {', '.join(model.languages())}")
    knobs = tones.settings(tone, exaggeration, cfg_weight, temperature)
    paras = paragraphs(text)
    if not paras:
        raise ValueError("There is no text to say.")

    log(f"Loading the {engine} model (first time downloads ~3 GB)...")
    m = model.load(engine)
    m.prepare_conditionals(str(ref), exaggeration=knobs["exaggeration"])

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out) if out else OUTPUT / f"{voice}-{tone}-{stamp}.wav"
    groups = [[p] for p in paras] if split else [paras]
    reports = []
    for g, group in enumerate(groups, start=1):
        pieces: list[np.ndarray] = []
        for p_i, para in enumerate(group):
            for s_i, (sentence, pause) in enumerate(sentences(para)):
                log(f"Saying: {sentence}")
                pieces.append(_generate(m, engine, sentence, language, knobs, seed + 100 * p_i + s_i, retries, log))
                pieces.append(np.zeros(int(PAUSE[pause] * m.sr), dtype=np.float32))
            pieces[-1] = np.zeros(int(PAUSE["paragraph"] * m.sr), dtype=np.float32)
        wav = A.trim(np.concatenate(pieces[:-1]), m.sr)
        wav = A.fade(A.normalize(A.change_speed(wav, m.sr, speed), m.sr, lufs=-16), m.sr)
        path = out if not split else out.parent / out.stem / f"{g:02d}{out.suffix or '.wav'}"
        A.save(path, wav, m.sr)
        said = "\n\n".join(group)
        report = {"file": str(path), "text": said, "seconds": round(len(wav) / m.sr, 2), "voice": voice, "tone": tone,
                  "language": language, "style": style, "seed": seed, "settings": knobs, "speed": speed}
        if verify:
            report["similarity"] = round(check.similarity(voice, wav, m.sr), 3)
            heard = check.transcribe(wav, m.sr, language, hint=said)
            if heard is not None:
                report["heard"] = heard
                report["word_accuracy"] = round(check.word_accuracy(said, heard), 3)
        reports.append(report)
        log(f"Saved {path} ({report['seconds']} s)"
            + (f", similarity {report['similarity']:.2f}" if "similarity" in report else "")
            + (f", words {report['word_accuracy']:.0%}" if "word_accuracy" in report else ""))
    if split:
        (out.parent / out.stem / "manifest.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
                                                            encoding="utf-8")
    return reports

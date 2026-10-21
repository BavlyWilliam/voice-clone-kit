"""Create a voice profile from a recording (1–2 minutes of natural speech is ideal).

What it does:
  1. decodes the recording (any format ffmpeg reads: .m4a, .mp3, .wav, even a video), removes rumble and
     levels it;
  2. finds the stretches of fluent speech and builds ~10-second candidate clips that start and end on natural
     pauses (the model only listens to about the first 10 seconds of a reference);
  3. scores every candidate with a speaker encoder against the whole recording and keeps the one that sounds most
     like you overall, plus two runners-up;
  4. saves the clips, the cleaned recording, a speaker fingerprint and a profile.json report.

Only enrol a voice you own or have explicit permission to use; `consent=True` records that confirmation.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import librosa
import numpy as np

from . import audio as A
from . import model, profiles

SR = 24000           # the model's reference sample rate
CLIP_SECONDS = 10.0  # the model uses about the first 10 s of the reference


class ConsentRequired(ValueError):
    pass


def embed(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Speaker fingerprint (unit vector) of a piece of audio."""
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    e = model.encoder().embeds_from_wavs([w16.astype(np.float32)], sample_rate=16000)
    e = np.asarray(e, dtype=np.float32).reshape(-1, e.shape[-1]).mean(axis=0)
    return e / (np.linalg.norm(e) + 1e-9)


def candidates(mask: np.ndarray, seconds: float = CLIP_SECONDS) -> list[tuple[int, int, float]]:
    """(start_frame, end_frame, quality) windows of ~`seconds` that begin at a speech onset and end at a pause."""
    want = int(seconds / A.FRAME)
    shortest = int((seconds - 3) / A.FRAME)
    onsets = [i for i in range(len(mask)) if mask[i] and (i == 0 or not mask[i - 1])]
    offsets = np.array([i for i in range(1, len(mask)) if mask[i - 1] and not mask[i]])
    out = []
    for s in onsets:
        e = s + want
        if e > len(mask):
            break
        ends = offsets[(offsets >= s + shortest) & (offsets <= e)] if len(offsets) else []
        e = int(ends[-1]) if len(ends) else e
        window = mask[s:e]
        idx = np.flatnonzero(window)
        longest = (np.max(np.diff(idx)) - 1) * A.FRAME if len(idx) > 1 else seconds
        quality = window.mean() - 0.4 * max(0.0, longest - 0.6)
        if window.mean() >= 0.6:
            out.append((s, e, float(quality)))
    out.sort(key=lambda c: -c[2])
    picked: list[tuple[int, int, float]] = []
    for c in out:  # keep candidates that do not overlap each other by more than half
        if all(min(c[1], p[1]) - max(c[0], p[0]) < 0.5 * (c[1] - c[0]) for p in picked):
            picked.append(c)
        if len(picked) == 12:
            break
    return picked


def enroll(name: str, recording: str | Path, consent: bool, style: str | None = None,
           clip: tuple[float, float] | None = None, log=A.say) -> dict:
    """Create (or, with `style`, extend) a voice profile. Returns the profile."""
    if not consent:
        raise ConsentRequired("Confirm that this is your own voice, or that the speaker gave you permission, "
                              "before cloning it (--i-have-consent in the command line, the checkbox in the app).")
    recording = Path(recording)
    name = profiles.slug(name)
    if style and not profiles.exists(name):
        raise ValueError(f"Create the voice '{name}' first (without --style), then add styles to it.")

    log(f"Reading {recording.name}...")
    wav = A.normalize(A.highpass(A.load(recording, SR), SR), SR, lufs=-20)
    duration = len(wav) / SR
    if duration < 15:
        raise ValueError(f"The recording is only {duration:.0f} s. Record at least 15 s of speech (1-2 minutes is best).")
    mask, floor, level = A.speech_frames(wav, SR)
    speech = float(mask.sum() * A.FRAME)
    snr = level - floor
    warnings = []
    if duration < 45:
        warnings.append(f"Short recording ({duration:.0f} s): 1–2 minutes gives a better choice of clips.")
    if snr < 20:
        warnings.append(f"Noticeable background noise (speech is only {snr:.0f} dB above it): record somewhere quieter.")
    if np.mean(np.abs(wav) > 0.98) > 0.001:
        warnings.append("Parts of the recording are distorted (clipped): record a little further from the mic.")

    if clip:
        a, b = int(clip[0] / A.FRAME), int(clip[1] / A.FRAME)
        options = [(a, b, 1.0)]
    else:
        options = candidates(mask)
        if not options:
            raise ValueError("Could not find 10 seconds of continuous speech. Speak without long pauses and try again.")

    log(f"Comparing {len(options)} candidate clips with your whole recording...")
    whole = embed(wav)
    scored = []
    for s, e, quality in options:
        piece = wav[int(s * A.FRAME * SR): int(e * A.FRAME * SR)]
        sim = float(embed(piece) @ whole)
        scored.append({"start": round(s * A.FRAME, 2), "end": round(e * A.FRAME, 2), "similarity": round(sim, 3),
                       "fluency": round(quality, 3), "audio": piece})
    scored.sort(key=lambda c: -(c["similarity"] + 0.05 * c["fluency"]))

    base = profiles.folder(name)
    base.mkdir(parents=True, exist_ok=True)

    def keep(c: dict, file: str) -> dict:
        A.save(base / file, A.fade(A.trim(c["audio"], SR), SR), SR)
        return {"file": file, "start": c["start"], "end": c["end"], "similarity": c["similarity"]}

    now = dt.datetime.now().isoformat(timespec="seconds")
    if style:
        profile = profiles.read(name)
        profile.setdefault("styles", {})[style] = {**keep(scored[0], f"styles/{profiles.slug(style)}.wav"),
                                                   "source": recording.name, "added": now}
        profile.setdefault("warnings_by_style", {})[style] = warnings
    else:
        A.save(base / "source.wav", wav, SR)
        np.save(base / "embedding.npy", whole)
        profile = {
            "name": name,
            "created": now,
            "consent": {"confirmed": True, "date": now,
                        "statement": "The person enrolling confirmed this is their own voice or that they have the speaker's permission."},
            "source": {"file": recording.name, "duration": round(duration, 1), "speech_seconds": round(speech, 1),
                       "noise_margin_db": round(snr, 1)},
            "reference": keep(scored[0], "reference.wav"),
            "alternates": [keep(c, f"reference-{i}.wav") for i, c in enumerate(scored[1:3], start=2)],
            "styles": profiles.read(name).get("styles", {}) if profiles.exists(name) else {},
            "warnings": warnings,
            "engine": "Chatterbox (Resemble AI, MIT), run locally",
        }
    profiles.write(name, profile)
    for w in warnings:
        log(f"Warning: {w}")
    best = scored[0]
    log(f"Voice '{name}'{f' style {style}' if style else ''} ready: clip {best['start']:.1f}-{best['end']:.1f} s, "
        f"similarity to the whole recording {best['similarity']:.2f}.")
    return profile

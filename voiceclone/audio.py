"""Audio helpers: decode anything ffmpeg can read, find speech, level loudness, trim, save."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile
from scipy.signal import butter, sosfilt

FRAME = 0.02  # seconds per analysis frame


def say(message: str) -> None:
    """print() that never crashes on text the console cannot show (e.g. Arabic in a Windows console)."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is not on the PATH. Install it (e.g. `winget install Gyan.FFmpeg`) and try again.")
    return exe


def load(path: str | Path, sr: int) -> np.ndarray:
    """Any audio or video file -> mono float32 at `sr`."""
    raw = subprocess.run(
        [require_ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def save(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write .wav (16-bit) directly; any other extension (.mp3, .m4a, .ogg…) via ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    if path.suffix.lower() == ".wav":
        soundfile.write(str(path), audio, sr, subtype="PCM_16")
        return path
    tmp = path.with_suffix(".tmp.wav")
    soundfile.write(str(tmp), audio, sr, subtype="PCM_16")
    subprocess.run([require_ffmpeg(), "-v", "error", "-y", "-i", str(tmp), "-b:a", "192k", str(path)], check=True)
    tmp.unlink()
    return path


def highpass(audio: np.ndarray, sr: int, hz: float = 70.0) -> np.ndarray:
    return sosfilt(butter(4, hz, btype="highpass", fs=sr, output="sos"), audio).astype(np.float32)


def normalize(audio: np.ndarray, sr: int, lufs: float = -18.0, peak_db: float = -1.0) -> np.ndarray:
    """Level to a target loudness, then make sure peaks stay below `peak_db`."""
    if len(audio) < sr * 0.5:
        return audio
    meter = pyloudnorm.Meter(sr)
    loud = meter.integrated_loudness(audio)
    if np.isfinite(loud):
        audio = audio * (10 ** ((lufs - loud) / 20))
    peak = np.max(np.abs(audio)) or 1.0
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        audio = audio * (limit / peak)
    return audio.astype(np.float32)


def frame_db(audio: np.ndarray, sr: int) -> np.ndarray:
    n = int(FRAME * sr)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def speech_frames(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """Boolean speech mask per 20 ms frame, plus the noise floor and typical speech level (dB)."""
    db = frame_db(audio, sr)
    floor = float(np.percentile(db, 10))
    loud = float(np.percentile(db, 95))
    threshold = max(floor + 10, loud - 32)
    mask = db > threshold
    # Close tiny gaps inside words (< 120 ms) so short consonant dips are not treated as pauses.
    gap = int(0.12 / FRAME)
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            mask[a:b] = True
    speech_level = float(np.median(db[mask])) if mask.any() else loud
    return mask, floor, speech_level


def trim(audio: np.ndarray, sr: int, pad: float = 0.05) -> np.ndarray:
    """Remove silence before the first and after the last speech."""
    mask, _, _ = speech_frames(audio, sr)
    if not mask.any():
        return audio
    idx = np.flatnonzero(mask)
    a = max(0, int((idx[0] * FRAME - pad) * sr))
    b = min(len(audio), int(((idx[-1] + 1) * FRAME + pad) * sr))
    return audio[a:b]


def longest_pause(audio: np.ndarray, sr: int) -> float:
    """Longest stretch of non-speech (seconds) between the first and last speech."""
    mask, _, _ = speech_frames(audio, sr)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return 0.0
    return float(np.max(np.diff(idx)) - 1) * FRAME


def fade(audio: np.ndarray, sr: int, seconds: float = 0.015) -> np.ndarray:
    n = min(len(audio) // 2, int(seconds * sr))
    if n > 0:
        audio = audio.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Faster/slower without changing pitch (ffmpeg atempo)."""
    if abs(speed - 1.0) < 1e-3:
        return audio
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    proc = subprocess.run(
        [require_ffmpeg(), "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-af", f"atempo={speed}", "-f", "f32le", "-"],
        input=audio.astype(np.float32).tobytes(), check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()

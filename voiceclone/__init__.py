"""Voice Clone Kit: clone a voice from a 1–2 minute recording and make it say any text, in any of several tones
and 23 languages, locally and for free (open-source Chatterbox models, MIT licence).

    from voiceclone import enroll, speak
    enroll("bavly", "recording.m4a", consent=True)
    speak("bavly", "Hello there!", tone="energetic")
"""
import os

# Plain HTTP downloads for the model weights: the newer chunked ("xet") transfer can stall on some connections.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from .enroll import enroll  # noqa: E402
from .speak import speak  # noqa: E402

__all__ = ["enroll", "speak"]
__version__ = "1.0.0"

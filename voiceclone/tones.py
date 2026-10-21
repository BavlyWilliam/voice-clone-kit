"""Tone presets: how emotional, how fast and how varied the delivery is.

The model has three knobs:
  exaggeration  emotional intensity (0.25 flat … 0.5 natural … 1.0+ animated; very high values can get unstable)
  cfg_weight    how tightly it follows the voice and pacing (lower = slower, more deliberate, more expressive)
  temperature   take-to-take variation (lower = steadier, higher = livelier but riskier)

A preset only shapes delivery. The clip a voice was enrolled from sets its baseline character, so for a
delivery the presets cannot reach (whispering, shouting, laughing) record a short clip in that style and add it
with `enroll <name> <clip> --style <style>`, then speak with `--style <style>`.
"""
from __future__ import annotations

TONES: dict[str, dict] = {
    "neutral":   {"exaggeration": 0.50, "cfg_weight": 0.50, "temperature": 0.80, "about": "natural, like the recording"},
    "calm":      {"exaggeration": 0.30, "cfg_weight": 0.50, "temperature": 0.70, "about": "relaxed and even"},
    "warm":      {"exaggeration": 0.55, "cfg_weight": 0.45, "temperature": 0.80, "about": "friendly, a little softer"},
    "confident": {"exaggeration": 0.65, "cfg_weight": 0.50, "temperature": 0.75, "about": "assured presenter"},
    "serious":   {"exaggeration": 0.35, "cfg_weight": 0.60, "temperature": 0.70, "about": "formal and measured"},
    "energetic": {"exaggeration": 0.85, "cfg_weight": 0.35, "temperature": 0.85, "about": "upbeat, ad-read energy"},
    "excited":   {"exaggeration": 1.10, "cfg_weight": 0.30, "temperature": 0.90, "about": "big enthusiasm"},
    "dramatic":  {"exaggeration": 1.40, "cfg_weight": 0.25, "temperature": 0.90, "about": "trailer-style, very animated"},
}


def settings(tone: str = "neutral", exaggeration: float | None = None, cfg_weight: float | None = None,
             temperature: float | None = None) -> dict:
    """A preset, with any knob overridden by an explicit value."""
    if tone not in TONES:
        raise ValueError(f"Unknown tone '{tone}'. Choose one of: {', '.join(TONES)}")
    s = {k: v for k, v in TONES[tone].items() if k != "about"}
    for key, value in (("exaggeration", exaggeration), ("cfg_weight", cfg_weight), ("temperature", temperature)):
        if value is not None:
            s[key] = float(value)
    return s

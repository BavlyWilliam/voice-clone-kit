"""Command line for Voice Clone Kit.

    voiceclone enroll  NAME RECORDING --i-have-consent [--style STYLE] [--clip 12.5-22.0]
    voiceclone speak   NAME "Any text you like." [--tone energetic] [--language it] [--out file.wav]
    voiceclone speak   NAME --file script.txt [--split] [--style excited] [--speed 1.05] [--seed 21]
    voiceclone voices                      list voices and their clips
    voiceclone tones                       list tone presets
    voiceclone languages                   list languages for --language
    voiceclone check   NAME AUDIO [--text "what it should say"]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio as A
from . import profiles, tones


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="voiceclone", description="Clone a voice from a 1–2 minute recording and make it say anything.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="create a voice from a recording (1–2 minutes of natural speech)")
    e.add_argument("name", help="a short name for the voice, e.g. bavly")
    e.add_argument("recording", help="audio or video file (.m4a .mp3 .wav .mp4 …)")
    e.add_argument("--i-have-consent", action="store_true", dest="consent",
                   help="confirm this is your own voice or you have the speaker's permission (required)")
    e.add_argument("--style", help="add this recording as a delivery style of an existing voice (e.g. excited)")
    e.add_argument("--clip", help="use this exact part as the reference instead of choosing automatically, e.g. 12.5-22")

    s = sub.add_parser("speak", help="say text in a voice")
    s.add_argument("name")
    s.add_argument("text", nargs="?", help="the words to say (or use --file)")
    s.add_argument("--file", help="a .txt file to read; blank lines separate paragraphs")
    s.add_argument("--tone", default="neutral", choices=list(tones.TONES))
    s.add_argument("--language", default="en", help="language code (en, ar, it, fr, es…); see `voiceclone languages`")
    s.add_argument("--style", help="use a style clip added with `enroll --style`")
    s.add_argument("--alternate", type=int, help="use reference clip 2 or 3 instead of the best one")
    s.add_argument("--out", help="output file (.wav, .mp3, .m4a…); default output/<voice>-<tone>-<time>.wav")
    s.add_argument("--split", action="store_true", help="one file per paragraph, plus manifest.json")
    s.add_argument("--speed", type=float, default=1.0, help="0.5–2.0, pitch is kept")
    s.add_argument("--seed", type=int, default=7, help="change it to get a different take")
    s.add_argument("--exaggeration", type=float, help="override the tone's emotional intensity (0.25–2.0)")
    s.add_argument("--cfg", type=float, dest="cfg_weight", help="override pacing/adherence (0–1, lower = slower)")
    s.add_argument("--temperature", type=float, help="override variation (0.5–1.2)")
    s.add_argument("--no-check", action="store_true", help="skip the similarity and word checks")

    sub.add_parser("voices", help="list voices")
    sub.add_parser("tones", help="list tone presets")
    sub.add_parser("languages", help="list languages")

    c = sub.add_parser("check", help="score any audio against a voice")
    c.add_argument("name")
    c.add_argument("audio")
    c.add_argument("--text", help="what it should say, to also check the words")
    c.add_argument("--language", default="en")

    args = ap.parse_args(argv)

    if args.cmd == "enroll":
        from .enroll import ConsentRequired, enroll

        clip = tuple(float(x) for x in args.clip.split("-")) if args.clip else None
        try:
            enroll(args.name, args.recording, consent=args.consent, style=args.style, clip=clip)
        except ConsentRequired as err:
            print(err)
            return 2
        return 0

    if args.cmd == "speak":
        from .speak import speak

        if not args.text and not args.file:
            ap.error("give the text to say, or --file script.txt")
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        reports = speak(args.name, text, tone=args.tone, language=args.language, style=args.style,
                        alternate=args.alternate, seed=args.seed, speed=args.speed, out=args.out, split=args.split,
                        verify=not args.no_check, exaggeration=args.exaggeration, cfg_weight=args.cfg_weight,
                        temperature=args.temperature)
        for r in reports:
            if r.get("word_accuracy", 1) < 0.9:
                print(f"Note: {Path(r['file']).name} may have skipped or changed words (heard: '{r['heard']}'). "
                      "Try again with another --seed.")
        return 0

    if args.cmd == "voices":
        names = profiles.list_names()
        if not names:
            print("No voices yet. Create one: voiceclone enroll <name> <recording> --i-have-consent")
        for n in names:
            p = profiles.read(n)
            ref = p["reference"]
            styles = ", ".join(p.get("styles", {})) or "none"
            print(f"{n}: from {p['source']['file']} ({p['source']['duration']} s); clip {ref['start']}-{ref['end']} s, "
                  f"match {ref['similarity']}; styles: {styles}")
        return 0

    if args.cmd == "tones":
        for name, t in tones.TONES.items():
            print(f"{name:10} {t['about']:32} exaggeration {t['exaggeration']}, cfg {t['cfg_weight']}, temperature {t['temperature']}")
        return 0

    if args.cmd == "languages":
        from .model import languages

        for code, lang in languages().items():
            print(f"{code}  {lang}" + ("  (best quality, english engine)" if code == "en" else ""))
        return 0

    if args.cmd == "check":
        from . import check

        wav = A.load(args.audio, 24000)
        result = {"similarity": round(check.similarity(args.name, wav, 24000), 3)}
        if args.text:
            heard = check.transcribe(wav, 24000, args.language, hint=args.text)
            if heard is not None:
                result.update(heard=heard, word_accuracy=round(check.word_accuracy(args.text, heard), 3))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

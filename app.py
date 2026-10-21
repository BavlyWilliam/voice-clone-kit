"""Voice Clone Kit: the local web app.

Double-click start-app.bat (or run `python app.py` in the voice-clone environment). It opens
http://127.0.0.1:7860 in your browser. Everything runs on this PC; nothing is uploaded anywhere.
"""
from __future__ import annotations

import faulthandler
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
faulthandler.enable()

import gradio as gr  # noqa: E402

from voiceclone import model, profiles, tones  # noqa: E402
from voiceclone.enroll import enroll as enroll_voice  # noqa: E402
from voiceclone.speak import speak as speak_text  # noqa: E402

LANGUAGES = {f"{name} ({code})": code for code, name in model.languages().items()}
MAIN = "(main voice)"
CLIPS = {"Best clip": None, "Clip 2": 2, "Clip 3": 3}


def voice_choices() -> list[str]:
    return profiles.list_names()


def style_choices(voice: str | None) -> list[str]:
    if not voice or not profiles.exists(voice):
        return [MAIN]
    return [MAIN, *profiles.read(voice).get("styles", {})]


def create_voice(name: str, recording: str | None, consent: bool, style: str):
    if not recording:
        return "Add or record a recording first.", None, gr.update()
    lines: list[str] = []
    try:
        profile = enroll_voice(name, recording, consent=consent, style=style.strip() or None, log=lines.append)
    except Exception as err:  # shown to the user in the page
        return f"**Could not create the voice.** {err}", None, gr.update()
    slug = profiles.slug(name)
    if style.strip():
        clip = profile["styles"][style.strip()]
        ref = profiles.reference(slug, style=style.strip())
    else:
        clip = profile["reference"]
        ref = profiles.reference(slug)
    src = profile["source"]
    report = [
        f"### Voice **{slug}** {'style **' + style.strip() + '**' if style.strip() else ''} is ready",
        f"- Recording: {src['duration']} s, {src['speech_seconds']} s of speech, background noise "
        f"{src['noise_margin_db']} dB below the voice" if not style.strip() else f"- Style clip from {clip['source']}",
        f"- Reference clip: {clip['start']}–{clip['end']} s (match to the whole recording {clip['similarity']:.2f})",
    ]
    report += [f"- ⚠️ {w}" for w in profile.get("warnings", [])]
    report.append("\nListen to the clip below, then go to **2 · Speak**.")
    return "\n".join(report), str(ref), gr.update(choices=voice_choices(), value=slug)


def say(voice, text, tone, language, style, clip, speed, seed, custom, exaggeration, cfg_weight, temperature):
    if not voice:
        return None, None, "Create a voice first (tab 1), or pick one from the list."
    if not text or not text.strip():
        return None, None, "Type what the voice should say."
    try:
        reports = speak_text(
            voice, text, tone=tone, language=LANGUAGES.get(language, "en"),
            style=None if style in (None, MAIN) else style, alternate=CLIPS.get(clip), seed=int(seed or 7),
            speed=float(speed), verify=True,
            exaggeration=exaggeration if custom else None, cfg_weight=cfg_weight if custom else None,
            temperature=temperature if custom else None, log=lambda *_: None,
        )
    except Exception as err:
        return None, None, f"**Could not generate.** {err}"
    r = reports[0]
    lines = [f"**{r['seconds']} s** · tone {r['tone']} · {language} · take {r['seed']}"]
    if "similarity" in r:
        lines.append(f"- Sounds like the voice: **{r['similarity']:.2f}** (0.80+ is a close match)")
    if "word_accuracy" in r:
        ok = r["word_accuracy"] >= 0.9
        lines.append(f"- Words said correctly: **{r['word_accuracy']:.0%}**"
                     + ("" if ok else f" — heard: “{r['heard']}”. Try another take number."))
    return r["file"], r["file"], "\n".join(lines)


def tone_values(tone: str):
    t = tones.TONES[tone]
    return t["exaggeration"], t["cfg_weight"], t["temperature"]


HELP = """
### Recording a good sample
- **1–2 minutes** of you talking naturally, the way you want to sound (reading anything is fine).
- A quiet room, phone or mic 15–20 cm away, no music or other voices.
- For a different delivery (excited, calm, whispering…), record 20–60 s in that style and add it in tab 1 with a
  **style** name. Then pick that style when speaking.

### Getting the tone you want
- **Tone** presets set emotion, pacing and variation. *Fine-tune the tone* lets you set them by hand.
- **Take number**: same text + same take = the same audio. Change it for a different reading.
- **Clip 2 / Clip 3**: try another reference clip from your recording if the voice does not sound quite right.

### Languages
English uses the best model. Other languages use a multilingual model (downloaded once, ~3.3 GB). If you speak
the language, a sample recorded in it gives the most natural accent.

### Use responsibly
Only clone your own voice or a voice you have permission to use, and say when audio is AI-generated.
Every file carries an inaudible watermark marking it as AI-generated.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Voice Clone Kit", analytics_enabled=False) as app:
        gr.Markdown("# Voice Clone Kit\nClone a voice from a 1–2 minute recording, then make it say anything. "
                    "Runs on this PC; nothing is uploaded.")
        with gr.Tab("1 · Create a voice"):
            name = gr.Textbox(label="Voice name", placeholder="e.g. bavly")
            recording = gr.Audio(label="Recording (1–2 minutes of natural speech)", sources=["upload", "microphone"],
                                 type="filepath")
            style = gr.Textbox(label="Optional: save this as a style of an existing voice (e.g. excited, calm, whisper)")
            consent = gr.Checkbox(label="This is my own voice, or I have the speaker's permission to clone it")
            create = gr.Button("Create voice", variant="primary")
            report = gr.Markdown()
            reference = gr.Audio(label="Reference clip that was chosen", interactive=False)

        with gr.Tab("2 · Speak"):
            with gr.Row():
                voice = gr.Dropdown(voice_choices(), value=(voice_choices() or [None])[0], label="Voice")
                style_pick = gr.Dropdown(style_choices((voice_choices() or [None])[0]), value=MAIN, label="Style")
                refresh = gr.Button("Refresh voices", scale=0)
            text = gr.Textbox(label="What should it say?", lines=6,
                              placeholder="Type anything. Leave a blank line between paragraphs.")
            with gr.Row():
                tone = gr.Dropdown(list(tones.TONES), value="neutral", label="Tone")
                language = gr.Dropdown(list(LANGUAGES), value="English (en)", label="Language")
                speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Speed")
                seed = gr.Number(value=7, precision=0, label="Take number")
            with gr.Accordion("Fine-tune the tone", open=False):
                custom = gr.Checkbox(label="Use these values instead of the preset")
                exaggeration = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Emotion intensity")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pacing (lower = slower, more expressive)")
                temperature = gr.Slider(0.5, 1.2, value=0.8, step=0.05, label="Variation")
                clip = gr.Radio(list(CLIPS), value="Best clip", label="Reference clip")
            go = gr.Button("Speak", variant="primary")
            result = gr.Audio(label="Result", interactive=False)
            download = gr.File(label="Download")
            details = gr.Markdown()

        with gr.Tab("Help"):
            gr.Markdown(HELP)

        create.click(create_voice, [name, recording, consent, style], [report, reference, voice])
        refresh.click(lambda: gr.update(choices=voice_choices()), None, voice)
        voice.change(lambda v: gr.update(choices=style_choices(v), value=MAIN), voice, style_pick)
        tone.change(tone_values, tone, [exaggeration, cfg_weight, temperature])
        go.click(say, [voice, text, tone, language, style_pick, clip, speed, seed, custom, exaggeration, cfg_weight,
                       temperature], [result, download, details])
    return app


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("VOICECLONE_PORT", 7860)),
        inbrowser=os.environ.get("VOICECLONE_NO_BROWSER") != "1",
    )

# Voice Clone Kit

Clone a voice from a **1–2 minute recording**, then make it say **any text**, in a choice of **tones** and
**23 languages**. Everything runs **on your own PC, for free**: no account, no subscription, nothing uploaded.

- **In:** a recording of someone talking naturally (.m4a, .mp3, .wav, even a video), 1–2 minutes is ideal.
- **Out:** speech in that voice saying whatever you type, as .wav or .mp3, with a score for how much it sounds
  like the voice and whether every word was said.

It is built on the open-source **Chatterbox** models by Resemble AI (MIT licence) and grew out of the narration

---

## 1. Set up (once)

Needs Windows 10/11, Python 3.10–3.13, `ffmpeg` on the PATH, and ~8 GB of free disk. An NVIDIA GPU makes it fast
(about real time); without one it still works on the CPU, just several times slower.

Double-click **`setup.bat`**. It creates a Python environment at `%USERPROFILE%\voice-clone-env` (a short path,
because PyTorch's files break Windows' 260-character path limit in deep folders), installs PyTorch (the GPU build
when it finds an NVIDIA card) and the models' code. The model weights download on first use (~3.2 GB for English,
~3.3 GB more for other languages) into `%USERPROFILE%\.cache\huggingface`.

> On Bavly's PC this is already done (`C:\Users\ACER\voice-clone-env`, RTX 3060).

## 2. Use the app

Double-click **`start-app.bat`**. Your browser opens <http://127.0.0.1:7860> (only reachable from this PC).

1. **Create a voice**: give it a name, upload or record 1–2 minutes of speech, tick the consent box, press
   *Create voice*. The kit picks the best ~10-second clip automatically and plays it back.
2. **Speak**: pick the voice, type the text (blank lines separate paragraphs), choose a tone, language, speed and
   take number, press *Speak*. Listen, download, or change the take number for a different reading.

## 3. Or use the command line

Open a terminal in this folder (or put it on your PATH) and use **`voiceclone.bat`**:

```bash
voiceclone enroll bavly "..\Voice recoderings\Claude Video Script.m4a" --i-have-consent
voiceclone speak bavly "Hello, nice to meet you." --tone warm
voiceclone speak bavly --file examples\script-example.txt --tone energetic --split --out output\ad.wav
voiceclone speak bavly "Ciao, piacere di conoscerti." --language it
voiceclone enroll bavly "excited-take.m4a" --style excited --i-have-consent
voiceclone speak bavly "This is amazing news!" --style excited --tone excited
voiceclone voices      &  voiceclone tones      &  voiceclone languages
voiceclone check bavly some-audio.wav --text "what it should say"
```

Useful options for `speak`: `--out file.mp3` (any format ffmpeg writes), `--split` (one file per paragraph plus
`manifest.json`, handy for video narration), `--speed 0.9`, `--seed 21` (a different take; same seed = same audio),
`--alternate 2` (use the second-best reference clip), `--exaggeration/--cfg/--temperature` (fine-tune the tone),
`--no-check`.

From Python:

```python
from voiceclone import enroll, speak
enroll("bavly", "recording.m4a", consent=True)
reports = speak("bavly", "Any text you like.", tone="confident", out="hello.wav")
print(reports[0]["similarity"], reports[0].get("word_accuracy"))
```

## Recording a good sample

- **1–2 minutes** of natural speech in the voice you want to hear back. Any words work; it does not need to be a script.
- A **quiet room** with soft furnishings, no music, fans or other voices. Phone or mic **15–20 cm** away, same distance throughout.
- Speak at your normal pace and energy. The clone copies the delivery as well as the voice.
- Check the report after *Create voice*: it warns about short, noisy or distorted recordings.

## Tones: what "any tone" means here

| Tone | Feels like |
|---|---|
| neutral | natural, like the recording |
| calm | relaxed and even |
| warm | friendly, a little softer |
| confident | assured presenter |
| serious | formal and measured |
| energetic | upbeat, ad-read energy |
| excited | big enthusiasm |
| dramatic | trailer-style, very animated |

Each preset sets three knobs: **emotion intensity** (`exaggeration`), **pacing** (`cfg`; lower is slower and
more deliberate) and **variation** (`temperature`). You can set them yourself (*Fine-tune the tone* in the app,
or the flags above).

The baseline character still comes from the recording. For a delivery the presets cannot produce (whispering,
shouting, laughing, a news-anchor read…), record 20–60 seconds **in that style** and add it as a style:
`enroll NAME clip.m4a --style whisper`, then `speak NAME "…" --style whisper`. That is the reliable way to get
genuinely different tones.

## Languages

English (`en`) uses the English model, which gives the best quality. Arabic, Chinese, Danish, Dutch, Finnish,
French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian,
Spanish, Swahili, Swedish and Turkish use the multilingual model (`voiceclone languages` lists the codes). With an
English sample the voice carries over to other languages, sometimes with a slight accent. If you speak the
language, a sample recorded in it sounds most natural. Lowering pacing (`--cfg 0`) reduces accent carry-over.

## How it works

1. **Enroll** (`voiceclone/enroll.py`): decode → high-pass and level → find speech → build ~10-second candidate
   clips that start and end on natural pauses (the model listens to about the first 10 seconds) → score each with
   a speaker encoder against the whole recording → keep the best three. Everything is saved in `voices/<name>/`.
2. **Speak** (`voiceclone/speak.py`): split into paragraphs and sentences → generate each sentence → retry a take
   with a new seed if its length is implausible for its words or it contains a long dead pause → join with
   natural pauses → trim, level to −16 LUFS, optional speed change → save.
3. **Check** (`voiceclone/check.py`): voice similarity against the whole original recording (0.80+ is a close
   match; unrelated voices score well below 0.7) and a word check with a local speech recogniser (faster-whisper).

`voiceclone/model.py` loads the models tensor by tensor. The stock loader reserves each 2 GB weights file in
Windows' page file and fails with *"The paging file is too small"* when other apps are using memory.

## Folders

```
voice-clone-kit/
  start-app.bat  voiceclone.bat  setup.bat  requirements.txt
  app.py                 the local web app
  voiceclone/            the code: audio, model, enroll, speak, check, tones, profiles, cli
  examples/              a sample script
  voices/<name>/         profile.json, reference clips, styles/, cleaned source, voice fingerprint  (private)
  output/                generated audio                                                           (private)
```

`voices/` and `output/` are in `.gitignore`: they hold personal recordings. Set `VOICECLONE_HOME` to keep voices elsewhere.

## Use it responsibly

- Clone only **your own voice**, or a voice whose owner has given you **explicit permission**. The kit asks for
  that confirmation and records it in the profile.
- Say when audio is AI-generated (for example "Narration: AI-generated voice"). Every file also carries an
  inaudible **Perth watermark** from Resemble AI that marks it as AI-generated.
- Don't use a cloned voice to impersonate, mislead, or get past voice-based security.

## Troubleshooting

| Problem | Fix |
|---|---|
| *ffmpeg is not on the PATH* | `winget install Gyan.FFmpeg`, then open a new terminal |
| *The paging file is too small* | Already avoided by the kit's loader; if it still happens, close some browser tabs |
| GPU out of memory | The kit retries automatically. Close GPU-heavy apps (games, many browser tabs) if it keeps happening |
| The app window closes with a crash right after "Loading the model" | Rare native crash on Windows; `voiceclone.bat` retries once automatically, otherwise run again |
| A word is skipped or changed | Try another take (`--seed` / take number); the report shows what was heard |
| It does not sound quite like the person | Try `--alternate 2` or `3`, or re-record a cleaner, longer sample |
| Very slow | No NVIDIA GPU detected: it runs on the CPU. Short texts are fine; long scripts take a while |
| First use takes long | One-time model download (~3 GB per model) |

## Credits and licences

Chatterbox TTS and the Perth watermarker: Resemble AI, MIT. PyTorch: BSD-3. faster-whisper: MIT (models by
OpenAI, MIT). Gradio: Apache-2.0. librosa: ISC. This kit's code: written for Bavly William, free to reuse.

## Verified on this PC (21 Oct 2026, RTX 3060)

| Test | Length | Sounds like the voice | Words correct |
|---|---|---|---|
| English, neutral (command line) | 7.3 s | 0.94 | 100% |
| English, energetic, 2 paragraphs (`--split`) | 4.1 s + 7.2 s | 0.90 / 0.92 | 100% |
| English, via the web app | 6.5 s | 0.95 | 100% |
| Italian, warm | 7.8 s | 0.94 | 100% |
| Arabic, neutral | 9.6 s | 0.93 | 100% |

The demo files are in `output/`.

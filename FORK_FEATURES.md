# What this fork adds

This is a fork of [mutonby/openshorts](https://github.com/mutonby/openshorts).
Everything upstream still works the same; the additions below are aimed at
**anime / foreign-language sources** and at **running locally on a consumer GPU**.
Each one is off by default and opt-in per job.

## 1. Burned-in subtitle handling (`hardsubs.py`)

Fansub and simulcast rips carry subtitles in the pixels. The 9:16 crop keeps
the middle third of the width, so those lines come out cut on both sides, and
OpenShorts' own captions then land on top of them.

`POST /api/process` takes `hardsubs` (dashboard: advanced options →
"burned-in subtitles"):

| Mode | What it does | Trade-off |
|---|---|---|
| `crop` | Drops the measured subtitle band from each clip | Fast, no artifacts, loses the bottom ~15-25% |
| `inpaint` | Paints the glyphs out frame by frame (OpenCV Telea) | Full picture, faint smudge, ~1 min/clip on CPU |
| `keep` | Leaves the original subtitles and frames around them | Scenes with dialogue render full width; our captions are turned off |

Detection is geometric, not OCR: thin bright strokes, a dark outline around
each glyph, at least four glyphs on one baseline, bottom 40% of the frame
only. A clip with no subtitles found is left untouched.

## 2. Separate language for hooks and titles (`copy_language`)

`copy_language` on `/api/process` (env `COPY_LANGUAGE`, dashboard
"hook & title language") sets the language of the hook, title and
descriptions independently of the audio, e.g. a Japanese anime clipped
for an English audience.

## 3. Audio pitch shift (`pitch`)

`pitch` on `/api/process` (semitones, clamped to ±6, env
`AUDIO_PITCH_SEMITONES`, dashboard "audio pitch") shifts the voice with
ffmpeg `rubberband`. Tempo is untouched, so audio stays in sync with the
picture and any burned-in subtitles. Applied once, when the clip is cut from
the source, so later re-encodes never stack it.

## 4. Better CJK and non-Latin language support

- **Japanese/Chinese/Korean speech is no longer treated as silence.** These
  scripts have no spaces between words, so a talky 24-minute episode used to
  count as ~87 "words" and fall back to the silent-video path. Words are now
  estimated from character count.
- **CJK hook text renders with a CJK font** when one is installed (and warns
  instead of burning in empty boxes when none is).
- **`fonts-noto-core` in the Docker image**, so subtitles in Hindi, Arabic,
  Thai, Bengali, Tamil, etc. (the dubbing languages) render as text instead of □□□.
- **`WHISPER_TASK=translate`** makes Whisper output English for any spoken
  language. The transcript is then labelled `en`, so hooks are written in
  English to match.

## 5. Local GPU setup (`docker-compose.gpu.yml`)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Runs faster-whisper on CUDA and encodes with NVENC (`FFMPEG_ENCODER=auto`
falls back to x264 if the card can't). Tuned to fit a 4 GB card (tested on an
RTX 2050): `WHISPER_MODEL=medium` at `int8_float16` alongside YOLO and NVENC.
It also stops publishing the renderer port, which Windows/WSL sometimes
reserves.

## 6. Dashboard tweaks

- **An Upload-Post key is no longer required to generate clips.** It is only
  needed to publish to social media, so it is now checked only for that.
- Frontend dev port is `6100` (was `5175`).

## Tests

New: `tests/test_hardsubs.py`, `tests/test_audio_pitch.py`,
`tests/test_copy_language.py`, plus CJK cases in `tests/test_sparse_speech.py`.

## License

Unchanged from upstream: MIT for the core, and the OpenShorts Commercial
License for everything under `cloud/` (see `LICENSE` and `cloud/LICENSE`).

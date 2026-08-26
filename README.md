# VizoLearn — Real-Time AI Visual Tutor

Point a webcam at something, click on it, and get a spoken, structured lesson
about it — then keep asking follow-up questions out loud or by typing.

VizoLearn is a Streamlit app that combines live object detection, a
vision-language model, and a small instruction-tuned LLM into one
click-to-learn workflow, entirely in the browser tab.

---

## How it works

```
Webcam ──▶ YOLO11n + ByteTrack ──▶ live bounding boxes
                                        │
                          user clicks a detected object
                                        │
                                        ▼
                          clean (unannotated) crop of that object
                                        │
                                        ▼
                        BLIP image captioning ──▶ visual description
                                        │
                                        ▼
                Qwen2.5-0.5B-Instruct ──▶ structured lesson (text)
                                        │
                                        ▼
                  browser SpeechSynthesis ──▶ spoken aloud
                                        │
                        user asks a follow-up (voice or typed)
                                        │
                                        ▼
                Qwen2.5-0.5B-Instruct ──▶ contextual answer, repeat
```

**Detection & tracking.** Every incoming webcam frame is run through a
YOLO11n model with ByteTrack (`model.track(persist=True)`), so each detected
object keeps a stable `track_id` across frames instead of flickering as a
new detection every frame.

**Click-to-select.** The processed frame is shown as a clickable image
(`streamlit-image-coordinates`). A click is hit-tested against the current
detections — if the click lands in more than one overlapping box, the
**smallest** (most specific) box wins.

**Clean crop.** The app keeps two versions of every frame: one with boxes
and confidence labels drawn on it (for the user to look at and click) and a
completely unannotated one (for the AI pipeline). Only the unannotated crop
is ever sent to the vision model — the box outlines and label text drawn
for the human are not part of what the model reasons about, so its
description is never contaminated by the app's own UI drawing.

**Visual description.** The clean crop goes through
`Salesforce/blip-image-captioning-base` (BLIP) to get a short natural-
language description of the actual object crop, not just the YOLO class
name.

**Lesson generation.** `Qwen/Qwen2.5-0.5B-Instruct` receives the YOLO class
name plus the BLIP description and produces a structured explanation
(what it is → what it's for → how it works → why it's useful → an
interesting fact), instructed to explicitly flag anything that can't be
determined from the image alone rather than invent detail.

**Speech.** The lesson (and every follow-up answer) has a **Read aloud**
button. This uses the browser's own `SpeechSynthesis` API — no server-side
TTS engine, no generated audio files, and it picks up whatever voices are
already installed on your OS/browser.

**Follow-up questions.** You can either type a question or tap the mic and
speak — voice input uses the browser's `SpeechRecognition` /
`webkitSpeechRecognition` API, shows a live transcript as you talk, and
auto-submits the question the moment you stop speaking (no separate
"confirm" step). Only the most recent answer in the conversation shows the
follow-up box; earlier answers just show their own **Read aloud** control.

---

## Project structure

```
VizoLearn/
├── app.py                          # Streamlit UI, session state, orchestration
├── yolo11n.pt                      # YOLO11-nano weights (committed, ~5.6 MB)
├── requirements.txt
├── env.example
├── pytest.ini
├── .gitignore
├── .streamlit/
│   └── config.toml                 # server config — see "Deployment note"
├── src/
│   ├── detector.py                 # YOLO + ByteTrack wrapper
│   ├── selection.py                # click hit-testing / track-id lookup
│   ├── crop.py                     # bounds-safe object cropping
│   ├── vision.py                   # BLIP image captioning
│   ├── explain.py                  # Qwen lesson + follow-up Q&A generation
│   ├── voice_ui.py                 # browser TTS + browser STT components
│   ├── utils.py                    # small path helper
│   ├── speak.py                    # not wired in — see "Inactive modules"
│   └── listen.py                   # not wired in — see "Inactive modules"
├── scripts/
│   └── download_whisper_model.py   # downloads a model listen.py would use
└── tests/
    ├── test_selection.py           # hit-testing logic, no ML deps needed
    └── test_crop.py                # crop bounds/clamping logic, no ML deps needed
```

---

## Setup

```bash
pip install -r requirements.txt
```

If you already have a working CUDA-enabled PyTorch install, keep it —
don't let a generic `pip install` silently replace it with a CPU-only
build. Install everything else first, then check `torch.cuda.is_available()`
before assuming you need to reinstall torch.

The first run downloads two Hugging Face models the first time each is
used:

- `Salesforce/blip-image-captioning-base`
- `Qwen/Qwen2.5-0.5B-Instruct`

Both are cached locally afterward (standard `transformers`/`huggingface_hub`
cache behavior) — subsequent runs don't re-download them.

`yolo11n.pt` is already committed in the repo, so no separate detector
download is needed.

## Run

```bash
streamlit run app.py
```

Open the printed URL and allow camera access when the browser asks.
Voice input (the mic button) needs a browser with `SpeechRecognition`
support: Chrome and Edge fully support it, Safari supports it (behind the
`webkitSpeechRecognition` prefix, which `voice_ui.py` already checks for),
and Firefox has it implemented but disabled by default behind a flag — so
in practice, expect it to work in Chrome, Edge, and Safari, and not in a
default Firefox install. `voice_ui.py` shows an inline message rather than
failing silently if the browser doesn't support it. One more thing worth
knowing: on most browsers/versions, `SpeechRecognition` sends audio to a
cloud service for transcription (Google's for Chrome, Apple's for Safari)
rather than running fully on-device, so voice input needs an internet
connection even though camera detection and the AI lessons can, in
principle, run fully offline once models are cached. (Chrome 139+ has
started rolling out an on-device option, so this may not hold for every
Chrome install going forward.) Text-to-speech (`SpeechSynthesis`) doesn't
have this limitation — it runs locally using OS/browser voices and works
offline.

## Configuration

**Nothing in `env.example` is currently read by the app.** Confidence
threshold (`0.40`), YOLO image size (`640`), and both Hugging Face model
IDs are hardcoded in `app.py` / `src/vision.py` / `src/explain.py`.
`env.example` documents the intended-but-not-yet-wired variable names as a
placeholder for future work, and the file itself says so at the top. If you
want to actually make these configurable, that's a small, contained change:
each hardcoded value would need to become an `os.getenv(...)` call at its
one call site.

The only environment variables actually read by any code right now are
`VIZO_WHISPER_MODEL` and `VIZO_WHISPER_MODEL_PATH`, inside `src/listen.py`
— and that module isn't currently used by the running app (see below).

## Tests

```bash
pytest -q
```

Covers the pure-logic modules that need no GPU, webcam, or downloaded
model: click hit-testing (including overlapping-box and boundary cases)
and crop bounds/clamping (including degenerate and inverted boxes). The
detection, vision, and language modules aren't unit-tested here since
exercising them meaningfully needs the real models and, in the detector's
case, real image input — that's better suited to manual/integration
testing than a fast unit suite.

## Hardware

Every model call checks `torch.cuda.is_available()` and uses the GPU when
present, falling back to CPU otherwise. Developed and tested against an
RTX 4050 Laptop GPU (~6GB VRAM) — at that VRAM budget, YOLO11n and
Qwen2.5-0.5B are deliberately the smallest variants in their families;
BLIP-base is the heavier of the two AI stages. If you have more VRAM to
spare, `MODEL_PATH` in `app.py` and `MODEL_ID` in `src/explain.py` are the
two places to point at larger variants.

---

## Inactive modules

Two complete, working subsystems exist in `src/` but are **not currently
used by `app.py`** — they were superseded by browser-native alternatives
during development, and were intentionally left in the codebase rather
than deleted:

| Module | What it does | Superseded by |
|---|---|---|
| `src/speak.py` | Server-side TTS via `pyttsx3`, synthesized to a WAV file | `src/voice_ui.py`'s browser `SpeechSynthesis` |
| `src/listen.py` + `scripts/download_whisper_model.py` | Offline STT via `faster-whisper`, local-model-first with online fallback | `src/voice_ui.py`'s browser `SpeechRecognition` |

The browser-native approach is simpler (no model download, no GPU needed
for voice, no generated audio files to manage) and is what you'll actually
experience running the app today. The offline paths are kept because
they're genuinely useful for a deployment where the browser's Web Speech
APIs aren't a good fit — no internet connectivity (voice input needs it,
per the network note above), a browser without `SpeechRecognition` support
(default Firefox, mainly), or an offline/air-gapped environment generally.
Re-enabling either one means wiring its function calls into `app.py`
yourself; neither is reachable through the UI as the code stands.

---

## Deployment note

`.streamlit/config.toml` sets `enableXsrfProtection = false`. This is a
real security trade-off, not just cosmetic config — Streamlit's own docs
frame CSRF protection as an intentional security feature, and turning it
off reduces the app's protection against cross-site request forgery. It's
a common (if blunt) fix for 403/upload errors that show up specifically
when a Streamlit app is embedded, reverse-proxied, or run behind certain
auth setups — which fits `streamlit-webrtc` and this app's custom `v2`
components fairly well — but it isn't something to carry into a
public-facing deployment without deciding that trade-off deliberately. For
purely local, single-user use (`streamlit run app.py` on your own machine)
it's low-stakes; if you ever deploy this somewhere multiple people can
reach, it's worth revisiting.

---

## Known limitations

- **Voice input browser/network requirements.** `SpeechRecognition` works
  in Chrome, Edge, and Safari but is off by default in Firefox. On top of
  that, most implementations send audio to a cloud service to transcribe
  it, so the mic button needs an internet connection even if everything
  else in the app is running fully offline. Users without a working
  `SpeechRecognition` or without connectivity can still type follow-up
  questions — only the mic button is affected.
- **No persistence.** All state (selected object, lesson, chat history)
  lives in Streamlit's session state and is lost on page refresh or
  server restart. There's no database or file-based save of past
  sessions.
- **Single active object.** Selecting a new object clears the previous
  lesson and conversation. There's no way to keep multiple objects'
  lessons open side by side.
- **English-centric.** `SpeechRecognition` is configured for `en-US`
  (`src/voice_ui.py`), and the tutor prompt (`src/explain.py`) is written
  and instructed in English. Other languages aren't currently supported
  end-to-end.
- **Vision model doesn't know brands/models.** The tutor prompt explicitly
  instructs Qwen not to guess at brand names, exact models, or internal
  components from appearance alone, and to say so when a detail can't be
  determined from the image — this is intentional (avoiding confidently
  wrong specifics), not a bug, but it does mean lessons stay at the
  general-object level rather than identifying, say, a specific phone
  model.

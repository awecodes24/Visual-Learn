# VizoLearn

**Point. Learn. Ask.**

VizoLearn is an interactive AI learning application that turns a live camera feed into a hands-on learning experience. Point the camera at an everyday object, let the system detect it, click the object you want to explore, and VizoLearn builds a lesson around it. You can then ask follow-up questions by typing or speaking while watching your words appear live as you talk.

The project is built with **Streamlit** and combines computer vision, image understanding, a local language model, browser speech recognition, and browser text-to-speech into one continuous learning workflow.

---

## ✨ What VizoLearn Does

VizoLearn is designed around a simple interaction loop:

```text
Live Camera
    ↓
YOLO Object Detection + Tracking
    ↓
Click / Tap an Object
    ↓
Crop the Selected Object
    ↓
BLIP Visual Description
    ↓
Qwen AI Tutor Explanation
    ↓
Read the Lesson Aloud
    ↓
Ask a Follow-up Question
   ↙                ↘
Voice              Typing
   ↓                  ↓
Live Transcript   Text Question
   └──────────┬───────┘
              ↓
       Qwen Follow-up Answer
```

### Core capabilities

- **Live camera object detection** using YOLO11.
- **Persistent object tracking** using ByteTrack so detected objects can be associated with the live scene.
- **Clickable object selection** directly from the camera preview.
- **Object selection buttons** below the preview as an alternative to clicking the bounding box.
- **Visual context generation** using Salesforce BLIP.
- **AI teaching and explanations** using `Qwen/Qwen2.5-0.5B-Instruct`.
- **Conversational follow-up questions** that maintain recent chat context.
- **Interactive voice input** using browser speech recognition.
- **Live/interim transcript display** while the user is speaking.
- **Typed question fallback** when voice is unavailable or not preferred.
- **Browser-native text-to-speech** with voice selection, replay, pause/resume, and stop controls.
- **CPU and CUDA support** where supported by the installed PyTorch build.
- **Local model loading and caching** so AI models are reused after their first load during the application session.

---

## 🎯 Project Goal

Traditional object-learning applications usually separate **seeing**, **searching**, and **asking** into different steps. VizoLearn brings these interactions together.

The goal is to create a more natural learning flow:

> **See something → select it → understand it → ask about it → keep learning.**

This makes the application suitable for demonstrations, exploratory learning, AI/computer-vision coursework, interactive education prototypes, and hands-on experimentation with multimodal AI.

---

## 🧠 AI / ML Components

| Component | Technology | Purpose |
|---|---|---|
| Object detection | **YOLO11n** via Ultralytics | Detects objects in the live camera feed |
| Object tracking | **ByteTrack** | Keeps detections trackable across frames |
| Image understanding | **Salesforce BLIP** | Produces visual context for the selected object |
| Language model | **Qwen2.5-0.5B-Instruct** | Generates explanations and answers follow-up questions |
| Speech recognition | **Browser SpeechRecognition / webkitSpeechRecognition** | Converts spoken follow-up questions into text |
| Text-to-speech | **Browser SpeechSynthesis** | Reads AI answers aloud |
| UI / orchestration | **Streamlit + Streamlit WebRTC** | Runs the application and live camera interaction |

### Models used by the application

The project includes the YOLO11n weights in the repository as:

```text
yolo11n.pt
```

The image-captioning and language models are downloaded from Hugging Face when they are first loaded:

```text
Salesforce/blip-image-captioning-base
Qwen/Qwen2.5-0.5B-Instruct
```

Model loading is cached in the application, so the same process does not repeatedly reload the models for every question.

---

## 🏗️ Architecture

The application is intentionally split into small modules so that detection, selection, visual understanding, language generation, and voice interaction can be developed independently.

```text
VizoLearn/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── env.example                    # Optional environment configuration
├── pytest.ini                     # Pytest configuration
├── yolo11n.pt                     # YOLO11n detector weights
├── .streamlit/
│   └── config.toml                # Streamlit server configuration
├── src/
│   ├── __init__.py
│   ├── detector.py                # YOLO detection + ByteTrack
│   ├── crop.py                    # Crop selected detection
│   ├── selection.py               # Bounding-box hit testing / selection
│   ├── vision.py                  # BLIP image description
│   ├── explain.py                 # Qwen tutor generation + Q&A
│   ├── voice_ui.py                # Live voice UI + TTS controls
│   ├── listen.py                  # Optional faster-whisper path
│   ├── speak.py                   # Optional pyttsx3 TTS path
│   └── utils.py                   # Project utilities
└── scripts/
    └── download_whisper_model.py  # Optional Whisper model helper
```

### Main execution flow

1. `app.py` starts the Streamlit interface.
2. `streamlit-webrtc` opens the live camera stream.
3. `ObjectDetector` runs YOLO11 detection and ByteTrack tracking on incoming frames.
4. The camera preview displays bounding boxes around detected objects.
5. `streamlit-image-coordinates` captures where the user clicks on the preview.
6. `selection.py` determines which bounding box contains the click.
7. `crop.py` extracts a clean image crop from the selected object.
8. `vision.py` uses BLIP to create a visual description.
9. `explain.py` sends the detected class and visual context to Qwen.
10. The generated explanation is shown in the tutor workspace.
11. `voice_ui.py` provides speaker controls plus the voice/typing follow-up interface.
12. Spoken questions are transcribed in the browser and automatically submitted when the user finishes speaking.
13. Qwen answers the new question using the selected object, visual context, and recent conversation history.

---

## 🖥️ User Experience

### 1. Start the camera

Launch the application and press **Start Camera**.

Once camera permission is granted, the live detector begins processing frames.

### 2. Select an object

There are two ways to select an object:

- Click directly inside its detected bounding box in the camera preview.
- Click the corresponding detected-object button shown below the preview.

The selected object is shown with its detection confidence and a clean crop prepared for AI analysis.

### 3. Learn about the object

Press **Analyze & Teach Me** to generate a lesson.

You can also use the quick-learning actions:

- **How does it work?**
- **What is it for?**
- **Fun fact**
- **Teach me something surprising**

### 4. Listen to the explanation

Every answer provides browser-native speech controls. Depending on the browser and available voices, you can select a voice and use:

- **Read aloud**
- **Replay**
- **Pause / Resume**
- **Stop**

### 5. Ask by voice

The follow-up panel places **Ask by voice** before the typing option.

While speaking, the current recognition result is shown as a live transcript. When speech recognition ends, the final transcript is submitted as the next question.

### 6. Ask by typing

The same follow-up panel includes a text input and **Ask →** button, providing a normal keyboard-based path when needed.

### 7. Continue the conversation

Each question and answer is added to the tutor workspace so the interaction behaves like a continuous lesson instead of isolated requests.

---

## ⚙️ Requirements

Recommended environment:

- Python **3.11+**
- A webcam for live object detection
- A modern Chromium-based browser for the best speech-recognition experience
- Internet access on first run to download Hugging Face models
- Optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation for faster inference

The application can run on CPU, but object detection and local transformer inference will generally be slower than on a suitable GPU.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd VizoLearn
```

### 2. Create a virtual environment

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run VizoLearn

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

> **Note:** `streamlit-webrtc` needs browser camera permissions. When prompted, allow the site to use your camera.

---

## 🔐 Optional Configuration

The repository contains an `env.example` file with application settings.

Copy it to `.env` when you want to maintain local configuration values:

```bash
cp env.example .env
```

Current configuration examples include:

```text
VIZOLEARN_CONF=0.40
VIZOLEARN_IMGSZ=640
VIZOLEARN_FRAME_MAX_WIDTH=960

VIZOLEARN_ENABLE_VISION=1
VIZOLEARN_ENABLE_LLM=1

VIZOLEARN_VISION_MODEL=Salesforce/blip-image-captioning-base
VIZOLEARN_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

### Important configuration note

The current application code uses the bundled `yolo11n.pt` detector and defines the BLIP/Qwen model IDs directly in `src/vision.py` and `src/explain.py`. Therefore, the `env.example` values should be treated as configuration documentation for the project rather than a guarantee that every variable is dynamically read by the current implementation.

---

## 🎙️ Voice Input Details

The current live voice experience uses the browser's native speech-recognition API rather than sending recorded audio to a Python transcription service.

This gives the UI a more interactive experience:

1. Press the microphone control.
2. Speak naturally.
3. Interim recognition text is displayed while speaking.
4. The interface keeps the visible transcript updated without waiting for the whole sentence to finish.
5. When recognition ends, the final transcript is submitted to the tutor.

### Browser compatibility

Speech recognition support differs by browser and operating system. Chromium-based browsers are the primary target for the current implementation.

If browser speech recognition is unavailable, the typing input remains available.

### Microphone permissions

When speech recognition is started for the first time, the browser may request microphone access. Allow access for the VizoLearn page.

---

## 🔊 Text-to-Speech Details

VizoLearn uses browser-native `SpeechSynthesis` for answer playback.

The implementation deliberately keeps this client-side so that audio playback starts directly in the browser instead of generating server-side WAV files for every response.

A separate `src/speak.py` module contains a `pyttsx3`-based path for offline/no-JavaScript experiments, but that path is **not the active speech output path in the current Streamlit application**.

---

## 🗣️ Optional Whisper Path

The project also contains `src/listen.py` and `scripts/download_whisper_model.py` for a `faster-whisper` transcription path.

These modules are kept as an optional/offline-friendly alternative, but the current application uses browser-native speech recognition for live voice follow-up questions.

To inspect or prepare the optional Whisper model path:

```bash
python scripts/download_whisper_model.py --help
```

The exact model download command depends on the options supported by the script.

---

## 🧪 Testing

The project is configured for pytest with:

```text
pytest.ini
```

Run:

```bash
pytest
```

Lightweight utility modules such as detection selection and cropping are separated from the Streamlit UI, which makes them easier to test independently.

---

## 🐛 Troubleshooting

### Camera does not start

Check that:

- The browser has camera permission.
- No other application is exclusively using the webcam.
- You opened the actual Streamlit page rather than a blocked/embedded browser context.
- `streamlit-webrtc` installed successfully.

### No objects are detected

Try moving the camera closer, improving lighting, or pointing it at a common object supported by the YOLO model.

Detection confidence is currently initialized around `0.40` in the application.

### The first AI explanation is slow

The BLIP and Qwen models are loaded lazily. The first inference can therefore take noticeably longer while model files are downloaded and initialized. Later requests reuse cached models within the running process.

### GPU is not being used

Check that:

```python
import torch
print(torch.cuda.is_available())
```

returns `True` and that the installed PyTorch build matches your CUDA environment.

The application automatically falls back to CPU when CUDA is unavailable.

### Voice input does not work

Try a Chromium-based browser, allow microphone access, and make sure the page is served in a browser context where speech recognition is supported.

The typing field can always be used as a fallback.

### Speech playback does not work

Verify that the browser supports `speechSynthesis` and that your system has an available speech voice. The application also exposes a voice selector when the browser provides multiple voices.

---

## 🔒 Privacy / Data Flow Notes

The current application is designed around local inference for object detection, image captioning, and language generation after model files are available locally.

Browser camera frames are processed through the application's live WebRTC pipeline for object detection. The selected image crop is then passed to the local BLIP model for visual description, and the resulting context is supplied to the local Qwen model.

Voice follow-up input currently uses the browser speech-recognition API. Browser speech-recognition implementations may use browser/vendor services depending on the browser and platform; VizoLearn itself does not currently route the live voice transcript through a server-side Whisper model in `app.py`.

Do not use the application with sensitive environments or information unless you have reviewed the privacy behavior of your browser, operating system, model downloads, and deployment environment.

---

## 📦 Dependency Overview

The main packages in `requirements.txt` are:

```text
streamlit
streamlit-webrtc
streamlit-image-coordinates
ultralytics
opencv-python
numpy
Pillow
torch
transformers
accelerate
safetensors
av
faster-whisper
huggingface_hub
```

Their roles are broadly:

- **Streamlit** — application UI and state management.
- **streamlit-webrtc** — live camera streaming.
- **streamlit-image-coordinates** — click coordinates on the preview.
- **Ultralytics** — YOLO11 model inference and tracking.
- **OpenCV / NumPy / Pillow** — image processing.
- **PyTorch** — neural-network execution.
- **Transformers / Accelerate / Safetensors** — BLIP and Qwen inference/model loading.
- **PyAV (`av`)** — video-frame handling.
- **faster-whisper** — optional alternative transcription path.
- **huggingface_hub** — Hugging Face model access.

---

## 📁 File-by-File Overview

### `app.py`

The main entry point. It creates the Streamlit page, manages session state, starts the live WebRTC stream, renders object-selection controls, generates lessons, and connects the tutor workspace to voice and typed follow-up questions.

### `src/detector.py`

Defines the `ObjectDetector` abstraction and detection data structure. YOLO11 is executed with ByteTrack persistence so detections can retain track IDs across frames.

### `src/selection.py`

Contains hit-testing and selection helpers for deciding which detected object corresponds to a user click.

### `src/crop.py`

Extracts a padded crop around a selected bounding box. The application prefers the clean camera frame so detector labels/boxes are not accidentally included in the visual model input.

### `src/vision.py`

Loads BLIP and generates a visual description of the selected image crop.

### `src/explain.py`

Loads Qwen and generates the initial lesson plus conversational follow-up answers. The system prompt also tells the model not to invent visual details that cannot be supported by the provided context.

### `src/voice_ui.py`

Contains the interactive voice controls, live speech-recognition component, live transcript behavior, typed fallback, and browser-native speech playback.

### `src/listen.py`

Provides an alternative `faster-whisper` transcription implementation that is currently not wired into the live Streamlit app.

### `src/speak.py`

Provides an alternative `pyttsx3` text-to-speech implementation that is currently not used by the live app.

### `scripts/download_whisper_model.py`

Helper for preparing a local faster-whisper model when using the optional offline transcription path.

---

## 🔮 Future Improvements

Possible next steps for the project include:

- More accurate object tracking and selection across difficult camera motion.
- Support for additional detection models and custom datasets.
- Better grounding between the image crop and language-model answer.
- Multi-language speech recognition and speech output.
- Persistent lesson history and user profiles.
- Saved learning sessions and exportable notes.
- Confidence-aware explanations when the detector is uncertain.
- Improved mobile support and responsive camera controls.
- Optional fully offline speech recognition using faster-whisper.
- More educational response modes such as quizzes, hints, and teach-back activities.

---

## ⚠️ Limitations

VizoLearn is an educational prototype and should not be treated as an authoritative source for safety-critical, medical, legal, or technical decisions.

Object recognition can be wrong, image descriptions can be incomplete, and language-model explanations can contain errors. The tutor prompt intentionally asks the model to avoid unsupported visual claims, but model output should still be reviewed when accuracy matters.

Performance also depends heavily on available CPU/GPU resources, camera resolution, browser behavior, and the time required to load the local transformer models.

---

## 📜 License

No explicit license file is included in the current project archive. Add a `LICENSE` file before distributing the project publicly if you want to specify reuse, modification, or redistribution terms.

---

## 👨‍💻 Project Status

**Current status:** Functional interactive prototype.

The current repository contains the live Streamlit camera workflow, interactive object selection, AI-generated teaching responses, browser voice input with live transcript feedback, typed follow-ups, and browser speech playback.

---

## 🙌 Acknowledgements

VizoLearn builds on the work of the open-source and research communities behind:

- Streamlit
- Ultralytics YOLO
- ByteTrack
- Hugging Face Transformers
- Salesforce BLIP
- Qwen
- faster-whisper
- WebRTC / streamlit-webrtc

The project brings these technologies together as an educational interaction layer focused on learning from the world around you.

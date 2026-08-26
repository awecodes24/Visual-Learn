# VizoLearn

**See it. Select it. Learn about it. Ask more.**

VizoLearn is an interactive AI learning project built with Streamlit. The idea is simple: point your camera at something, let the app recognize it, select the object you want to learn about, and let the AI explain it to you.

What makes the project a little different from a normal object detector is that you do not have to stop at the detection. After selecting an object, you can ask questions about it by typing or speaking. When you speak, the transcript appears while you are talking, so the interaction feels more like a conversation than a traditional speech-to-text form.

The project combines computer vision, image understanding, a local language model, speech recognition, and text-to-speech in one Streamlit application.

---

## What VizoLearn does

The main workflow looks like this:

```text
Camera
  ↓
YOLO detects objects
  ↓
Choose an object
  ↓
VizoLearn crops the object
  ↓
BLIP describes what is visible
  ↓
Qwen explains it like a tutor
  ↓
Listen to the explanation
  ↓
Ask a follow-up question
      ↙        ↘
   Speak       Type
     ↓           ↓
Live transcript  Text input
      └─────┬─────┘
            ↓
       Qwen answers
```

### Current features

- Live camera feed through Streamlit WebRTC
- Real-time object detection with YOLO11n
- ByteTrack-based tracking for detected objects
- Clickable objects in the camera view
- Object-selection buttons below the camera preview
- Automatic cropping of the selected object
- BLIP image understanding for visual context
- Qwen2.5-0.5B-Instruct for explanations and follow-up answers
- Quick learning prompts such as **How does it work?** and **Fun fact**
- Voice questions using the browser speech-recognition API
- Live/interim transcript while speaking
- Typing as a fallback when voice is unavailable
- Browser text-to-speech for reading answers aloud
- Voice selection and replay/pause/stop controls
- Recent conversation context for follow-up questions
- CPU fallback and CUDA support when the installed PyTorch build supports it

---

## Why I built it

The project started from a simple idea: learning should not always require opening a textbook or searching for an answer somewhere else.

If you can see an object, you should be able to point at it and start learning about it.

So the goal of VizoLearn is to connect these steps into one flow:

> **See something → select it → learn about it → ask a question → keep learning.**

It is mainly an AI/computer-vision project and a working prototype rather than a production education platform. It is useful for demonstrating how different AI capabilities can work together in a single application.

---

## Technologies used

| Part | Technology | What it does |
|---|---|---|
| UI | Streamlit | Builds the application interface |
| Camera | streamlit-webrtc | Streams live camera frames |
| Detection | YOLO11n / Ultralytics | Finds objects in the camera feed |
| Tracking | ByteTrack | Keeps object detections trackable across frames |
| Image understanding | Salesforce BLIP | Describes the selected image crop |
| Tutor model | Qwen2.5-0.5B-Instruct | Generates explanations and answers questions |
| Voice input | Browser SpeechRecognition | Turns spoken questions into text |
| Voice output | Browser SpeechSynthesis | Reads AI responses aloud |
| Image processing | OpenCV, Pillow, NumPy | Handles frames and image crops |
| Model runtime | PyTorch, Transformers | Runs the AI models |

### Models

The repository contains the YOLO weights:

```text
yolo11n.pt
```

The other models are loaded from Hugging Face when they are first needed:

```text
Salesforce/blip-image-captioning-base
Qwen/Qwen2.5-0.5B-Instruct
```

The models are cached by the application so they do not need to be loaded from scratch for every interaction during the same run.

---

## Project structure

```text
VizoLearn/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── env.example                    # Example configuration values
├── pytest.ini                     # Pytest configuration
├── yolo11n.pt                     # YOLO11n weights
│
├── .streamlit/
│   └── config.toml                # Streamlit configuration
│
├── src/
│   ├── __init__.py
│   ├── detector.py                # Detection and tracking
│   ├── crop.py                    # Selected-object cropping
│   ├── selection.py               # Click / bounding-box selection
│   ├── vision.py                  # BLIP image understanding
│   ├── explain.py                 # Qwen tutor and Q&A logic
│   ├── voice_ui.py                # Live voice UI and browser TTS
│   ├── listen.py                  # Optional faster-whisper path
│   ├── speak.py                   # Optional pyttsx3 path
│   └── utils.py                   # Utility functions
│
└── scripts/
    └── download_whisper_model.py  # Optional Whisper helper
```

---

## How the application works

### 1. Start the camera

Open the Streamlit app and press **Start Camera**. After the browser gives camera permission, frames begin flowing through the detection pipeline.

### 2. Detect and track objects

YOLO11n looks for objects in each frame. ByteTrack is used so detections can keep track IDs as objects move from frame to frame.

### 3. Select an object

There are two ways to choose what you want to learn about:

- Click directly inside a detected bounding box.
- Use the detected-object buttons shown below the camera preview.

Once selected, the application keeps the object crop separate from the visual overlays used on the camera preview.

### 4. Generate a lesson

When you choose **Analyze & Teach Me**, the selected crop is passed to BLIP first. BLIP provides visual context, which is then given to Qwen along with the detected object name.

Qwen uses that context to generate a short explanation instead of trying to answer from the object label alone.

### 5. Ask follow-up questions

After the first explanation, the tutor area lets you continue the conversation.

You can type a question or use **Ask by voice**. During voice input, the browser shows the recognised words as you speak. Once recognition finishes, the final transcript is sent to the tutor.

### 6. Listen to the answer

The application uses the browser's built-in speech synthesis to read answers aloud. The UI also provides playback controls such as replay, pause/resume, and stop.

---

## Getting started

### Requirements

You will need:

- Python 3.11 or newer
- A webcam for the live camera feature
- A modern browser (Chromium-based browsers are the main target for the current voice experience)
- Internet access the first time the Hugging Face models are downloaded
- A suitable NVIDIA GPU if you want faster inference through CUDA

The project can run on CPU, but model loading and inference will usually be slower.

### 1. Clone the project

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

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Start VizoLearn

```bash
python -m streamlit run app.py
```

Open the Streamlit address shown in the terminal. In most setups it will be:

```text
http://localhost:8501
```

Allow camera and microphone access in the browser when prompted.

---

## Using the voice feature

Voice input is handled in the browser instead of recording an audio file and sending the recording to a Python transcription service.

The normal flow is:

1. Press the microphone / **Ask by voice** control.
2. Start speaking.
3. Watch the transcript update while you speak.
4. Finish speaking.
5. The final recognised question is sent to the tutor.

This is also why the experience can vary from one browser to another. Speech-recognition support is not identical everywhere.

### If voice input is unavailable

You can still use the typing field in the follow-up section. The text-question path does not depend on browser speech recognition.

---

## Text-to-speech

The current app uses the browser's `SpeechSynthesis` API for spoken answers.

That means the response is read directly by the browser rather than generating a new audio file on the server for every answer.

There is also a `src/speak.py` module containing a `pyttsx3` implementation. It is kept as an alternative path for experimentation, but it is not the main speech-output path used by the current Streamlit interface.

---

## Optional Whisper support

The project also contains an alternative transcription path based on `faster-whisper`:

```text
src/listen.py
scripts/download_whisper_model.py
```

This path is not currently used by the live voice interaction in `app.py`. The current UI uses browser speech recognition because it provides the live, word-by-word/interim transcript experience used by the project.

You can inspect the helper script with:

```bash
python scripts/download_whisper_model.py --help
```

---

## Configuration

The project includes an `env.example` file with example settings such as detector confidence, image size, and model names.

For example:

```text
VIZOLEARN_CONF=0.40
VIZOLEARN_IMGSZ=640
VIZOLEARN_FRAME_MAX_WIDTH=960

VIZOLEARN_ENABLE_VISION=1
VIZOLEARN_ENABLE_LLM=1

VIZOLEARN_VISION_MODEL=Salesforce/blip-image-captioning-base
VIZOLEARN_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

One important detail: the current application still defines some model settings directly in the Python source files. So `env.example` should be treated as a guide for configuration, not as proof that every variable is automatically read at runtime.

---

## Testing

The project includes a `pytest.ini` configuration file.

Run the test suite with:

```bash
pytest
```

The code is split so that pieces such as object selection and image cropping can be tested independently from the Streamlit UI.

---

## Troubleshooting

### The camera does not start

Make sure:

- the browser has permission to use the camera,
- another application is not already locking the webcam,
- `streamlit-webrtc` installed correctly, and
- you are opening the actual Streamlit page.

### Nothing is detected

Try pointing the camera at a common object in better lighting and moving close enough for the object to be clearly visible.

The detector confidence is currently around `0.40` by default.

### The first explanation takes a while

That is normal on the first run. BLIP and Qwen have to be downloaded and loaded before they can generate an answer. Later requests can reuse the loaded models while the application is running.

### CUDA is not being used

Check your PyTorch installation:

```python
import torch
print(torch.cuda.is_available())
```

If it prints `False`, the application will use CPU inference instead.

### Voice input does not work

Try a Chromium-based browser, allow microphone permission, and make sure the browser supports speech recognition.

The text input is always available as a fallback.

### Text-to-speech does not work

Check that your browser supports `speechSynthesis` and that your operating system/browser has at least one available voice.

---

## A note about privacy

VizoLearn is designed so that object detection, image understanding, and language generation can happen locally after the required model files are available.

However, the voice feature is different. The current live voice input uses the browser's speech-recognition API. Depending on the browser and operating system, speech recognition may involve browser or vendor services outside the Python application itself.

For that reason, you should avoid using the project with sensitive audio, images, or information unless you have reviewed the privacy behavior of your browser and deployment environment.

---

## Main dependencies

The project uses packages including:

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

In simple terms:

- **Streamlit** handles the app interface.
- **streamlit-webrtc** handles the live camera stream.
- **Ultralytics** runs YOLO11 detection and tracking.
- **OpenCV / Pillow / NumPy** handle image processing.
- **PyTorch / Transformers** run BLIP and Qwen.
- **faster-whisper** is available for the optional transcription path.

---

## A quick look at the main files

### `app.py`

This is the entry point of the project. It connects the camera, detector, object selection, AI explanation, conversation state, and voice interface.

### `src/detector.py`

Contains the YOLO detection and ByteTrack tracking logic.

### `src/selection.py`

Handles the logic for figuring out which detected object the user clicked.

### `src/crop.py`

Creates a clean crop of the selected object before sending it to the vision model.

### `src/vision.py`

Loads BLIP and generates a description of the selected crop.

### `src/explain.py`

Loads Qwen and generates the initial lesson and follow-up answers.

### `src/voice_ui.py`

Contains the interactive voice input, live transcript behaviour, typed follow-up input, and browser speech controls.

### `src/listen.py`

Contains the optional faster-whisper transcription implementation.

### `src/speak.py`

Contains the optional pyttsx3 speech-output implementation.

---

## Current limitations

VizoLearn is a working prototype, so there are still a few practical limitations:

- YOLO can miss objects or classify them incorrectly.
- BLIP can misunderstand what is visible in a crop.
- Qwen can generate an incorrect explanation even when the detected object is correct.
- Browser speech recognition behaves differently across browsers and operating systems.
- CPU-only machines can feel slow when loading or running the transformer models.
- The project is not intended for medical, legal, safety-critical, or other high-stakes decisions.

The goal is to make the interaction useful and enjoyable, not to pretend that every AI output is automatically correct.

---

## What could be added next

There are several directions that would make the project more complete:

- Better object selection when several objects overlap
- Support for custom object classes and custom YOLO datasets
- Stronger grounding between the selected image and the generated explanation
- More languages for speech input and output
- Saved learning sessions and lesson history
- Quiz and teach-back modes
- Better mobile/responsive camera support
- A fully offline speech-recognition option using faster-whisper
- Confidence-aware answers when the detector is unsure

---

## Project status

**Status: Functional prototype**

The current version includes the main end-to-end workflow: live camera detection, interactive object selection, AI-generated teaching, follow-up conversation, live voice transcription, typed questions, and browser-based speech playback.

It is best thought of as a project/demo that shows how multiple AI capabilities can be combined into one useful interaction.

---

## Credits

This project builds on several excellent open-source and research projects, including:

- Streamlit
- Ultralytics YOLO
- ByteTrack
- Hugging Face Transformers
- Salesforce BLIP
- Qwen
- faster-whisper
- WebRTC / streamlit-webrtc

The interesting part of VizoLearn is not trying to replace these tools. It is putting them together into one simple idea: **point at something, learn about it, and keep asking questions.**

---

## License

There is currently no `LICENSE` file in the project repository. If you plan to publish or redistribute VizoLearn, add an appropriate license file so that the reuse terms are clear.

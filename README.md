# VizoLearn — Streamlit Real-Time AI Visual Tutor

VizoLearn combines:

- Streamlit UI
- `streamlit-webrtc` live webcam streaming
- YOLO11n detection + ByteTrack tracking
- click-to-select detected objects
- selected-object cropping
- BLIP visual description
- Qwen2.5-0.5B-Instruct educational explanations
- browser-native SpeechSynthesis text-to-speech with selectable voices, pause/resume, and stop
- local faster-whisper speech-to-text
- follow-up conversational tutoring

## Important Streamlit design choice

`st.camera_input()` captures camera pictures; it is not a continuous video source. VizoLearn therefore uses `streamlit-webrtc` for the live stream and `streamlit-image-coordinates` for click coordinates on the most recent processed frame.

## Setup

Use your existing CUDA/PyTorch environment when possible. Do not blindly replace a working CUDA-enabled PyTorch installation just to satisfy a generic pip command.

```bash
pip install -r requirements.txt
```

The first run downloads the Hugging Face models:

- `Salesforce/blip-image-captioning-base`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Systran/faster-whisper` / CTranslate2 assets as required by the Whisper model

## Run

```bash
streamlit run app.py
```

Open the URL printed by Streamlit and allow webcam access.

## User flow

1. Start the live camera.
2. YOLO detects/tracks objects.
3. Click the actual object in the processed-frame panel.
4. The object crop becomes the multimodal input.
5. BLIP supplies visual context.
6. Qwen generates a structured lesson.
7. The browser speaks the lesson using its native high-quality SpeechSynthesis voice.
8. Type or record a follow-up question.
9. Whisper transcribes recorded audio.
10. Qwen answers using the selected object's context and conversation history.


## Interaction improvements

- Tap the live bounding box **or** select the detected-object button beneath the camera.
- Use quick actions such as **How does it work?**, **What is it for?**, and **Fun fact** without typing a question.
- Every tutor answer has **Read aloud**, **Pause**, and **Stop** controls plus a selectable browser voice.
- Voice playback runs locally in the browser, avoiding the robotic Linux `espeak` playback used by the older version.

### Voice requirements

Playback uses the browser's Web Speech API, so sound quality depends on the voices available in the browser/operating system. Chrome on Linux usually exposes the installed system voices; installing additional high-quality voices in the OS can improve the voice options shown by VizoLearn.

## Tests

```bash
pytest -q
```

## Hardware

The project is GPU-aware. YOLO, BLIP, Qwen and Whisper automatically use CUDA when PyTorch reports CUDA availability, otherwise they fall back to CPU where supported.

For an RTX 4050 laptop, keep YOLO at `yolo11n` and Qwen at `0.5B` initially. BLIP and Whisper are the heavier optional stages.

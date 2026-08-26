from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from streamlit_image_coordinates import streamlit_image_coordinates

from src.crop import crop_detection
from src.detector import ObjectDetector
from src.explain import answer_question, explain_object
from src.selection import hit_test
from src.voice_ui import answer_voice_controls
from src.vision import describe_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolo11n.pt"

st.set_page_config(
    page_title="VisualizeLearning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Styling — deliberately keeps the page close to the requested wireframe.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* Make spinner text black */
.stSpinner > div > div {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
:root {
  --vz-bg: #f6f8fc;
  --vz-card: #ffffff;
  --vz-border: #e4e9f2;
  --vz-text: #142033;
  --vz-muted: #667085;
  --vz-primary: #5b5ce2;
  --vz-primary-dark: #4647bd;
  --vz-soft: #f0f1ff;
  --vz-success: #12a36f;
}
.stApp { background: var(--vz-bg); }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }

.vz-brand { text-align: center; margin-bottom: 0.25rem; }
.vz-brand h1 { font-size: 2.45rem; line-height: 1.1; margin: 0; color: var(--vz-text); font-weight: 800; letter-spacing: -0.03em; }
.vz-brand p { margin: 0.45rem 0 0; color: var(--vz-muted); font-size: 1.03rem; }

.vz-section {
  background: var(--vz-card);
  border: 1px solid var(--vz-border);
  border-radius: 22px;
  padding: 1.5rem;
  box-shadow: 0 10px 30px rgba(20,32,51,.05);
  margin-top: 1.1rem;
}
.vz-section-title { font-size: 1.32rem; font-weight: 750; color: var(--vz-text); margin: 0; }
.vz-section-subtitle { margin: 0.4rem 0 0; color: var(--vz-muted); font-size: .94rem; }

.vz-camera-bar { display:flex; justify-content:center; margin: 1.1rem 0 .85rem; }
.vz-camera-note { text-align:center; color:var(--vz-muted); font-size:.88rem; margin-top:.6rem; }

.vz-selected-card {
  border: 1px solid #d9defe;
  background: linear-gradient(135deg, #f8f8ff, #f2f7ff);
  border-radius: 18px;
  padding: 1.2rem;
}
.vz-selected-name { font-size: 1.55rem; font-weight: 800; color: var(--vz-text); }
.vz-selected-meta { color: var(--vz-muted); font-size: .9rem; margin-top: .2rem; }

.vz-tutor {
  background: var(--vz-card);
  border: 1px solid var(--vz-border);
  border-radius: 22px;
  padding: 1.55rem;
  box-shadow: 0 10px 30px rgba(20,32,51,.05);
  margin-top: 1.1rem;
}
.vz-tutor-title { font-size: 1.5rem; font-weight: 800; color:var(--vz-text); }
.vz-tutor-caption { color:var(--vz-muted); margin-top:.2rem; }
.vz-lesson-label { font-size:.78rem; letter-spacing:.12em; font-weight:800; color:var(--vz-primary); text-transform:uppercase; margin-bottom:.55rem; }
.vz-answer-card {
  border: 1px solid var(--vz-border);
  border-radius: 18px;
  padding: 1.2rem 1.25rem;
  background: #fff;
  margin-top: .95rem;
}
.vz-answer-user { color:#5b5ce2; font-weight:750; font-size:.86rem; margin-bottom:.25rem; }
.vz-answer-tutor { color:var(--vz-text); font-weight:750; font-size:.86rem; margin-bottom:.55rem; }
.vz-answer-text { color:#273449; font-size:1rem; line-height:1.75; white-space:pre-wrap; }
.vz-follow-label { color:var(--vz-text); font-weight:750; font-size:.9rem; margin-top:1rem; margin-bottom:.35rem; }

.vz-empty {
  text-align:center;
  padding: 1.6rem 1rem;
  color:var(--vz-muted);
  border:1px dashed #cfd6e4;
  border-radius:16px;
  background:#fbfcff;
}

/* Make ordinary Streamlit buttons feel like deliberate product controls. */
.stButton > button, .stDownloadButton > button {
  border-radius: 12px !important;
  min-height: 42px !important;
  font-weight: 700 !important;
}
</style>

<div class="vz-brand">
  <h1>Visual Learn</h1>
  <p>Point and Learn</p>
</div>
""",
    unsafe_allow_html=True,
)


class VizoVideoProcessor(VideoProcessorBase):
    def __init__(self, confidence: float = 0.40):
        self.detector = ObjectDetector(str(MODEL_PATH), confidence=confidence)
        self.latest_frame: np.ndarray | None = None
        # Clean, pre-annotation frame — used for cropping so the vision model
        # never sees the drawn bounding box or confidence-label text.
        self.latest_clean_frame: np.ndarray | None = None
        self.latest_detections: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.frames = 0
        self.last_error: str | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        try:
            detections = self.detector.detect(image)
            annotated = self._annotate(image.copy(), detections)
            with self.lock:
                self.latest_frame = annotated
                self.latest_clean_frame = image
                self.latest_detections = detections
                self.frames += 1
                self.last_error = None
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as exc:
            with self.lock:
                self.latest_frame = image.copy()
                self.latest_clean_frame = image.copy()
                self.latest_detections = []
                self.last_error = str(exc)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    @staticmethod
    def _annotate(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label = f"{d['name']}  {d['confidence']:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (91, 92, 226), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .58, 2)
            top = max(0, y1 - th - 12)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 10, y1), (91, 92, 226), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, .58, (255,255,255), 2)
        return frame

    def snapshot(self):
        with self.lock:
            return (
                None if self.latest_frame is None else self.latest_frame.copy(),
                None if self.latest_clean_frame is None else self.latest_clean_frame.copy(),
                [dict(d) for d in self.latest_detections],
                self.last_error,
                self.frames,
            )


def init_state() -> None:
    defaults = {
        "camera_on": False,
        "selected_detection": None,
        "selected_crop": None,
        "visual_context": "",
        "explanation": "",
        "chat_history": [],
        "last_click_signature": None,
        "last_voice_question": "",
        "confidence": 0.40,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def reset_learning() -> None:
    st.session_state.selected_detection = None
    st.session_state.selected_crop = None
    st.session_state.visual_context = ""
    st.session_state.explanation = ""
    st.session_state.chat_history = []
    st.session_state.last_voice_question = ""
    st.session_state.last_click_signature = None


def select_detection(detection: dict[str, Any], clean_frame: np.ndarray | None, padding: int = 12) -> None:
    """Select a live detection and prepare a clean crop for the AI pipeline."""
    st.session_state.selected_detection = detection
    if clean_frame is not None:
        st.session_state.selected_crop = crop_detection(clean_frame, detection, padding=padding)
    else:
        st.session_state.selected_crop = None
    st.session_state.visual_context = ""
    st.session_state.explanation = ""
    st.session_state.chat_history = []
    st.session_state.last_voice_question = ""


def analyze_selected() -> None:
    selected = st.session_state.get("selected_detection")
    crop = st.session_state.get("selected_crop")
    if selected is None or crop is None:
        st.warning("Select a detected object first.")
        return
    try:
        with st.spinner("Looking closely at the object…"):
            context = describe_image(crop)
        with st.spinner("Building a personalized lesson…"):
            explanation = explain_object(selected["name"], context)
        st.session_state.visual_context = context
        st.session_state.explanation = explanation
        st.session_state.chat_history = []
        st.session_state.last_voice_question = ""
        st.rerun()
    except Exception as exc:
        st.error(f"AI analysis failed: {exc}")


def ask_quick_question(question: str) -> None:
    selected = st.session_state.get("selected_detection")
    if selected is None:
        return
    try:
        with st.spinner("💡 Thinking…"):
            answer = answer_question(
                selected["name"],
                st.session_state.get("visual_context", ""),
                st.session_state.get("chat_history", []),
                question,
            )
        if not st.session_state.get("explanation"):
            st.session_state.explanation = answer
            st.session_state.chat_history = []
        else:
            st.session_state.chat_history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
        st.rerun()
    except Exception as exc:
        st.error(f"I couldn't answer that right now: {exc}")


# -----------------------------------------------------------------------------
# 1. EXACT requested starting section: no "Live camera" section.
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="vz-section">
  <div class="vz-section-title">Click the object you want to learn about</div>
  <div class="vz-section-subtitle">Turn the camera on, point it at your surroundings, then tap an object box or choose it from the live object buttons below.</div>
</div>
""",
    unsafe_allow_html=True,
)

camera_col_l, camera_col, camera_col_r = st.columns([1, 1, 1])
with camera_col:
    st.markdown("<div class='vz-camera-bar'>", unsafe_allow_html=True)
    camera_on = st.toggle(
        "📷 Camera",
        value=st.session_state.camera_on,
        key="camera_toggle",
        help="Turn the live camera and object detection on or off.",
    )
    st.markdown("</div>", unsafe_allow_html=True)
st.session_state.camera_on = camera_on

if not camera_on:
    st.markdown(
        "<div class='vz-empty'><strong>Camera is off.</strong><br>Turn it on using the centered toggle above to begin object detection.</div>",
        unsafe_allow_html=True,
    )
else:
    # No standalone heading: the camera lives ONLY inside the click-object section.
    confidence = float(st.session_state.get("confidence", 0.40))
    ctx = webrtc_streamer(
        key="visualizeLearning-camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        video_processor_factory=lambda: VizoVideoProcessor(confidence=confidence),
        desired_playing_state=True,
        async_processing=True,
    )

    @st.fragment(run_every="0.8s")
    def click_surface() -> None:
        processor = ctx.video_processor if ctx.state.playing else None
        if processor is None:
            st.markdown(
                "<div class='vz-empty'>Allow camera access to start the interactive detector.</div>",
                unsafe_allow_html=True,
            )
            return

        frame, clean_frame, detections, error, _ = processor.snapshot()
        if error:
            st.warning(f"Frame-processing issue: {error}")
        if frame is None:
            st.markdown("<div class='vz-empty'>Preparing the camera…</div>", unsafe_allow_html=True)
            return

        # `frame` (annotated, with boxes drawn) is used for the clickable
        # preview so the user can see what to click. `clean_frame` (no boxes
        # or labels burned in) is what gets cropped and sent to the vision
        # model, so its description is never contaminated by overlay text.
        h, w = frame.shape[:2]
        display_width = min(980, w)
        scale = display_width / w
        display_h = int(round(h * scale))
        preview = cv2.resize(frame, (display_width, display_h), interpolation=cv2.INTER_AREA)

        click = streamlit_image_coordinates(
            preview,
            key="visualizeLearning-click-target",
            width=display_width,
            cursor="crosshair",
        )

        if detections:
            st.markdown("<div style='margin-top:.65rem;font-weight:800;color:var(--vz-text);'>Detected right now</div>", unsafe_allow_html=True)
            object_cols = st.columns(min(4, len(detections)))
            for idx, detection in enumerate(detections):
                x1, y1, x2, y2 = detection["box"]
                label = f"{detection['name']} · {detection['confidence']:.0%}"
                with object_cols[idx % len(object_cols)]:
                    if st.button(
                        f"🎯 {label}",
                        key=f"live-object-{detection.get('track_id', 'noid')}-{idx}",
                        use_container_width=True,
                        help=f"Select the {detection['name']} at ({x1}, {y1})",
                    ):
                        crop_source = clean_frame if clean_frame is not None else frame
                        select_detection(detection, crop_source)
                        st.rerun()
        else:
            st.info("No objects are confidently detected yet. Move the camera slightly or get closer.")

        st.markdown(
            f"<div class='vz-camera-note'>{len(detections)} object(s) detected · Click the box or tap an object button</div>",
            unsafe_allow_html=True,
        )

        if click:
            signature = (int(click["x"]), int(click["y"]), int(click.get("unix_time", 0)))
            if signature != st.session_state.last_click_signature:
                st.session_state.last_click_signature = signature
                raw_x = int(round(click["x"] / scale))
                raw_y = int(round(click["y"] / scale))
                matched = hit_test(detections, raw_x, raw_y)
                if matched is not None:
                    crop_source = clean_frame if clean_frame is not None else frame
                    select_detection(matched, crop_source)
                    st.rerun()
                else:
                    st.toast("Click inside a detected bounding box to select an object.")

    click_surface()


# -----------------------------------------------------------------------------
# Selected object + Analyze button.
# -----------------------------------------------------------------------------
selected = st.session_state.selected_detection
crop = st.session_state.selected_crop

if selected is None:
    st.markdown(
        "<div class='vz-empty' style='margin-top:1rem'>Your selected object will appear here after you click one in the camera.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='vz-section'>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        if crop is not None:
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
    with right:
        st.markdown(
            f"<div class='vz-selected-card'><div class='vz-selected-name'>{selected['name']}</div><div class='vz-selected-meta'>Detected with {selected['confidence']:.0%} confidence · ready to teach</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Analyze & Teach Me", type="primary", use_container_width=True, key="analyze_button"):
            analyze_selected()

        st.markdown("<div style='margin-top:.85rem;font-size:.86rem;font-weight:800;color:var(--vz-text);'>Or jump straight into learning</div>", unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            if st.button("How does it work?", use_container_width=True, key="quick-how"):
                if not st.session_state.visual_context and crop is not None:
                    try:
                        st.session_state.visual_context = describe_image(crop)
                    except Exception:
                        pass
                ask_quick_question("How does this object work? Explain it simply with an everyday example.")
            if st.button("Fun fact", use_container_width=True, key="quick-fun"):
                if not st.session_state.visual_context and crop is not None:
                    try:
                        st.session_state.visual_context = describe_image(crop)
                    except Exception:
                        pass
                ask_quick_question("Tell me one accurate, interesting fun fact about this object.")
        with q2:
            if st.button("What is it for?", use_container_width=True, key="quick-use"):
                if not st.session_state.visual_context and crop is not None:
                    try:
                        st.session_state.visual_context = describe_image(crop)
                    except Exception:
                        pass
                ask_quick_question("What is this object mainly used for, and when would a person use it?")
            if st.button("Teach me something surprising", use_container_width=True, key="quick-surprise"):
                if not st.session_state.visual_context and crop is not None:
                    try:
                        st.session_state.visual_context = describe_image(crop)
                    except Exception:
                        pass
                ask_quick_question("Teach me one surprising but reliable thing about this object that a student would remember.")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Tutor workspace — one continuous panel exactly as requested.
#
# The initial lesson and every follow-up Q&A pair are treated as one ordered
# list of "answer cards" so that exactly one rule decides which card shows
# the follow-up section: whichever card is rendered last. Every card gets
# its own speaker button regardless of position.
# -----------------------------------------------------------------------------
if st.session_state.explanation:
    st.markdown(
        """
<div class="vz-tutor">
  <div class="vz-tutor-title">Tutor Workspace</div>
  <div class="vz-tutor-caption">Learn about the selected object, listen to any answer, and keep asking follow-up questions from the same panel.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Build the ordered list of cards: the lesson first, then each Q&A pair.
    cards: list[dict[str, str]] = [
        {"kind": "lesson", "user": "", "answer": st.session_state.explanation}
    ]
    for idx in range(0, len(st.session_state.chat_history), 2):
        user_message = st.session_state.chat_history[idx]
        assistant_message = (
            st.session_state.chat_history[idx + 1]
            if idx + 1 < len(st.session_state.chat_history)
            else None
        )
        if assistant_message is None:
            continue
        cards.append(
            {
                "kind": "qa",
                "user": user_message["content"],
                "answer": assistant_message["content"],
            }
        )

    new_question = ""
    last_index = len(cards) - 1

    for i, card in enumerate(cards):
        is_last = i == last_index
        st.markdown("<div class='vz-answer-card'>", unsafe_allow_html=True)

        if card["kind"] == "lesson":
            st.markdown("<div class='vz-lesson-label'>Object Lesson</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='vz-answer-user'>You</div><div class='vz-answer-text'>{card['user']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='vz-answer-tutor'>VisualizeLearning</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='vz-answer-text'>{card['answer']}</div>", unsafe_allow_html=True)

        # Every card gets a speaker button. Only the LAST card gets the
        # follow-up question section — this is what show_question controls.
        voice_result = answer_voice_controls(
            text=card["answer"],
            key=f"answer-voice-{i}",
            show_question=is_last,
            placeholder="Ask a follow-up question…" if card["kind"] == "lesson" else "Continue the conversation…",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if is_last and voice_result.question:
            new_question = voice_result.question

    if new_question and new_question != st.session_state.last_voice_question:
        st.session_state.last_voice_question = new_question
        try:
            with st.spinner("Thinking…"):
                answer = answer_question(
                    selected["name"],
                    st.session_state.visual_context,
                    st.session_state.chat_history,
                    new_question,
                )
        except Exception as exc:
            answer = f"I couldn't answer that right now: {exc}"
        st.session_state.chat_history.extend(
            [
                {"role": "user", "content": new_question},
                {"role": "assistant", "content": answer},
            ]
        )
        st.rerun()
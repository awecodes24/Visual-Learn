from __future__ import annotations

# STATUS: not currently used by app.py. The live app's speech-to-text is
# browser-native (SpeechRecognition/webkitSpeechRecognition, in
# src/voice_ui.py:_live_voice_component), which needs no server-side model,
# no download step, and no GPU. This faster-whisper-based path (together
# with scripts/download_whisper_model.py) is kept because it works and may
# be useful for an offline or non-Chromium deployment, but importing it
# does nothing unless you wire it into app.py yourself.

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_SOURCE: Optional[str] = None


def _candidate_local_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    configured = os.getenv("VIZO_WHISPER_MODEL_PATH", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            root / "models" / "faster-whisper-base",
            root / "models" / "faster-whisper-small",
            root / "models" / "faster-whisper-tiny",
        ]
    )
    return candidates


def _local_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (path / "model.bin").exists()


def _get_model():
    """Load Whisper once, preferring a fully downloaded local model.

    A local model avoids Hugging Face network checks during every Streamlit rerun.
    If no local model exists, we allow a one-time online download and provide a
    clear error explaining how to install the model locally if the download fails.
    """
    global _MODEL, _MODEL_SOURCE
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        from faster_whisper import WhisperModel
        import torch

        requested = os.getenv("VIZO_WHISPER_MODEL", "base").strip() or "base"
        local_candidates = _candidate_local_paths()
        for path in local_candidates:
            if _local_model(path):
                try:
                    _MODEL = WhisperModel(
                        str(path),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        compute_type="float16" if torch.cuda.is_available() else "int8",
                    )
                    _MODEL_SOURCE = str(path)
                    return _MODEL
                except Exception as exc:
                    # Try the next local path rather than immediately failing.
                    last_local_error = exc
                    continue

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            _MODEL = WhisperModel(
                requested,
                device=device,
                compute_type=compute_type,
            )
            _MODEL_SOURCE = requested
            return _MODEL
        except Exception as exc:
            local_dirs = "\n".join(f"  - {p}" for p in local_candidates)
            raise RuntimeError(
                "Whisper model could not be loaded. The model is not available locally "
                "and Hugging Face could not be reached.\n\n"
                f"Requested model: {requested}\n"
                f"Original error: {exc}\n\n"
                "Fix: download the model once with:\n"
                "  python scripts/download_whisper_model.py --model base\n\n"
                "Then restart Streamlit. The downloader saves a complete local model "
                "and VizoLearn will use it without contacting Hugging Face during transcription.\n\n"
                "Expected local model locations:\n"
                f"{local_dirs}"
            ) from exc


def whisper_status() -> dict[str, str | bool]:
    """Return lightweight status information without forcing a model download."""
    local = [p for p in _candidate_local_paths() if _local_model(p)]
    return {
        "installed": True,
        "local_model": bool(local),
        "local_path": str(local[0]) if local else "",
        "model_source": _MODEL_SOURCE or "not loaded",
    }


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not audio_bytes:
        return ""

    # Unique filename prevents concurrent Streamlit reruns from overwriting audio.
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(prefix="vizolearn_input_", suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = Path(tmp.name)

    try:
        model = _get_model()
        segments, _info = model.transcribe(
            str(audio_path),
            vad_filter=True,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
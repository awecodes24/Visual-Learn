from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import tempfile
import threading
import time
import uuid


_TTS_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _engine():
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    return engine


def speak_to_file(text: str) -> str:
    """Synthesize text to a guaranteed-writable temporary WAV file.

    A unique filename is used for every synthesis because reusing one fixed
    /tmp filename can race with Streamlit reruns and can cause pyttsx3/espeak
    to return before the expected file exists.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot synthesize empty text.")

    tmp_dir = Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output = tmp_dir / f"vizolearn_tts_{uuid.uuid4().hex}.wav"

    with _TTS_LOCK:
        engine = _engine()

        # Remove a pathological pre-existing path, although UUID makes this
        # practically impossible.
        try:
            output.unlink(missing_ok=True)
        except TypeError:
            if output.exists():
                output.unlink()

        engine.save_to_file(text, str(output))
        engine.runAndWait()

        # Some Linux speech drivers finalize the WAV immediately after
        # runAndWait(). Give the filesystem/driver a brief window to settle.
        deadline = time.monotonic() + 5.0
        last_size = -1
        stable_reads = 0

        while time.monotonic() < deadline:
            if output.exists():
                size = output.stat().st_size
                if size > 44 and size == last_size:
                    stable_reads += 1
                    if stable_reads >= 2:
                        break
                else:
                    stable_reads = 0
                    last_size = size
            time.sleep(0.05)

    if not output.exists():
        raise RuntimeError(
            "TTS synthesis completed but no WAV file was produced. "
            "On Linux, verify that espeak-ng is installed and working."
        )

    if output.stat().st_size <= 44:
        try:
            output.unlink()
        except OSError:
            pass
        raise RuntimeError("TTS produced an empty or invalid WAV file.")

    return str(output)
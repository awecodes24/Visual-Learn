from __future__ import annotations

import json
import html
import re
from dataclasses import dataclass

import streamlit as st
import streamlit.components.v1 as components


@dataclass
class VoiceResult:
    """Result from the interactive follow-up controls."""

    question: str = ""


def _safe_text(text: str) -> str:
    """Normalize text before it is placed in a browser-side speech widget."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def browser_speech(text: str, key: str) -> None:
    """Render high-quality browser-native speech controls.

    This uses the browser's SpeechSynthesis voices instead of Linux espeak.
    The controls are intentionally client-side, so playback starts immediately
    and does not regenerate WAV files on Streamlit reruns.
    """
    text = _safe_text(text)
    if not text:
        return

    payload = json.dumps(text, ensure_ascii=False)
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "-", key)
    components.html(
        f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:transparent; }}
  .bar {{ display:flex; align-items:center; gap:7px; flex-wrap:wrap; }}
  button, select {{
    border:1px solid #d9def0; background:#f7f8ff; color:#26314a;
    border-radius:10px; padding:7px 10px; font-size:13px; font-weight:700;
    cursor:pointer;
  }}
  button:hover, select:hover {{ background:#eef0ff; border-color:#bfc5f3; }}
  button.primary {{ background:#5b5ce2; border-color:#5b5ce2; color:white; }}
  button.primary:hover {{ background:#4647bd; }}
  select {{ max-width:240px; font-weight:600; }}
  .status {{ color:#667085; font-size:12px; min-width:84px; }}
</style>
</head>
<body>
<div class="bar" id="voice-{safe_key}">
  <button class="primary" id="play">🔊 Read aloud</button>
  <button id="pause">⏸ Pause</button>
  <button id="stop">⏹ Stop</button>
  <select id="voice" aria-label="Speech voice"><option>Loading voices…</option></select>
  <span class="status" id="status">Ready</span>
</div>
<script>
(() => {{
  const text = {payload};
  const play = document.getElementById('play');
  const pause = document.getElementById('pause');
  const stop = document.getElementById('stop');
  const voiceSelect = document.getElementById('voice');
  const status = document.getElementById('status');
  const synth = window.speechSynthesis;
  let voices = [];

  function loadVoices() {{
    if (!('speechSynthesis' in window)) {{
      status.textContent = 'Speech unsupported';
      play.disabled = true; pause.disabled = true; stop.disabled = true;
      return;
    }}
    voices = synth.getVoices();
    voiceSelect.innerHTML = '';
    const preferred = voices.filter(v => /en-(US|GB|AU|IN)|english/i.test(v.lang + ' ' + v.name));
    const ordered = [...preferred, ...voices.filter(v => !preferred.includes(v))];
    ordered.slice(0, 30).forEach((v, i) => {{
      const option = document.createElement('option');
      option.value = i;
      option.textContent = `${{v.name}} — ${{v.lang}}`;
      voiceSelect.appendChild(option);
    }});
    if (preferred.length) voiceSelect.value = '0';
  }}

  function speak() {{
    if (!('speechSynthesis' in window)) return;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const idx = Number(voiceSelect.value);
    const ordered = [...voices.filter(v => /en-(US|GB|AU|IN)|english/i.test(v.lang + ' ' + v.name)), ...voices.filter(v => !/en-(US|GB|AU|IN)|english/i.test(v.lang + ' ' + v.name))];
    if (ordered[idx]) u.voice = ordered[idx];
    u.rate = 0.96;
    u.pitch = 1.02;
    u.volume = 1.0;
    u.onstart = () => {{ status.textContent = 'Speaking…'; play.textContent = '🔊 Replay'; }};
    u.onend = () => {{ status.textContent = 'Finished'; }};
    u.onerror = () => {{ status.textContent = 'Could not play'; }};
    synth.speak(u);
  }}

  play.addEventListener('click', speak);
  pause.addEventListener('click', () => {{
    if (synth.speaking && !synth.paused) {{ synth.pause(); status.textContent = 'Paused'; }}
    else if (synth.paused) {{ synth.resume(); status.textContent = 'Speaking…'; }}
  }});
  stop.addEventListener('click', () => {{ synth.cancel(); status.textContent = 'Stopped'; }});
  if ('onvoiceschanged' in synth) synth.addEventListener('voiceschanged', loadVoices);
  loadVoices();
}})();
</script>
</body>
</html>
""",
        height=78,
        scrolling=False,
    )


def answer_voice_controls(
    text: str,
    key: str,
    show_question: bool = True,
    placeholder: str = "Ask a follow-up question…",
) -> VoiceResult:
    """Render premium browser speech controls and an optional follow-up input."""
    browser_speech(text, key)

    if not show_question:
        return VoiceResult()

    st.markdown("<div class='vz-follow-label'>Ask a follow-up question</div>", unsafe_allow_html=True)
    type_col, mic_col = st.columns([3, 2])

    enter_key = f"_enter_submitted_{key}"

    def _mark_enter_submitted() -> None:
        st.session_state[enter_key] = True

    with type_col:
        typed = st.text_input(
            "Ask a follow-up question",
            key=f"{key}-typed-question",
            placeholder=placeholder,
            label_visibility="collapsed",
            on_change=_mark_enter_submitted,
        )
        clicked_ask = st.button("Ask →", key=f"{key}-typed-submit", use_container_width=True)

    submitted_typed = clicked_ask or st.session_state.pop(enter_key, False)

    with mic_col:
        st.markdown(
            "<div style='font-size:.78rem;color:var(--vz-muted);margin-bottom:.2rem;'>🎙 Ask by voice</div>",
            unsafe_allow_html=True,
        )
        recording = st.audio_input(
            "Ask by voice",
            key=f"{key}-mic",
            label_visibility="collapsed",
            help="Record a short question, then VizoLearn will transcribe it.",
        )

    if submitted_typed and typed.strip():
        return VoiceResult(question=typed.strip())

    if recording is not None:
        last_processed_key = f"_last_processed_recording_{key}"
        if st.session_state.get(last_processed_key) != recording.file_id:
            st.session_state[last_processed_key] = recording.file_id
            from src.listen import transcribe_audio

            with st.spinner("Listening and transcribing…"):
                try:
                    transcript = transcribe_audio(recording.getvalue(), suffix=".wav")
                except Exception as exc:
                    st.error(f"Couldn't transcribe that: {exc}")
                    transcript = ""
            if transcript.strip():
                return VoiceResult(question=transcript.strip())
            st.warning("I didn't catch any speech. Try a short, clear question.")

    return VoiceResult()

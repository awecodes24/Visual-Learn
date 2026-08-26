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


def _live_voice_component(key: str):
    """Live browser speech-recognition input with interim transcript display.

    The visible transcript updates on every speech-recognition result via
    direct DOM writes (free — no Python round-trip). Only a throttled subset
    of those updates is pushed to Python via setStateValue, so a whole
    sentence of continuous speech costs a handful of Streamlit reruns
    instead of one per recognized word. The final transcript is sent via
    setTriggerValue exactly once, when the mic naturally stops (silence,
    error, or the user clicking stop) — callers should treat that trigger
    as "the user just finished asking a question," not merely as a value to
    display.
    """
    LIVE_VOICE = st.components.v2.component(
        name="vizolearn_live_voice_input",
        html="""
        <div class="vv-shell">
          <button id="mic" class="vv-mic" aria-label="Ask by voice" type="button">
            <span class="vv-mic-ring"></span>
            <svg class="vv-mic-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M12 15.5c1.93 0 3.5-1.57 3.5-3.5V6c0-1.93-1.57-3.5-3.5-3.5S8.5 4.07 8.5 6v6c0 1.93 1.57 3.5 3.5 3.5Z" fill="currentColor"/>
              <path d="M18.5 12a6.5 6.5 0 0 1-13 0" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
              <path d="M12 18.5V21.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
              <path d="M8.5 21.5H15.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
            </svg>
          </button>
          <div class="vv-body">
            <div id="heard" class="vv-text vv-text--idle">Tap to ask a question</div>
            <div id="note" class="vv-note"></div>
          </div>
        </div>
        """,
        css="""
        .vv-shell { display:flex; align-items:flex-start; gap:14px; padding:14px 16px; border-radius:18px; background:var(--st-secondary-background-color); border:1px solid var(--st-border-color); }

        .vv-mic {
          position:relative; flex:0 0 auto; width:44px; height:44px; border-radius:50%;
          display:flex; align-items:center; justify-content:center; cursor:pointer;
          border:1.5px solid var(--vv-accent); background:transparent; color:var(--vv-accent);
          transition:background-color .18s ease, color .18s ease, border-color .18s ease, transform .12s ease;
        }
        .vv-mic:hover { background:var(--vv-accent-soft); }
        .vv-mic:active { transform:scale(.96); }
        .vv-mic-icon { width:19px; height:19px; position:relative; z-index:1; }
        .vv-mic-ring { position:absolute; inset:-1.5px; border-radius:50%; border:1.5px solid var(--vv-accent); opacity:0; }

        .vv-mic.is-listening {
          background:var(--vv-live); border-color:var(--vv-live); color:#ffffff;
        }
        .vv-mic.is-listening .vv-mic-ring {
          border-color:var(--vv-live); opacity:.55;
          animation:vv-pulse 1.6s cubic-bezier(.33,0,.2,1) infinite;
        }
        @keyframes vv-pulse {
          0%   { transform:scale(1);    opacity:.55; }
          70%  { transform:scale(1.65); opacity:0;   }
          100% { transform:scale(1.65); opacity:0;   }
        }
        @media (prefers-reduced-motion: reduce) {
          .vv-mic.is-listening .vv-mic-ring { animation:none; opacity:0; }
        }

        .vv-body { flex:1 1 auto; min-width:0; padding-top:9px; }
        .vv-text {
          font-size:.95rem; line-height:1.55; color:var(--st-text-color);
          word-wrap:break-word;
        }
        .vv-text--idle { color:var(--st-text-color); opacity:.5; }
        .vv-note { margin-top:5px; font-size:.76rem; color:var(--st-text-color); opacity:.6; min-height:0; }
        """,
        js="""
        export default function(component) {
          const { parentElement, setStateValue, setTriggerValue } = component;
          const root = parentElement.querySelector('.vv-shell');
          root.style.setProperty('--vv-accent', '#5b5ce2');
          root.style.setProperty('--vv-accent-soft', 'rgba(91,92,226,0.1)');
          root.style.setProperty('--vv-live', '#dc2626');

          const mic = parentElement.querySelector('#mic');
          const heard = parentElement.querySelector('#heard');
          const note = parentElement.querySelector('#note');
          const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          let recognition = null;
          let listening = false;
          let finalText = '';
          let throttleTimer = null;
          let pendingLive = null;

          const IDLE_TEXT = 'Tap to ask a question';

          const setNote = (value) => {
            note.textContent = value || '';
            // Low-frequency (only changes on start/stop/error), so sending
            // it straight through on every change is fine.
            setStateValue('status', value || '');
          };

          const paintTranscript = (value) => {
            // Free: direct DOM write, no Python round-trip, no rerun cost.
            if (value) {
              heard.textContent = value;
              heard.classList.remove('vv-text--idle');
            } else {
              heard.textContent = IDLE_TEXT;
              heard.classList.add('vv-text--idle');
            }
          };

          // setStateValue triggers a Streamlit rerun, so it must not fire on
          // every single recognition event during continuous speech — that
          // would rerun the whole app many times a second. Instead, the
          // visible text updates immediately (paintTranscript, free) and the
          // Python-side mirror updates at most every 400ms while speaking.
          const pushLiveThrottled = (value) => {
            pendingLive = value;
            if (throttleTimer) return;
            throttleTimer = setTimeout(() => {
              throttleTimer = null;
              setStateValue('transcript_live', pendingLive || '');
            }, 400);
          };

          function stop() {
            recognition?.stop?.();
          }

          function start() {
            if (!Recognition) {
              setNote('Voice isn\\'t supported in this browser — try Chrome or Edge, or type below.');
              return;
            }
            finalText = '';
            recognition = new Recognition();
            recognition.lang = 'en-US';
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.maxAlternatives = 1;
            listening = true;
            mic.classList.add('is-listening');
            mic.setAttribute('aria-label', 'Stop');
            setNote('');
            paintTranscript('');

            recognition.onresult = (event) => {
              let interimText = '';
              for (let i = event.resultIndex; i < event.results.length; i++) {
                const text = event.results[i][0].transcript;
                if (event.results[i].isFinal) finalText += text + ' ';
                else interimText += text;
              }
              const live = `${finalText}${interimText}`.replace(/\\s+/g, ' ').trim();
              paintTranscript(live);
              pushLiveThrottled(live);
            };

            recognition.onerror = (event) => {
              listening = false;
              mic.classList.remove('is-listening');
              mic.setAttribute('aria-label', 'Ask by voice');
              if (event.error === 'no-speech') {
                setNote('Didn\\'t catch that — tap to try again.');
              } else if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                setNote('Microphone access is blocked — allow it and try again.');
              } else {
                setNote('Something went wrong — tap to try again.');
              }
            };

            recognition.onend = () => {
              listening = false;
              mic.classList.remove('is-listening');
              mic.setAttribute('aria-label', 'Ask by voice');
              const result = finalText.replace(/\\s+/g, ' ').trim();
              if (result) {
                paintTranscript(result);
                // Fires exactly once per stop; this is what auto-submits
                // the question the moment the user stops speaking.
                setTriggerValue('question_ready', result);
              }
            };

            recognition.start();
          }

          mic.onclick = () => listening ? stop() : start();
          setStateValue('status', '');
          setStateValue('transcript_live', '');
          return () => recognition?.stop?.();
        }
        """,
    )
    return LIVE_VOICE(
        default={"status": "", "transcript_live": ""},
        key=key,
        on_status_change=lambda: None,
        on_transcript_live_change=lambda: None,
        on_question_ready_change=lambda: None,
    )


def answer_voice_controls(
    text: str,
    key: str,
    show_question: bool = True,
    placeholder: str = "Ask a follow-up question…",
) -> VoiceResult:
    """Render the existing speaker plus a live-transcribing voice question UI."""
    # IMPORTANT: leave the existing speaker implementation untouched.
    browser_speech(text, key)

    if not show_question:
        return VoiceResult()

    st.markdown("<div class='vz-follow-label'>Ask a follow-up question</div>", unsafe_allow_html=True)

    # Voice comes first so the user can see the transcript while speaking.
    live = _live_voice_component(f"{key}-live-voice")

    # question_ready is a trigger value: it holds the final transcript for
    # exactly the one script run right after the user stops speaking, then
    # resets to None on every run after. That makes it safe to treat as an
    # immediate "the user just asked this" signal with no extra dedup logic.
    question_ready = ""
    if isinstance(live, dict):
        question_ready = str(live.get("question_ready", "") or "").strip()

    if question_ready:
        return VoiceResult(question=question_ready)

    st.markdown("<div style='margin-top:.6rem;font-size:.78rem;color:var(--vz-muted);'>Or type your question</div>", unsafe_allow_html=True)
    typed_col, ask_col = st.columns([4, 1])
    enter_key = f"_enter_submitted_{key}"

    def _mark_enter_submitted() -> None:
        st.session_state[enter_key] = True

    with typed_col:
        typed = st.text_input(
            "Ask a follow-up question",
            key=f"{key}-typed-question",
            placeholder=placeholder,
            label_visibility="collapsed",
            on_change=_mark_enter_submitted,
        )
    with ask_col:
        clicked_ask = st.button("Ask →", key=f"{key}-typed-submit", use_container_width=True)

    submitted_typed = clicked_ask or st.session_state.pop(enter_key, False)
    if submitted_typed and typed.strip():
        return VoiceResult(question=typed.strip())

    return VoiceResult()
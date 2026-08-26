#!/usr/bin/env python3
from __future__ import annotations

# STATUS: downloads a model for src/listen.py's offline faster-whisper path,
# which app.py does not currently use — live voice input is the browser's
# own SpeechRecognition API (src/voice_ui.py) and needs no downloaded model
# at all. Only run this if you're re-enabling the offline STT path yourself.

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a faster-whisper CTranslate2 model locally for VizoLearn.")
    parser.add_argument("--model", choices=MODEL_REPOS, default="base")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to models/faster-whisper-<model>.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output = Path(args.output).expanduser() if args.output else project_root / "models" / f"faster-whisper-{args.model}"
    output.mkdir(parents=True, exist_ok=True)

    repo_id = MODEL_REPOS[args.model]
    print(f"Downloading {repo_id}")
    print(f"Destination: {output}")
    print("This is a one-time download; later VizoLearn runs use the local files.")

    # local_dir makes the model directly usable by WhisperModel(path).
    # max_workers=1 is gentler on unstable connections and easier to resume.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output),
        local_dir_use_symlinks=False,
        max_workers=1,
        resume_download=True,
    )

    required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise SystemExit(f"Download finished but required files are missing: {', '.join(missing)}")

    print("\nWhisper model ready:")
    print(output.resolve())
    print("\nSet this optionally in .env:")
    print(f"VIZO_WHISPER_MODEL_PATH={output.resolve()}")


if __name__ == "__main__":
    main()
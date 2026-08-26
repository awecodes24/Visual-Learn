#!/bin/bash

CUDA12_LIBS="$(
python - <<'PY'
from importlib import import_module
from pathlib import Path

paths = []

for name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    mod = import_module(name)

    package_paths = list(getattr(mod, "__path__", []))

    if package_paths:
        paths.append(str(Path(package_paths[0]).resolve()))
    elif getattr(mod, "__file__", None):
        paths.append(str(Path(mod.__file__).resolve().parent))

print(":".join(paths))
PY
)"

export LD_LIBRARY_PATH="$CUDA12_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec python -m streamlit run app.py
from __future__ import annotations
from typing import Any
import numpy as np


def crop_detection(frame: np.ndarray, detection: dict[str, Any], padding: int = 8) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, detection["box"])
    x1 = max(0, x1 - padding); y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding); y2 = min(h, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()
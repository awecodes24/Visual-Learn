from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import threading



@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_id: int
    name: str
    confidence: float
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObjectDetector:
    def __init__(self, model_name: str = "yolo11n.pt", confidence: float = 0.40) -> None:
        # Lazy imports keep lightweight unit tests (crop/selection utilities)
        # usable even when the heavy vision stack is not installed.
        import torch
        from ultralytics import YOLO

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.confidence = float(confidence)
        self._lock = threading.Lock()

    @property
    def device_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    def detect(self, frame) -> list[dict[str, Any]]:
        with self._lock:
            results = self.model.track(
                source=frame,
                device=self.device,
                conf=self.confidence,
                imgsz=640,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            track_id = int(box.id[0].item()) if box.id is not None else None
            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=class_id,
                    name=str(result.names[class_id]),
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                ).to_dict()
            )
        return detections
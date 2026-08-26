from __future__ import annotations
from typing import Any

Detection = dict[str, Any]


def hit_test(detections: list[Detection], x: int, y: int) -> Detection | None:
    # Prefer the smallest containing box when detections overlap.
    candidates = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            candidates.append((max(1, (x2 - x1) * (y2 - y1)), d))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_by_track_id(detections: list[Detection], track_id: int | None) -> Detection | None:
    if track_id is None:
        return None
    return next((d for d in detections if d.get("track_id") == track_id), None)


class Selection:
    def __init__(self) -> None:
        self.track_id: int | None = None

    def select_at(self, detections: list[Detection], x: int, y: int) -> Detection | None:
        matched = hit_test(detections, x, y)
        self.track_id = matched.get("track_id") if matched else None
        return matched

    def current(self, detections: list[Detection]) -> Detection | None:
        return find_by_track_id(detections, self.track_id)

    def clear(self) -> None:
        self.track_id = None
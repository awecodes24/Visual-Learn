import numpy as np
from src.selection import hit_test, find_by_track_id
from src.crop import crop_detection


def test_hit_test_prefers_smallest_overlap():
    detections = [
        {"track_id": 1, "box": (0, 0, 100, 100)},
        {"track_id": 2, "box": (25, 25, 75, 75)},
    ]
    assert hit_test(detections, 50, 50)["track_id"] == 2


def test_find_by_track_id():
    ds = [{"track_id": 7, "box": (1, 2, 10, 11)}]
    assert find_by_track_id(ds, 7) == ds[0]
    assert find_by_track_id(ds, 99) is None


def test_crop_clamps():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_detection(frame, {"box": (-10, -10, 15, 12)}, padding=0)
    assert crop.shape == (12, 15, 3)
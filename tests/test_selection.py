"""Unit tests for src/selection.py.

These cover pure logic only (no torch/ultralytics/streamlit imports), so
they run fast and need no GPU, webcam, or downloaded model.
"""
from __future__ import annotations

from src.selection import Selection, find_by_track_id, hit_test


def _det(track_id, box, name="thing"):
    x1, y1, x2, y2 = box
    return {
        "track_id": track_id,
        "class_id": 0,
        "name": name,
        "confidence": 0.9,
        "box": (x1, y1, x2, y2),
    }


def test_hit_test_returns_none_when_point_outside_all_boxes():
    detections = [_det(1, (10, 10, 50, 50))]
    assert hit_test(detections, x=100, y=100) is None


def test_hit_test_returns_none_for_empty_detections():
    assert hit_test([], x=5, y=5) is None


def test_hit_test_matches_single_containing_box():
    detections = [_det(1, (10, 10, 50, 50))]
    result = hit_test(detections, x=30, y=30)
    assert result is not None
    assert result["track_id"] == 1


def test_hit_test_prefers_smallest_box_when_overlapping():
    # A large box and a small box both contain the click point; the smaller
    # (more specific) box should win, matching app.py's click-to-select UX.
    big = _det(1, (0, 0, 200, 200), name="big")
    small = _det(2, (40, 40, 60, 60), name="small")
    result = hit_test([big, small], x=50, y=50)
    assert result is not None
    assert result["track_id"] == 2


def test_hit_test_boundary_inclusive():
    # A click exactly on the box edge should still count as a hit.
    detections = [_det(1, (10, 10, 50, 50))]
    assert hit_test(detections, x=10, y=10) is not None
    assert hit_test(detections, x=50, y=50) is not None


def test_find_by_track_id_returns_none_for_none_track_id():
    detections = [_det(1, (0, 0, 10, 10))]
    assert find_by_track_id(detections, None) is None


def test_find_by_track_id_finds_match():
    detections = [_det(1, (0, 0, 10, 10)), _det(2, (20, 20, 30, 30))]
    result = find_by_track_id(detections, 2)
    assert result is not None
    assert result["track_id"] == 2


def test_find_by_track_id_returns_none_when_absent():
    detections = [_det(1, (0, 0, 10, 10))]
    assert find_by_track_id(detections, 99) is None


def test_selection_class_tracks_selected_id_across_frames():
    sel = Selection()
    frame1 = [_det(7, (0, 0, 20, 20))]
    matched = sel.select_at(frame1, x=10, y=10)
    assert matched["track_id"] == 7

    # Next frame: same track_id, object has moved — current() should still
    # resolve it by track_id rather than needing another click.
    frame2 = [_det(7, (5, 5, 25, 25))]
    assert sel.current(frame2)["track_id"] == 7


def test_selection_clear_resets_current():
    sel = Selection()
    sel.select_at([_det(3, (0, 0, 10, 10))], x=5, y=5)
    sel.clear()
    assert sel.current([_det(3, (0, 0, 10, 10))]) is None

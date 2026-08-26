"""Unit tests for src/crop.py.

Pure numpy logic only — no torch/ultralytics/streamlit imports needed.
"""
from __future__ import annotations

import numpy as np

from src.crop import crop_detection


def _frame(h=100, w=100):
    # BGR-shaped uint8 frame filled with a recognizable ramp so we can
    # check that the correct region was actually sliced out.
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def test_crop_detection_returns_none_for_empty_frame():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert crop_detection(empty, {"box": (0, 0, 10, 10)}) is None


def test_crop_detection_returns_none_for_none_frame():
    assert crop_detection(None, {"box": (0, 0, 10, 10)}) is None


def test_crop_detection_basic_shape():
    frame = _frame(100, 100)
    result = crop_detection(frame, {"box": (10, 10, 30, 30)}, padding=0)
    assert result is not None
    assert result.shape == (20, 20, 3)


def test_crop_detection_padding_is_applied_and_clamped_to_frame_bounds():
    frame = _frame(100, 100)
    # Box near the top-left corner: padding should clamp to 0, not go negative.
    result = crop_detection(frame, {"box": (2, 2, 20, 20)}, padding=10)
    assert result is not None
    # x1,y1 clamp to 0 (2-10 -> 0); x2,y2 = 20+10 = 30
    assert result.shape == (30, 30, 3)


def test_crop_detection_padding_clamped_at_frame_edge():
    frame = _frame(100, 100)
    # Box near the bottom-right corner: padding should clamp to frame size.
    result = crop_detection(frame, {"box": (80, 80, 95, 95)}, padding=20)
    assert result is not None
    # x2,y2 clamp to 100; x1,y1 = 80-20 = 60
    assert result.shape == (40, 40, 3)


def test_crop_detection_returns_none_for_degenerate_zero_area_box():
    frame = _frame(100, 100)
    assert crop_detection(frame, {"box": (50, 50, 50, 50)}, padding=0) is None


def test_crop_detection_returns_none_for_inverted_box():
    frame = _frame(100, 100)
    # x2 < x1: should be treated as invalid rather than wrapping/erroring.
    assert crop_detection(frame, {"box": (50, 50, 10, 10)}, padding=0) is None


def test_crop_detection_returns_a_copy_not_a_view():
    frame = _frame(50, 50)
    result = crop_detection(frame, {"box": (5, 5, 15, 15)}, padding=0)
    assert result is not None
    original_value = int(frame[5, 5, 0])
    result[0, 0, 0] = 255
    # Mutating the crop must not mutate the source frame.
    assert int(frame[5, 5, 0]) == original_value

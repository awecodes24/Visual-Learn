from __future__ import annotations
import cv2


def open_camera(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {index}.")
    return cap
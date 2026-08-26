"""Phase 4+5+6 integration test: click a detected object, selection
survives motion (fixed via track_id — see src/selection.py), press S
to save the current crop to outputs/crops/ so you can eyeball that
Phase 6 is actually cropping the right thing.

This supersedes clickable_camera_test.py for testing selection: that
file is left as-is since it still demonstrates the original
click-to-select interaction, but its selected-object highlight does
not survive the object moving (see src/selection.py docstring for
why). Use THIS script going forward for anything involving selection
or crop.

Controls:
  click   - select the object under the cursor
  s       - save a crop of the currently selected object
  q       - quit
"""

import os
import sys
import time

import cv2

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.crop import crop_detection
from src.detector import ObjectDetector
from src.selection import Selection


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "crops")

# Shared across the mouse callback and the main loop, same pattern
# as the original clickable_camera_test.py.
selection = Selection()
detections = []


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    matched = selection.select_at(detections, x, y)

    if matched is not None:
        print(
            f"\nSelected: {matched['name']} "
            f"(track_id={matched['track_id']}, "
            f"confidence={matched['confidence']:.2f})"
        )
    else:
        print("\nNo detected object at that location.")


def main() -> None:
    global detections

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detector = ObjectDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "VizoLearn - Click to Select, S to Save Crop"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("VizoLearn selection+crop test started.")
    print("Click a detected object. Move it around — the highlight")
    print("should keep following it (this is the track_id fix).")
    print("Press S to save a crop of the current selection.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read camera frame.")
            break

        detections = detector.detect(frame)

        current_selection = selection.current(detections)

        # Draw all detections, highlighting the selected one.
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            name = detection["name"]
            confidence = detection["confidence"]

            thickness = 2
            line_color = (0, 255, 0)

            is_selected = (
                current_selection is not None
                and detection["track_id"] == current_selection["track_id"]
            )

            if is_selected:
                line_color = (0, 0, 255)
                thickness = 4

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                line_color,
                thickness,
            )

            label = f"{name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_color,
                2,
                cv2.LINE_AA,
            )

        if current_selection is not None:
            text = f"Selected: {current_selection['name']}  (S to save crop)"

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        elif selection.track_id is not None:
            # We had a selection but the tracker lost it this frame
            # (object left the view, occlusion, or ID reassigned).
            # See src/selection.py NOTE for why this can happen.
            cv2.putText(
                frame,
                "Selection lost (object left view / occluded)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            if current_selection is None:
                print("\nNothing selected — click an object first.")
            else:
                crop = crop_detection(frame, current_selection)

                if crop is None:
                    print(
                        "\nCould not crop — selected box is degenerate "
                        "(likely right at the frame edge)."
                    )
                else:
                    filename = (
                        f"{current_selection['name']}_"
                        f"{current_selection['track_id']}_"
                        f"{int(time.time())}.jpg"
                    )
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(filepath, crop)
                    print(f"\nSaved crop: {filepath}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
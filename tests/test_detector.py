import os
import sys

import cv2

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import ObjectDetector


def main() -> None:
    detector = ObjectDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("VizoLearn detector test started.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        detections = detector.detect(frame)

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            label = (
                f"{detection['name']} "
                f"{detection['confidence']:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            "VizoLearn - YOLO Test",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
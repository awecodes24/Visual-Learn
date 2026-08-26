import cv2

from src.detector import ObjectDetector


selected_object = None
detections = []


def mouse_callback(event, x, y, flags, param):
    global selected_object

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    selected_object = None

    # Check from the last detection to the first so that
    # overlapping boxes favor the most recently processed object.
    for detection in reversed(detections):
        x1, y1, x2, y2 = detection["box"]

        if x1 <= x <= x2 and y1 <= y <= y2:
            selected_object = detection

            print(
                f"\nSelected: {detection['name']} "
                f"(confidence={detection['confidence']:.2f})"
            )

            break

    if selected_object is None:
        print("\nNo detected object at that location.")


def main():
    global detections

    detector = ObjectDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "VizoLearn - Click an Object"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("VizoLearn clickable-camera test started.")
    print("Click a detected object.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read camera frame.")
            break

        detections = detector.detect(frame)

        # Draw detections.
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            name = detection["name"]
            confidence = detection["confidence"]

            # Default box.
            thickness = 2
            line_color = (0, 255, 0)

            # Highlight selected object.
            if (
                selected_object is not None
                and detection["box"] == selected_object["box"]
            ):
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

        # Display selected object.
        if selected_object is not None:
            text = f"Selected: {selected_object['name']}"

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

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
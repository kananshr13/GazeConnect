
import cv2
import mediapipe as mp
import time

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

# Eye contour landmarks
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173,
            144, 145, 153, 154, 155, 246]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398,
             373, 374, 380, 381, 382, 466]

# Iris landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def normalized_eye_position(points, eye_indices, iris_indices, width, height):
    """Estimate iris position relative to the eye's bounding box."""

    eye_x = [points[i].x * width for i in eye_indices]
    eye_y = [points[i].y * height for i in eye_indices]

    iris_x = [points[i].x * width for i in iris_indices]
    iris_y = [points[i].y * height for i in iris_indices]

    eye_left = min(eye_x)
    eye_right = max(eye_x)
    eye_top = min(eye_y)
    eye_bottom = max(eye_y)

    iris_center_x = sum(iris_x) / len(iris_x)
    iris_center_y = sum(iris_y) / len(iris_y)

    eye_width = eye_right - eye_left
    eye_height = eye_bottom - eye_top

    if eye_width < 1 or eye_height < 1:
        return None

    # Clamp values to 0..1 for readable output.
    horizontal = (iris_center_x - eye_left) / eye_width
    vertical = (iris_center_y - eye_top) / eye_height

    horizontal = max(0.0, min(1.0, horizontal))
    vertical = max(0.0, min(1.0, vertical))

    return horizontal, vertical, int(iris_center_x), int(iris_center_y)


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        print("GazeConnect eye-position tracking started.")
        print("Press Q on the preview window to quit.")

        start_time = time.monotonic()
        last_timestamp = -1

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the camera for a natural preview.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            timestamp_ms = max(timestamp_ms, last_timestamp + 1)
            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                points = result.face_landmarks[0]
                h, w, _ = frame.shape

                eye_configs = [
                    ("Left eye", LEFT_EYE, LEFT_IRIS, 20),
                    ("Right eye", RIGHT_EYE, RIGHT_IRIS, 65),
                ]

                for name, eye_indices, iris_indices, text_y in eye_configs:
                    measurement = normalized_eye_position(
                        points, eye_indices, iris_indices, w, h
                    )

                    if measurement is None:
                        continue

                    horizontal, vertical, cx, cy = measurement

                    # Green eye contour landmarks.
                    for index in eye_indices:
                        p = points[index]
                        x, y = int(p.x * w), int(p.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # Blue iris landmarks and red iris center.
                    for index in iris_indices:
                        p = points[index]
                        x, y = int(p.x * w), int(p.y * h)
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    label = (
                        f"{name}: H={horizontal:.2f} V={vertical:.2f}"
                    )
                    cv2.putText(
                        frame, label, (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2
                    )

            else:
                cv2.putText(
                    frame, "Face not detected", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            cv2.imshow("GazeConnect - Normalized Eye Position", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()


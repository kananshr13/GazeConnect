
import cv2
import mediapipe as mp

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access webcam.")
    raise SystemExit

print("Starting GazeConnect landmark detection...")
print("Press Q to quit.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            results = landmarker.detect(mp_image)

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]

                height, width, _ = frame.shape

                # Draw all 478 facial landmarks
                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame, (x, y), 1, (0, 255, 0), -1
                    )

                # Highlight the 10 iris landmarks
                for index in range(468, 478):
                    point = landmarks[index]
                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame, (x, y), 3, (0, 0, 255), -1
                    )

                cv2.putText(
                    frame,
                    "Face detected | 478 landmarks",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("GazeConnect - Face & Iris Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

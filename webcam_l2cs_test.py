
import cv2
import mediapipe as mp
import time
import numpy as np

from hybrid_gaze_m5 import HybridFeatureExtractor

MODEL_DIR = "models"

print("Loading trained Binocular L2CS...")
extractor = HybridFeatureExtractor(model_dir=MODEL_DIR)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    extractor.close()
    raise RuntimeError("Cannot access webcam.")

print("Live L2CS prediction started. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        start = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = extractor.landmarker.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            # Run the trained binocular L2CS model
            predictions = extractor._appearance_vector(
                frame, landmarks
            )

            yaw_left, pitch_left, yaw_right, pitch_right = predictions

            yaw_avg = (yaw_left + yaw_right) / 2
            pitch_avg = (pitch_left + pitch_right) / 2

            elapsed = (time.perf_counter() - start) * 1000

            lines = [
                f"Left eye:  Yaw {yaw_left:.2f}  Pitch {pitch_left:.2f}",
                f"Right eye: Yaw {yaw_right:.2f}  Pitch {pitch_right:.2f}",
                f"Average:   Yaw {yaw_avg:.2f}  Pitch {pitch_avg:.2f}",
                f"Inference: {elapsed:.1f} ms"
            ]

            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (15, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
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

        cv2.imshow("GazeConnect - Live L2CS Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("L2CS test stopped.")

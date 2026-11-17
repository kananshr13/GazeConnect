
import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7
]

RIGHT_EYE = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382
]


def crop_eye(image, landmarks, indices, padding=0.35):
    h, w = image.shape[:2]

    points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in indices
    ])

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    eye_w = x_max - x_min
    eye_h = y_max - y_min

    if eye_w < 3 or eye_h < 3:
        return None

    pad_x = eye_w * padding
    pad_y = eye_h * padding

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot access webcam.")

print("Live eye-crop test started. Press Q to quit.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                if len(landmarks) >= 478:

                    eye1 = crop_eye(rgb, landmarks, LEFT_EYE)
                    eye2 = crop_eye(rgb, landmarks, RIGHT_EYE)

                    if eye1 is not None and eye2 is not None:

                        centers = [
                            (
                                np.mean([
                                    landmarks[i].x
                                    for i in LEFT_EYE
                                ]),
                                eye1
                            ),
                            (
                                np.mean([
                                    landmarks[i].x
                                    for i in RIGHT_EYE
                                ]),
                                eye2
                            )
                        ]

                        centers.sort(key=lambda item: item[0])

                        # Same ordering as the Colab notebook
                        left_crop = centers[0][1]
                        right_crop = centers[1][1]

                        cv2.imshow(
                            "Image-left eye crop",
                            cv2.cvtColor(
                                left_crop, cv2.COLOR_RGB2BGR
                            )
                        )

                        cv2.imshow(
                            "Image-right eye crop",
                            cv2.cvtColor(
                                right_crop, cv2.COLOR_RGB2BGR
                            )
                        )# Enlarge both crops for visual inspection
                        left_display = cv2.resize(
                            cv2.cvtColor(left_crop, cv2.COLOR_RGB2BGR),
                            (256, 256),
                            interpolation=cv2.INTER_NEAREST
                        )   

                        right_display = cv2.resize(
                            cv2.cvtColor(right_crop, cv2.COLOR_RGB2BGR),
                            (256, 256),
                            interpolation=cv2.INTER_NEAREST
                        )

                        # Add labels
                        cv2.putText(
                            left_display, "LEFT EYE",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2
                        )

                        cv2.putText(
                            right_display, "RIGHT EYE",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2
                        )

                        # Display both crops side by side
                        eye_view = np.hstack([left_display, right_display])

                        cv2.imshow("GazeConnect - Eye Crop Inspection", eye_view)

                        combined = np.concatenate(
                            [left_crop, right_crop],
                            axis=2
                        )

                        assert combined.shape == (64, 64, 6)

                        cv2.putText(
                            frame,
                            "Both eye crops extracted: 64x64x6",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2
                        )

            cv2.imshow(
                "GazeConnect - Live Webcam",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

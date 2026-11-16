
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    print("Check your Mac's camera permissions.")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from webcam.")
        break

    cv2.imshow("GazeConnect - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


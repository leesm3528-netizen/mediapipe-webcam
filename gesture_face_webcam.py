"""MediaPipe Gesture Recognizer + Face Landmarker - webcam demo.

Usage:
    python gesture_face_webcam.py [--camera 0] [--num-hands 2] [--num-faces 1]
Press 'q' or ESC to quit.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).parent / "models"
GESTURE_MODEL = MODELS_DIR / "gesture_recognizer.task"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
FACE_CONTOURS = [(c.start, c.end) for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]


def to_pixels(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def draw_gestures(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        points = to_pixels(landmarks, w, h)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
        for p in points:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        hand = result.handedness[i][0].category_name if result.handedness else ""
        gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
        text = f"{hand}: {gesture.category_name} ({gesture.score:.2f})" if gesture else hand
        x0 = min(p[0] for p in points)
        y0 = min(p[1] for p in points)
        cv2.putText(frame, text, (x0, max(y0 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 205, 54), 2)


def draw_faces(frame, result):
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        points = to_pixels(landmarks, w, h)
        for p in points:
            cv2.circle(frame, p, 1, (200, 200, 0), -1)
        for a, b in FACE_CONTOURS:
            cv2.line(frame, points[a], points[b], (0, 255, 255), 1)

    # Show the top blendshapes (expressions) of the first face
    if result.face_blendshapes:
        top = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)[:5]
        for j, c in enumerate(top):
            cv2.putText(frame, f"{c.category_name}: {c.score:.2f}", (10, 60 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--num-faces", type=int, default=1)
    args = parser.parse_args()

    gesture_options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=args.num_faces,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    start = time.monotonic()
    prev = start
    with vision.GestureRecognizer.create_from_options(gesture_options) as recognizer, \
            vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"Failed to read a frame from camera {args.camera} "
                      "(is another app using it? try --camera 1)")
                break
            frame = cv2.flip(frame, 1)  # mirror view

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            gesture_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            draw_faces(frame, face_result)
            draw_gestures(frame, gesture_result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Gesture + Face", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Run the custom gesture classifier on the webcam.

Usage:
    python custom_gesture_webcam.py [--camera 0] [--num-hands 2] [--threshold 0.7]
Press 'q' or ESC to quit.
"""
import argparse
import time

import cv2
import joblib
import mediapipe as mp

from gesture_features import CLASSIFIER_PATH, create_hand_landmarker, draw_hand, landmarks_to_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="below this confidence the gesture is shown as Unknown")
    args = parser.parse_args()

    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"{CLASSIFIER_PATH} not found. Run train_gestures.py first.")
    clf = joblib.load(CLASSIFIER_PATH)
    print("labels:", list(clf.classes_))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    start = time.monotonic()
    with create_hand_landmarker(args.num_hands) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"Failed to read a frame from camera {args.camera}")
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            for landmarks, world, handedness in zip(
                    result.hand_landmarks, result.hand_world_landmarks, result.handedness):
                hand = handedness[0].category_name
                features = landmarks_to_features(world, hand)
                probs = clf.predict_proba([features])[0]
                best = probs.argmax()
                label = clf.classes_[best] if probs[best] >= args.threshold else "Unknown"

                points = draw_hand(frame, landmarks)
                x0 = min(p[0] for p in points)
                y0 = min(p[1] for p in points)
                cv2.putText(frame, f"{hand}: {label} ({probs[best]:.2f})", (x0, max(y0 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 205, 54), 2)

            cv2.imshow("Custom gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

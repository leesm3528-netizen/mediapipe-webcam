"""Collect hand-landmark samples for a custom gesture from the webcam.

Usage:
    python collect_gestures.py --label heart
Keys (click the webcam window first):
    SPACE  start / pause recording (one sample per frame while a hand is visible)
    q/ESC  save and quit
Samples are appended to data/gestures.csv, so run it once per gesture label.
"""
import argparse
import csv
import time

import cv2
import mediapipe as mp

from gesture_features import DATA_CSV, NUM_FEATURES, create_hand_landmarker, draw_hand, landmarks_to_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="gesture name, e.g. heart, ok, none")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max", type=int, default=500, help="stop after this many samples")
    args = parser.parse_args()

    DATA_CSV.parent.mkdir(exist_ok=True)
    new_file = not DATA_CSV.exists()
    f = DATA_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    recording = False
    count = 0
    start = time.monotonic()
    with create_hand_landmarker() as landmarker:
        while count < args.max:
            ok, frame = cap.read()
            if not ok:
                print(f"Failed to read a frame from camera {args.camera}")
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            if result.hand_landmarks:
                draw_hand(frame, result.hand_landmarks[0])
                if recording:
                    handedness = result.handedness[0][0].category_name
                    features = landmarks_to_features(result.hand_world_landmarks[0], handedness)
                    assert len(features) == NUM_FEATURES
                    writer.writerow([args.label] + [f"{v:.5f}" for v in features])
                    count += 1

            status = "REC" if recording else "PAUSED (SPACE to record)"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{args.label}] {status}  samples: {count}/{args.max}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if recording and not result.hand_landmarks:
                cv2.putText(frame, "no hand detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Collect gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key in (ord("q"), 27):
                break

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples for '{args.label}' to {DATA_CSV}")


if __name__ == "__main__":
    main()

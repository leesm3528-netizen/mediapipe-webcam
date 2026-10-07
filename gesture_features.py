"""Shared helpers for the custom gesture pipeline (collect -> train -> infer)."""
from pathlib import Path

import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).parent
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
DATA_CSV = ROOT / "data" / "gestures.csv"
CLASSIFIER_PATH = ROOT / "models" / "custom_gesture.joblib"

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
NUM_FEATURES = 21 * 3


def create_hand_landmarker(num_hands=1):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(world_landmarks, handedness):
    """Turn 21 world landmarks into a 63-dim, position/scale/hand-invariant vector.

    - translate so the wrist is the origin
    - mirror left hands so one model works for both hands
    - scale so the farthest joint is at distance 1
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def draw_hand(frame, landmarks, color=(255, 255, 255)):
    import cv2

    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color, 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return points

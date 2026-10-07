"""Download the MediaPipe .task models into ./models."""
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "gesture_recognizer.task": f"{BASE}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

MODELS_DIR = Path(__file__).parent / "models"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"skip {name} (already exists)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

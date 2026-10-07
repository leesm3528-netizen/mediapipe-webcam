"""Export the trained custom gesture classifier to web/model.json for the browser demo.

Usage:
    python export_model_web.py
"""
import json

import joblib

from gesture_features import CLASSIFIER_PATH, ROOT

OUT = ROOT / "web" / "model.json"


def main():
    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"{CLASSIFIER_PATH} not found. Train a model first.")
    clf = joblib.load(CLASSIFIER_PATH)
    if clf.activation != "relu":
        raise SystemExit(f"Only relu MLPs are supported (got {clf.activation}).")

    model = {
        "classes": [str(c) for c in clf.classes_],
        "out_activation": clf.out_activation_,  # softmax (3+ classes) or logistic (2 classes)
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(clf.coefs_, clf.intercepts_)
        ],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(model), encoding="utf-8")
    print(f"exported {model['classes']} -> {OUT}")


if __name__ == "__main__":
    main()

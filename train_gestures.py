"""Train a custom gesture classifier from data/gestures.csv.

Usage:
    python train_gestures.py
Saves the model to models/custom_gesture.joblib.
"""
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from gesture_features import CLASSIFIER_PATH, DATA_CSV


def load_data():
    if not DATA_CSV.exists():
        raise SystemExit(f"{DATA_CSV} not found. Run collect_gestures.py first.")
    labels, features = [], []
    with DATA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])
    return np.array(features, dtype=np.float32), np.array(labels)


def train_and_save(X, y):
    """Train, evaluate on a held-out split, refit on all data and save. Returns (clf, report)."""
    counts = Counter(y)
    lines = [f"samples per label: { {str(k): v for k, v in counts.items()} }"]
    if len(counts) < 2:
        raise ValueError("Need at least 2 different labels (tip: collect a 'none' class too).")
    if min(counts.values()) < 5:
        raise ValueError("Each label needs at least 5 samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                        early_stopping=len(X_train) >= 100, random_state=42)
    clf.fit(X_train, y_train)

    lines.append(f"\ntest accuracy: {clf.score(X_test, y_test):.3f}\n")
    lines.append(classification_report(y_test, clf.predict(X_test), zero_division=0))

    # Refit on all data for the final model
    clf.fit(X, y)
    CLASSIFIER_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    lines.append(f"saved model -> {CLASSIFIER_PATH}")
    return clf, "\n".join(lines)


def main():
    X, y = load_data()
    try:
        _, report = train_and_save(X, y)
    except ValueError as e:
        raise SystemExit(str(e))
    print(report)


if __name__ == "__main__":
    main()

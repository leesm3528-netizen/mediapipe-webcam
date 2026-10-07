"""Gesture Studio - collect, train and test custom hand gestures in one window.

Usage:
    python gesture_studio.py [--camera 0]
"""
import argparse
import csv
import queue
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from gesture_features import CLASSIFIER_PATH, DATA_CSV, create_hand_landmarker, draw_hand, landmarks_to_features
from train_gestures import train_and_save

VIDEO_W, VIDEO_H = 640, 480
HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
TAB_COLLECT, TAB_TRAIN, TAB_INFER = range(3)


def load_font(size):
    for name in ("malgun.ttf", "malgunbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class GestureStudio:
    def __init__(self, root, camera):
        self.root = root
        self.camera_index = camera
        self.cap = None
        self.landmarker = create_hand_landmarker(num_hands=2)
        self.start = time.monotonic()
        self.prev = self.start
        self.font = load_font(24)

        # dataset: list of rows [label, "f0", ..., "f62"]
        self.rows = self._load_rows()
        self.recording = False
        self.session_count = 0

        self.clf = joblib.load(CLASSIFIER_PATH) if CLASSIFIER_PATH.exists() else None
        self.train_queue = queue.Queue()
        self.training = False

        self._build_ui()
        self._refresh_counts()
        self._refresh_model_info()
        self._open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._tick()

    # ---------- data ----------
    def _load_rows(self):
        if not DATA_CSV.exists():
            return []
        with DATA_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            return [row for row in reader if row]

    def _save_rows(self):
        DATA_CSV.parent.mkdir(exist_ok=True)
        with DATA_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(self.rows)

    def _append_row(self, row):
        self.rows.append(row)
        new_file = not DATA_CSV.exists()
        DATA_CSV.parent.mkdir(exist_ok=True)
        with DATA_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(HEADER)
            writer.writerow(row)

    # ---------- UI ----------
    def _build_ui(self):
        self.root.title("Gesture Studio")
        self.root.option_add("*Font", ("Malgun Gothic", 10))
        style = ttk.Style()
        style.configure("Rec.TButton", font=("Malgun Gothic", 11, "bold"))

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # left: video
        left = ttk.Frame(main)
        left.pack(side="left", fill="both")
        self.video = ttk.Label(left)
        self.video.pack()
        self.status = ttk.Label(left, text="", foreground="#666")
        self.status.pack(anchor="w", pady=(4, 0))

        cam_row = ttk.Frame(left)
        cam_row.pack(anchor="w", pady=(4, 0))
        ttk.Label(cam_row, text="카메라 번호").pack(side="left")
        self.camera_var = tk.IntVar(value=self.camera_index)
        ttk.Spinbox(cam_row, from_=0, to=5, width=4, textvariable=self.camera_var).pack(side="left", padx=4)
        ttk.Button(cam_row, text="다시 연결", command=self._open_camera).pack(side="left")

        # right: tabs
        self.tabs = ttk.Notebook(main, width=340)
        self.tabs.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.tabs.add(self._build_collect_tab(), text=" ① 수집 ")
        self.tabs.add(self._build_train_tab(), text=" ② 학습 ")
        self.tabs.add(self._build_infer_tab(), text=" ③ 인식 ")
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._stop_recording())
        self.root.bind("<space>", self._on_space)

    def _build_collect_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)

        ttk.Label(tab, text="제스처 이름").pack(anchor="w")
        self.label_var = tk.StringVar(value="heart")
        ttk.Entry(tab, textvariable=self.label_var).pack(fill="x", pady=(2, 8))

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="목표 개수").pack(side="left")
        self.target_var = tk.IntVar(value=300)
        ttk.Spinbox(row, from_=50, to=2000, increment=50, width=6,
                    textvariable=self.target_var).pack(side="left", padx=4)

        self.rec_button = ttk.Button(tab, text="● 녹화 시작 (Space)", style="Rec.TButton",
                                     command=self._toggle_recording)
        self.rec_button.pack(fill="x", pady=8, ipady=6)
        self.session_label = ttk.Label(tab, text="이번 녹화: 0개")
        self.session_label.pack(anchor="w")
        self.session_bar = ttk.Progressbar(tab, maximum=300)
        self.session_bar.pack(fill="x", pady=(2, 10))

        ttk.Label(tab, text="수집된 데이터").pack(anchor="w")
        self.count_tree = ttk.Treeview(tab, columns=("count",), height=8)
        self.count_tree.heading("#0", text="제스처")
        self.count_tree.heading("count", text="개수")
        self.count_tree.column("#0", width=180)
        self.count_tree.column("count", width=80, anchor="e")
        self.count_tree.pack(fill="both", expand=True, pady=(2, 4))
        ttk.Button(tab, text="선택한 제스처 데이터 삭제", command=self._delete_selected).pack(fill="x")

        tip = ("팁: 'none' 제스처(아무 동작 아닌 손)도 꼭 수집하세요.\n"
               "녹화 중 손 각도·거리·왼손/오른손을 바꿔 주세요.")
        ttk.Label(tab, text=tip, foreground="#666", wraplength=310).pack(anchor="w", pady=(8, 0))
        return tab

    def _build_train_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.train_button = ttk.Button(tab, text="학습 시작", style="Rec.TButton", command=self._start_training)
        self.train_button.pack(fill="x", ipady=6)
        self.train_bar = ttk.Progressbar(tab, mode="indeterminate")
        self.train_bar.pack(fill="x", pady=8)
        self.model_info = ttk.Label(tab, text="", foreground="#666")
        self.model_info.pack(anchor="w")
        self.log = tk.Text(tab, height=20, font=("Consolas", 9), wrap="none")
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        return tab

    def _build_infer_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.pred_label = ttk.Label(tab, text="-", font=("Malgun Gothic", 22, "bold"), anchor="center")
        self.pred_label.pack(fill="x", pady=(4, 10))

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="인식 기준 확률").pack(side="left")
        self.threshold_var = tk.DoubleVar(value=0.7)
        self.threshold_text = ttk.Label(row, text="0.70", width=5)
        self.threshold_text.pack(side="right")
        ttk.Scale(tab, from_=0.3, to=0.99, variable=self.threshold_var,
                  command=lambda v: self.threshold_text.config(text=f"{float(v):.2f}")).pack(fill="x", pady=(2, 10))

        ttk.Label(tab, text="제스처별 확률 (첫 번째 손)").pack(anchor="w")
        self.prob_frame = ttk.Frame(tab)
        self.prob_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.prob_bars = {}
        return tab

    def _rebuild_prob_bars(self):
        for w in self.prob_frame.winfo_children():
            w.destroy()
        self.prob_bars = {}
        if self.clf is None:
            ttk.Label(self.prob_frame, text="학습된 모델이 없습니다. ② 학습 탭에서 학습하세요.",
                      foreground="#c00", wraplength=310).pack(anchor="w")
            return
        for label in self.clf.classes_:
            row = ttk.Frame(self.prob_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=str(label), width=12).pack(side="left")
            bar = ttk.Progressbar(row, maximum=1.0)
            bar.pack(side="left", fill="x", expand=True)
            self.prob_bars[label] = bar

    def _refresh_counts(self):
        self.count_tree.delete(*self.count_tree.get_children())
        for label, n in sorted(Counter(r[0] for r in self.rows).items()):
            self.count_tree.insert("", "end", iid=label, text=label, values=(n,))

    def _refresh_model_info(self):
        if self.clf is None:
            self.model_info.config(text="저장된 모델: 없음")
        else:
            self.model_info.config(text="저장된 모델 제스처: " + ", ".join(map(str, self.clf.classes_)))
        self._rebuild_prob_bars()

    # ---------- camera ----------
    def _open_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.camera_index = self.camera_var.get()
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.status.config(text=f"카메라 {self.camera_index}번을 열 수 없습니다", foreground="#c00")

    # ---------- collect ----------
    def _on_space(self, event):
        # Don't steal Space while typing a label name
        # (a focused button already handles Space itself)
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox, ttk.Button)):
            return
        if self.tabs.index("current") == TAB_COLLECT:
            self._toggle_recording()

    def _toggle_recording(self):
        if self.recording:
            self._stop_recording()
            return
        label = self.label_var.get().strip()
        if not label or "," in label:
            messagebox.showwarning("제스처 이름", "제스처 이름을 입력하세요 (쉼표 사용 불가).")
            return
        self.recording = True
        self.session_count = 0
        self.session_bar.config(maximum=self.target_var.get(), value=0)
        self.rec_button.config(text="■ 녹화 중지 (Space)")
        self.root.focus_set()

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.rec_button.config(text="● 녹화 시작 (Space)")
        self.root.focus_set()
        self._refresh_counts()

    def _delete_selected(self):
        selected = self.count_tree.selection()
        if not selected:
            return
        label = selected[0]
        if not messagebox.askyesno("삭제", f"'{label}' 데이터를 모두 삭제할까요?"):
            return
        self.rows = [r for r in self.rows if r[0] != label]
        self._save_rows()
        self._refresh_counts()

    # ---------- train ----------
    def _start_training(self):
        if self.training:
            return
        if not self.rows:
            messagebox.showinfo("학습", "수집된 데이터가 없습니다. ① 수집 탭에서 먼저 수집하세요.")
            return
        self.training = True
        self.train_button.config(state="disabled")
        self.train_bar.start(10)
        self.log.delete("1.0", "end")
        self.log.insert("end", "학습 중...\n")

        X = np.array([[float(v) for v in r[1:]] for r in self.rows], dtype=np.float32)
        y = np.array([r[0] for r in self.rows])

        def work():
            try:
                self.train_queue.put(("ok", *train_and_save(X, y)))
            except Exception as e:  # report any failure in the UI
                self.train_queue.put(("error", None, str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _poll_training(self):
        try:
            status, clf, text = self.train_queue.get_nowait()
        except queue.Empty:
            return
        self.training = False
        self.train_bar.stop()
        self.train_button.config(state="normal")
        self.log.delete("1.0", "end")
        if status == "ok":
            self.clf = clf
            self.log.insert("end", text + "\n\n학습 완료! ③ 인식 탭에서 테스트하세요.")
            self._refresh_model_info()
        else:
            self.log.insert("end", "학습 실패: " + text)

    # ---------- main loop ----------
    def _tick(self):
        self._poll_training()
        ok, frame = (self.cap.read() if self.cap is not None and self.cap.isOpened() else (False, None))
        if ok:
            self._process(cv2.flip(frame, 1))
        self.root.after(10, self._tick)

    def _process(self, frame):
        frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, int((time.monotonic() - self.start) * 1000))
        tab = self.tabs.index("current")

        captions = []  # (x, y, text, color) drawn with PIL so Korean labels render
        for i, (landmarks, world, handed) in enumerate(zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness)):
            hand = handed[0].category_name
            points = draw_hand(frame, landmarks)
            x0, y0 = min(p[0] for p in points), min(p[1] for p in points)

            if tab == TAB_COLLECT and self.recording and i == 0:
                features = landmarks_to_features(world, hand)
                self._append_row([self.label_var.get().strip()] + [f"{v:.5f}" for v in features])
                self.session_count += 1

            if tab == TAB_INFER and self.clf is not None:
                probs = self.clf.predict_proba([landmarks_to_features(world, hand)])[0]
                best = probs.argmax()
                ok = probs[best] >= self.threshold_var.get()
                label = str(self.clf.classes_[best]) if ok else "Unknown"
                captions.append((x0, max(y0 - 32, 0), f"{hand}: {label} {probs[best]:.2f}",
                                 (54, 205, 88) if ok else (200, 200, 200)))
                if i == 0:
                    self.pred_label.config(text=label)
                    for cls, p in zip(self.clf.classes_, probs):
                        self.prob_bars[cls].config(value=float(p))

        if tab == TAB_INFER and not result.hand_landmarks:
            self.pred_label.config(text="손이 보이지 않음")

        if tab == TAB_COLLECT and self.recording:
            self.session_label.config(text=f"이번 녹화: {self.session_count}개")
            self.session_bar.config(value=self.session_count)
            cv2.circle(frame, (VIDEO_W - 30, 30), 12, (0, 0, 255), -1)
            if not result.hand_landmarks:
                captions.append((10, 10, "손이 보이지 않음", (255, 165, 0)))
            if self.session_count >= self.target_var.get():
                self._stop_recording()

        now = time.monotonic()
        fps = 1.0 / max(now - self.prev, 1e-6)
        self.prev = now
        self.status.config(text=f"카메라 {self.camera_index}번  |  FPS {fps:.0f}  |  손 {len(result.hand_landmarks)}개",
                           foreground="#666")

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if captions:
            draw = ImageDraw.Draw(image)
            for x, y, text, color in captions:
                draw.text((x, y), text, font=self.font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        self.photo = ImageTk.PhotoImage(image)
        self.video.config(image=self.photo)

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    root = tk.Tk()
    GestureStudio(root, args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()

# MediaPipe Webcam Demos

[MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide) 기반 실시간 웹캠 데모 (Python + OpenCV).

| 스크립트 | 기능 |
| --- | --- |
| `hand_landmarker_webcam.py` | 손 21개 랜드마크 + 왼손/오른손 표시 |
| `gesture_face_webcam.py` | 제스처 인식 + 얼굴 478개 랜드마크 + 표정(blendshape) 상위 5개 |

## 설치

```bash
pip install -r requirements.txt
python download_models.py   # models/ 에 .task 모델 3개 다운로드
```

## 실행

```bash
python hand_landmarker_webcam.py [--camera 0] [--num-hands 2]
python gesture_face_webcam.py    [--camera 0] [--num-hands 2] [--num-faces 1]
```

웹캠 창을 클릭한 뒤 `q` 또는 `ESC`로 종료합니다.

인식되는 제스처: `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`

## 문제 해결

- **`No module named 'cv2'` / `'mediapipe'`**: PC에 파이썬이 여러 개 설치되어 있으면 실행되는 파이썬과 패키지를 설치한 파이썬이 다를 수 있습니다. 실행할 파이썬으로 직접 설치하세요: `python -m pip install -r requirements.txt`
- **창이 안 뜨거나 바로 꺼짐**: 다른 앱(Zoom, 카메라 앱 등)이 카메라를 쓰고 있는지 확인하고, `--camera 1` 로 다른 카메라를 시도하세요.
- **PowerShell이 명령을 안 받음**: 웹캠 프로그램이 실행 중이면 정상입니다. 창에서 `q`를 누르거나 터미널에서 `Ctrl + C`로 종료하세요.

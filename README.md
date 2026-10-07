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

## 나만의 제스처 학습

손 랜드마크 21개(3D)를 수집해서 작은 분류기(scikit-learn MLP)를 학습합니다.

### UI로 하기 (추천)

```bash
python gesture_studio.py
```

한 창에서 탭으로 진행합니다.

1. **① 수집**: 제스처 이름 입력 → `녹화 시작`(또는 Space) → 손 모양 유지. 목표 개수가 차면 자동 정지. 제스처마다 반복 (`none` 포함).
2. **② 학습**: `학습 시작` → 정확도·제스처별 성능 표시, `models/custom_gesture.joblib` 저장.
3. **③ 인식**: 실시간 인식 결과와 제스처별 확률 막대. 슬라이더로 인식 기준 확률 조절.

### 터미널로 하기

```bash
# 1) 제스처마다 한 번씩 수집 (창에서 SPACE로 녹화 시작/일시정지, q로 저장 후 종료)
python collect_gestures.py --label heart
python collect_gestures.py --label ok
python collect_gestures.py --label none     # 아무 제스처도 아닌 손 모양 (오인식 방지용)

# 2) 학습 -> models/custom_gesture.joblib
python train_gestures.py

# 3) 웹캠 추론
python custom_gesture_webcam.py [--threshold 0.7]
```

- 수집 데이터는 `data/gestures.csv` 에 계속 추가됩니다. 라벨 하나를 다시 찍으려면 CSV에서 해당 줄을 지우세요.
- 제스처당 200~500개, 손 각도·거리·왼손/오른손을 바꿔가며 찍으면 인식이 안정적입니다.
- 확률이 `--threshold` 미만이면 `Unknown` 으로 표시됩니다.

## 웹 버전: 제스처 → 이모지/로고

학습한 모델을 브라우저에서 실행해서, 인식된 제스처 위치에 스티커를 띄웁니다.

| 제스처 라벨 | 표시 |
| --- | --- |
| `Nike` | Nike 스우시 로고 (`web/nike.png` 를 넣으면 그 이미지 사용) |
| `ok` | 👌 |
| `Heart` | ❤️ 하트가 손에서 화면 앞으로 연속 발사 |
| `ㅗ` | 🖕 |

```bash
python export_model_web.py   # models/custom_gesture.joblib -> web/model.json (재학습할 때마다 실행)
python serve_web.py          # http://localhost:8000/web/ 자동으로 열림
```

- 브라우저에서 카메라 권한을 허용하세요. 파이썬 웹캠 앱이 켜져 있으면 카메라를 같이 못 쓰니 먼저 닫으세요.
- 라벨 이름은 대소문자 구분 없이 매칭됩니다. 다른 라벨을 추가하려면 `web/app.js` 의 `stickerContent()` 에 한 줄 추가하면 됩니다.

## 문제 해결

- **`No module named 'cv2'` / `'mediapipe'`**: PC에 파이썬이 여러 개 설치되어 있으면 실행되는 파이썬과 패키지를 설치한 파이썬이 다를 수 있습니다. 실행할 파이썬으로 직접 설치하세요: `python -m pip install -r requirements.txt`
- **창이 안 뜨거나 바로 꺼짐**: 다른 앱(Zoom, 카메라 앱 등)이 카메라를 쓰고 있는지 확인하고, `--camera 1` 로 다른 카메라를 시도하세요.
- **PowerShell이 명령을 안 받음**: 웹캠 프로그램이 실행 중이면 정상입니다. 창에서 `q`를 누르거나 터미널에서 `Ctrl + C`로 종료하세요.

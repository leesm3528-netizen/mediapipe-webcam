# MediaPipe Webcam Demos

[MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide) 기반 실시간 웹캠 데모 (Python + OpenCV), 나만의 제스처 학습, 브라우저 데모.

**웹 데모: https://leesm3528-netizen.github.io/mediapipe-webcam/**

| 파일 | 기능 |
| --- | --- |
| `hand_landmarker_webcam.py` | 손 21개 랜드마크 + 왼손/오른손 표시 |
| `gesture_face_webcam.py` | 기본 제스처 인식 + 얼굴 478개 랜드마크 + 표정(blendshape) 상위 5개 |
| `gesture_studio.py` | 나만의 제스처 수집 → 학습 → 인식 UI 앱 |
| `collect_gestures.py` / `train_gestures.py` / `custom_gesture_webcam.py` | 같은 과정을 터미널로 |
| `web/` | 학습한 모델로 제스처별 이모지·로고를 띄우는 브라우저 데모 (GitHub Pages 배포) |
| `export_model_web.py` / `serve_web.py` | 모델을 `web/model.json`으로 내보내기 / 로컬 웹 서버 |
| `download_models.py` | MediaPipe `.task` 모델 다운로드 |

## 작업 기록: 2026-10-07

### 1. MediaPipe 웹캠 데모 (커밋 `09ff333`)
- Hand Landmarker 문서의 공식 모델(`hand_landmarker.task`)과 `mediapipe` 1.1.0, `opencv-python`으로 웹캠 손 랜드마크 데모를 만들었습니다.
- Gesture Recognizer + Face Landmarker를 한 화면에 띄우는 데모를 추가했습니다.
- 웹캠이 안 켜지던 원인은 **파이썬이 두 개 설치되어 있던 것**이었습니다. 3.14.6에만 패키지가 있고, 새로 설치된 3.14.8에는 없었습니다. 양쪽 모두에 설치해서 해결했습니다.

### 2. 나만의 제스처 학습 (커밋 `a797ea7`)
- **방식**: HandLandmarker의 3D world 랜드마크 21개를 아래처럼 정규화해서 63차원 특징 벡터로 만들고, scikit-learn MLP로 분류합니다.
  - 손목을 원점으로 옮깁니다.
  - 왼손은 좌우를 뒤집어 한 모델로 양손을 처리합니다.
  - 손목에서 가장 먼 관절까지의 거리가 1이 되게 크기를 맞춥니다.
- **도구**: `gesture_studio.py` (Tkinter UI)로 수집, 학습, 인식을 한 창에서 합니다.
- **학습 데이터** (`data/gestures.csv`, 저장소에는 포함하지 않음):

  | 제스처 | 샘플 수 |
  | --- | --- |
  | `Heart` | 300 |
  | `Nike` | 300 |
  | `ok` | 300 |
  | `ㅗ` | 300 |
  | **합계** | **1,200** |

- **모델**: MLP `63 → 64 → 32 → 4` (ReLU, softmax), 학습/테스트 80:20 분할 (stratified, `random_state=42`)
- **결과**: 테스트 정확도 **97.1%** (240개 중 233개)

  | 제스처 | precision | recall | f1 |
  | --- | --- | --- | --- |
  | Heart | 1.000 | 0.967 | 0.983 |
  | Nike | 0.952 | 0.983 | 0.967 |
  | ok | 0.983 | 0.983 | 0.983 |
  | ㅗ | 0.950 | 0.950 | 0.950 |

- **한계**: `none`(아무 제스처도 아닌 손) 클래스가 없어서, 평범한 손도 4개 중 하나로 분류될 수 있습니다. 지금은 확률 기준값(기본 0.8)으로 거르고 있고, `none`을 수집해서 다시 학습하는 것이 다음 개선 과제입니다.

### 3. 웹 데모 + GitHub Pages 배포 (커밋 `a797ea7`, `9559d21`)
- 학습한 MLP 가중치를 `web/model.json`으로 내보내고, 브라우저에서 MediaPipe JS(`@mediapipe/tasks-vision` 1.1.0)와 같은 특징 추출·추론을 그대로 구현했습니다. 파이썬과 확률이 소수점 4자리까지 같은 것을 확인했습니다.
- 제스처별 효과: Nike → 스우시 로고, ok → 👌, Heart → 하트가 화면 앞으로 연속 발사, ㅗ → 🖕
- `web/`을 GitHub Actions(`.github/workflows/pages.yml`)로 GitHub Pages에 자동 배포합니다.

### 4. PowerShell 단축 명령 (로컬 PC 설정, 저장소에는 포함하지 않음)
`$PROFILE`에 등록해서 어느 폴더에서나 쓸 수 있습니다.

| 명령 | 동작 |
| --- | --- |
| `studio` (`gesture-studio`) | Gesture Studio 실행 |
| `web` (`gesture-web`) | 모델이 바뀌었으면 다시 내보내고 → 로컬 서버를 켜고 → 브라우저 열기 |
| `gesture-web-stop` | 로컬 웹 서버 종료 |

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

**배포 주소: https://leesm3528-netizen.github.io/mediapipe-webcam/**
(`web/` 이 바뀌어 main에 푸시되면 GitHub Actions가 자동으로 다시 배포합니다. 재학습 후에는 `python export_model_web.py` 로 `web/model.json` 을 갱신해서 커밋하세요.)

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

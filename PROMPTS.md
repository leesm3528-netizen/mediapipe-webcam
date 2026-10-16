# 사용한 프롬프트 모음

이 프로젝트를 만들 때 입력한 프롬프트를 정리했습니다. 작업 기간은 2026-10-07 ~ 2026-10-14입니다.
문장은 입력한 그대로 옮겼습니다. 1번의 URL만 추적용 파라미터를 뺐습니다.

- `#`은 실제로 입력한 순서이고, 표는 주제별로 묶었습니다.
- 결과에 적힌 파일과 기능의 자세한 내용은 [README](README.md)의 작업 기록에 있습니다.

## 1. 시작: 손 랜드마크 웹캠

| # | 프롬프트 | 결과 |
| --- | --- | --- |
| 1 | https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker 해당하는 라이브러리 받고 파이썬 실행 코드 webcam 기반으로 짜줘 클로인크롬써서 다운받아 | `hand_landmarker_webcam.py`, `hand_landmarker.task` 모델 |
| 2 | 웹캠실행링크가 어딨니 | 링크가 아니라 로컬 파이썬 프로그램이라는 설명, 실행 방법 |

## 2. 제스처·얼굴 인식으로 확장

| # | 프롬프트 | 결과 |
| --- | --- | --- |
| 3 | Gesture Recognition 과 face landmark 둘다 웹캠에 듸워줘 | `gesture_face_webcam.py` |
| 4 | 코드 작성시키기 | 원하는 내용을 되물음 |
| 5 | 파이썬 기반으로 웹캡이 안켜지는데 무슨 문제인지 알려줘 | 원인: 파이썬이 두 개 설치됨 (3.14.8에 패키지 없음) → 설치로 해결 |
| 6 | 파워쉘에 명령어를 입력해도 듣질않아. | 프로그램 실행 중에는 PowerShell이 기다리는 게 정상이라는 설명 (`q` / `Ctrl+C`) |
| 7 | 방금한것도 깃허브에도 또 올리고 정리해줘 | GitHub 저장소 생성, README·`requirements.txt`·`download_models.py` |
| 8 | (기본 제스처 목록 붙여넣기) 7번째 Iloveyou 모듈은 어떻게 제스처를해야 인식이되는거지 | 🤟 손 모양 (엄지·검지·새끼 펴기) 설명 |

## 3. 나만의 제스처 학습

| # | 프롬프트 | 결과 |
| --- | --- | --- |
| 9 | 제스쳐인식 코드가 잘 동작하지만 내가 원하는 제스쳐를 추가 학습해서 추론해보고 싶다. 수집하는 코드와 훈련하는 코드를 각각 작성해주고 사용방법을 알려줘 | `collect_gestures.py`, `train_gestures.py`, `custom_gesture_webcam.py` |
| 10 | UI형식으로 만들어줘 | `gesture_studio.py` (수집·학습·인식 탭 UI) |
| 12 | gesture_studio.py 실행시켜 | 실행 |
| 15 | 제스처 마저 학습시킬 스튜디오켜줘 | 실행 → Heart·Nike·ok·ㅗ 각 300개, 테스트 정확도 97.1% |

## 4. 웹 버전

| # | 프롬프트 | 결과 |
| --- | --- | --- |
| 11 | 훈련한 모델로Nike 가 나오면 Nike 로고가 뜨고 ok 를 하면 ok 이모지가, Heart하면 Heart이모지가 뜨도록 웹버전으로 만들어줘 | `web/` 데모, `export_model_web.py`, `serve_web.py` |
| 16 | 학습한 ㅗ는 Fuck you 모양 이모지로 뜨게하고 하트이모지는 하트가 앞으로 발사하게 되는 걸로 수정해 | 🖕 스티커, 하트 연속 발사 효과 |
| 17 | 웹 버전 켜줘 | 실행 |

## 5. PowerShell 단축 명령

| # | 프롬프트 | 결과 |
| --- | --- | --- |
| 13 | 지금 제스처 스튜디오를 파워셀에 입력하면 켜지는 코드를 만들어줘 | `$PROFILE`에 `studio` / `gesture-studio` 등록 |
| 14 | 웹버전도 파워쉘에 입력하면 켜지게 해줘 | `web` / `gesture-web`, `gesture-web-stop` 등록 |

## 6. 배포·정리

| # | 프롬프트 | 결과 |
| --- | --- | --- |
| 18 | 깃허브에 깃페이지에다가 올려줘 그리고 배포할 URL만들어줘 | GitHub Pages 배포: https://leesm3528-netizen.github.io/mediapipe-webcam/ |
| 19 | 오늘 깃허브에 올리고 학습한거 정리해줘 README에 | README에 작업 기록 추가 |
| 20 | 내가 지금 landmaker webcam 프로젝트할때 사용한 프롬포트 모아보고싶어 | 프롬프트 목록 정리 |
| 21 | 잘 정리해서 깃허브에 올려줘 | 이 문서 (`PROMPTS.md`) |

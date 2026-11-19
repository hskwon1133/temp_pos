# Subject 10 — 2D 자세 추정 기반 재활 운동 동작 분류

스마트폰으로 찍은 운동 영상에서 **사람의 관절 위치(Keypoint)** 를 뽑아내고,
그 움직임을 보고 **어떤 운동을 하고 있는지 자동으로 맞히는** 프로젝트입니다.

```
운동 영상(MP4) ─▶ YOLO Pose로 관절 17개 추출 ─▶ 품질 검사(QA) ─▶ 사람별로 학습/검증/테스트 분리
             ─▶ 30프레임 시퀀스로 변환 ─▶ Random Forest 학습 ─▶ 성능 평가 · 오답 분석
```

현재 분류 대상 운동은 3가지입니다.

| ID | 운동 이름 | 설명 |
|----|-----------|------|
| A001 | Seated_Knee_Extension | 앉아서 다리 앞으로 뻗기 |
| A002 | Shoulder_Reach_Up_&Down | 팔 위아래로 뻗기 |
| A003 | Foam_Roller_Hip_Hinge | 폼롤러 힙힌지 |

> `configs/action_classes.csv`에는 A001~A010까지 10개 동작이 정의되어 있지만, 영상이 준비된 것은 A001~A003입니다.

---

## 1. 준비물

| 항목 | 내용 |
|------|------|
| 운영체제 | Linux 또는 Windows의 **WSL2(Ubuntu)** 권장 (개발 환경: WSL2 Ubuntu) |
| Python | 3.11 이상 (개발 환경: 3.12) |
| 패키지 관리자 | [uv](https://docs.astral.sh/uv/) |
| GPU | 없어도 실행됩니다. NVIDIA GPU가 있으면 자동으로 사용합니다 (개발 환경: RTX 3060) |

### uv 설치 (처음 한 번만)

터미널(Ubuntu)을 열고 아래를 붙여넣으세요.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 터미널을 껐다 켜고 `uv --version`이 나오면 성공입니다.

---

## 2. 설치

```bash
# 1) 프로젝트 폴더로 이동
cd subject10_pose_action

# 2) 필요한 라이브러리 설치 (.venv 폴더가 자동으로 만들어집니다)
uv sync

# 3) 환경 확인
uv run python scripts/00_check_environment.py
```

3번을 실행하면 Python/PyTorch/CUDA 버전과 `Selected device`가 출력됩니다.
`Selected device: 0` 이면 GPU, `cpu` 이면 CPU로 실행됩니다. 결과는 `reports/day05/environment.txt`에 저장됩니다.

> **포즈 모델 파일(`yolo11n-pose.pt`)** 은 프로젝트 맨 위 폴더에 있습니다. 없으면 처음 실행할 때 자동으로 다운로드됩니다.

> 이 README의 모든 명령은 **프로젝트 맨 위 폴더(`subject10_pose_action/`)에서** 실행합니다.
> `uv run` 대신 `source .venv/bin/activate`로 가상환경을 켠 뒤 `python ...`으로 실행해도 됩니다.

---

## 3. 데이터 준비

`data/`, `artifacts/` 폴더와 영상·모델 파일은 용량 때문에 Git에 올라가지 않습니다(`.gitignore`).
처음 받았다면 영상을 아래 규칙대로 직접 넣어야 합니다.

```
data/day05/videos/
├── S01/
│   ├── S01_A001_T01.mp4
│   ├── S01_A001_T02.mp4
│   ├── S01_A002_T01.mp4
│   └── ...
├── S02/
├── S03/
└── S04/
```

파일 이름 규칙: **`{사람}_{동작}_{촬영회차}.mp4`**

- 사람(Subject): `S01` ~ `S04`
- 동작(Action): `A001` ~ `A003`
- 촬영회차(Take): `T01`, `T02`

→ 4명 × 3동작 × 2회 = **영상 24개**

---

## 4. 전체 실행 순서 (Day05 파이프라인)

아래 순서대로 한 줄씩 실행하면 됩니다. 각 단계는 앞 단계의 결과 파일을 입력으로 씁니다.

### STEP 1. 영상 확인

```bash
uv run python scripts/25_check_day05_data.py
```
영상 24개가 규칙대로 있는지, 열리는지 검사합니다.
→ 결과: `reports/day05/tables/video_inventory.csv`

### STEP 2. (선택) 관절이 잘 잡히는지 눈으로 확인

```bash
uv run python scripts/04_pose_demo_video.py                      # 24개 전부
uv run python scripts/04_pose_demo_video.py --subject S01        # S01만
uv run python scripts/04_pose_demo_video.py --action A002 --take T01
```
영상 위에 뼈대(Skeleton)를 그린 결과 영상을 만듭니다.

| 옵션 | 의미 |
|------|------|
| `--source 경로` | 영상 1개만 처리 |
| `--subject` / `--action` / `--take` | 원하는 영상만 골라서 처리 |
| `--max-frames N` | 영상당 최대 N프레임만 처리 (기본 2000, `0`이면 끝까지) |
| `--rotate cw` / `ccw` | 누워서 하는 동작일 때 영상을 90도 돌려서 추정 |

→ 결과: `reports/day05/pose_demo/S01/S01_A001_T01_pose.mp4` …, `reports/day05/pose_demo/pose_demo_summary.csv`

### STEP 3. 관절 좌표 추출 ⏱️ 가장 오래 걸림

```bash
uv run python scripts/26_extract_keypoint_dataset.py
```
모든 영상의 모든 프레임에서 관절 17개 좌표와 신뢰도를 CSV로 저장합니다.
→ 결과: `data/day05/keypoints/{사람}/{동작이름}/{clip_id}.csv`, `data/day05/manifests/keypoint_dataset_manifest.csv`

### STEP 4. 자동 품질 검사(QA)

```bash
uv run python scripts/27_keypoint_qa.py
```
사람이 잘 검출됐는지, 중요 관절의 신뢰도가 낮지 않은지를 보고 각 영상에 `AUTO_PASS` 또는 `CHECK`를 붙입니다.
→ 결과: `data/day05/qa/keypoint_clip_qa.csv`, `data/day05/qa/keypoint_clip_review.csv`

### STEP 5. 사람이 직접 검토 ✍️ (수작업)

`data/day05/qa/keypoint_clip_review.csv`를 엑셀 등으로 열고,
STEP 2에서 만든 결과 영상을 보면서 **`human_decision`** 칸을 채웁니다.

| 값 | 의미 |
|----|------|
| `PASS` | 학습에 사용 |
| `RESHOOT` | 다시 촬영 필요 |
| `EXCLUDE` | 데이터에서 제외 |

처음에는 모두 `UNREVIEWED`로 되어 있습니다. 메모가 있으면 `human_note`에 적으세요.
다 채웠으면 빠진 곳이 없는지 검사합니다.

```bash
uv run python scripts/28_validate_human_review.py
```

### STEP 6. 승인된 영상 목록 만들기

```bash
uv run python scripts/29_build_approved_manifest.py
```
`PASS`인 영상만 골라냅니다.
→ 결과: `data/day05/manifests/approved_clip_manifest.csv`

### STEP 7. 학습 / 검증 / 테스트 나누기

```bash
uv run python scripts/30_build_subject_split.py
uv run python scripts/31_check_split_leakage.py
```
**사람 단위**로 데이터를 나눕니다 (`configs/day05_subject_split.json`).

| 구분 | 사람 |
|------|------|
| Train (학습) | S01 |
| Validation (검증) | S02 |
| Test (최종 시험) | S03, S04 |

같은 사람의 영상이 학습과 평가에 동시에 들어가면 성능이 부풀려지기 때문입니다.
31번은 이런 섞임(Leakage)이 없는지 확인합니다.
→ 결과: `data/day05/splits/train.csv`, `val.csv`, `test.csv`

> (참고) `uv run python scripts/32_demo_frame_split_leakage.py` — 프레임 단위로 무작위 분리하면 왜 위험한지 보여주는 시연용 스크립트입니다.

### STEP 8. 모델 입력 데이터 만들기

```bash
uv run python scripts/33_check_day06_inputs.py
uv run python scripts/34_build_sequence_dataset.py
```
33번은 입력 파일이 모두 있는지 점검하고(→ `reports/day05/tables/day05_input_check.csv`),
34번은 각 영상을 **30프레임으로 맞춰(리샘플링)** 정규화된 관절 좌표 + 팔꿈치·무릎 각도를 한 줄로 펼칩니다.
→ 결과: `data/day05/sequences/train.csv`, `val.csv`, `test.csv` (영상 1개 = 1줄, Feature 1,140개)

### STEP 9. 모델 학습

```bash
uv run python scripts/35_train_rf_baseline.py
```
Random Forest 분류기를 학습합니다 (설정: `configs/day06_baseline.json`).
→ 결과: `artifacts/day05/`
- `rf_action_baseline.joblib` — 학습된 모델
- `rf_feature_columns.json` — 입력 Feature 이름과 순서
- `rf_training_info.json` — 학습에 쓴 데이터와 설정 요약

### STEP 10. 검증 및 오답 분석

```bash
uv run python scripts/36_evaluate_validation.py
uv run python scripts/37_analyze_misclassified.py
```
→ 결과:
- `reports/day05/metrics/validation_metrics.txt` — 정확도, F1 점수
- `reports/day05/metrics/validation_predictions.csv` — 영상별 예측 결과
- `reports/day05/confusion/validation_confusion_matrix.png` — 혼동행렬 그림
- `reports/day05/misclassified/validation_misclassified.csv` — 틀린 영상 목록

### 한 번에 실행하기

STEP 5(수작업 검토)가 이미 끝나 있다면 아래처럼 이어서 실행할 수 있습니다.

```bash
for s in 28_validate_human_review 29_build_approved_manifest 30_build_subject_split \
         31_check_split_leakage 33_check_day06_inputs 34_build_sequence_dataset \
         35_train_rf_baseline 36_evaluate_validation 37_analyze_misclassified; do
  echo "▶ $s"; uv run python scripts/$s.py || break
done
```

---

## 5. 현재 결과 (Validation: S02)

| 지표 | 값 |
|------|-----|
| Accuracy | **0.833** (6개 중 5개 정답) |
| Macro F1 | **0.822** |

| 동작 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Foam_Roller_Hip_Hinge | 0.67 | 1.00 | 0.80 |
| Seated_Knee_Extension | 1.00 | 0.50 | 0.67 |
| Shoulder_Reach_Up_&Down | 1.00 | 1.00 | 1.00 |

오답 1건: `S02_A001_T02` (실제 Seated_Knee_Extension → 예측 Foam_Roller_Hip_Hinge, 확신도 0.40)

> 학습 데이터가 S01 한 명의 영상 6개뿐이라 수치 변동이 큽니다. Test(S03, S04) 평가는 아직 진행하지 않았습니다.

---

## 6. 폴더 구조

```
subject10_pose_action/
├── configs/                 # 설정 파일
│   ├── project.json             # 포즈 모델, 이미지 크기, 신뢰도 기준, 장치(auto=GPU 자동)
│   ├── action_classes.csv       # 동작 ID ↔ 이름 ↔ 설명 (A001~A010)
│   ├── day05_subject_split.json # 사람별 Train/Val/Test 배정
│   └── day06_baseline.json      # 시퀀스 길이(30), 최소 유효 프레임(10), RF 트리 수(300) 등
│
├── src/                     # 여러 스크립트가 함께 쓰는 공통 함수
│   ├── common.py                # 설정 읽기, GPU/CPU 선택, 기본 폴더 생성
│   ├── pose_utils.py            # COCO 17개 관절 이름, 뼈대 연결, 신뢰도 판정(VALID/WARNING/LOW)
│   ├── feature_utils.py         # 관절 각도, 몸 중심, 어깨너비 정규화, 좌우 반전
│   └── sequence_utils.py        # 프레임 Feature 생성, 30프레임 리샘플링, 시퀀스 펼치기
│
├── scripts/                 # 실행 스크립트 (번호 순서대로 진행)
│   ├── 00~05   기초: 환경 확인, 이미지/영상 읽기, 포즈 추정, IP Webcam 캡처
│   ├── 11~17   (Day03) 관절 각도, 정규화, 좌우 반전, 프레임 Feature, 팔 들기 규칙
│   ├── 19~24   (Day04) 동작 영상 점검, 키프레임 추출, 시퀀스 생성·요약
│   ├── 25~37   (Day05) ★ 위 4장의 파이프라인
│   └── 38~40   
│
├── data/day05/              # 입력 데이터와 중간 산출물
│   ├── videos/                  # 원본 영상 (S01~S04)
│   ├── keypoints/               # 영상별 관절 좌표 CSV
│   ├── manifests/               # 전체 / 승인 영상 목록
│   ├── qa/                      # 자동 QA 결과, 사람 검토표
│   ├── splits/                  # Train/Val/Test 영상 목록
│   └── sequences/               # 모델 입력용 시퀀스 데이터
│
├── artifacts/day05/         # 학습된 모델과 Feature 정보
├── reports/day05/           # 결과물 (표, 시연 영상, 성능 지표, 그림)
│
├── pyproject.toml / uv.lock # 라이브러리 목록 (uv sync로 설치)
├── yolo11n-pose.pt          # YOLO 포즈 추정 모델
└── 추가자료.zip             # 
```

---

## 7. 자주 겪는 문제

| 증상 | 해결 방법 |
|------|-----------|
| `uv: command not found` | uv 설치 후 터미널을 다시 열어 주세요. |
| `ModuleNotFoundError: No module named 'src'` | 프로젝트 **맨 위 폴더**에서 실행했는지 확인하세요. |
| `FileNotFoundError` | 앞 단계를 건너뛰었는지 확인하세요. 각 단계는 이전 단계의 결과 파일이 필요합니다. |
| 너무 느림 | GPU가 없으면 STEP 3이 오래 걸립니다. `00_check_environment.py`에서 `Selected device`를 확인하세요. |
| 28번에서 미검토 오류 | `keypoint_clip_review.csv`의 `human_decision`에 `UNREVIEWED`가 남아 있습니다. |

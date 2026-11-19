from __future__ import annotations

# ==================================================
# ① 필요한 라이브러리 가져오기
# ==================================================
import csv
from pathlib import Path


# ==================================================
# ② 프로젝트 경로와 입력 파일 위치 설정
# ==================================================
# ACTION_FILE : Action 정의 CSV (ID, Label, 설명)
# VIDEO_DIR   : 사람별 폴더(S01~S04) 안에 영상이 있는 폴더
#
# 우리 데이터는 Action 1개에 영상이 여러 개(4명 x 2테이크)라서
# 영상 목록은 CSV의 video_file 칸이 아니라
# 아래 EXPECTED_ACTIONS에서 관리한다.

ROOT = Path(__file__).resolve().parents[1]
ACTION_FILE = ROOT / "configs" / "action_classes.csv"
VIDEO_DIR = ROOT / "data" / "day05" / "videos"

SUBJECTS = ["S01", "S02", "S03", "S04"]
TAKES = ["T01", "T02"]


# ==================================================
# ③ 정상적인 Action ID, Label, 영상 목록 기준 정의
# ==================================================
# Key   : Action ID
# Value : (Action Label, 영상 경로 목록)
# 영상 경로는 VIDEO_DIR 기준 상대 경로 (예: S01/S01_A001_T01.mp4)

def videos_for(action_id: str) -> list[str]:
    """동작 하나의 영상 경로 목록 (4명 x 2테이크 = 8개)."""
    return [f"{s}/{s}_{action_id}_{t}.mp4" for s in SUBJECTS for t in TAKES]


EXPECTED_ACTIONS = {
    "A001": ("Seated_Knee_Extension", videos_for("A001")),
    "A002": ("Shoulder_Reach_Up_&Down", videos_for("A002")),
    "A003": ("Foam_Roller_Hip_Hinge", videos_for("A003")),
    # 아래는 RGB 영상이 아직 day05 폴더에 없음 → 빈 목록
    # 영상을 넣으면 [] 를 videos_for("A00x")로 바꾸기
    "A004": ("Assisted_Shoulder_Flexion", []),
    "A005": ("Straight_Leg_Raise", []),
    "A006": ("Side-lying_Leg_Raise", []),
    "A007": ("Hip_Rotation_to_Extension", []),
    "A008": ("Supine_Leg_Raise_n_Abduction", []),
    "A009": ("Adductor_Bridge", []),   # 11/17 촬영함. 폴더에 넣으면 바꾸기
    "A010": ("Bird_Dog", []),          # 11/17 촬영함. 폴더에 넣으면 바꾸기
}


# ==================================================
# ④ 중복된 값을 찾는 함수
# ==================================================
def find_duplicates(values: list[str]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


def main() -> None:

    # ⑤ Action 설정 CSV 파일 존재 여부 확인
    if not ACTION_FILE.exists():
        raise FileNotFoundError(ACTION_FILE)

    # ⑥ CSV 파일 읽기 및 필수 컬럼 확인
    # video_file 열은 남겨 두지만 비어 있어도 된다 (검사하지 않음)
    with ACTION_FILE.open("r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"action_id", "action_label", "type", "description"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise RuntimeError("필수 열이 없습니다: " + ", ".join(sorted(missing_columns)))
        rows = list(reader)

    # ⑦ Action 데이터 존재 여부와 개수 확인
    if not rows:
        raise RuntimeError("Action 정의가 없습니다.")

    problems = []

    if len(rows) != len(EXPECTED_ACTIONS):
        problems.append(
            f"Action 개수는 {len(EXPECTED_ACTIONS)}개여야 합니다. 현재 {len(rows)}개입니다."
        )

    # ⑧ ID, Label 중복 검사 (video_file 중복 검사는 뺐다)
    ids = [row["action_id"].strip() for row in rows]
    labels = [row["action_label"].strip() for row in rows]

    duplicate_ids = find_duplicates(ids)
    duplicate_labels = find_duplicates(labels)

    if duplicate_ids:
        problems.append("중복 Action ID: " + ", ".join(sorted(duplicate_ids)))
    if duplicate_labels:
        problems.append("중복 Action Label: " + ", ".join(sorted(duplicate_labels)))

    # ⑨ Action마다 Label 비교 + 영상 존재 확인
    print(f"Action count: {len(rows)}\n")

    total_videos = 0
    total_found = 0

    for row in rows:
        action_id = row["action_id"].strip()
        action_label = row["action_label"].strip()
        action_type = row["type"].strip()

        expected = EXPECTED_ACTIONS.get(action_id)
        if expected is None:
            problems.append(f"알 수 없는 Action ID: {action_id}")
            continue

        expected_label, expected_videos = expected

        # ⑩ Label이 기준과 같은지
        if action_label != expected_label:
            problems.append(f"{action_id}의 Label은 {expected_label}이어야 합니다.")

        # ⑪ type 검사
        if action_type != "dynamic":
            problems.append(f"{action_id}의 type은 dynamic이어야 합니다.")

        # 영상 존재 확인: 기준 목록의 파일이 실제로 있는지
        found = 0
        for v in expected_videos:
            if (VIDEO_DIR / v).exists():
                found += 1
            else:
                problems.append(f"없는 동영상: {v}")

        total_videos += len(expected_videos)
        total_found += found

        status = "영상 없음 (건너뜀)" if not expected_videos else f"videos={found}/{len(expected_videos)}"
        print(f"{action_id} {action_label:30s} {status}")

    # ⑫ 필수 Action ID 누락 여부 확인
    missing_ids = set(EXPECTED_ACTIONS) - set(ids)
    if missing_ids:
        problems.append("누락 Action ID: " + ", ".join(sorted(missing_ids)))

    # ⑬ 전체 결과 출력
    print(f"\n영상 확인: {total_found}/{total_videos}")
    print()

    if problems:
        for problem in dict.fromkeys(problems):
            print("[ERROR]", problem)
        raise RuntimeError("Action 설정을 다시 확인하세요.")

    print("Action class check: OK")


if __name__ == "__main__":
    main()
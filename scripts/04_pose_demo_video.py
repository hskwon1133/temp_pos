from __future__ import annotations

# ==================================================
# 여러 영상에 YOLO Pose를 적용해 결과 영상과 요약표를 저장한다.
#
# 영상 위치: data/day05/videos/S01/S01_A001_T01.mp4 ...
# 결과 위치: reports/day05/pose_demo/S01/S01_A001_T01_pose.mp4 ...
#           reports/day05/pose_demo/pose_demo_summary.csv
#
# 사용 예 (프로젝트 맨 위 폴더에서):
#   python scripts/pose_demo_videos.py                          # 24개 전부, 영상마다 150 Frame
#   python scripts/pose_demo_videos.py --action A001            # A001만 (8개)
#   python scripts/pose_demo_videos.py --subject S02 --take T02 # S02의 T02만
#   python scripts/pose_demo_videos.py --source data/day05/videos/S01/S01_A001_T01.mp4 --max-frames 0
#   python scripts/pose_demo_videos.py --action A009 --rotate cw   # 누운 동작: 90도 돌려서 추정
# ==================================================

# ① 필요한 라이브러리 불러오기
import argparse
import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ② 프로젝트 경로 설정 및 공통 함수 불러오기
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)

VIDEO_DIR = ROOT / "data" / "day05" / "videos"
OUTPUT_DIR = ROOT / "reports" / "day05" / "pose_demo"

# 회전 방향: (추정 전에 돌리는 방향, 결과를 원래대로 되돌리는 방향)
ROTATIONS = {
    "cw": (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE),   # 머리가 화면 왼쪽일 때
    "ccw": (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE),  # 머리가 화면 오른쪽일 때
}


# ③ 터미널 입력 받기
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="영상 1개만 처리할 때 그 MP4 경로")
    parser.add_argument("--subject", help="예: S01 (생략하면 전체)")
    parser.add_argument("--action", help="예: A001 (생략하면 전체)")
    parser.add_argument("--take", help="예: T01 (생략하면 전체)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="영상마다 처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )
    parser.add_argument(
        "--rotate",
        choices=["none", "cw", "ccw"],
        default="none",
        help="누운 동작용. cw=시계 방향, ccw=반시계 방향으로 돌려서 추정",
    )
    return parser.parse_args()


# ④ 처리할 영상 목록 만들기
def collect_videos(args) -> list[Path]:
    if args.source:
        return [Path(args.source)]

    subject = args.subject or "S*"
    action = args.action or "A*"
    take = args.take or "T*"
    # data/day05/videos/S01/S01_A001_T01.mp4 형식
    pattern = f"{subject}/{subject}_{action}_{take}.mp4"
    return sorted(VIDEO_DIR.glob(pattern))


# ⑤ 영상 1개 처리
def process_video(model, config, device, source: Path, max_frames: int, rotate: str) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) 

    output_fps = fps if fps > 0 else 30.0

    success, frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError(f"첫 Frame을 읽지 못했습니다: {source}")

    height, width = frame.shape[:2]

    # 결과 경로: 사람 폴더를 그대로 유지 (S01/S01_A001_T01_pose.mp4)
    suffix = "_pose" if rotate == "none" else f"_pose_rot{rotate}"
    output_path = OUTPUT_DIR / source.parent.name / f"{source.stem}{suffix}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),          # 원본 크기 그대로 저장 (회전해도 다시 되돌림)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"결과 MP4 파일을 만들 수 없습니다: {output_path}")

    frame_number = 0
    person_frames = 0      # 사람을 1명 이상 찾은 Frame 수
    conf_sum = 0.0         # 첫 번째 사람의 관절 평균 confidence 합

    while success:
        frame_number += 1

        # 누운 동작이면 돌려서 추정
        input_frame = frame if rotate == "none" else cv2.rotate(frame, ROTATIONS[rotate][0])

        results = model.predict(
            source=input_frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )
        result = results[0]

        # 사람을 찾았는지, 관절 confidence는 얼마인지 기록
        if result.keypoints is not None and result.keypoints.conf is not None and len(result.keypoints.conf) > 0:
            person_frames += 1
            conf_sum += float(result.keypoints.conf[0].mean())

        # BBox + Keypoint + Skeleton 그리기, 돌렸으면 원래 방향으로 되돌려 저장
        plotted = result.plot()
        if rotate != "none":
            plotted = cv2.rotate(plotted, ROTATIONS[rotate][1])
        writer.write(plotted)

        if max_frames > 0 and frame_number >= max_frames:
            break
        success, frame = capture.read()

    capture.release()
    writer.release()

    return {
        "video": source.name,
        "subject": source.parent.name,
        "rotate": rotate,
        "processed_frames": frame_number,
        "person_frames": person_frames,
        "person_ratio": round(person_frames / frame_number, 3) if frame_number else 0,
        "mean_keypoint_conf": round(conf_sum / person_frames, 3) if person_frames else 0,
        "output": str(output_path.relative_to(ROOT)),
    }


def main() -> None:
    args = parse_args()

    videos = collect_videos(args)
    if not videos:
        raise FileNotFoundError(
            f"처리할 영상이 없습니다. 폴더와 조건을 확인하세요: {VIDEO_DIR}"
        )
    for v in videos:
        if not v.exists():
            raise FileNotFoundError(f"파일이 없습니다: {v}")

    # ⑥ 설정과 모델은 한 번만 불러온다 (영상마다 불러오면 느림)
    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    print(f"처리할 영상: {len(videos)}개, 영상마다 최대 {args.max_frames or '전체'} Frame, rotate={args.rotate}\n")

    # ⑦ 영상마다 처리하고 결과를 모은다
    rows = []
    for i, source in enumerate(videos, start=1):
        try:
            row = process_video(model, config, device, source, args.max_frames, args.rotate)
        except RuntimeError as e:
            print(f"[{i}/{len(videos)}] [건너뜀] {e}")
            continue
        rows.append(row)
        print(
            f"[{i}/{len(videos)}] {row['video']:22s} "
            f"frames={row['processed_frames']:5d}  "
            f"person={row['person_ratio']:.3f}  "
            f"conf={row['mean_keypoint_conf']:.3f}"
        )

    # ⑧ 요약표 저장
    if rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_DIR / "pose_demo_summary.csv"
        # 같은 이름이 있으면 덮어쓰므로, 회전 결과는 이름을 나눠 저장
        if args.rotate != "none":
            summary_path = OUTPUT_DIR / f"pose_demo_summary_rot{args.rotate}.csv"
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved videos : {OUTPUT_DIR}")
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
"""
Build frame-level dataset for BrazingSense.

Input:
    data/annotations/stage_intervals.csv

Expected CSV columns:
    video_id,video_path,stage_name,start_s,end_s

Output:
    data/processed/frames/<video_id>/*.jpg
    data/annotations/frame_labels.csv

Example:
    python3 scripts/build_frame_dataset.py \
        --intervals data/annotations/stage_intervals.csv \
        --output-frames data/processed/frames \
        --output-labels data/annotations/frame_labels.csv \
        --fps 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2


STAGE_TO_ID: Dict[str, int] = {
    "inactive_preparation": 0,
    "flux_activation": 1,
    "active_brazing": 2,
    "stabilization": 3,
}


@dataclass(frozen=True)
class StageInterval:
    video_id: str
    video_path: Path
    stage_name: str
    start_s: float
    end_s: float

    @property
    def stage_id(self) -> int:
        if self.stage_name not in STAGE_TO_ID:
            raise ValueError(
                f"Unknown stage_name='{self.stage_name}'. "
                f"Known stages: {list(STAGE_TO_ID.keys())}"
            )
        return STAGE_TO_ID[self.stage_name]

    def contains(self, timestamp_s: float) -> bool:
        # Левая граница включается, правая исключается.
        # Это помогает избежать двойной разметки на стыках интервалов.
        return self.start_s <= timestamp_s < self.end_s


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frame-level dataset from stage interval annotations."
    )

    parser.add_argument(
        "--intervals",
        type=Path,
        default=Path("data/annotations/stage_intervals.csv"),
        help="Path to stage_intervals.csv",
    )

    parser.add_argument(
        "--output-frames",
        type=Path,
        default=Path("data/processed/frames"),
        help="Directory where extracted frames will be saved",
    )

    parser.add_argument(
        "--output-labels",
        type=Path,
        default=Path("data/annotations/frame_labels.csv"),
        help="Path to output frame_labels.csv",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Target frame extraction FPS. Example: 3 means 3 frames per second.",
    )

    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image extension.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality, used only for jpg output.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames and labels.",
    )

    parser.add_argument(
        "--save-stats",
        type=Path,
        default=Path("data/annotations/frame_dataset_stats.json"),
        help="Path to save dataset statistics as JSON.",
    )

    return parser.parse_args()


def read_intervals(csv_path: Path) -> List[StageInterval]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Intervals CSV not found: {csv_path}")

    intervals: List[StageInterval] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"video_id", "video_path", "stage_name", "start_s", "end_s"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing columns in {csv_path}: {sorted(missing)}. "
                f"Required columns: {sorted(required_columns)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            try:
                video_id = row["video_id"].strip()
                video_path = Path(row["video_path"].strip())
                stage_name = row["stage_name"].strip()
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except Exception as exc:
                raise ValueError(f"Failed to parse row {row_idx}: {row}") from exc

            if not video_id:
                raise ValueError(f"Empty video_id at row {row_idx}")

            if stage_name not in STAGE_TO_ID:
                raise ValueError(
                    f"Unknown stage_name='{stage_name}' at row {row_idx}. "
                    f"Known stages: {list(STAGE_TO_ID.keys())}"
                )

            if end_s <= start_s:
                raise ValueError(
                    f"Invalid interval at row {row_idx}: "
                    f"start_s={start_s}, end_s={end_s}"
                )

            intervals.append(
                StageInterval(
                    video_id=video_id,
                    video_path=video_path,
                    stage_name=stage_name,
                    start_s=start_s,
                    end_s=end_s,
                )
            )

    if not intervals:
        raise ValueError(f"No intervals found in {csv_path}")

    return intervals


def group_intervals_by_video(
    intervals: List[StageInterval],
) -> Dict[str, List[StageInterval]]:
    grouped: Dict[str, List[StageInterval]] = {}

    for interval in intervals:
        grouped.setdefault(interval.video_id, []).append(interval)

    for video_id, video_intervals in grouped.items():
        video_intervals.sort(key=lambda x: x.start_s)

        # Проверяем, что у одного video_id не указаны разные video_path.
        paths = {str(x.video_path) for x in video_intervals}
        if len(paths) != 1:
            raise ValueError(
                f"Video '{video_id}' has multiple video_path values: {sorted(paths)}"
            )

        # Проверяем пересечения интервалов.
        for prev, cur in zip(video_intervals, video_intervals[1:]):
            if cur.start_s < prev.end_s:
                raise ValueError(
                    f"Overlapping intervals for video '{video_id}': "
                    f"{prev.stage_name} [{prev.start_s}, {prev.end_s}] and "
                    f"{cur.stage_name} [{cur.start_s}, {cur.end_s}]"
                )

    return grouped


def find_stage_for_timestamp(
    timestamp_s: float,
    intervals: List[StageInterval],
) -> Optional[StageInterval]:
    for interval in intervals:
        if interval.contains(timestamp_s):
            return interval

    # На последнем кадре timestamp может быть почти равен end_s последнего интервала.
    # Разрешаем небольшую погрешность.
    eps = 1e-3
    last = intervals[-1]
    if abs(timestamp_s - last.end_s) <= eps:
        return last

    return None


def ensure_output_ready(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}. "
            f"Use --overwrite if you want to regenerate it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)


def save_frame(frame_path: Path, frame, image_ext: str, jpeg_quality: int) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    if image_ext == "jpg":
        ok = cv2.imwrite(
            str(frame_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    else:
        ok = cv2.imwrite(str(frame_path), frame)

    if not ok:
        raise RuntimeError(f"Failed to save frame: {frame_path}")


def build_dataset_for_video(
    video_id: str,
    intervals: List[StageInterval],
    output_frames_dir: Path,
    target_fps: float,
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
) -> List[Dict[str, object]]:
    video_path = intervals[0].video_path

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found for video_id='{video_id}': {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / source_fps if source_fps > 0 else 0.0

    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video '{video_id}': {source_fps}")

    if target_fps <= 0:
        cap.release()
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    step_s = 1.0 / target_fps
    max_end_s = max(x.end_s for x in intervals)

    # Не выходим за пределы видео и разметки.
    effective_end_s = min(duration_s, max_end_s)

    logging.info(
        "Processing %s: source_fps=%.3f, frames=%d, duration=%.2fs, annotated_end=%.2fs",
        video_id,
        source_fps,
        frame_count,
        duration_s,
        max_end_s,
    )

    video_output_dir = output_frames_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    timestamp_s = 0.0
    sample_idx = 0

    while timestamp_s < effective_end_s:
        stage_interval = find_stage_for_timestamp(timestamp_s, intervals)

        if stage_interval is None:
            # Это не обязательно ошибка: могут быть намеренные пропуски в разметке.
            # Но для первого датасета лучше их видеть в логах.
            logging.warning(
                "No stage found for %s at %.3fs. Skipping frame.",
                video_id,
                timestamp_s,
            )
            timestamp_s += step_s
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = cap.read()

        if not ok or frame is None:
            logging.warning(
                "Could not read frame for %s at %.3fs. Stopping video processing.",
                video_id,
                timestamp_s,
            )
            break

        frame_name = f"{video_id}_{sample_idx:06d}_{timestamp_s:08.3f}s.{image_ext}"
        frame_path = video_output_dir / frame_name

        if frame_path.exists() and not overwrite:
            raise FileExistsError(
                f"Frame already exists: {frame_path}. "
                f"Use --overwrite if you want to regenerate dataset."
            )

        save_frame(frame_path, frame, image_ext, jpeg_quality)

        rows.append(
            {
                "video_id": video_id,
                "video_path": str(video_path),
                "frame_path": str(frame_path),
                "timestamp_s": round(timestamp_s, 6),
                "stage_id": stage_interval.stage_id,
                "stage_name": stage_interval.stage_name,
                "source_fps": round(source_fps, 6),
                "target_fps": target_fps,
            }
        )

        sample_idx += 1
        timestamp_s += step_s

    cap.release()

    logging.info("Extracted %d frames for %s", len(rows), video_id)

    return rows


def write_frame_labels(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "video_path",
        "frame_path",
        "timestamp_s",
        "stage_id",
        "stage_name",
        "source_fps",
        "target_fps",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_stage: Dict[str, int] = {}
    by_video: Dict[str, int] = {}

    for row in rows:
        stage_name = str(row["stage_name"])
        video_id = str(row["video_id"])

        by_stage[stage_name] = by_stage.get(stage_name, 0) + 1
        by_video[video_id] = by_video.get(video_id, 0) + 1

    return {
        "total_frames": len(rows),
        "stage_to_id": STAGE_TO_ID,
        "frames_by_stage": by_stage,
        "frames_by_video": by_video,
    }


def save_stats(stats: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()
    args = parse_args()

    ensure_output_ready(args.output_labels, overwrite=args.overwrite)

    intervals = read_intervals(args.intervals)
    grouped = group_intervals_by_video(intervals)

    all_rows: List[Dict[str, object]] = []

    for video_id, video_intervals in grouped.items():
        rows = build_dataset_for_video(
            video_id=video_id,
            intervals=video_intervals,
            output_frames_dir=args.output_frames,
            target_fps=args.fps,
            image_ext=args.image_ext,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No frames were extracted. Check videos and annotations.")

    write_frame_labels(all_rows, args.output_labels)

    stats = collect_stats(all_rows)
    save_stats(stats, args.save_stats)

    logging.info("Saved labels to: %s", args.output_labels)
    logging.info("Saved stats to: %s", args.save_stats)
    logging.info("Total frames: %d", stats["total_frames"])

    logging.info("Frames by stage:")
    for stage_name, count in stats["frames_by_stage"].items():
        logging.info("  %s: %d", stage_name, count)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Split BrazingSense frame-level dataset by video_id.

Input:
    data/annotations/frame_labels.csv

Output:
    data/annotations/splits/train.csv
    data/annotations/splits/val.csv
    data/annotations/splits/test.csv
    data/annotations/splits/split_stats.json

Important:
    Split is performed by video_id, not by frame.
    This prevents leakage between train/val/test caused by near-duplicate frames
    from the same video.

Optional:
    You can pass --blacklist data/annotations/blacklist.txt to exclude videos
    from train/val/test split.

Blacklist format:
    One video_id per line.

Example:
    MVI_6270
    MVI_6273
    MVI_6278

Example:
    python3 scripts/split_frame_dataset.py \
        --labels data/annotations/frame_labels.csv \
        --output-dir data/annotations/splits \
        --blacklist data/annotations/blacklist.txt \
        --train-ratio 0.70 \
        --val-ratio 0.15 \
        --test-ratio 0.15 \
        --seed 42 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


STAGE_ORDER = [
    "inactive_preparation",
    "flux_activation",
    "active_brazing",
    "stabilization",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split frame-level dataset by video_id."
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/frame_labels.csv"),
        help="Path to frame_labels.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/splits"),
        help="Directory for train/val/test CSV files",
    )

    parser.add_argument(
        "--blacklist",
        type=Path,
        default=None,
        help=(
            "Optional path to blacklist.txt. "
            "Each line should contain one video_id to exclude from split."
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.67,
        help="Train video ratio",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.17,
        help="Validation video ratio",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.16,
        help="Test video ratio",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files",
    )

    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum:.6f}: "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    for name, value in {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


def read_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)

    required_columns = {
        "video_id",
        "video_path",
        "frame_path",
        "timestamp_s",
        "stage_id",
        "stage_name",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {labels_path}: {sorted(missing)}. "
            f"Required columns: {sorted(required_columns)}"
        )

    if df.empty:
        raise ValueError(f"Labels file is empty: {labels_path}")

    unknown_stages = sorted(set(df["stage_name"]) - set(STAGE_ORDER))
    if unknown_stages:
        raise ValueError(f"Unknown stage names found: {unknown_stages}")

    return df


def read_blacklist(blacklist_path: Path | None) -> Set[str]:
    if blacklist_path is None:
        return set()

    if not blacklist_path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")

    blacklisted: Set[str] = set()

    with blacklist_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            blacklisted.add(line)

    return blacklisted


def apply_blacklist(
    df: pd.DataFrame,
    blacklist: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not blacklist:
        empty_blacklisted_df = df.iloc[0:0].copy()
        return df.copy(), empty_blacklisted_df

    clean_df = df[~df["video_id"].isin(blacklist)].copy()
    blacklisted_df = df[df["video_id"].isin(blacklist)].copy()

    return clean_df, blacklisted_df


def split_video_ids(
    video_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if len(video_ids) < 3:
        raise ValueError(
            f"Need at least 3 videos for train/val/test split, got {len(video_ids)}"
        )

    video_ids = list(video_ids)
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n_total = len(video_ids)

    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)

    # Гарантируем, что каждый split получит хотя бы одно видео.
    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))
    n_test = n_total - n_train - n_val

    if n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}"
        )

    train_videos = sorted(video_ids[:n_train])
    val_videos = sorted(video_ids[n_train : n_train + n_val])
    test_videos = sorted(video_ids[n_train + n_val :])

    return train_videos, val_videos, test_videos


def make_split_df(df: pd.DataFrame, video_ids: List[str]) -> pd.DataFrame:
    split_df = df[df["video_id"].isin(video_ids)].copy()
    split_df = split_df.sort_values(["video_id", "timestamp_s"]).reset_index(drop=True)
    return split_df


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["stage_name"].value_counts().to_dict()
    return {stage: int(counts.get(stage, 0)) for stage in STAGE_ORDER}


def video_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["video_id"].value_counts().sort_index().to_dict()
    return {str(video_id): int(count) for video_id, count in counts.items()}


def build_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_videos: List[str],
    val_videos: List[str],
    test_videos: List[str],
    seed: int,
    original_total_frames: int,
    original_total_videos: int,
    blacklist: Set[str],
    blacklisted_df: pd.DataFrame,
) -> Dict[str, object]:
    splits = {
        "train": {
            "videos": train_videos,
            "num_videos": len(train_videos),
            "num_frames": int(len(train_df)),
            "frames_by_stage": class_counts(train_df),
            "frames_by_video": video_counts(train_df),
        },
        "val": {
            "videos": val_videos,
            "num_videos": len(val_videos),
            "num_frames": int(len(val_df)),
            "frames_by_stage": class_counts(val_df),
            "frames_by_video": video_counts(val_df),
        },
        "test": {
            "videos": test_videos,
            "num_videos": len(test_videos),
            "num_frames": int(len(test_df)),
            "frames_by_stage": class_counts(test_df),
            "frames_by_video": video_counts(test_df),
        },
    }

    total_frames = len(train_df) + len(val_df) + len(test_df)
    total_videos = len(train_videos) + len(val_videos) + len(test_videos)

    return {
        "seed": seed,
        "stage_order": STAGE_ORDER,
        "original_total_frames": int(original_total_frames),
        "original_total_videos": int(original_total_videos),
        "total_frames": int(total_frames),
        "total_videos": int(total_videos),
        "blacklist": {
            "enabled": bool(blacklist),
            "requested_videos": sorted(blacklist),
            "matched_videos": sorted(blacklisted_df["video_id"].unique().tolist()),
            "num_matched_videos": int(blacklisted_df["video_id"].nunique()),
            "num_excluded_frames": int(len(blacklisted_df)),
            "excluded_frames_by_stage": class_counts(blacklisted_df),
            "excluded_frames_by_video": video_counts(blacklisted_df),
        },
        "splits": splits,
    }


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        output_dir / "train.csv",
        output_dir / "val.csv",
        output_dir / "test.csv",
        output_dir / "split_stats.json",
    ]

    existing = [path for path in output_files if path.exists()]

    if existing and not overwrite:
        existing_text = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Split output files already exist:\n"
            f"{existing_text}\n"
            "Use --overwrite to regenerate them."
        )


def save_split(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_stats(stats: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def print_split_summary(name: str, split_df: pd.DataFrame, videos: List[str]) -> None:
    logging.info(
        "%s: %d videos, %d frames",
        name,
        len(videos),
        len(split_df),
    )
    logging.info("  videos: %s", ", ".join(videos))

    counts = class_counts(split_df)
    for stage_name, count in counts.items():
        logging.info("  %-22s %d", stage_name + ":", count)


def main() -> None:
    setup_logging()
    args = parse_args()

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    original_df = read_labels(args.labels)
    original_video_ids = sorted(original_df["video_id"].unique().tolist())

    logging.info(
        "Original dataset: %d videos, %d frames",
        len(original_video_ids),
        len(original_df),
    )

    blacklist = read_blacklist(args.blacklist)

    if blacklist:
        logging.info("Blacklist enabled: %d requested videos", len(blacklist))
        logging.info("  requested: %s", ", ".join(sorted(blacklist)))

    df, blacklisted_df = apply_blacklist(original_df, blacklist)

    if blacklist:
        matched = sorted(blacklisted_df["video_id"].unique().tolist())
        unmatched = sorted(blacklist - set(matched))

        logging.info(
            "Blacklisted matched: %d videos, %d frames",
            len(matched),
            len(blacklisted_df),
        )

        if matched:
            logging.info("  matched: %s", ", ".join(matched))

        if unmatched:
            logging.warning(
                "Blacklist contains video_id values not found in labels: %s",
                ", ".join(unmatched),
            )

    if df.empty:
        raise ValueError("Dataset is empty after blacklist filtering.")

    video_ids = sorted(df["video_id"].unique().tolist())
    logging.info("Dataset for split: %d videos, %d frames", len(video_ids), len(df))

    train_videos, val_videos, test_videos = split_video_ids(
        video_ids=video_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_df = make_split_df(df, train_videos)
    val_df = make_split_df(df, val_videos)
    test_df = make_split_df(df, test_videos)

    # Проверка, что кадры не потерялись и не задублировались между split.
    total_after_split = len(train_df) + len(val_df) + len(test_df)
    if total_after_split != len(df):
        raise RuntimeError(
            f"Frame count mismatch after split: {total_after_split} != {len(df)}"
        )

    all_split_videos = set(train_videos) | set(val_videos) | set(test_videos)
    if all_split_videos != set(video_ids):
        raise RuntimeError("Video IDs mismatch after split.")

    if set(train_videos) & set(val_videos):
        raise RuntimeError("Train and val video sets overlap.")
    if set(train_videos) & set(test_videos):
        raise RuntimeError("Train and test video sets overlap.")
    if set(val_videos) & set(test_videos):
        raise RuntimeError("Val and test video sets overlap.")

    save_split(train_df, args.output_dir / "train.csv")
    save_split(val_df, args.output_dir / "val.csv")
    save_split(test_df, args.output_dir / "test.csv")

    stats = build_stats(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_videos=train_videos,
        val_videos=val_videos,
        test_videos=test_videos,
        seed=args.seed,
        original_total_frames=len(original_df),
        original_total_videos=len(original_video_ids),
        blacklist=blacklist,
        blacklisted_df=blacklisted_df,
    )
    save_stats(stats, args.output_dir / "split_stats.json")

    print_split_summary("train", train_df, train_videos)
    print_split_summary("val", val_df, val_videos)
    print_split_summary("test", test_df, test_videos)

    logging.info("Saved split files to: %s", args.output_dir)


if __name__ == "__main__":
    main()
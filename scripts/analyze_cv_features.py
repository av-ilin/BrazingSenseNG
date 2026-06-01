#!/usr/bin/env python3
"""
Analyze classical computer vision features for BrazingSense dataset.

Input:
    data/annotations/frame_labels.csv

Output:
    reports/cv_features/frame_features.csv
    reports/cv_features/summary_by_stage.csv
    reports/figures/cv_features/<video_id>_features.png
    reports/figures/cv_features/<video_id>_stage_timeline.png

Example:
    python scripts/analyze_cv_features.py \
        --labels data/annotations/frame_labels.csv \
        --output-csv reports/cv_features/frame_features.csv \
        --figures-dir reports/figures/cv_features \
        --overwrite
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd


STAGE_ORDER = [
    "inactive_preparation",
    "flux_activation",
    "active_brazing",
    "stabilization",
]

STAGE_TO_ID = {
    "inactive_preparation": 0,
    "flux_activation": 1,
    "active_brazing": 2,
    "stabilization": 3,
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OpenCV-based features from frame-level dataset."
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/frame_labels.csv"),
        help="Path to frame_labels.csv",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/cv_features/frame_features.csv"),
        help="Path to output frame features CSV",
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/cv_features/summary_by_stage.csv"),
        help="Path to output summary CSV grouped by stage",
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures/cv_features"),
        help="Directory for output feature plots",
    )

    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help=(
            "Optional ROI in format x,y,w,h. "
            "Example: --roi 350,250,600,350. "
            "If not provided, full frame is used."
        ),
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit for number of videos to plot. Features are still computed for all frames.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def parse_roi(roi_value: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if roi_value is None:
        return None

    parts = [p.strip() for p in roi_value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid ROI format: {roi_value}. Expected format: x,y,w,h"
        )

    x, y, w, h = map(int, parts)

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI size: w={w}, h={h}")

    return x, y, w, h


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Use --overwrite to regenerate it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def read_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)

    required_columns = {
        "video_id",
        "frame_path",
        "timestamp_s",
        "stage_id",
        "stage_name",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {labels_path}: {sorted(missing)}"
        )

    unknown_stages = sorted(set(df["stage_name"]) - set(STAGE_ORDER))
    if unknown_stages:
        raise ValueError(f"Unknown stages found: {unknown_stages}")

    return df.sort_values(["video_id", "timestamp_s"]).reset_index(drop=True)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def apply_roi(image: np.ndarray, roi: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return image

    x, y, w, h = roi
    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"ROI is outside image bounds. ROI={roi}, image_size={width}x{height}"
        )

    return image[y1:y2, x1:x2]


def calc_laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def calc_frame_diff_score(
    current_gray: np.ndarray,
    previous_gray: Optional[np.ndarray],
) -> float:
    if previous_gray is None:
        return 0.0

    if current_gray.shape != previous_gray.shape:
        previous_gray = cv2.resize(
            previous_gray,
            (current_gray.shape[1], current_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    diff = cv2.absdiff(current_gray, previous_gray)
    return float(diff.mean())


def extract_features(
    image_bgr: np.ndarray,
    previous_gray: Optional[np.ndarray] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> tuple[Dict[str, float], np.ndarray]:
    image_bgr = apply_roi(image_bgr, roi)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    r = image_rgb[:, :, 0]
    g = image_rgb[:, :, 1]
    b = image_rgb[:, :, 2]

    h = image_hsv[:, :, 0]
    s = image_hsv[:, :, 1]
    v = image_hsv[:, :, 2]

    l_channel = image_lab[:, :, 0]
    a_channel = image_lab[:, :, 1]
    b_lab_channel = image_lab[:, :, 2]

    # Белёсые области: высокая яркость и низкая насыщенность.
    white_mask = (v > 170) & (s < 70)
    white_area_ratio = float(np.mean(white_mask))

    # Блики/расплав: очень высокая яркость.
    # Это грубая эвристика, но для первого анализа полезна.
    specular_mask = v > 235
    specular_highlight_ratio = float(np.mean(specular_mask))

    # Тёмные области.
    dark_mask = v < 50
    dark_area_ratio = float(np.mean(dark_mask))

    # Красно-оранжевые/тёплые области.
    # В OpenCV Hue: red around 0/179, orange around 5-25.
    warm_mask = ((h <= 25) | (h >= 170)) & (s > 70) & (v > 80)
    warm_area_ratio = float(np.mean(warm_mask))

    # Границы/текстура.
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(np.mean(edges > 0))

    features = {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "value_mean": float(v.mean()),
        "value_std": float(v.std()),
        "saturation_mean": float(s.mean()),
        "saturation_std": float(s.std()),
        "hue_mean": float(h.mean()),
        "lab_l_mean": float(l_channel.mean()),
        "lab_a_mean": float(a_channel.mean()),
        "lab_b_mean": float(b_lab_channel.mean()),
        "red_mean": float(r.mean()),
        "green_mean": float(g.mean()),
        "blue_mean": float(b.mean()),
        "red_green_diff": float(r.mean() - g.mean()),
        "red_blue_diff": float(r.mean() - b.mean()),
        "white_area_ratio": white_area_ratio,
        "specular_highlight_ratio": specular_highlight_ratio,
        "dark_area_ratio": dark_area_ratio,
        "warm_area_ratio": warm_area_ratio,
        "edge_density": edge_density,
        "laplacian_var": calc_laplacian_variance(gray),
        "frame_diff_score": calc_frame_diff_score(gray, previous_gray),
    }

    return features, gray


def compute_features(df: pd.DataFrame, roi: Optional[tuple[int, int, int, int]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for video_id, video_df in df.groupby("video_id"):
        logging.info("Computing features for %s (%d frames)", video_id, len(video_df))

        previous_gray: Optional[np.ndarray] = None

        for _, row in video_df.sort_values("timestamp_s").iterrows():
            frame_path = resolve_path(str(row["frame_path"]))
            image_bgr = read_image_bgr(frame_path)

            features, current_gray = extract_features(
                image_bgr=image_bgr,
                previous_gray=previous_gray,
                roi=roi,
            )

            output_row: Dict[str, object] = {
                "video_id": row["video_id"],
                "frame_path": row["frame_path"],
                "timestamp_s": row["timestamp_s"],
                "stage_id": int(row["stage_id"]),
                "stage_name": row["stage_name"],
            }
            output_row.update(features)

            rows.append(output_row)
            previous_gray = current_gray

    return pd.DataFrame(rows)


def save_summary_by_stage(features_df: pd.DataFrame, summary_csv: Path) -> None:
    feature_columns = [
        col
        for col in features_df.columns
        if col
        not in {
            "video_id",
            "frame_path",
            "timestamp_s",
            "stage_id",
            "stage_name",
        }
    ]

    summary = (
        features_df.groupby("stage_name")[feature_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.to_csv(summary_csv, index=False)


def plot_video_features(
    video_df: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    video_id = str(video_df["video_id"].iloc[0])
    video_df = video_df.sort_values("timestamp_s")

    feature_groups = [
        (
            "Brightness / saturation",
            ["brightness_mean", "saturation_mean", "value_mean"],
        ),
        (
            "Area ratios",
            ["white_area_ratio", "specular_highlight_ratio", "warm_area_ratio"],
        ),
        (
            "Texture / motion",
            ["edge_density", "laplacian_var", "frame_diff_score"],
        ),
    ]

    fig, axes = plt.subplots(
        nrows=len(feature_groups) + 1,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
    )

    x = video_df["timestamp_s"].values

    for ax, (title, columns) in zip(axes[:-1], feature_groups):
        for col in columns:
            if col not in video_df.columns:
                continue

            y = video_df[col].values.astype(float)

            # Для признаков с сильно разным масштабом делаем min-max нормировку
            # только для визуального сравнения на одном графике.
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                y_plot = (y - y_min) / (y_max - y_min)
            else:
                y_plot = y

            ax.plot(x, y_plot, label=col)

        ax.set_title(title)
        ax.set_ylabel("Normalized value")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    stage_ax = axes[-1]
    stage_ax.scatter(
        x,
        video_df["stage_id"].values,
        s=18,
    )

    stage_ax.set_yticks([0, 1, 2, 3])
    stage_ax.set_yticklabels(STAGE_ORDER)
    stage_ax.set_xlabel("Time, s")
    stage_ax.set_ylabel("Stage")
    stage_ax.set_title("Stage timeline")
    stage_ax.grid(alpha=0.3)

    fig.suptitle(f"OpenCV features over time: {video_id}", fontsize=14)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stage_timeline(
    video_df: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    video_id = str(video_df["video_id"].iloc[0])
    video_df = video_df.sort_values("timestamp_s")

    fig, ax = plt.subplots(figsize=(14, 3))

    ax.scatter(
        video_df["timestamp_s"],
        video_df["stage_id"],
        s=18,
    )

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(STAGE_ORDER)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Stage")
    ax.set_title(f"Stage timeline: {video_id}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_all_videos(
    features_df: pd.DataFrame,
    figures_dir: Path,
    max_videos: Optional[int],
) -> None:
    video_ids = sorted(features_df["video_id"].unique())

    if max_videos is not None:
        video_ids = video_ids[:max_videos]

    for video_id in video_ids:
        video_df = features_df[features_df["video_id"] == video_id]

        logging.info("Saving plots for %s", video_id)

        plot_video_features(
            video_df=video_df,
            output_path=figures_dir / f"{video_id}_features.png",
        )

        plot_stage_timeline(
            video_df=video_df,
            output_path=figures_dir / f"{video_id}_stage_timeline.png",
        )


def main() -> None:
    setup_logging()
    args = parse_args()

    roi = parse_roi(args.roi)

    ensure_can_write(args.output_csv, overwrite=args.overwrite)
    ensure_can_write(args.summary_csv, overwrite=args.overwrite)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading labels: %s", args.labels)
    df = read_labels(args.labels)

    logging.info("Frames: %d", len(df))
    logging.info("Videos: %d", df["video_id"].nunique())
    logging.info("ROI: %s", roi if roi is not None else "full frame")

    features_df = compute_features(df, roi=roi)

    features_df.to_csv(args.output_csv, index=False)
    logging.info("Saved frame features to: %s", args.output_csv)

    save_summary_by_stage(features_df, args.summary_csv)
    logging.info("Saved summary by stage to: %s", args.summary_csv)

    plot_all_videos(
        features_df=features_df,
        figures_dir=args.figures_dir,
        max_videos=args.max_videos,
    )

    logging.info("Saved figures to: %s", args.figures_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
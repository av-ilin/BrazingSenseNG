from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".MP4",
    ".MOV",
    ".AVI",
    ".MKV",
    ".M4V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time-grid images for all videos in a directory."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with source videos",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/grid"),
        help="Directory for output PNG grids",
    )
    parser.add_argument(
        "--make-grid-script",
        type=Path,
        default=Path("scripts/make_time_grid.py"),
        help="Path to single-video grid generator script",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start second for every video",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End second for every video",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.2,
        help="Step inside each second",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for frames",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop region as x1,y1,x2,y2",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos for which the output PNG already exists",
    )
    return parser.parse_args()


def find_videos(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    videos = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def make_output_path(video_path: Path, raw_dir: Path, out_dir: Path) -> Path:
    relative_path = video_path.relative_to(raw_dir)
    relative_no_suffix = relative_path.with_suffix("")
    return out_dir / relative_no_suffix.with_suffix(".png")


def build_command(
    python_executable: str,
    make_grid_script: Path,
    video_path: Path,
    output_path: Path,
    start: float,
    end: float | None,
    step: float,
    scale: float,
    crop: str | None,
) -> list[str]:
    cmd = [
        python_executable,
        str(make_grid_script),
        "--video",
        str(video_path),
        "--out",
        str(output_path),
        "--start",
        str(start),
        "--step",
        str(step),
        "--scale",
        str(scale),
    ]

    if end is not None:
        cmd.extend(["--end", str(end)])

    if crop is not None:
        cmd.extend(["--crop", crop])

    return cmd


def main() -> None:
    args = parse_args()

    videos = find_videos(args.raw_dir)

    if not videos:
        print(f"No videos found in: {args.raw_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found videos: {len(videos)}")
    print(f"Output dir: {args.out_dir}")
    print()

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, video_path in enumerate(videos, start=1):
        output_path = make_output_path(video_path, args.raw_dir, args.out_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(videos)}] {video_path}")

        if args.skip_existing and output_path.exists():
            print(f"  Skipped: output already exists -> {output_path}")
            skipped_count += 1
            print()
            continue

        cmd = build_command(
            python_executable=sys.executable,
            make_grid_script=args.make_grid_script,
            video_path=video_path,
            output_path=output_path,
            start=args.start,
            end=args.end,
            step=args.step,
            scale=args.scale,
            crop=args.crop,
        )

        try:
            subprocess.run(cmd, check=True)
            print(f"  Saved: {output_path}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {video_path}")
            print(f"  Error: {e}")
            failed_count += 1

        print()

    print("Done")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
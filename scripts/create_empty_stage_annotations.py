# python3 scripts/create_empty_stage_annotations.py

from pathlib import Path
import csv
import yaml


RAW_DIR = Path("data/raw")
CONFIG_PATH = Path("configs/stages.yaml")
OUT_CSV = Path("data/annotations/stage_intervals.csv")

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


def find_videos(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    videos = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix in VIDEO_EXTENSIONS
    ]

    return sorted(videos)


def make_video_id(video_path: Path) -> str:
    return video_path.stem


def load_stages(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Stages config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stages = config.get("stages")

    if not stages:
        raise ValueError(f"No 'stages' section found in {config_path}")

    stage_items = []

    for stage_id, stage_data in stages.items():
        if "name" not in stage_data:
            raise ValueError(f"Stage {stage_id} has no 'name' field")

        stage_items.append((int(stage_id), stage_data["name"]))

    stage_items.sort(key=lambda item: item[0])

    return [stage_name for _, stage_name in stage_items]


def create_empty_annotations() -> None:
    videos = find_videos(RAW_DIR)
    stages = load_stages(CONFIG_PATH)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "video_path",
                "stage_name",
                "start_s",
                "end_s",
            ],
        )
        writer.writeheader()

        for video_path in videos:
            video_id = make_video_id(video_path)
            rel_video_path = video_path.as_posix()

            for stage_name in stages:
                writer.writerow(
                    {
                        "video_id": video_id,
                        "video_path": rel_video_path,
                        "stage_name": stage_name,
                        "start_s": "",
                        "end_s": "",
                    }
                )

    print(f"Found videos: {len(videos)}")
    print(f"Loaded stages: {len(stages)}")
    print(f"Created annotation file: {OUT_CSV}")


if __name__ == "__main__":
    create_empty_annotations()
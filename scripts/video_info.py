# python3 scripts/video_info.py data/raw/MVI_6265.MOV

from pathlib import Path
import argparse
import cv2

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps else 0

    print(f"video: {args.video_path}")
    print(f"fps: {fps:.3f}")
    print(f"frames: {frame_count}")
    print(f"duration_s: {duration_s:.2f}")

    cap.release()


if __name__ == "__main__":
    main()
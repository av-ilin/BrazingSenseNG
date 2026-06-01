# python scripts/make_time_grid.py \
#   --video data/raw/MVI_6265.MOV \
#   --out reports/figures/grid/MVI_6265.png \
#   --start 25 \
#   --end 45 \
#   --crop 360,260,900,610 \
#   --scale 0.5

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a large time grid image from a video for manual stage annotation."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--out", type=Path, required=True, help="Path to output PNG image")
    parser.add_argument("--start", type=float, default=0.0, help="Start second (inclusive)")
    parser.add_argument("--end", type=float, default=None, help="End second (exclusive)")
    parser.add_argument(
        "--step",
        type=float,
        default=0.2,
        help="Step inside each second. Example: 0.2",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for frames. 1.0 = original size",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop region as x1,y1,x2,y2. Example: 300,200,900,650",
    )
    return parser.parse_args()


def load_video_info(video_path: Path) -> tuple[cv2.VideoCapture, float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps else 0.0
    return cap, fps, frame_count, duration_s


def parse_crop(crop_str: str | None) -> tuple[int, int, int, int] | None:
    if crop_str is None:
        return None

    parts = [p.strip() for p in crop_str.split(",")]
    if len(parts) != 4:
        raise ValueError("Crop must be in format x1,y1,x2,y2")

    x1, y1, x2, y2 = map(int, parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates")

    return x1, y1, x2, y2


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def read_frame_at_time(
    cap: cv2.VideoCapture,
    timestamp_s: float,
    crop: tuple[int, int, int, int] | None,
    scale: float,
) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame at {timestamp_s:.3f}s")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if crop is not None:
        x1, y1, x2, y2 = crop
        frame_rgb = frame_rgb[y1:y2, x1:x2]

    if scale != 1.0:
        h, w = frame_rgb.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        frame_rgb = cv2.resize(
            frame_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )

    return frame_rgb


def build_offsets(step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be > 0")

    offsets = []
    value = 0.0
    while value < 1.0 - 1e-9:
        offsets.append(round(value, 6))
        value += step
    return offsets


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(0, 0, 0),
) -> None:
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2
    draw.text((tx, ty), text, font=font, fill=fill)


def draw_text_with_background(
    image: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_fill=(255, 255, 255),
    bg_fill=(0, 0, 0),
    padding=4,
) -> None:
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = position
    draw.rectangle(
        [x, y, x + tw + 2 * padding, y + th + 2 * padding],
        fill=bg_fill,
    )
    draw.text((x + padding, y + padding), text, font=font, fill=text_fill)


def main() -> None:
    args = parse_args()

    cap, fps, frame_count, duration_s = load_video_info(args.video)
    crop = parse_crop(args.crop)
    offsets = build_offsets(args.step)

    start_sec = int(math.floor(args.start))
    end_limit = args.end if args.end is not None else duration_s
    end_sec = int(math.ceil(end_limit))

    if start_sec >= end_sec:
        raise ValueError("Start must be < end")

    sample_ts = min(start_sec + offsets[0], max(0.0, duration_s - 1.0 / max(fps, 1.0)))
    sample_frame = read_frame_at_time(cap, sample_ts, crop, args.scale)
    cell_h, cell_w = sample_frame.shape[:2]

    # Layout parameters
    top_margin = 70
    left_margin = 140
    pad = 6

    n_rows = end_sec - start_sec
    n_cols = len(offsets)

    canvas_w = left_margin + pad + n_cols * (cell_w + pad)
    canvas_h = top_margin + pad + n_rows * (cell_h + pad)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_title = get_font(24)
    font_header = get_font(28)
    font_side = get_font(28)
    font_cell = get_font(18)

    # Background for header areas
    draw.rectangle([0, 0, canvas_w, top_margin], fill=(235, 235, 235))
    draw.rectangle([0, 0, left_margin, canvas_h], fill=(235, 235, 235))

    # Title
    title = f"{args.video.name} | fps={fps:.3f} | duration={duration_s:.2f}s"
    draw.text((10, 10), title, font=font_title, fill=(0, 0, 0))

    # Column headers
    for col_idx, offset in enumerate(offsets):
        x1 = left_margin + pad + col_idx * (cell_w + pad)
        x2 = x1 + cell_w
        draw_text_centered(
            draw,
            (x1, 28, x2, top_margin),
            f"{offset:.1f}",
            font_header,
            fill=(0, 0, 0),
        )

    # Fill frames
    for row_idx, sec in enumerate(range(start_sec, end_sec)):
        y1 = top_margin + pad + row_idx * (cell_h + pad)
        y2 = y1 + cell_h

        # Row label
        draw_text_centered(
            draw,
            (0, y1, left_margin - 10, y2),
            f"{sec}s",
            font_side,
            fill=(0, 0, 0),
        )

        for col_idx, offset in enumerate(offsets):
            ts = sec + offset
            x1 = left_margin + pad + col_idx * (cell_w + pad)
            x2 = x1 + cell_w

            if ts >= duration_s:
                draw.rectangle([x1, y1, x2, y2], outline=(180, 180, 180), width=2)
                draw_text_centered(
                    draw,
                    (x1, y1, x2, y2),
                    "N/A",
                    font_cell,
                    fill=(100, 100, 100),
                )
                continue

            frame = read_frame_at_time(cap, ts, crop, args.scale)
            frame_img = Image.fromarray(frame)
            canvas.paste(frame_img, (x1, y1))

            # Border around cell
            draw.rectangle([x1, y1, x2, y2], outline=(120, 120, 120), width=2)

            # Timestamp inside cell
            draw_text_with_background(
                canvas,
                position=(x1 + 6, y1 + 6),
                text=f"{ts:.1f}s",
                font=font_cell,
                text_fill=(255, 255, 255),
                bg_fill=(0, 0, 0),
                padding=3,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, format="PNG", compress_level=0)

    cap.release()

    print(f"Saved: {args.out}")
    print(f"Rows: {n_rows}")
    print(f"Cols: {n_cols}")
    print(f"Cell size: {cell_w}x{cell_h}")
    print(f"Canvas size: {canvas_w}x{canvas_h}")


if __name__ == "__main__":
    main()
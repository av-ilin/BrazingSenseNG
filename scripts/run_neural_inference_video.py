#!/usr/bin/env python3
"""
Run final BrazingSense inference on a video and save demo-video with overlay.

Final pipeline:
    frame
    -> ROI crop
    -> neural stage classifier
    -> raw stage
    -> state machine / majority / raw postprocess
    -> stable stage
    -> active_brazing trigger from P(active_brazing)
    -> optional CV fallback trigger
    -> demo-video overlay

Example without CV trigger:
    PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
        --video data/raw/MVI_6266.MOV \
        --checkpoint models/checkpoints/final_neural_stage_classification_10fps/resnet18_10fps_balanced_best_10fps.pt \
        --output reports/demo/MVI_6266_final_system_demo.mp4 \
        --model resnet18 \
        --roi 470,280,430,290 \
        --image-size 224 \
        --postprocess state_machine \
        --trigger-threshold 0.2 \
        --trigger-confirm-frames 7 \
        --device auto

Example with optional CV trigger:
    PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
        --video data/raw/MVI_6266.MOV \
        --checkpoint models/checkpoints/final_neural_stage_classification_10fps/resnet18_10fps_balanced_best_10fps.pt \
        --output reports/demo/MVI_6266_final_system_demo_cv.mp4 \
        --model resnet18 \
        --roi 470,280,430,290 \
        --image-size 224 \
        --postprocess state_machine \
        --trigger-threshold 0.2 \
        --trigger-confirm-frames 7 \
        --enable-cv-trigger \
        --cv-threshold 0.5 \
        --cv-confirm-frames 5 \
        --device auto
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import time
from collections import Counter, deque
from typing import Deque, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

from brazing_sense.control.state_machine import BrazingStageStateMachine


STAGE_ORDER = [
    "inactive_preparation",
    "flux_activation",
    "active_brazing",
    "stabilization",
]

ID_TO_STAGE = {
    0: "inactive_preparation",
    1: "flux_activation",
    2: "active_brazing",
    3: "stabilization",
}

ACTIVE_BRAZING_STAGE_ID = 2

STAGE_COLORS_BGR = {
    "inactive_preparation": (180, 180, 180),
    "flux_activation": (0, 255, 255),
    "active_brazing": (0, 120, 255),
    "stabilization": (0, 255, 0),
}


class ConsecutiveTrigger:
    """
    Trigger that switches ON after score >= threshold for N consecutive frames.
    Once triggered, stays ON.
    """

    def __init__(self, threshold: float, confirm_frames: int):
        if confirm_frames <= 0:
            raise ValueError("confirm_frames must be positive")

        self.threshold = float(threshold)
        self.confirm_frames = int(confirm_frames)
        self.counter = 0
        self.triggered = False

    def reset(self) -> None:
        self.counter = 0
        self.triggered = False

    def update(self, score: float) -> bool:
        if self.triggered:
            return True

        if score >= self.threshold:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.confirm_frames:
            self.triggered = True

        return self.triggered
    
class AvgMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


def make_activation(name: str):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "hardswish":
        return nn.Hardswish(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


def make_norm(norm_name: str, num_features: int):
    if norm_name == "none":
        return None
    if norm_name == "batchnorm":
        return nn.BatchNorm1d(num_features)
    raise ValueError(f"Unknown head_norm: {norm_name}")


class MobileNetV3SmallEvo(nn.Module):
    def __init__(
        self,
        num_classes: int,
        head_type: str,
        head_hidden: int,
        head_bottleneck_ratio: float,
        dropout: float,
        activation: str,
        pooling: str,
        head_norm: str,
        freeze_mode: str,
        pretrained: bool = False,
        unfreeze_last_n_blocks: int = 3,
    ):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)

        self.features = base.features

        backbone_feature_dim = base.classifier[0].in_features

        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
            pooled_dim = backbone_feature_dim
        elif pooling == "avg_max":
            self.pool = AvgMaxPool2d()
            pooled_dim = backbone_feature_dim * 2
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.flatten = nn.Flatten(1)

        self.classifier = self._make_head(
            in_features=pooled_dim,
            num_classes=num_classes,
            head_type=head_type,
            head_hidden=int(head_hidden),
            head_bottleneck_ratio=float(head_bottleneck_ratio),
            dropout=float(dropout),
            activation=activation,
            head_norm=head_norm,
        )

        self.apply_freeze_mode(
            freeze_mode=freeze_mode,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

        self.arch_config = {
            "head_type": head_type,
            "head_hidden": int(head_hidden),
            "head_bottleneck_ratio": float(head_bottleneck_ratio),
            "dropout": float(dropout),
            "activation": activation,
            "pooling": pooling,
            "head_norm": head_norm,
            "freeze_mode": freeze_mode,
            "pretrained": pretrained,
            "unfreeze_last_n_blocks": unfreeze_last_n_blocks,
        }

    def _make_head(
        self,
        in_features: int,
        num_classes: int,
        head_type: str,
        head_hidden: int,
        head_bottleneck_ratio: float,
        dropout: float,
        activation: str,
        head_norm: str,
    ) -> nn.Sequential:
        act = make_activation(activation)

        def maybe_norm(n: int):
            norm = make_norm(head_norm, n)
            return [] if norm is None else [norm]

        if head_type == "linear":
            layers = [
                *maybe_norm(in_features),
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            ]

        elif head_type == "mlp_1":
            layers = [
                nn.Linear(in_features, head_hidden),
                *maybe_norm(head_hidden),
                act,
                nn.Dropout(p=dropout),
                nn.Linear(head_hidden, num_classes),
            ]

        elif head_type == "mlp_2":
            bottleneck_hidden = max(
                32,
                int(round(head_hidden * float(head_bottleneck_ratio))),
            )

            layers = [
                nn.Linear(in_features, head_hidden),
                *maybe_norm(head_hidden),
                act,
                nn.Dropout(p=dropout),
                nn.Linear(head_hidden, bottleneck_hidden),
                *maybe_norm(bottleneck_hidden),
                make_activation(activation),
                nn.Dropout(p=dropout),
                nn.Linear(bottleneck_hidden, num_classes),
            ]

        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        return nn.Sequential(*layers)

    def apply_freeze_mode(self, freeze_mode: str, unfreeze_last_n_blocks: int = 3):
        for p in self.classifier.parameters():
            p.requires_grad = True

        if freeze_mode == "full":
            for p in self.features.parameters():
                p.requires_grad = True

        elif freeze_mode == "last_blocks":
            for p in self.features.parameters():
                p.requires_grad = False

            for block in self.features[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def normalize_arch_candidate(candidate: dict) -> dict:
    required_keys = [
        "head_type",
        "head_hidden",
        "head_bottleneck_ratio",
        "dropout",
        "activation",
        "pooling",
        "head_norm",
        "freeze_mode",
    ]

    missing = [key for key in required_keys if key not in candidate]
    if missing:
        raise ValueError(f"Missing architecture candidate keys: {missing}")

    return {
        "head_type": str(candidate["head_type"]),
        "head_hidden": int(candidate["head_hidden"]),
        "head_bottleneck_ratio": float(candidate["head_bottleneck_ratio"]),
        "dropout": float(candidate["dropout"]),
        "activation": str(candidate["activation"]),
        "pooling": str(candidate["pooling"]),
        "head_norm": str(candidate["head_norm"]),
        "freeze_mode": str(candidate["freeze_mode"]),
    }


def extract_arch_candidate_from_checkpoint(checkpoint: dict) -> dict:
    # Варианты на случай разных форматов сохранения checkpoint
    for key in ["candidate", "best_candidate", "arch_config", "model_config", "config"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            try:
                return normalize_arch_candidate(value)
            except ValueError:
                pass

    # Частый вариант: config лежит внутри checkpoint["metadata"] или checkpoint["training_config"]
    for outer_key in ["metadata", "training_config", "fixed_training_config"]:
        outer = checkpoint.get(outer_key)
        if isinstance(outer, dict):
            for key in ["candidate", "best_candidate", "arch_config", "model_config", "config"]:
                value = outer.get(key)
                if isinstance(value, dict):
                    try:
                        return normalize_arch_candidate(value)
                    except ValueError:
                        pass

    # Если параметры лежат прямо на верхнем уровне checkpoint
    direct_candidate = {}
    for key in [
        "head_type",
        "head_hidden",
        "head_bottleneck_ratio",
        "dropout",
        "activation",
        "pooling",
        "head_norm",
        "freeze_mode",
    ]:
        if key in checkpoint:
            direct_candidate[key] = checkpoint[key]

    if direct_candidate:
        return normalize_arch_candidate(direct_candidate)

    raise ValueError(
        "Could not find MobileNetV3 architecture evolution config in checkpoint. "
        "Expected one of: candidate, best_candidate, arch_config, model_config, config, "
        "or direct head_type/head_hidden/... keys."
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final neural inference on brazing video and save overlay demo."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained .pt checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output demo video, usually .mp4.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small", "mobilenet_v3_small_architecture_evo"],
        help="Model architecture.",
    )

    parser.add_argument(
        "--roi",
        type=str,
        default="470,280,430,290",
        help="ROI in x,y,w,h format.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size for neural model.",
    )

    parser.add_argument(
        "--postprocess",
        type=str,
        default="state_machine",
        choices=["raw", "majority", "state_machine"],
        help="Postprocessing mode for stable stage output.",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Majority-vote smoothing window. Used only with --postprocess majority.",
    )

    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.2,
        help="Threshold for neural active_brazing trigger based on P(active_brazing).",
    )

    parser.add_argument(
        "--trigger-confirm-frames",
        type=int,
        default=7,
        help="Number of consecutive frames to confirm neural active_brazing trigger.",
    )

    parser.add_argument(
        "--enable-cv-trigger",
        action="store_true",
        help="Enable optional OpenCV fallback trigger.",
    )

    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.5,
        help="Threshold for optional OpenCV active_brazing trigger.",
    )

    parser.add_argument(
        "--cv-confirm-frames",
        type=int,
        default=5,
        help="Number of consecutive frames to confirm CV trigger.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )

    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="OpenCV VideoWriter codec, e.g. mp4v or XVID.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for quick debug run.",
    )

    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Optional output FPS. If omitted, input video FPS is used.",
    )

    parser.add_argument(
        "--draw-probabilities",
        action="store_true",
        help="Draw probabilities for all stages on the frame.",
    )

    return parser.parse_args()


def parse_roi(roi_value: str) -> Dict[str, int]:
    parts = [p.strip() for p in roi_value.split(",")]

    if len(parts) != 4:
        raise ValueError(f"Invalid ROI format: {roi_value}. Expected x,y,w,h")

    x, y, w, h = map(int, parts)

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI size: w={w}, h={h}")

    return {"x": x, "y": y, "w": w, "h": h}


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    return torch.device(device_arg)

def create_model(
    model_name: str,
    num_classes: int = 4,
    candidate: dict | None = None,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small_architecture_evo":
        if candidate is None:
            raise ValueError(
                "candidate config is required for mobilenet_v3_small_architecture_evo"
            )

        candidate = normalize_arch_candidate(candidate)

        return MobileNetV3SmallEvo(
            num_classes=num_classes,
            head_type=candidate["head_type"],
            head_hidden=candidate["head_hidden"],
            head_bottleneck_ratio=candidate["head_bottleneck_ratio"],
            dropout=candidate["dropout"],
            activation=candidate["activation"],
            pooling=candidate["pooling"],
            head_norm=candidate["head_norm"],
            freeze_mode=candidate["freeze_mode"],
            pretrained=False,
        )

    raise ValueError(f"Unknown model_name: {model_name}")

def load_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name", model_name)

    # Если checkpoint сам говорит, что это evo-архитектура, лучше использовать её.
    if (
        checkpoint_model_name == "mobilenet_v3_small_architecture_evo"
        and model_name == "mobilenet_v3_small"
    ):
        print(
            "WARNING: checkpoint is mobilenet_v3_small_architecture_evo, "
            "but --model=mobilenet_v3_small was provided. "
            "Switching to mobilenet_v3_small_architecture_evo."
        )
        model_name = "mobilenet_v3_small_architecture_evo"

    elif checkpoint_model_name != model_name:
        print(
            f"WARNING: checkpoint model_name={checkpoint_model_name}, "
            f"but argument --model={model_name}. Using --model={model_name}."
        )

    candidate = None

    if model_name == "mobilenet_v3_small_architecture_evo":
        candidate = extract_arch_candidate_from_checkpoint(checkpoint)
        print(f"Loaded MobileNetV3 architecture-evo config: {candidate}")

    model = create_model(
        model_name=model_name,
        num_classes=4,
        candidate=candidate,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model

def crop_roi(frame_bgr: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    height, width = frame_bgr.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI {roi} for frame size {width}x{height}")

    return frame_bgr[y1:y2, x1:x2]


def make_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_frame(
    model: nn.Module,
    frame_bgr: np.ndarray,
    roi: Dict[str, int],
    transform,
    device: torch.device,
) -> Tuple[int, str, float, np.ndarray, float]:
    start_time = time.perf_counter()

    roi_bgr = crop_roi(frame_bgr, roi)
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    tensor = transform(roi_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_ms = (time.perf_counter() - start_time) * 1000.0

    pred_id = int(torch.argmax(probs).detach().cpu().item())
    pred_stage = ID_TO_STAGE[pred_id]
    confidence = float(probs[pred_id].detach().cpu().item())
    probs_np = probs.detach().cpu().numpy()

    return pred_id, pred_stage, confidence, probs_np, inference_ms


def majority_vote_smooth(
    history: Deque[int],
    current_pred_id: int,
    window_size: int,
) -> int:
    if window_size <= 1:
        return current_pred_id

    history.append(current_pred_id)
    most_common_id = Counter(history).most_common(1)[0][0]

    return int(most_common_id)


def compute_cv_trigger_score(
    frame_bgr: np.ndarray,
    roi: Dict[str, int],
    previous_roi_gray: np.ndarray | None,
) -> Tuple[float, np.ndarray]:
    """
    Online approximation of score_cv_v1_motion_texture.

    Notebook version used per-video minmax normalization.
    In online/demo mode, fixed normalization constants are used instead.
    """
    roi_bgr = crop_roi(frame_bgr, roi)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    _, _, v = cv2.split(hsv)

    if previous_roi_gray is None:
        frame_diff_score = 0.0
    else:
        if previous_roi_gray.shape != gray.shape:
            previous_roi_gray = cv2.resize(
                previous_roi_gray,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        frame_diff_score = float(cv2.absdiff(gray, previous_roi_gray).mean())

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    specular_highlight_ratio = float(np.mean(v > 235))

    frame_diff_norm = float(np.clip(frame_diff_score / 25.0, 0.0, 1.0))
    laplacian_norm = float(np.clip(laplacian_var / 120.0, 0.0, 1.0))
    edge_norm = float(np.clip(edge_density / 0.10, 0.0, 1.0))
    specular_norm = float(np.clip(specular_highlight_ratio / 0.35, 0.0, 1.0))

    score = (
        frame_diff_norm
        + laplacian_norm
        + edge_norm
        + 0.5 * specular_norm
    ) / 3.5

    return float(score), gray


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.65,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 7),
        (x + text_w + 5, y + baseline + 5),
        bg_color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    roi: Dict[str, int],
    raw_stage: str,
    stable_stage: str,
    confidence: float,
    p_active_brazing: float,
    neural_trigger_on: bool,
    cv_trigger_on: bool,
    hold_signal_on: bool,
    inference_ms: float,
    timestamp_s: float,
    probs: np.ndarray,
    draw_probabilities: bool,
    cv_score: float | None = None,
) -> np.ndarray:
    output = frame_bgr.copy()

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        3,
    )

    line_x = 25
    line_y = 40
    line_step = 32

    draw_text_with_background(
        output,
        f"Raw Stage: {raw_stage}",
        (line_x, line_y),
        text_color=(255, 255, 255),
        bg_color=(30, 30, 30),
    )

    draw_text_with_background(
        output,
        f"Stable Stage: {stable_stage}",
        (line_x, line_y + line_step),
        text_color=STAGE_COLORS_BGR.get(stable_stage, (255, 255, 255)),
        bg_color=(30, 30, 30),
    )

    draw_text_with_background(
        output,
        f"Confidence: {confidence:.2f}",
        (line_x, line_y + line_step * 2),
        text_color=(255, 255, 255),
        bg_color=(30, 30, 30),
    )

    draw_text_with_background(
        output,
        f"P(active_brazing): {p_active_brazing:.2f}",
        (line_x, line_y + line_step * 3),
        text_color=(255, 255, 255),
        bg_color=(30, 30, 30),
    )

    neural_color = (0, 255, 0) if neural_trigger_on else (180, 180, 180)
    draw_text_with_background(
        output,
        f"Neural Trigger: {'ON' if neural_trigger_on else 'OFF'}",
        (line_x, line_y + line_step * 4),
        text_color=neural_color,
        bg_color=(30, 30, 30),
    )

    next_line = 5

    if cv_score is not None:
        cv_color = (0, 255, 0) if cv_trigger_on else (180, 180, 180)

        draw_text_with_background(
            output,
            f"CV Score: {cv_score:.2f}",
            (line_x, line_y + line_step * next_line),
            text_color=(255, 255, 255),
            bg_color=(30, 30, 30),
        )
        next_line += 1

        draw_text_with_background(
            output,
            f"CV Trigger: {'ON' if cv_trigger_on else 'OFF'}",
            (line_x, line_y + line_step * next_line),
            text_color=cv_color,
            bg_color=(30, 30, 30),
        )
        next_line += 1

    hold_color = (0, 255, 255) if hold_signal_on else (180, 180, 180)
    hold_bg = (0, 70, 70) if hold_signal_on else (30, 30, 30)

    draw_text_with_background(
        output,
        f"HOLD TEMPERATURE: {'ON' if hold_signal_on else 'OFF'}",
        (line_x, line_y + line_step * next_line),
        text_color=hold_color,
        bg_color=hold_bg,
    )
    next_line += 1

    draw_text_with_background(
        output,
        f"Inference: {inference_ms:.1f} ms",
        (line_x, line_y + line_step * next_line),
        text_color=(255, 255, 255),
        bg_color=(30, 30, 30),
    )
    next_line += 1

    draw_text_with_background(
        output,
        f"Time: {timestamp_s:.2f} s",
        (line_x, line_y + line_step * next_line),
        text_color=(255, 255, 255),
        bg_color=(30, 30, 30),
    )

    if draw_probabilities:
        prob_x = 25
        prob_y = output.shape[0] - 125
        bar_w = 220
        bar_h = 18
        gap = 8

        for i, stage_name in enumerate(STAGE_ORDER):
            p = float(probs[i])
            yy = prob_y + i * (bar_h + gap)

            cv2.rectangle(
                output,
                (prob_x, yy),
                (prob_x + bar_w, yy + bar_h),
                (60, 60, 60),
                -1,
            )

            cv2.rectangle(
                output,
                (prob_x, yy),
                (prob_x + int(bar_w * p), yy + bar_h),
                STAGE_COLORS_BGR.get(stage_name, (255, 255, 255)),
                -1,
            )

            cv2.putText(
                output,
                f"{stage_name}: {p:.2f}",
                (prob_x + bar_w + 10, yy + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return output


def open_video(video_path: Path) -> cv2.VideoCapture:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    return cap


def create_writer(
    output_path: Path,
    codec: str,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {output_path}. "
            f"Try another codec, e.g. --codec XVID."
        )

    return writer


def main() -> None:
    args = parse_args()

    roi = parse_roi(args.roi)
    device = select_device(args.device)

    print(f"Input video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"ROI: {roi}")
    print(f"Image size: {args.image_size}")
    print(f"Postprocess: {args.postprocess}")
    print(f"Smoothing window: {args.smoothing_window}")
    print(f"Neural trigger: threshold={args.trigger_threshold}, confirm={args.trigger_confirm_frames}")
    print(f"CV trigger enabled: {args.enable_cv_trigger}")
    if args.enable_cv_trigger:
        print(f"CV trigger: threshold={args.cv_threshold}, confirm={args.cv_confirm_frames}")
    print(f"Device: {device}")

    state_machine = BrazingStageStateMachine(
        min_confirm_frames=3,
        window_size=7,
        confidence_threshold=0.0,
    )

    neural_trigger = ConsecutiveTrigger(
        threshold=args.trigger_threshold,
        confirm_frames=args.trigger_confirm_frames,
    )

    cv_trigger = ConsecutiveTrigger(
        threshold=args.cv_threshold,
        confirm_frames=args.cv_confirm_frames,
    )

    previous_roi_gray = None

    model = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=device,
    )

    transform = make_transform(args.image_size)

    cap = open_video(args.video)

    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = args.output_fps if args.output_fps is not None else input_fps

    print(f"Video FPS: {input_fps:.3f}")
    print(f"Frames: {frame_count}")
    print(f"Frame size: {width}x{height}")
    print(f"Output FPS: {output_fps:.3f}")

    writer = create_writer(
        output_path=args.output,
        codec=args.codec,
        fps=output_fps,
        frame_size=(width, height),
    )

    smoothing_history: Deque[int] = deque(maxlen=max(1, args.smoothing_window))

    processed_frames = 0
    total_inference_ms = 0.0

    start_total = time.perf_counter()

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

        timestamp_s = processed_frames / input_fps if input_fps > 0 else 0.0

        pred_id, raw_stage, confidence, probs, inference_ms = predict_frame(
            model=model,
            frame_bgr=frame,
            roi=roi,
            transform=transform,
            device=device,
        )

        p_active_brazing = float(probs[ACTIVE_BRAZING_STAGE_ID])
        neural_trigger_on = neural_trigger.update(p_active_brazing)

        if args.enable_cv_trigger:
            cv_score, current_roi_gray = compute_cv_trigger_score(
                frame_bgr=frame,
                roi=roi,
                previous_roi_gray=previous_roi_gray,
            )
            previous_roi_gray = current_roi_gray
            cv_trigger_on = cv_trigger.update(cv_score)
        else:
            cv_score = None
            cv_trigger_on = False

        if args.postprocess == "raw":
            stable_stage = raw_stage

        elif args.postprocess == "majority":
            smooth_id = majority_vote_smooth(
                history=smoothing_history,
                current_pred_id=pred_id,
                window_size=args.smoothing_window,
            )
            stable_stage = ID_TO_STAGE[smooth_id]

        elif args.postprocess == "state_machine":
            sm_result = state_machine.update(
                raw_stage_id=pred_id,
                confidence=confidence,
            )
            stable_stage = sm_result["stable_stage_name"]

        else:
            raise ValueError(f"Unknown postprocess mode: {args.postprocess}")

        # In the current prototype, hold signal is controlled by neural trigger.
        # CV trigger is an optional fallback / diagnostic signal.
        hold_signal_on = neural_trigger_on

        output_frame = draw_overlay(
            frame_bgr=frame,
            roi=roi,
            raw_stage=raw_stage,
            stable_stage=stable_stage,
            confidence=confidence,
            p_active_brazing=p_active_brazing,
            neural_trigger_on=neural_trigger_on,
            cv_trigger_on=cv_trigger_on,
            hold_signal_on=hold_signal_on,
            inference_ms=inference_ms,
            timestamp_s=timestamp_s,
            probs=probs,
            draw_probabilities=args.draw_probabilities,
            cv_score=cv_score,
        )

        writer.write(output_frame)

        processed_frames += 1
        total_inference_ms += inference_ms

        if processed_frames % 100 == 0:
            avg_inference_ms = total_inference_ms / processed_frames
            print(
                f"Processed {processed_frames}/{frame_count} frames | "
                f"avg inference: {avg_inference_ms:.2f} ms"
            )

    elapsed_total = time.perf_counter() - start_total

    cap.release()
    writer.release()

    avg_inference_ms = (
        total_inference_ms / processed_frames if processed_frames > 0 else 0.0
    )
    avg_total_ms = (
        elapsed_total / processed_frames * 1000.0 if processed_frames > 0 else 0.0
    )

    print("Done.")
    print(f"Saved to: {args.output}")
    print(f"Processed frames: {processed_frames}")
    print(f"Average model+preprocess inference: {avg_inference_ms:.2f} ms/frame")
    print(f"Average full script time: {avg_total_ms:.2f} ms/frame")
    print(f"Approx full-script FPS: {1000.0 / avg_total_ms:.2f}" if avg_total_ms > 0 else "FPS: n/a")


if __name__ == "__main__":
    main()
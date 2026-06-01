#!/usr/bin/env python3
"""
Benchmark final BrazingSense inference latency.

Final pipeline:
    frame read
    -> ROI crop
    -> RGB conversion
    -> resize / normalize / tensor
    -> neural stage classifier
    -> neural active_brazing trigger
    -> optional CV fallback trigger
    -> postprocess: raw / majority / state_machine
    -> result

Example without CV trigger:
    PYTHONPATH=src python3 scripts/benchmark_neural_latency.py \
        --video data/raw/MVI_6266.MOV \
        --checkpoint models/checkpoints/final_neural_stage_classification_10fps/resnet18_10fps_balanced_best_10fps.pt \
        --model resnet18 \
        --roi 470,280,430,290 \
        --image-size 224 \
        --postprocess state_machine \
        --trigger-threshold 0.4 \
        --trigger-confirm-frames 1 \
        --device auto \
        --output reports/neural_latency/final_system_MVI_6266_no_cv_latency.csv \
        --summary-output reports/neural_latency/final_system_MVI_6266_no_cv_summary.json

Example with CV trigger:
    PYTHONPATH=src python3 scripts/benchmark_neural_latency.py \
        --video data/raw/MVI_6266.MOV \
        --checkpoint models/checkpoints/final_neural_stage_classification_10fps/resnet18_10fps_balanced_best_10fps.pt \
        --model resnet18 \
        --roi 470,280,430,290 \
        --image-size 224 \
        --postprocess state_machine \
        --trigger-threshold 0.4 \
        --trigger-confirm-frames 1 \
        --enable-cv-trigger \
        --cv-threshold 0.38 \
        --cv-confirm-frames 2 \
        --device auto \
        --output reports/neural_latency/final_system_MVI_6266_with_cv_latency.csv \
        --summary-output reports/neural_latency/final_system_MVI_6266_with_cv_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import time
from collections import Counter, deque
from typing import Deque, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

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
    """
    MobileNetV3 Small with an evolutionary classifier head.

    This class must match the architecture used in 11_mobilenet_nas.ipynb.
    The MobileNetV3 backbone is standard; only pooling, classifier head and
    freeze mode are configurable.
    """

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
    # Checkpoints from 11_mobilenet_nas.ipynb store the architecture
    # in checkpoint["candidate"] and full training config in checkpoint["config"].
    for key in ["candidate", "best_candidate", "arch_config", "model_config", "config"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            try:
                return normalize_arch_candidate(value)
            except ValueError:
                pass

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
        "Could not find MobileNetV3 architecture-evo config in checkpoint. "
        "Expected checkpoint['candidate'], checkpoint['config'] or direct "
        "head_type/head_hidden/... keys."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark final neural inference latency on brazing video frames."
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
        help="Input image size.",
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
        default=0.4,
        help="Threshold for neural active_brazing trigger based on P(active_brazing).",
    )

    parser.add_argument(
        "--trigger-confirm-frames",
        type=int,
        default=1,
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
        default=0.38,
        help="Threshold for optional OpenCV active_brazing trigger.",
    )

    parser.add_argument(
        "--cv-confirm-frames",
        type=int,
        default=2,
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
        "--max-frames",
        type=int,
        default=None,
        help="Optional max number of frames to benchmark.",
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Number of first frames to skip from final latency statistics.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/neural_latency/frame_latency.csv"),
        help="Output CSV with per-frame latency.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/neural_latency/latency_summary.json"),
        help="Output JSON summary.",
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


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values, q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_latency(df: pd.DataFrame, warmup_frames: int) -> Dict[str, object]:
    if warmup_frames > 0:
        effective_df = df[df["frame_idx"] >= warmup_frames].copy()
    else:
        effective_df = df.copy()

    if effective_df.empty:
        effective_df = df.copy()

    latency_columns = [
        "frame_read_ms",
        "preprocess_ms",
        "model_ms",
        "neural_trigger_ms",
        "cv_trigger_ms",
        "postprocess_ms",
        "total_ms",
    ]

    metrics = {}

    for col in latency_columns:
        values = effective_df[col].values.astype(float)

        metrics[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
        }

    total_ms_mean = metrics["total_ms"]["mean"]
    total_ms_p95 = metrics["total_ms"]["p95"]

    metrics["fps_estimate_mean"] = (
        float(1000.0 / total_ms_mean) if total_ms_mean > 0 else 0.0
    )
    metrics["fps_estimate_p95"] = (
        float(1000.0 / total_ms_p95) if total_ms_p95 > 0 else 0.0
    )

    metrics["target_50ms"] = {
        "mean_total_ms_le_50": bool(total_ms_mean <= 50.0),
        "p95_total_ms_le_50": bool(total_ms_p95 <= 50.0),
    }

    metrics["trigger_counts"] = {
        "neural_trigger_on_frames": int(effective_df["neural_trigger_on"].sum()),
        "cv_trigger_on_frames": int(effective_df["cv_trigger_on"].sum()),
        "hold_signal_on_frames": int(effective_df["hold_signal_on"].sum()),
    }

    metrics["predicted_stage_counts"] = {
        str(k): int(v)
        for k, v in effective_df["stable_stage_name"]
        .value_counts()
        .sort_index()
        .to_dict()
        .items()
    }

    return {
        "num_frames_total": int(len(df)),
        "num_frames_used_after_warmup": int(len(effective_df)),
        "warmup_frames": int(warmup_frames),
        "metrics": metrics,
    }


def benchmark_video(
    video_path: Path,
    model: nn.Module,
    transform,
    roi: Dict[str, int],
    device: torch.device,
    postprocess: str,
    smoothing_window: int,
    trigger_threshold: float,
    trigger_confirm_frames: int,
    enable_cv_trigger: bool,
    cv_threshold: float,
    cv_confirm_frames: int,
    max_frames: int | None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    state_machine = BrazingStageStateMachine(
        min_confirm_frames=3,
        window_size=7,
        confidence_threshold=0.0,
    )

    neural_trigger = ConsecutiveTrigger(
        threshold=trigger_threshold,
        confirm_frames=trigger_confirm_frames,
    )

    cv_trigger = ConsecutiveTrigger(
        threshold=cv_threshold,
        confirm_frames=cv_confirm_frames,
    )

    smoothing_history: Deque[int] = deque(maxlen=max(1, smoothing_window))
    previous_roi_gray = None

    rows = []
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        t_total_start = time.perf_counter()

        t_read_start = time.perf_counter()
        ok, frame_bgr = cap.read()
        frame_read_ms = (time.perf_counter() - t_read_start) * 1000.0

        if not ok or frame_bgr is None:
            break

        timestamp_s = frame_idx / input_fps if input_fps > 0 else 0.0

        t_pre_start = time.perf_counter()

        roi_bgr = crop_roi(frame_bgr, roi)
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(roi_rgb).unsqueeze(0).to(device)

        synchronize_if_needed(device)
        preprocess_ms = (time.perf_counter() - t_pre_start) * 1000.0

        t_model_start = time.perf_counter()

        with torch.no_grad():
            logits = model(tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0]

        synchronize_if_needed(device)
        model_ms = (time.perf_counter() - t_model_start) * 1000.0

        probs = probs_tensor.detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        raw_stage_name = ID_TO_STAGE[pred_id]
        confidence = float(probs[pred_id])
        p_active_brazing = float(probs[ACTIVE_BRAZING_STAGE_ID])

        t_neural_trigger_start = time.perf_counter()
        neural_trigger_on = neural_trigger.update(p_active_brazing)
        neural_trigger_ms = (time.perf_counter() - t_neural_trigger_start) * 1000.0

        t_cv_trigger_start = time.perf_counter()

        if enable_cv_trigger:
            cv_score, current_roi_gray = compute_cv_trigger_score(
                frame_bgr=frame_bgr,
                roi=roi,
                previous_roi_gray=previous_roi_gray,
            )
            previous_roi_gray = current_roi_gray
            cv_trigger_on = cv_trigger.update(cv_score)
        else:
            cv_score = np.nan
            cv_trigger_on = False

        cv_trigger_ms = (time.perf_counter() - t_cv_trigger_start) * 1000.0

        t_post_start = time.perf_counter()

        if postprocess == "raw":
            stable_stage_id = pred_id
            stable_stage_name = raw_stage_name

        elif postprocess == "majority":
            stable_stage_id = majority_vote_smooth(
                history=smoothing_history,
                current_pred_id=pred_id,
                window_size=smoothing_window,
            )
            stable_stage_name = ID_TO_STAGE[stable_stage_id]

        elif postprocess == "state_machine":
            sm_result = state_machine.update(
                raw_stage_id=pred_id,
                confidence=confidence,
            )
            stable_stage_id = int(sm_result["stable_stage_id"])
            stable_stage_name = str(sm_result["stable_stage_name"])

        else:
            raise ValueError(f"Unknown postprocess: {postprocess}")

        hold_signal_on = neural_trigger_on

        postprocess_ms = (time.perf_counter() - t_post_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "frame_read_ms": frame_read_ms,
                "preprocess_ms": preprocess_ms,
                "model_ms": model_ms,
                "neural_trigger_ms": neural_trigger_ms,
                "cv_trigger_ms": cv_trigger_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
                "raw_stage_id": pred_id,
                "raw_stage_name": raw_stage_name,
                "stable_stage_id": stable_stage_id,
                "stable_stage_name": stable_stage_name,
                "confidence": confidence,
                "p_active_brazing": p_active_brazing,
                "neural_trigger_on": bool(neural_trigger_on),
                "cv_score": cv_score,
                "cv_trigger_on": bool(cv_trigger_on),
                "hold_signal_on": bool(hold_signal_on),
            }
        )

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")

    cap.release()

    df = pd.DataFrame(rows)

    video_info = {
        "video_path": str(video_path),
        "input_fps": input_fps,
        "frame_count_reported": frame_count,
        "frame_width": width,
        "frame_height": height,
        "frames_processed": int(len(df)),
    }

    return df, video_info


def main() -> None:
    args = parse_args()

    roi = parse_roi(args.roi)
    device = select_device(args.device)

    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
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
    print(f"Max frames: {args.max_frames}")
    print(f"Warmup frames: {args.warmup_frames}")

    model = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=device,
    )

    transform = make_transform(args.image_size)

    frame_latency_df, video_info = benchmark_video(
        video_path=args.video,
        model=model,
        transform=transform,
        roi=roi,
        device=device,
        postprocess=args.postprocess,
        smoothing_window=args.smoothing_window,
        trigger_threshold=args.trigger_threshold,
        trigger_confirm_frames=args.trigger_confirm_frames,
        enable_cv_trigger=args.enable_cv_trigger,
        cv_threshold=args.cv_threshold,
        cv_confirm_frames=args.cv_confirm_frames,
        max_frames=args.max_frames,
    )

    if frame_latency_df.empty:
        raise RuntimeError("No frames were processed.")

    summary = summarize_latency(
        frame_latency_df,
        warmup_frames=args.warmup_frames,
    )

    summary_output = {
        "video_info": video_info,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "roi": roi,
        "image_size": args.image_size,
        "postprocess": args.postprocess,
        "smoothing_window": args.smoothing_window,
        "trigger_threshold": args.trigger_threshold,
        "trigger_confirm_frames": args.trigger_confirm_frames,
        "enable_cv_trigger": args.enable_cv_trigger,
        "cv_threshold": args.cv_threshold,
        "cv_confirm_frames": args.cv_confirm_frames,
        "max_frames": args.max_frames,
        **summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    frame_latency_df.to_csv(args.output, index=False)

    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved per-frame latency: {args.output}")
    print(f"Saved summary: {args.summary_output}")

    metrics = summary_output["metrics"]

    print()
    print("Latency summary after warmup:")
    print(f"total_ms mean:          {metrics['total_ms']['mean']:.3f}")
    print(f"total_ms p95:           {metrics['total_ms']['p95']:.3f}")
    print(f"frame_read_ms mean:     {metrics['frame_read_ms']['mean']:.3f}")
    print(f"preprocess_ms mean:     {metrics['preprocess_ms']['mean']:.3f}")
    print(f"model_ms mean:          {metrics['model_ms']['mean']:.3f}")
    print(f"neural_trigger_ms mean: {metrics['neural_trigger_ms']['mean']:.6f}")
    print(f"cv_trigger_ms mean:     {metrics['cv_trigger_ms']['mean']:.3f}")
    print(f"postprocess_ms mean:    {metrics['postprocess_ms']['mean']:.3f}")
    print(f"estimated FPS mean:     {metrics['fps_estimate_mean']:.2f}")
    print(f"mean total <= 50 ms:    {metrics['target_50ms']['mean_total_ms_le_50']}")
    print(f"p95 total <= 50 ms:     {metrics['target_50ms']['p95_total_ms_le_50']}")


if __name__ == "__main__":
    main()
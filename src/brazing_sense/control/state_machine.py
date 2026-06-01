from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


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

STAGE_TO_ID = {v: k for k, v in ID_TO_STAGE.items()}


@dataclass
class BrazingStageStateMachine:
    """
    Irreversible stage state machine for induction brazing monitoring.

    Logic:
        - process starts from inactive_preparation;
        - only transition to the next stage is allowed;
        - backward transitions are forbidden;
        - stage skipping is forbidden;
        - transition is confirmed by majority-like vote:
          the next stage must appear at least min_confirm_frames
          times in the recent window.
    """

    min_confirm_frames: int = 5
    window_size: int = 7
    confidence_threshold: float = 0.0
    initial_stage_id: int = 0
    current_stage_id: int = field(init=False)
    history: Deque[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_confirm_frames <= 0:
            raise ValueError("min_confirm_frames must be positive")
        if self.min_confirm_frames > self.window_size:
            raise ValueError("min_confirm_frames must be <= window_size")

        self.current_stage_id = self.initial_stage_id
        self.history = deque(maxlen=self.window_size)

    @property
    def current_stage_name(self) -> str:
        return ID_TO_STAGE[self.current_stage_id]

    def reset(self) -> None:
        self.current_stage_id = self.initial_stage_id
        self.history.clear()

    def update(
        self,
        raw_stage_id: int,
        confidence: float = 1.0,
    ) -> Dict[str, object]:
        """
        Update state machine using raw model prediction.

        Args:
            raw_stage_id: model argmax stage id.
            confidence: probability/confidence of raw prediction.

        Returns:
            dict with stable stage and transition information.
        """
        raw_stage_id = int(raw_stage_id)
        confidence = float(confidence)

        if raw_stage_id not in ID_TO_STAGE:
            raise ValueError(f"Unknown raw_stage_id: {raw_stage_id}")

        previous_stage_id = self.current_stage_id
        transitioned = False

        self.history.append(raw_stage_id)

        # Final stage is absorbing.
        if self.current_stage_id == len(STAGE_ORDER) - 1:
            return {
                "raw_stage_id": raw_stage_id,
                "raw_stage_name": ID_TO_STAGE[raw_stage_id],
                "stable_stage_id": self.current_stage_id,
                "stable_stage_name": self.current_stage_name,
                "previous_stage_id": previous_stage_id,
                "previous_stage_name": ID_TO_STAGE[previous_stage_id],
                "transitioned": transitioned,
                "confidence": confidence,
            }

        next_stage_id = self.current_stage_id + 1

        next_votes = sum(1 for stage_id in self.history if stage_id == next_stage_id)

        if (
            next_votes >= self.min_confirm_frames
            and confidence >= self.confidence_threshold
        ):
            self.current_stage_id = next_stage_id
            transitioned = True

        return {
            "raw_stage_id": raw_stage_id,
            "raw_stage_name": ID_TO_STAGE[raw_stage_id],
            "stable_stage_id": self.current_stage_id,
            "stable_stage_name": self.current_stage_name,
            "previous_stage_id": previous_stage_id,
            "previous_stage_name": ID_TO_STAGE[previous_stage_id],
            "transitioned": transitioned,
            "confidence": confidence,
        }


def build_default_state_machine() -> BrazingStageStateMachine:
    return BrazingStageStateMachine(
        min_confirm_frames=5,
        window_size=7,
        confidence_threshold=0.0,
        initial_stage_id=0,
    )
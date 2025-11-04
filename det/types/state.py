from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingState:
    """
    Conceptually represents the state of a training session
    to be persisted in a checkpoint.
    """

    model_state_dict: dict[str, Any]
    optimizer_states: list[dict[str, Any]]
    lr_scheduler_states: list[dict[str, Any]]
    hparams: dict[str, Any]

    epoch: int
    global_step: int

    torch_rng_state: Any
    cuda_rng_state: Any
    numpy_rng_state: Any

    best_model_score: float | None = None

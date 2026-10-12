"""Paths and parameters of the supplied Phase 4 experiment."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Default values reproduce the executed configuration in the source notebook."""

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "paper_model_best.keras"
    )
    subject: int = 1
    window_size: int = 600
    step: int = 100
    sampling_rate: int = 2000
    cutoff_hz: float = 5.0
    label_sigma: float = 400.0
    prediction_sigma: float = 10.0
    gain_factor: float = 1.35
    train_repetitions: tuple[int, ...] = (1, 3, 4, 6)
    test_repetitions: tuple[int, ...] = (2, 5)
    movement_codes: tuple[int, ...] = (9, 10, 13, 14, 15, 16, 0)
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 0.00025

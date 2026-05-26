"""Central configuration values for dataset, training and artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration used by scripts, tests and API helpers."""

    data_dir: Path = DATA_DIR / "processed" / "synthetic_defects"
    model_path: Path = ARTIFACTS_DIR / "models" / "defect_cnn.pt"
    report_path: Path = ARTIFACTS_DIR / "reports" / "metrics.json"
    confusion_matrix_path: Path = ARTIFACTS_DIR / "reports" / "confusion_matrix.png"
    image_size: int = 64
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "cpu"

"""Classification metric and reporting utilities for model evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cv-defect-detector-matplotlib")
)

# Force a non-interactive backend so training and tests can render plots in CI
# or headless shells without a desktop session.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from defect_detector.data import LABELS


def classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float | list[list[int]]]:
    # zero_division=0 keeps precision/recall stable on tiny smoke-test datasets
    # where one class may be absent from predictions.
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion_matrix(matrix: list[list[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix)
    # The saved image is used both as a training artifact and as the README
    # screenshot, so all labels are rendered directly into the plot.
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(array, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), labels=LABELS)
    ax.set_yticks(range(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            ax.text(col, row, str(array[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

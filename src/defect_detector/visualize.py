"""Visualization helpers for inspecting generated image samples."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cv-defect-detector-matplotlib")
)

# Use the same headless plotting setup as evaluation so prediction screenshots
# can be regenerated from scripts on any machine.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def save_prediction_plot(image_path: Path, prediction: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    # The title mirrors the JSON prediction, making the plot useful even when
    # it is viewed without the terminal output beside it.
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.axis("off")
    title = f"{prediction['label']} ({prediction['confidence']:.2%})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

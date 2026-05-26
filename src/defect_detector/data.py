"""Dataset generation and loading utilities for synthetic defect images."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset


LABELS = ["normal", "defective"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class DatasetSummary:
    """Counts and location for a generated defect image dataset."""

    root: Path
    train_normal: int
    train_defective: int
    val_normal: int
    val_defective: int


def generate_synthetic_dataset(
    output_dir: Path,
    train_per_class: int = 120,
    val_per_class: int = 40,
    image_size: int = 64,
    seed: int = 42,
    overwrite: bool = False,
) -> DatasetSummary:
    """Generate a simple visual inspection dataset.

    Normal images contain a clean metal-like tile. Defective images contain scratches,
    dark spots or dents. The goal is binary classification: normal vs defective.
    """
    output_dir = Path(output_dir)
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        for label in LABELS:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts = {"train": train_per_class, "val": val_per_class}
    for split, count in counts.items():
        for label in LABELS:
            for index in range(count):
                # Use a per-image seed so reruns keep the same dataset while
                # still giving each tile its own texture and defect placement.
                image_seed = rng.randint(0, 10_000_000)
                image = _make_tile(
                    image_size=image_size, defective=label == "defective", seed=image_seed
                )
                image.save(output_dir / split / label / f"{label}_{index:04d}.png")

    return DatasetSummary(
        root=output_dir,
        train_normal=train_per_class,
        train_defective=train_per_class,
        val_normal=val_per_class,
        val_defective=val_per_class,
    )


def _make_tile(image_size: int, defective: bool, seed: int) -> Image.Image:
    """Render one synthetic material tile with optional visual defects."""
    rng = random.Random(seed)
    base = rng.randint(175, 215)
    # The low-amplitude noise creates a material surface that is learnable but
    # not perfectly flat, which keeps the demo closer to real inspection data.
    noise = np.random.default_rng(seed).normal(loc=0, scale=8, size=(image_size, image_size, 3))
    array = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=0.4))
    draw = ImageDraw.Draw(image)

    # Subtle grid lines make the generated images look like inspected material tiles.
    for pos in range(0, image_size, 16):
        color = (base - 18, base - 18, base - 18)
        draw.line((pos, 0, pos, image_size), fill=color, width=1)
        draw.line((0, pos, image_size, pos), fill=color, width=1)

    if defective:
        defect_type = rng.choice(["scratch", "spot", "dent"])
        if defect_type == "scratch":
            # Scratches are drawn as long dark strokes, making them visually
            # distinct from spots while remaining simple enough for the CNN.
            for _ in range(rng.randint(1, 3)):
                x1 = rng.randint(4, image_size // 2)
                y1 = rng.randint(4, image_size - 8)
                x2 = rng.randint(image_size // 2, image_size - 4)
                y2 = min(image_size - 4, max(4, y1 + rng.randint(-12, 12)))
                draw.line((x1, y1, x2, y2), fill=(35, 35, 35), width=rng.randint(2, 4))
        elif defect_type == "spot":
            # Multiple dark blobs imitate contamination or burn marks.
            for _ in range(rng.randint(2, 5)):
                radius = rng.randint(3, 8)
                x = rng.randint(radius, image_size - radius)
                y = rng.randint(radius, image_size - radius)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(55, 50, 45))
        else:
            # A dent is a ring plus a darker center so the defect has shape,
            # not just color, and can be checked by eye in the screenshots.
            radius = rng.randint(8, 14)
            x = rng.randint(radius, image_size - radius)
            y = rng.randint(radius, image_size - radius)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), outline=(70, 70, 70), width=3
            )
            draw.ellipse(
                (x - radius + 4, y - radius + 4, x + radius - 4, y + radius - 4),
                fill=(145, 145, 145),
            )

    return image


class DefectImageDataset(Dataset):
    """PyTorch dataset that loads labeled PNG inspection images from class folders."""

    def __init__(self, root: Path, image_size: int = 64):
        self.root = Path(root)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for label in LABELS:
            for image_path in sorted((self.root / label).glob("*.png")):
                self.samples.append((image_path, LABEL_TO_INDEX[label]))
        if not self.samples:
            raise FileNotFoundError(f"No PNG images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        tensor = load_image_tensor(image_path, image_size=self.image_size)
        return tensor, torch.tensor(label, dtype=torch.long)


def load_image_tensor(image_path: Path, image_size: int = 64) -> torch.Tensor:
    """Load an image as a normalized CHW tensor suitable for the CNN."""
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    # Keep validation deterministic: shuffling is useful for training batches
    # but would make metric debugging harder for the fixed validation split.
    train_dataset = DefectImageDataset(Path(data_dir) / "train", image_size=image_size)
    val_dataset = DefectImageDataset(Path(data_dir) / "val", image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

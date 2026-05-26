"""Command-line utility for creating the synthetic defect image dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from defect_detector.config import TrainingConfig
from defect_detector.data import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic visual defect dataset.")
    # Defaults match TrainingConfig so dataset generation and training agree on
    # paths and image dimensions unless the caller overrides them.
    parser.add_argument("--output-dir", type=Path, default=TrainingConfig().data_dir)
    parser.add_argument("--train-per-class", type=int, default=120)
    parser.add_argument("--val-per-class", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The summary is printed for quick verification before starting a training
    # run that depends on the generated folder structure.
    summary = generate_synthetic_dataset(
        output_dir=args.output_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        image_size=args.image_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Dataset created at: {summary.root}")
    print(f"Train normal: {summary.train_normal}, train defective: {summary.train_defective}")
    print(f"Val normal: {summary.val_normal}, val defective: {summary.val_defective}")


if __name__ == "__main__":
    main()

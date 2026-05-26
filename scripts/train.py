"""Command-line entry point for training the defect classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_detector.config import TrainingConfig
from defect_detector.training import train_model


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the PyTorch defect classifier.")
    # Every artifact path is configurable so experiments can run without
    # overwriting the default README/demo outputs.
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--report-path", type=Path, default=defaults.report_path)
    parser.add_argument(
        "--confusion-matrix-path", type=Path, default=defaults.confusion_matrix_path
    )
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Convert CLI arguments back into the shared config object used by tests and
    # the API, keeping runtime behavior consistent across entry points.
    config = TrainingConfig(
        data_dir=args.data_dir,
        model_path=args.model_path,
        report_path=args.report_path,
        confusion_matrix_path=args.confusion_matrix_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    report = train_model(config)
    print(json.dumps(report["validation"], indent=2))
    print(f"Model saved to: {config.model_path}")
    print(f"Report saved to: {config.report_path}")


if __name__ == "__main__":
    main()

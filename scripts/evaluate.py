"""Command-line entry point for evaluating a trained defect classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_detector.config import TrainingConfig
from defect_detector.data import create_dataloaders
from defect_detector.metrics import plot_confusion_matrix, save_metrics
from defect_detector.training import evaluate_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Evaluate a trained defect classifier.")
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--report-path", type=Path, default=defaults.report_path)
    parser.add_argument(
        "--confusion-matrix-path", type=Path, default=defaults.confusion_matrix_path
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", type=str, default=defaults.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, checkpoint = load_checkpoint(args.model_path, device=args.device)
    # Use the checkpoint image size so evaluation matches the dimensions used
    # during training.
    _, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        image_size=int(checkpoint.get("image_size", 64)),
        batch_size=args.batch_size,
    )
    metrics = evaluate_model(model, val_loader, device=next(model.parameters()).device)
    # Refresh both machine-readable metrics and the plot artifact from the same
    # evaluation pass.
    save_metrics({"validation": metrics}, args.report_path)
    plot_confusion_matrix(metrics["confusion_matrix"], args.confusion_matrix_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

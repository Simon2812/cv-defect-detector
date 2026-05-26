"""Command-line entry point for running inference on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_detector.config import TrainingConfig
from defect_detector.inference import predict_image
from defect_detector.visualize import save_prediction_plot


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--plot-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Print JSON first so the command can be piped into other tools even when a
    # visual plot is also requested.
    prediction = predict_image(
        model_path=args.model_path,
        image_path=args.image_path,
        device=args.device,
    )
    print(json.dumps(prediction, indent=2))
    if args.plot_path:
        # The optional plot is a human-readable companion to the JSON response.
        save_prediction_plot(args.image_path, prediction, args.plot_path)
        print(f"Prediction plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()

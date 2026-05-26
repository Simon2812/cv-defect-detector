"""Inference helpers for classifying individual inspection images."""

from __future__ import annotations

from pathlib import Path

import torch

from defect_detector.data import INDEX_TO_LABEL, load_image_tensor
from defect_detector.training import load_checkpoint


@torch.no_grad()
def predict_image(model_path: Path, image_path: Path, device: str = "cpu") -> dict:
    model, checkpoint = load_checkpoint(model_path, device=device)
    image_size = int(checkpoint.get("image_size", 64))
    target_device = next(model.parameters()).device
    # Match the training image size saved in the checkpoint; this prevents
    # callers from accidentally serving images with a different tensor shape.
    image = load_image_tensor(image_path, image_size=image_size).unsqueeze(0).to(target_device)
    logits = model(image)
    # Softmax converts logits into user-facing class probabilities for both
    # the CLI JSON output and the FastAPI response.
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    predicted_index = int(probabilities.argmax().item())
    return {
        "image": str(image_path),
        "label": INDEX_TO_LABEL[predicted_index],
        "confidence": round(float(probabilities[predicted_index].item()), 4),
        "probabilities": {
            INDEX_TO_LABEL[index]: round(float(probability.item()), 4)
            for index, probability in enumerate(probabilities)
        },
    }

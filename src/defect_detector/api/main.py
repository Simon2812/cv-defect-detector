"""HTTP API exposing health checks and image classification endpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from defect_detector.config import TrainingConfig
from defect_detector.inference import predict_image


app = FastAPI(
    title="CV Defect Detector",
    version="0.1.0",
    description="Upload a product-tile image and classify it as normal or defective.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    model_path = Path(os.getenv("MODEL_PATH", TrainingConfig().model_path))
    device = os.getenv("DEVICE", "cpu")
    # Fail fast with a clear service error when the model artifact has not
    # been produced by the training script yet.
    if not model_path.exists():
        raise HTTPException(status_code=503, detail=f"Model not found at {model_path}")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")

    suffix = ".jpg" if file.content_type == "image/jpeg" else ".png"
    # Store the upload briefly on disk because the inference helper accepts a
    # path and shares the same image-loading code as the CLI.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        return predict_image(model_path=model_path, image_path=temp_path, device=device)
    finally:
        temp_path.unlink(missing_ok=True)

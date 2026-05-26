# CV Defect Detector

A compact PyTorch image-classification pipeline for visual quality inspection. It creates a small synthetic tile dataset, trains a CNN, stores evaluation artifacts, and serves single-image predictions through FastAPI.

## Quick Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_dataset.py --overwrite
python scripts/train.py --epochs 8
python scripts/predict.py data/processed/synthetic_defects/val/defective/defective_0000.png --plot-path artifacts/predictions/defective_prediction.png
pytest
```

Start the API with:

```bash
uvicorn defect_detector.api.main:app --reload
```

Open `http://localhost:8000/docs`.

## Screenshots

These images come from the generated dataset, saved training artifacts, prediction output, and the running FastAPI documentation page.

To refresh the same surfaces locally:

```bash
python scripts/generate_dataset.py --overwrite
python scripts/train.py --epochs 8
python scripts/predict.py data/processed/synthetic_defects/val/defective/defective_0000.png --plot-path artifacts/predictions/defective_prediction.png
PYTHONPATH=src uvicorn defect_detector.api.main:app --reload
```

Then open `http://localhost:8000/docs`. The screenshots below correspond to the generated files under `data/processed/`, `artifacts/reports/`, `artifacts/predictions/`, and the live Swagger UI.

![Dataset examples](docs/screenshots/dataset-examples.png)

![Training metrics](docs/screenshots/training-metrics.png)

![Confusion matrix](docs/screenshots/confusion-matrix.png)

![Prediction output](docs/screenshots/prediction-output.png)

![FastAPI docs](docs/screenshots/fastapi-docs.png)

## What It Does

- Builds reproducible `normal` and `defective` image folders.
- Trains and evaluates a binary CNN with accuracy, precision, recall, F1, and a confusion matrix.
- Saves the model to `artifacts/models/defect_cnn.pt`.
- Provides CLI scripts for data generation, training, evaluation, and prediction.
- Exposes `/health` and `/predict` endpoints for inference.

## Repository Layout

```text
src/defect_detector/      model, dataset, training, inference, API
scripts/                  command-line entry points
tests/                    data, model, training, and API tests
data/processed/           generated image dataset
artifacts/                model, metrics, plots, prediction output
docs/screenshots/         README images
```

## Useful Commands

```bash
python scripts/evaluate.py
python -m pytest tests/test_api.py
```

Keywords: python, pytorch, computer vision, image classification, fastapi, pytest

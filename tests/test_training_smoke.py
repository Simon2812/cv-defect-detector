"""End-to-end smoke tests for the defect training pipeline."""

from __future__ import annotations

from defect_detector.config import TrainingConfig
from defect_detector.data import generate_synthetic_dataset
from defect_detector.training import train_model


def test_training_smoke_run(tmp_path):
    data_dir = tmp_path / "dataset"
    generate_synthetic_dataset(
        output_dir=data_dir,
        train_per_class=4,
        val_per_class=2,
        image_size=32,
        seed=11,
    )
    config = TrainingConfig(
        data_dir=data_dir,
        model_path=tmp_path / "model.pt",
        report_path=tmp_path / "metrics.json",
        confusion_matrix_path=tmp_path / "confusion_matrix.png",
        image_size=32,
        batch_size=4,
        epochs=1,
        learning_rate=1e-3,
        seed=11,
        device="cpu",
    )

    report = train_model(config)

    assert config.model_path.exists()
    assert config.report_path.exists()
    assert config.confusion_matrix_path.exists()
    assert "validation" in report
    assert "accuracy" in report["validation"]

"""Dataset generation and loading tests for defect images."""

from __future__ import annotations

from defect_detector.data import DefectImageDataset, generate_synthetic_dataset


def test_generate_synthetic_dataset_creates_expected_split(tmp_path):
    output_dir = tmp_path / "dataset"

    summary = generate_synthetic_dataset(
        output_dir=output_dir,
        train_per_class=3,
        val_per_class=2,
        image_size=32,
        seed=7,
    )

    assert summary.train_normal == 3
    assert len(list((output_dir / "train" / "normal").glob("*.png"))) == 3
    assert len(list((output_dir / "train" / "defective").glob("*.png"))) == 3
    assert len(list((output_dir / "val" / "normal").glob("*.png"))) == 2
    assert len(list((output_dir / "val" / "defective").glob("*.png"))) == 2


def test_defect_dataset_returns_tensor_and_label(tmp_path):
    output_dir = tmp_path / "dataset"
    generate_synthetic_dataset(
        output_dir=output_dir, train_per_class=1, val_per_class=1, image_size=32
    )

    dataset = DefectImageDataset(output_dir / "train", image_size=32)
    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert label.item() in {0, 1}

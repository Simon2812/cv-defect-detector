"""Training, evaluation and checkpoint management for the defect classifier."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from defect_detector.config import TrainingConfig
from defect_detector.data import create_dataloaders
from defect_detector.metrics import classification_metrics, plot_confusion_matrix, save_metrics
from defect_detector.models import DefectCNN
from defect_detector.utils.reproducibility import set_seed


def train_model(config: TrainingConfig) -> dict:
    """Train the classifier and persist the checkpoint, metrics and confusion matrix."""
    set_seed(config.seed)
    # Fall back to CPU when a requested accelerator is not available so the
    # same command works on laptops, CI runners and GPU workstations.
    device = torch.device(
        config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    )
    train_loader, val_loader = create_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
    )

    model = DefectCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    started_at = perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Accumulate sample-weighted loss because the final batch may be
            # smaller than the configured batch size.
            running_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

        val_metrics = evaluate_model(model=model, data_loader=val_loader, device=device)
        # Store a compact history that is reused by the README screenshot
        # renderer and by anyone comparing short training runs.
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running_loss / total, 4),
                "train_accuracy": round(correct / total, 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

    final_metrics = evaluate_model(model=model, data_loader=val_loader, device=device)
    report = {
        "model": "DefectCNN",
        "image_size": config.image_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "device": str(device),
        "training_seconds": round(perf_counter() - started_at, 2),
        "history": history,
        "validation": final_metrics,
    }

    # Keep these three outputs together: the API needs the checkpoint, while
    # the README and report screenshots use the metrics and confusion matrix.
    save_checkpoint(model, config.model_path, config)
    save_metrics(report, config.report_path)
    plot_confusion_matrix(final_metrics["confusion_matrix"], config.confusion_matrix_path)
    return report


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader, device: torch.device) -> dict:
    """Run model evaluation and return standard classification metrics."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in data_loader:
        images = images.to(device)
        logits = model(images)
        # Move predictions back to CPU lists so metric code stays independent
        # from PyTorch tensors and can be tested with plain Python values.
        predictions = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(predictions)
        y_true.extend(labels.tolist())
    return classification_metrics(y_true, y_pred)


def save_checkpoint(model: nn.Module, output_path: Path, config: TrainingConfig) -> None:
    """Persist model weights and metadata required for later inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": config.image_size,
            "labels": ["normal", "defective"],
        },
        output_path,
    )


def load_checkpoint(model_path: Path, device: str = "cpu") -> tuple[DefectCNN, dict]:
    """Restore a trained model and its checkpoint metadata."""
    # The same fallback rule used in training keeps inference portable when a
    # checkpoint was produced on a different machine.
    target_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    checkpoint = torch.load(model_path, map_location=target_device)
    model = DefectCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint

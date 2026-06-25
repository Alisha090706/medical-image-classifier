"""
train.py
========
Trains a pneumonia-vs-normal chest X-ray classifier using transfer learning
on top of a pretrained EfficientNet-B0 backbone.

Training strategy
------------------
- The EfficientNet-B0 convolutional backbone (all feature-extraction layers)
  is frozen — its ImageNet-pretrained weights are used as a fixed feature
  extractor.
- Only the final classifier head (a small linear layer) is trained. This
  keeps the number of trainable parameters tiny, which is exactly what makes
  CPU training practical: each epoch only needs to update a few thousand
  weights via backprop, while the expensive convolutional feature maps are
  computed once per forward pass with no gradient tracking overhead beyond
  feeding the classifier.
- Adam optimizer + CrossEntropyLoss, with the model emitting raw 2-class
  logits (NORMAL=0, PNEUMONIA=1).

Reliability features
---------------------
- Automatic device detection: uses CUDA if available, otherwise falls back
  to CPU transparently. No code changes needed either way.
- Early stopping (patience=3): training halts if validation accuracy fails
  to improve for 3 consecutive epochs, preventing wasted CPU time once the
  model has converged.
- Model checkpointing: the model state is saved to disk every time
  validation accuracy reaches a new best, so the file on disk always
  reflects the best-performing checkpoint seen so far — even if training is
  interrupted partway through.
- Per-epoch logging of training/validation loss and accuracy to the console,
  plus two PNG plots saved to outputs/ for a visual training history.

Usage
-----
    python train.py
    python train.py --epochs 15 --batch-size 16 --lr 0.0005
    python train.py --data-dir /path/to/chest_xray
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for headless/CPU-only servers.
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from utils.dataset import CLASS_NAMES, get_dataloaders
from utils.transforms import get_eval_transforms, get_train_transforms

# Number of output classes for the final classifier layer (NORMAL, PNEUMONIA).
NUM_CLASSES: int = 2

# Default paths, relative to the project root (where this script is run from).
DEFAULT_DATA_DIR: str = "chest_xray"
MODEL_SAVE_PATH: str = os.path.join("models", "best_model.pth")
OUTPUTS_DIR: str = "outputs"
LOSS_PLOT_PATH: str = os.path.join(OUTPUTS_DIR, "training_loss.png")
ACCURACY_PLOT_PATH: str = os.path.join(OUTPUTS_DIR, "training_accuracy.png")


@dataclass
class EpochMetrics:
    """Container for the metrics produced by a single training epoch."""

    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainingHistory:
    """Accumulates per-epoch metrics across the full training run, used to
    generate the loss/accuracy plots at the end of training."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)

    def append(self, metrics: EpochMetrics) -> None:
        """Record one epoch's worth of metrics into the history."""
        self.train_loss.append(metrics.train_loss)
        self.val_loss.append(metrics.val_loss)
        self.train_accuracy.append(metrics.train_accuracy)
        self.val_accuracy.append(metrics.val_accuracy)


def get_device() -> torch.device:
    """
    Automatically select the best available compute device.

    Returns:
        torch.device("cuda") if a GPU is available, otherwise
        torch.device("cpu"). All training code is written to work correctly
        on either device without modification.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(device: torch.device) -> nn.Module:
    """
    Construct an EfficientNet-B0 model pretrained on ImageNet, with the
    convolutional backbone frozen and a fresh trainable classifier head for
    binary classification.

    Args:
        device: The torch device to move the model onto.

    Returns:
        An EfficientNet-B0 nn.Module ready for training, with:
            - All backbone (feature extraction) parameters frozen
              (requires_grad=False).
            - A new `classifier` head with requires_grad=True, mapping the
              1280-dim EfficientNet-B0 feature vector to NUM_CLASSES logits.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Freeze every parameter in the pretrained network. This includes both
    # the convolutional "features" block and the original classifier — we
    # will replace the classifier below with a fresh, trainable one.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # torchvision's EfficientNet-B0 classifier is:
    #   Sequential(Dropout(p=0.2), Linear(in_features=1280, out_features=1000))
    # We replace it with a new Sequential head sized for our 2-class problem.
    # Freshly constructed layers have requires_grad=True by default, so this
    # head is trainable while everything else remains frozen.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features, out_features=NUM_CLASSES),
    )

    model = model.to(device)
    return model


def compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Compute classification accuracy for a batch.

    Args:
        logits: Raw model outputs of shape (batch_size, NUM_CLASSES).
        labels: Ground-truth integer labels of shape (batch_size,).

    Returns:
        Fraction of correct predictions in [0.0, 1.0].
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam = None,
) -> Tuple[float, float]:
    """
    Run a single pass over the given dataloader, either in training mode
    (if an optimizer is provided) or evaluation mode (if optimizer is None).

    Args:
        model: The model to run.
        dataloader: DataLoader yielding (images, labels) batches.
        criterion: Loss function (CrossEntropyLoss).
        device: Device to run computation on.
        optimizer: If provided, the model is put in training mode and
            weights are updated via backpropagation. If None, the model is
            put in evaluation mode and no gradients are computed.

    Returns:
        Tuple of (average_loss, average_accuracy) across all batches,
        weighted by batch size to correctly handle a final partial batch.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Gradient tracking is only needed during training. Disabling it during
    # evaluation reduces memory usage and speeds up the forward pass, which
    # matters on CPU.
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            batch_accuracy = compute_accuracy(logits, labels)

            total_loss += loss.item() * batch_size
            total_correct += batch_accuracy * batch_size
            total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples
    return average_loss, average_accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    patience: int,
    model_save_path: str,
) -> TrainingHistory:
    """
    Run the full training loop with validation, early stopping, and best
    -model checkpointing.

    Args:
        model: The model to train (backbone frozen, classifier trainable).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        device: Device to train on.
        num_epochs: Maximum number of epochs to train for.
        learning_rate: Learning rate for the Adam optimizer.
        patience: Number of consecutive epochs without validation accuracy
            improvement to tolerate before stopping early.
        model_save_path: File path to save the best model's state_dict to.

    Returns:
        TrainingHistory containing per-epoch loss/accuracy for both splits,
        covering only the epochs that actually ran (i.e. truncated if early
        stopping triggered before num_epochs completed).
    """
    criterion = nn.CrossEntropyLoss()

    # Only the classifier head's parameters have requires_grad=True, so this
    # optimizer naturally updates just those weights, leaving the frozen
    # backbone untouched.
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_parameters, lr=learning_rate)

    history = TrainingHistory()
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        epoch_duration = time.time() - epoch_start_time

        history.append(
            EpochMetrics(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(
                f"  -> New best validation accuracy ({best_val_accuracy:.4f}). "
                f"Model checkpoint saved to '{model_save_path}'."
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No improvement for {epochs_without_improvement} "
                f"epoch(s) (best so far: {best_val_accuracy:.4f})."
            )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(no improvement in validation accuracy for {patience} "
                f"consecutive epochs)."
            )
            break

    return history


def plot_training_history(history: TrainingHistory, loss_path: str, accuracy_path: str) -> None:
    """
    Generate and save two PNG plots summarizing the training run:
        1. Training vs. validation loss per epoch.
        2. Training vs. validation accuracy per epoch.

    Args:
        history: The TrainingHistory collected during training.
        loss_path: File path to save the loss plot to.
        accuracy_path: File path to save the accuracy plot to.
    """
    os.makedirs(os.path.dirname(loss_path), exist_ok=True)
    epochs_range = range(1, len(history.train_loss) + 1)

    # --- Loss plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.train_loss, label="Train Loss", marker="o")
    plt.plot(epochs_range, history.val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # --- Accuracy plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.train_accuracy, label="Train Accuracy", marker="o")
    plt.plot(epochs_range, history.val_accuracy, label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_path)
    plt.close()

    print(f"\nSaved training loss plot to '{loss_path}'.")
    print(f"Saved training accuracy plot to '{accuracy_path}'.")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the training run.

    Returns:
        Parsed argparse.Namespace with fields: data_dir, epochs, batch_size,
        lr, patience, num_workers.
    """
    parser = argparse.ArgumentParser(
        description="Train a chest X-ray pneumonia classifier using transfer "
        "learning on a frozen EfficientNet-B0 backbone."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to the chest_xray dataset root directory (default: '{DEFAULT_DATA_DIR}').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in epochs (default: 3).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0, safest for CPU-only machines).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: parses arguments, builds datasets/model, runs training with
    early stopping and checkpointing, then saves training history plots.
    """
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{args.data_dir}' was not found. Pass the "
            f"correct path with --data-dir /path/to/chest_xray."
        )

    device = get_device()
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_loader, val_loader, _test_loader = get_dataloaders(
        dataset_root=args.data_dir,
        train_transform=get_train_transforms(),
        eval_transform=get_eval_transforms(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Loaded {len(train_loader.dataset)} training images and "
        f"{len(val_loader.dataset)} validation images across classes: "
        f"{CLASS_NAMES}."
    )

    print("Building EfficientNet-B0 model (backbone frozen, classifier trainable)...")
    model = build_model(device)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} total.")

    print(f"\nStarting training for up to {args.epochs} epoch(s)...\n")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        model_save_path=MODEL_SAVE_PATH,
    )

    plot_training_history(history, LOSS_PLOT_PATH, ACCURACY_PLOT_PATH)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

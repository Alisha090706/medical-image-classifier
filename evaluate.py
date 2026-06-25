"""
evaluate.py
===========
Loads the best saved model checkpoint and evaluates it on the held-out
chest_xray/test split, reporting accuracy, precision, recall, F1 score, a
full classification report, and a confusion matrix plot.

This script is intentionally separate from train.py: evaluation should be
re-runnable at any time against a saved checkpoint without re-training,
which is also how evaluate.py will later be reused by predict.py / app.py
for sanity-checking a deployed model.

Usage
-----
    python evaluate.py
    python evaluate.py --data-dir /path/to/chest_xray --model-path models/best_model.pth
"""

import argparse
import json
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for headless/CPU-only servers.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from train import build_model, get_device
from utils.dataset import CLASS_NAMES, ChestXrayDataset
from utils.transforms import get_eval_transforms

# Default paths, relative to the project root (where this script is run from).
DEFAULT_DATA_DIR: str = "chest_xray"
DEFAULT_MODEL_PATH: str = os.path.join("models", "best_model.pth")
OUTPUTS_DIR: str = "outputs"
CONFUSION_MATRIX_PATH: str = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
# Metrics are also persisted as JSON so app.py can display them instantly
# without re-running inference over the entire test set on every page load.
METRICS_JSON_PATH: str = os.path.join(OUTPUTS_DIR, "metrics.json")

# Default DataLoader batch size for evaluation. Independent from training
# batch size since no gradients are tracked here, so memory pressure is lower.
DEFAULT_BATCH_SIZE: int = 32


def load_trained_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Reconstruct the EfficientNet-B0 architecture (frozen backbone + custom
    classifier head, matching train.py exactly) and load saved weights into
    it.

    Args:
        model_path: Path to the saved model state_dict (.pth file produced
            by train.py).
        device: Device to load the model onto.

    Returns:
        The model with trained weights loaded, set to evaluation mode.

    Raises:
        FileNotFoundError: If no checkpoint exists at model_path.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No model checkpoint found at '{model_path}'. Run train.py "
            f"first to produce a trained model."
        )

    # build_model() constructs the exact same architecture used during
    # training (pretrained EfficientNet-B0 backbone + custom classifier
    # head), so the saved state_dict's keys line up correctly.
    model = build_model(device)
    # weights_only=True is safe and explicit here because train.py only ever
    # saves a plain model.state_dict() (just tensors, no optimizer state or
    # custom objects). Pinning this explicitly avoids relying on torch's
    # default value, which changed between versions (torch >= 2.6 defaults
    # to True; earlier versions defaulted to False).
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def get_predictions(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over an entire dataloader and collect predictions,
    ground-truth labels, and predicted class probabilities.

    Args:
        model: Trained model in evaluation mode.
        dataloader: DataLoader yielding (images, labels) batches.
        device: Device to run inference on.

    Returns:
        Tuple of three NumPy arrays, each of shape (num_samples,):
            - y_true: ground-truth integer labels.
            - y_pred: predicted integer labels (argmax of logits).
            - y_prob: predicted probability of the positive class
              (PNEUMONIA, index 1), useful for confidence reporting.
    """
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_positive_probabilities: List[float] = []

    for images, labels in dataloader:
        images = images.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_positive_probabilities.extend(probabilities[:, 1].cpu().numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_positive_probabilities),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute the core evaluation metrics for binary classification.

    PNEUMONIA (label 1) is treated as the positive class, which is the
    clinically relevant convention: precision/recall/F1 describe how well
    the model identifies pneumonia cases specifically.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Dict with keys: "accuracy", "precision", "recall", "f1_score".
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """
    Compute and save a confusion matrix plot, with class names (NORMAL,
    PNEUMONIA) labeled on both axes for readability.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        save_path: File path to save the resulting PNG to.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — Chest X-Ray Test Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Saved confusion matrix plot to '{save_path}'.")


def save_metrics_json(metrics: dict, num_samples: int, save_path: str) -> None:
    """
    Persist the headline metrics to a JSON file so other parts of the
    project (notably app.py) can display them instantly without needing to
    reload the model and re-run inference over the entire test set.

    Args:
        metrics: Dict produced by compute_metrics().
        num_samples: Total number of test samples evaluated.
        save_path: File path to write the JSON file to.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    payload = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "num_test_samples": num_samples,
    }

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics summary to '{save_path}'.")


def print_metrics_report(
    metrics: dict, y_true: np.ndarray, y_pred: np.ndarray, num_samples: int
) -> None:
    """
    Print a clean, human-readable summary of all evaluation results to the
    console, including the headline metrics and the full per-class
    scikit-learn classification report.

    Args:
        metrics: Dict produced by compute_metrics().
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        num_samples: Total number of test samples evaluated.
    """
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS — Test Set")
    print("=" * 50)
    print(f"Total test samples : {num_samples}")
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (PNEUMONIA): {metrics['precision']:.4f}")
    print(f"Recall (PNEUMONIA)    : {metrics['recall']:.4f}")
    print(f"F1 Score (PNEUMONIA)  : {metrics['f1_score']:.4f}")
    print("-" * 50)
    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        )
    )
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the evaluation run.

    Returns:
        Parsed argparse.Namespace with fields: data_dir, model_path,
        batch_size, num_workers.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained chest X-ray pneumonia classifier on "
        "the held-out test set."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to the chest_xray dataset root directory (default: '{DEFAULT_DATA_DIR}').",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model checkpoint (default: '{DEFAULT_MODEL_PATH}').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE}).",
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
    Entry point: loads the trained model and test set, runs inference,
    computes all required metrics, prints a report, and saves the
    confusion matrix plot.
    """
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{args.data_dir}' was not found. Pass the "
            f"correct path with --data-dir /path/to/chest_xray."
        )

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading trained model from '{args.model_path}'...")
    model = load_trained_model(args.model_path, device)

    print("Loading test dataset...")
    test_dataset = ChestXrayDataset(
        dataset_root=args.data_dir, split="test", transform=get_eval_transforms()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Loaded {len(test_dataset)} test images across classes: {CLASS_NAMES}.")

    print("Running inference on test set...")
    y_true, y_pred, _y_prob = get_predictions(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics_report(metrics, y_true, y_pred, num_samples=len(test_dataset))

    plot_confusion_matrix(y_true, y_pred, CONFUSION_MATRIX_PATH)
    save_metrics_json(metrics, num_samples=len(test_dataset), save_path=METRICS_JSON_PATH)


if __name__ == "__main__":
    main()

"""
predict.py
==========
Runs inference on a single chest X-ray image using the trained model saved
by train.py, returning the predicted class (NORMAL or PNEUMONIA) along with
a confidence score.

This module is designed to be used two ways:
    1. As a CLI script:      python predict.py --image path/to/xray.jpeg
    2. As an importable API: from predict import predict_image
       (used by both app.py and gradcam.py so prediction logic lives in one
       place and stays perfectly consistent across the whole project).

The preprocessing pipeline here is identical to the one used for validation
during training (utils.transforms.get_eval_transforms) — same resize,
same ImageNet normalization, same RGB conversion — which is essential for
the model's predictions to be meaningful on new images.
"""

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image

from train import MODEL_SAVE_PATH, build_model, get_device
from utils.dataset import CLASS_NAMES
from utils.transforms import get_eval_transforms


@dataclass
class PredictionResult:
    """
    Container for a single image prediction.

    Attributes:
        predicted_class: The predicted class name ("NORMAL" or "PNEUMONIA").
        confidence: Softmax probability of the predicted class, in [0, 1].
        normal_probability: Softmax probability assigned to NORMAL.
        pneumonia_probability: Softmax probability assigned to PNEUMONIA.
    """

    predicted_class: str
    confidence: float
    normal_probability: float
    pneumonia_probability: float


def load_model_for_inference(model_path: str, device: torch.device) -> nn.Module:
    """
    Reconstruct the EfficientNet-B0 architecture and load trained weights
    for inference.

    Args:
        model_path: Path to the saved model state_dict (.pth file).
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


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Apply the same preprocessing pipeline used during validation/testing to
    a single PIL image, producing a batch of size 1 ready for the model.

    Args:
        image: A PIL Image, in any mode (will be converted to RGB).

    Returns:
        A torch.Tensor of shape (1, 3, 224, 224), normalized with ImageNet
        statistics.
    """
    rgb_image = image.convert("RGB")
    transform = get_eval_transforms()
    tensor = transform(rgb_image)
    return tensor.unsqueeze(0)  # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224).


@torch.no_grad()
def predict_image(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
) -> PredictionResult:
    """
    Run a single image through the trained model and return its predicted
    class with confidence scores.

    Args:
        image: A PIL Image of the chest X-ray to classify.
        model: A trained model (already loaded with weights, in eval mode).
        device: Device the model lives on.

    Returns:
        A PredictionResult containing the predicted class and probabilities
        for both classes.
    """
    input_tensor = preprocess_image(image).to(device)

    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)

    normal_probability = probabilities[0].item()
    pneumonia_probability = probabilities[1].item()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        normal_probability=normal_probability,
        pneumonia_probability=pneumonia_probability,
    )


def predict_from_path(
    image_path: str,
    model_path: str = MODEL_SAVE_PATH,
) -> PredictionResult:
    """
    Convenience function: loads the model, opens an image from disk, and
    returns the prediction in a single call. Intended for simple programmatic
    use (e.g. quick scripts, notebooks) where the caller does not want to
    manage model loading themselves.

    Note: this reloads the model from disk on every call. For repeated
    predictions (e.g. inside app.py serving many requests), load the model
    once with load_model_for_inference() and call predict_image() directly
    instead, to avoid redundant disk I/O.

    Args:
        image_path: Path to the chest X-ray image file (.png, .jpg, .jpeg).
        model_path: Path to the trained model checkpoint.

    Returns:
        A PredictionResult containing the predicted class and probabilities.

    Raises:
        FileNotFoundError: If the image or model checkpoint does not exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'.")

    device = get_device()
    model = load_model_for_inference(model_path, device)
    image = Image.open(image_path)

    return predict_image(image, model, device)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the predict.py CLI entry point.

    Returns:
        Parsed argparse.Namespace with fields: image, model_path.
    """
    parser = argparse.ArgumentParser(
        description="Predict NORMAL or PNEUMONIA for a single chest X-ray image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the chest X-ray image file (.png, .jpg, .jpeg).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_SAVE_PATH,
        help=f"Path to the trained model checkpoint (default: '{MODEL_SAVE_PATH}').",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: loads the model, runs prediction on the given image path,
    and prints the result in a clean, human-readable format.
    """
    args = parse_args()

    result = predict_from_path(image_path=args.image, model_path=args.model_path)

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Image              : {args.image}")
    print(f"Predicted class     : {result.predicted_class}")
    print(f"Confidence          : {result.confidence * 100:.2f}%")
    print("-" * 50)
    print("Class probabilities:")
    print(f"  NORMAL            : {result.normal_probability * 100:.2f}%")
    print(f"  PNEUMONIA         : {result.pneumonia_probability * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

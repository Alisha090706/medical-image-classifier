"""
transforms.py
=============
Defines the image preprocessing and augmentation pipelines used across the
MediScan AI project (training, validation, testing, and single-image inference).

Design notes
------------
- EfficientNet-B0 (torchvision, pretrained on ImageNet) expects:
    * 3-channel RGB input
    * 224x224 spatial size
    * Normalization using ImageNet mean/std
- Chest X-rays are grayscale, but we convert them to 3-channel RGB so the
  pretrained EfficientNet-B0 weights (trained on RGB ImageNet data) remain
  compatible without modifying the first convolutional layer.
- Augmentations are intentionally lightweight (flip + small rotation) since:
    1. We are training on CPU only — heavy augmentation pipelines slow down
       every epoch significantly.
    2. The classifier head is small (backbone is frozen), so aggressive
       augmentation is unnecessary and can even hurt convergence speed.
- Validation/Test transforms never include augmentation — only deterministic
  resizing and normalization, so evaluation numbers are stable and reproducible.
"""

from typing import Tuple

import torchvision.transforms as T

# Target spatial resolution expected by EfficientNet-B0.
IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Standard ImageNet normalization statistics. EfficientNet-B0's pretrained
# weights were trained on ImageNet using these exact values, so we must reuse
# them for transfer learning to work correctly.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms() -> T.Compose:
    """
    Build the transformation pipeline used for the training split.

    Pipeline steps:
        1. Resize to 224x224 (fixed size required by EfficientNet-B0).
        2. Random horizontal flip (p=0.5) — chest X-rays are roughly
           left-right symmetric in terms of pathology presence, so this is a
           safe, label-preserving augmentation.
        3. Random rotation (+/- 10 degrees) — simulates minor patient
           positioning variation during the X-ray scan without distorting
           anatomy beyond realistic limits.
        4. Convert PIL image to a PyTorch tensor (scales pixel values to
           [0, 1] and reorders dimensions to C x H x W).
        5. Normalize using ImageNet mean/std.

    Returns:
        torchvision.transforms.Compose: callable transform pipeline that
        takes a PIL.Image (already converted to RGB) and returns a
        normalized torch.Tensor of shape (3, 224, 224).
    """
    return T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> T.Compose:
    """
    Build the transformation pipeline used for validation, testing, and
    single-image inference (e.g. an uploaded X-ray in the Streamlit app).

    This pipeline is fully deterministic (no randomness) so that:
        - Validation/test metrics are reproducible across runs.
        - A given uploaded image always produces the same prediction.

    Pipeline steps:
        1. Resize to 224x224.
        2. Convert PIL image to a PyTorch tensor.
        3. Normalize using ImageNet mean/std.

    Returns:
        torchvision.transforms.Compose: callable transform pipeline that
        takes a PIL.Image (already converted to RGB) and returns a
        normalized torch.Tensor of shape (3, 224, 224).
    """
    return T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

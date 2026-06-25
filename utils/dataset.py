"""
dataset.py
==========
Handles loading the Kaggle "Chest X-Ray Images (Pneumonia)" dataset from a
local directory and exposes it as PyTorch Dataset / DataLoader objects.

Expected directory layout (already extracted locally by the user):

    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/

Notes on robustness
--------------------
- Kaggle zip extraction on macOS often creates a stray "__MACOSX" folder and
  hidden "._*" resource-fork files alongside the real images. These are not
  valid images and must be skipped, otherwise PIL will raise errors or the
  model will be trained on garbage/duplicate entries.
- Some re-extractions create a nested duplicate folder such as
  "chest_xray/chest_xray/..." — this module searches for the real split
  folders rather than assuming a fixed nesting depth, so it works whether the
  dataset root passed in is the outer folder or the true data folder.
"""

import os
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

# The two class names, in a fixed order. This order defines the integer
# label mapping used everywhere in the project: NORMAL -> 0, PNEUMONIA -> 1.
CLASS_NAMES: List[str] = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Valid image extensions. Anything else (e.g. "._image.jpeg" resource forks,
# or non-image files) is skipped during indexing.
VALID_EXTENSIONS: Tuple[str, ...] = (".jpeg", ".jpg", ".png")


def _is_valid_image_file(filename: str) -> bool:
    """
    Determine whether a filename should be treated as a valid dataset image.

    Filters out:
        - macOS resource-fork files (start with "._").
        - Hidden files (start with ".").
        - Any file whose extension is not in VALID_EXTENSIONS.

    Args:
        filename: The base filename (not full path) to check.

    Returns:
        True if the file should be included in the dataset, False otherwise.
    """
    if filename.startswith("."):
        return False
    return filename.lower().endswith(VALID_EXTENSIONS)


def _resolve_split_dir(dataset_root: str, split: str) -> str:
    """
    Locate the actual directory for a given split ("train", "val", "test"),
    handling the common case where Kaggle extraction creates a nested
    duplicate folder (e.g. "chest_xray/chest_xray/train").

    Also skips any "__MACOSX" directories if encountered while searching.

    Args:
        dataset_root: Path to the top-level dataset folder as provided by
            the user (e.g. ".../chest_xray").
        split: One of "train", "val", or "test".

    Returns:
        The resolved absolute path to the split directory that directly
        contains the "NORMAL" and "PNEUMONIA" subfolders.

    Raises:
        FileNotFoundError: If no valid split directory can be found.
    """
    # Most common case: split folder is a direct child of dataset_root.
    direct_path = os.path.join(dataset_root, split)
    if _is_valid_split_dir(direct_path):
        return direct_path

    # Fallback: search up to 2 levels deep for a folder named `split` that
    # contains both NORMAL and PNEUMONIA subfolders. This handles nested
    # duplicate extraction folders (e.g. chest_xray/chest_xray/train).
    for root, dirnames, _ in os.walk(dataset_root):
        # Skip macOS extraction artifacts entirely.
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]

        if os.path.basename(root) == split and _is_valid_split_dir(root):
            return root

    raise FileNotFoundError(
        f"Could not locate a valid '{split}' split directory under "
        f"'{dataset_root}'. Expected to find '{split}/NORMAL' and "
        f"'{split}/PNEUMONIA' folders somewhere under this path."
    )


def _is_valid_split_dir(path: str) -> bool:
    """
    Check whether a given path is a valid split directory, i.e. it exists
    and directly contains both "NORMAL" and "PNEUMONIA" subfolders.

    Args:
        path: Candidate directory path.

    Returns:
        True if path/NORMAL and path/PNEUMONIA both exist as directories.
    """
    if not os.path.isdir(path):
        return False
    return all(os.path.isdir(os.path.join(path, cls)) for cls in CLASS_NAMES)


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the Chest X-Ray Pneumonia classification task.

    Each sample is a single chest X-ray image loaded from disk, converted to
    RGB (3-channel), optionally transformed, and paired with an integer
    label (0 = NORMAL, 1 = PNEUMONIA).

    The dataset indexes the file system once at construction time, building
    an in-memory list of (filepath, label) pairs. This is memory-light since
    only file paths are stored, not pixel data — actual images are loaded
    lazily in __getitem__, which keeps RAM usage low even for the ~5,800
    image dataset.
    """

    def __init__(self, dataset_root: str, split: str, transform: Compose = None) -> None:
        """
        Args:
            dataset_root: Path to the top-level "chest_xray" folder.
            split: One of "train", "val", or "test".
            transform: A torchvision transform pipeline (e.g. from
                utils.transforms) applied to each loaded image. If None,
                images are returned as raw RGB PIL Images converted to
                tensors via no-op (not recommended for training/eval).

        Raises:
            FileNotFoundError: If the split directory or class subfolders
                cannot be located under dataset_root.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform

        self.split_dir: str = _resolve_split_dir(dataset_root, split)
        self.samples: List[Tuple[str, int]] = self._index_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid images found in split '{split}' under "
                f"'{self.split_dir}'. Check that the dataset was extracted "
                f"correctly."
            )

    def _index_samples(self) -> List[Tuple[str, int]]:
        """
        Walk the NORMAL and PNEUMONIA subfolders of the split directory and
        build a list of (absolute_filepath, label) tuples, skipping any
        invalid files (hidden files, resource forks, non-image extensions).

        Returns:
            List of (filepath, label) tuples for every valid image found.
        """
        samples: List[Tuple[str, int]] = []

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.split_dir, class_name)
            label = CLASS_TO_IDX[class_name]

            for filename in sorted(os.listdir(class_dir)):
                if not _is_valid_image_file(filename):
                    continue
                filepath = os.path.join(class_dir, filename)
                if os.path.isfile(filepath):
                    samples.append((filepath, label))

        return samples

    def __len__(self) -> int:
        """Return the total number of valid images in this split."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Load and return a single (image_tensor, label) pair.

        Args:
            index: Index into the internal samples list.

        Returns:
            Tuple of (transformed_image, label) where transformed_image is
            a torch.Tensor of shape (3, 224, 224) and label is an int
            (0 = NORMAL, 1 = PNEUMONIA).
        """
        filepath, label = self.samples[index]

        # Convert to RGB: source X-rays are grayscale, but EfficientNet-B0's
        # pretrained weights expect 3-channel input.
        image = Image.open(filepath).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Compute the number of samples per class in this split. Useful for
        sanity-checking class imbalance before training (the Kaggle
        pneumonia dataset is notably imbalanced toward PNEUMONIA in train).

        Returns:
            Dict mapping class name to sample count, e.g.
            {"NORMAL": 1341, "PNEUMONIA": 3875}.
        """
        counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            class_name = CLASS_NAMES[label]
            counts[class_name] += 1
        return counts


def get_dataloaders(
    dataset_root: str,
    train_transform: Compose,
    eval_transform: Compose,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience factory that builds train, validation, and test DataLoaders
    in one call.

    Args:
        dataset_root: Path to the top-level "chest_xray" folder.
        train_transform: Transform pipeline applied to training images
            (typically includes augmentation — see utils.transforms).
        eval_transform: Transform pipeline applied to validation/test images
            (deterministic, no augmentation).
        batch_size: Number of samples per batch. 32 is a reasonable default
            for CPU training with frozen EfficientNet-B0 features.
        num_workers: Number of subprocess workers for data loading. Defaults
            to 0, which is the safest setting on CPU-only machines (avoids
            multiprocessing overhead/issues on some platforms, e.g. Windows).
            Increase to 2-4 if your machine has spare CPU cores and you want
            faster data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = ChestXrayDataset(dataset_root, split="train", transform=train_transform)
    val_dataset = ChestXrayDataset(dataset_root, split="val", transform=eval_transform)
    test_dataset = ChestXrayDataset(dataset_root, split="test", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

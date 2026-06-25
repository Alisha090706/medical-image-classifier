"""
gradcam.py
==========
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for the
trained EfficientNet-B0 chest X-ray classifier, producing a visual
explanation of which regions of an X-ray most influenced the model's
prediction.

How Grad-CAM works here
------------------------
1. We register a forward hook and a backward hook on the final convolutional
   block of EfficientNet-B0 (`model.features[-1]`), capturing both its
   output feature maps and the gradients that flow into them during
   backpropagation.
2. We run a forward pass and then backpropagate from the logit of the
   *predicted* class (not a fixed class), so the heatmap always explains
   "why did the model predict what it predicted" for that specific image.
3. Grad-CAM weights each feature map channel by the average gradient
   flowing into it (global-average-pooled gradients), then computes a
   weighted sum of the feature maps, followed by a ReLU (we only care about
   features that positively support the predicted class).
4. The resulting low-resolution activation map is resized to 224x224 and
   normalized to [0, 1] to form the heatmap.

Why the last conv block:
EfficientNet-B0's final feature block captures the highest-level, most
semantically meaningful spatial features (as opposed to early layers, which
respond to low-level edges/textures). This is the standard choice for
Grad-CAM across CNN architectures.

Output
------
For a given input image, this module saves two files to outputs/gradcam/:
    - "<name>_heatmap.png"  : the raw Grad-CAM heatmap (colorized).
    - "<name>_overlay.png"  : the heatmap blended on top of the original
      X-ray, which is the more clinically useful visualization.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from predict import load_model_for_inference, preprocess_image
from train import MODEL_SAVE_PATH, get_device
from utils.dataset import CLASS_NAMES
from utils.transforms import IMAGE_SIZE

# Directory where Grad-CAM visualizations are saved.
GRADCAM_OUTPUT_DIR: str = os.path.join("outputs", "gradcam")

# Opacity of the heatmap when blended onto the original image (0 = invisible
# heatmap, 1 = heatmap fully replaces the original image).
OVERLAY_ALPHA: float = 0.45


@dataclass
class GradCAMResult:
    """
    Container for the outputs of a Grad-CAM run on a single image.

    Attributes:
        heatmap: Normalized Grad-CAM activation map, shape (224, 224),
            values in [0, 1], where higher values indicate regions that
            more strongly influenced the predicted class.
        overlay: The heatmap blended onto the original RGB image, shape
            (224, 224, 3), dtype uint8, ready to save or display.
        predicted_class: The class name the Grad-CAM explanation corresponds
            to (the model's predicted class for this image).
        confidence: Softmax probability of the predicted class.
    """

    heatmap: np.ndarray
    overlay: np.ndarray
    predicted_class: str
    confidence: float


def quantify_heatmap_region(heatmap: np.ndarray) -> dict:
    """
    Convert a raw Grad-CAM heatmap into a small set of structured,
    objectively-measured facts about WHERE the activation is concentrated.

    This exists so that downstream consumers (e.g. an LLM-based explainer in
    app.py) describe the heatmap using numbers computed directly from the
    pixel data, rather than guessing from the image — the LLM's job is only
    to phrase these facts in plain language, not to perceive them.

    The heatmap is divided into a 3x3 grid of regions (rows: upper/middle/
    lower, columns: left/center/right). We report which grid cell has the
    single highest average activation, plus a simple "midline ratio" that
    measures how much of the total activation falls in the central column
    versus the two outer (left/right) columns — a high midline ratio can
    indicate the model is focusing on the spine/mediastinum rather than the
    lung fields on either side, which is a known failure pattern worth
    surfacing rather than hiding.

    Args:
        heatmap: Normalized Grad-CAM heatmap, shape (224, 224), values in
            [0, 1].

    Returns:
        Dict with keys:
            - "dominant_region": str, e.g. "upper-left", "center-center".
            - "dominant_region_strength": float in [0, 1], the average
              activation within that dominant grid cell.
            - "midline_ratio": float in [0, 1], fraction of total activation
              energy located in the central column (spine/mediastinum area)
              versus the left+center+right columns combined.
            - "is_midline_dominant": bool, True if midline_ratio > 0.45,
              a simple heuristic threshold for flagging spine-centered
              activation as worth mentioning.
    """
    height, width = heatmap.shape
    row_bounds = [0, height // 3, 2 * height // 3, height]
    col_bounds = [0, width // 3, 2 * width // 3, width]

    row_labels = ["upper", "middle", "lower"]
    col_labels = ["left", "center", "right"]

    region_means = {}
    for row_index, row_label in enumerate(row_labels):
        for col_index, col_label in enumerate(col_labels):
            r_start, r_end = row_bounds[row_index], row_bounds[row_index + 1]
            c_start, c_end = col_bounds[col_index], col_bounds[col_index + 1]
            cell = heatmap[r_start:r_end, c_start:c_end]
            region_means[f"{row_label}-{col_label}"] = float(cell.mean())

    dominant_region = max(region_means, key=region_means.get)
    dominant_region_strength = region_means[dominant_region]

    # Midline ratio: total activation energy in the center column divided
    # by total activation energy across all three columns.
    center_col_start, center_col_end = col_bounds[1], col_bounds[2]
    center_column_energy = float(heatmap[:, center_col_start:center_col_end].sum())
    total_energy = float(heatmap.sum())
    midline_ratio = center_column_energy / total_energy if total_energy > 0 else 0.0

    return {
        "dominant_region": dominant_region,
        "dominant_region_strength": dominant_region_strength,
        "midline_ratio": midline_ratio,
        "is_midline_dominant": midline_ratio > 0.45,
    }


class GradCAM:
    """
    Grad-CAM implementation targeting a specific convolutional layer of a
    given model.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap, predicted_index, confidence = cam.generate(input_tensor)
        cam.remove_hooks()  # always clean up registered hooks when done
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Args:
            model: The trained model to explain (must be in eval mode).
            target_layer: The specific layer (an nn.Module) whose
                activations and gradients will be captured. For
                EfficientNet-B0, this should be `model.features[-1]`, the
                final convolutional block.
        """
        self.model = model
        self.target_layer = target_layer

        self._activations: torch.Tensor = None
        self._gradients: torch.Tensor = None

        # Forward hook captures the layer's output during the forward pass.
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        # Full backward hook captures the gradient flowing into the layer's
        # output during backpropagation.
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(
        self, module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook callback: stores the target layer's output activations."""
        self._activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook callback: stores the gradient of the loss w.r.t. the
        target layer's output."""
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, int, float]:
        """
        Run a forward + backward pass to compute the Grad-CAM heatmap for
        the model's predicted class on the given input.

        Args:
            input_tensor: Preprocessed input image tensor of shape
                (1, 3, 224, 224), already on the correct device.

        Returns:
            Tuple of:
                - heatmap: np.ndarray of shape (224, 224), values in [0, 1].
                - predicted_index: int, the index of the predicted class.
                - confidence: float, softmax probability of that class.
        """
        # Grad-CAM requires gradients, so we explicitly enable gradient
        # tracking here even though inference elsewhere in the project runs
        # under torch.no_grad().
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = probabilities[0, predicted_index].item()

        # Backpropagate from the predicted class's logit. This tells us
        # which spatial regions, if changed, would most affect the model's
        # confidence in the class it actually predicted.
        self.model.zero_grad()
        class_score = logits[0, predicted_index]
        class_score.backward()

        # Activations/gradients shape: (1, num_channels, H, W) — for
        # EfficientNet-B0's final block with a 224x224 input, H = W = 7.
        activations = self._activations[0]  # (num_channels, H, W)
        gradients = self._gradients[0]  # (num_channels, H, W)

        # Global-average-pool the gradients across spatial dimensions to get
        # one importance weight per channel.
        channel_weights = gradients.mean(dim=(1, 2))  # (num_channels,)

        # Weighted sum of activation channels, using the gradient-derived
        # importance weights.
        weighted_activations = activations * channel_weights[:, None, None]
        raw_heatmap = weighted_activations.sum(dim=0)  # (H, W)

        # ReLU: we only care about features that positively contribute to
        # the predicted class, matching the original Grad-CAM formulation.
        raw_heatmap = F.relu(raw_heatmap)

        # Resize from the small feature-map resolution (e.g. 7x7) up to the
        # full input resolution (224x224) for a usable visualization.
        raw_heatmap = raw_heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        resized_heatmap = F.interpolate(
            raw_heatmap, size=IMAGE_SIZE, mode="bilinear", align_corners=False
        )
        resized_heatmap = resized_heatmap.squeeze().detach().cpu().numpy()  # (224, 224)

        # Normalize to [0, 1] for consistent visualization. Guard against a
        # degenerate all-zero heatmap (can happen if the predicted class
        # score has zero gradient w.r.t. this layer in rare edge cases).
        heatmap_max = resized_heatmap.max()
        if heatmap_max > 0:
            resized_heatmap = resized_heatmap / heatmap_max

        return resized_heatmap, predicted_index, confidence

    def remove_hooks(self) -> None:
        """
        Remove the registered forward/backward hooks from the target layer.

        Always call this once you are done generating Grad-CAM explanations
        to avoid leaving stale hooks attached to the model (which would
        otherwise keep capturing activations/gradients on every future
        forward/backward pass, wasting memory).
        """
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap_on_image(heatmap: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap on top of the original RGB image for a
    clinically interpretable visualization.

    Args:
        heatmap: Normalized heatmap, shape (224, 224), values in [0, 1].
        original_rgb: Original image resized to (224, 224, 3), dtype uint8,
            in RGB channel order.

    Returns:
        np.ndarray of shape (224, 224, 3), dtype uint8, RGB order — the
        heatmap colorized and blended onto the original image.
    """
    # Convert the normalized heatmap to an 8-bit single-channel image, then
    # apply a color map (JET: blue = low activation, red = high activation,
    # which is the conventional Grad-CAM color scheme).
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend: overlay = alpha * heatmap + (1 - alpha) * original.
    overlay = cv2.addWeighted(
        colored_heatmap_rgb, OVERLAY_ALPHA, original_rgb, 1 - OVERLAY_ALPHA, gamma=0
    )
    return overlay


def generate_gradcam(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
) -> GradCAMResult:
    """
    Full Grad-CAM pipeline for a single PIL image: preprocesses the image,
    runs Grad-CAM targeting EfficientNet-B0's final convolutional block,
    and produces both the raw heatmap and the blended overlay.

    Args:
        image: A PIL Image of the chest X-ray to explain.
        model: The trained model (in eval mode).
        device: Device the model lives on.

    Returns:
        A GradCAMResult containing the heatmap, overlay, and the prediction
        the explanation corresponds to.
    """
    # Reuse the exact same preprocessing used for prediction, so the
    # explanation corresponds to what the model actually saw.
    input_tensor = preprocess_image(image).to(device)

    # EfficientNet-B0's final convolutional block, as exposed by
    # torchvision's `features` Sequential container. This is the standard
    # Grad-CAM target layer for this architecture.
    target_layer = model.features[-1]

    cam = GradCAM(model, target_layer)
    try:
        heatmap, predicted_index, confidence = cam.generate(input_tensor)
    finally:
        # Ensure hooks are always removed, even if generation raises.
        cam.remove_hooks()

    # Prepare the original image at the same 224x224 resolution used by the
    # model, as a uint8 RGB NumPy array, for blending with the heatmap.
    resized_original = image.convert("RGB").resize(IMAGE_SIZE)
    original_rgb_array = np.array(resized_original)

    overlay = overlay_heatmap_on_image(heatmap, original_rgb_array)

    return GradCAMResult(
        heatmap=heatmap,
        overlay=overlay,
        predicted_class=CLASS_NAMES[predicted_index],
        confidence=confidence,
    )


def save_gradcam_outputs(
    result: GradCAMResult, output_name: str, output_dir: str = GRADCAM_OUTPUT_DIR
) -> Tuple[str, str]:
    """
    Save the Grad-CAM heatmap and overlay images to disk as PNG files.

    Args:
        result: The GradCAMResult to save.
        output_name: Base filename (without extension) to use for the
            saved files, typically derived from the input image's filename.
        output_dir: Directory to save the PNGs into. Created if it does
            not already exist.

    Returns:
        Tuple of (heatmap_path, overlay_path) — the paths the files were
        saved to.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Colorize the standalone heatmap (without blending) for the separate
    # heatmap-only output file.
    heatmap_uint8 = np.uint8(255 * result.heatmap)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    heatmap_path = os.path.join(output_dir, f"{output_name}_heatmap.png")
    overlay_path = os.path.join(output_dir, f"{output_name}_overlay.png")

    # cv2.imwrite expects BGR channel order.
    cv2.imwrite(heatmap_path, colored_heatmap_bgr)
    overlay_bgr = cv2.cvtColor(result.overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)

    return heatmap_path, overlay_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the gradcam.py CLI entry point.

    Returns:
        Parsed argparse.Namespace with fields: image, model_path, output_dir.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM heatmap and overlay for a chest "
        "X-ray image, explaining the trained model's prediction."
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=GRADCAM_OUTPUT_DIR,
        help=f"Directory to save Grad-CAM outputs to (default: '{GRADCAM_OUTPUT_DIR}').",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: loads the model, generates a Grad-CAM explanation for the
    given image, saves the heatmap and overlay to disk, and prints a
    summary.
    """
    args = parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image file not found: '{args.image}'.")

    device = get_device()
    model = load_model_for_inference(args.model_path, device)

    image = Image.open(args.image)
    result = generate_gradcam(image, model, device)

    output_name = os.path.splitext(os.path.basename(args.image))[0]
    heatmap_path, overlay_path = save_gradcam_outputs(result, output_name, args.output_dir)

    print("\n" + "=" * 50)
    print("GRAD-CAM RESULT")
    print("=" * 50)
    print(f"Image              : {args.image}")
    print(f"Predicted class     : {result.predicted_class}")
    print(f"Confidence          : {result.confidence * 100:.2f}%")
    print(f"Heatmap saved to    : {heatmap_path}")
    print(f"Overlay saved to    : {overlay_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def generate(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        logits.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu()

    def close(self) -> None:
        self._forward_handle.remove()


def tensor_to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
    image = image_tensor.detach().cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def save_cam_figure(
    original_image_path: str | Path,
    normalized_tensor: torch.Tensor,
    cam: torch.Tensor,
    probability: float,
    label: int,
    output_path: str | Path,
) -> None:
    original = Image.open(original_image_path).convert("RGB")
    original = original.resize((normalized_tensor.shape[-1], normalized_tensor.shape[-2]))

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_display_image(normalized_tensor))
    axes[1].imshow(cam.numpy(), cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM\nlabel={label}, p={probability:.3f}")
    axes[1].axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

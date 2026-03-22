from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from cxr_project.data.transforms import build_eval_transforms


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_rgb = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip(0.6 * image + 0.4 * heatmap_rgb, 0.0, 1.0)


def compute_gradcam(model, image_tensor: torch.Tensor) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module, inputs, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_forward = model.target_layer.register_forward_hook(forward_hook)
    handle_backward = model.target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(image_tensor)
    logits.sum().backward()

    handle_forward.remove()
    handle_backward.remove()

    activation = activations[-1]
    gradient = gradients[-1]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    denominator = cam.max()
    if denominator > 0:
        cam = cam / denominator
    return cam


def save_gradcam_examples(
    manifest_path: str | Path,
    model,
    output_dir: str | Path,
    image_size: int,
    device: torch.device,
    num_positive: int = 5,
    num_negative: int = 5,
    seed: int = 826,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path)
    test_frame = frame.loc[frame["split"] == "test"].copy()

    positives = test_frame.loc[test_frame["label"] == 1].sample(n=min(num_positive, (test_frame["label"] == 1).sum()), random_state=seed)
    negatives = test_frame.loc[test_frame["label"] == 0].sample(n=min(num_negative, (test_frame["label"] == 0).sum()), random_state=seed)
    selected = pd.concat([positives, negatives], ignore_index=True)

    transform = build_eval_transforms(image_size)
    model.eval()

    for _, row in selected.iterrows():
        original = Image.open(row["image_path"]).convert("RGB").resize((image_size, image_size))
        image_tensor = transform(original).unsqueeze(0).to(device)
        probability = float(model.predict_proba(image_tensor).item())
        heatmap = compute_gradcam(model, image_tensor)

        image_array = np.asarray(original).astype(np.float32) / 255.0
        overlay = _overlay_heatmap(image_array, heatmap)

        figure, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image_array)
        axes[0].set_title("Original")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(overlay)
        axes[2].set_title(f"label={int(row['label'])}, p={probability:.3f}")
        for axis in axes:
            axis.axis("off")

        figure.tight_layout()
        figure.savefig(output_dir / f"{row['dicom_id']}_gradcam.png", dpi=150)
        plt.close(figure)

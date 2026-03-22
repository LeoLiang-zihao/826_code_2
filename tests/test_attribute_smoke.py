from pathlib import Path

import torch

from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.data.synthetic import generate_synthetic_dataset
from cxr_project.models.attribution import GradCAM
from cxr_project.models.classifier import LightningBinaryClassifier


def test_gradcam_generates_heatmap(tmp_path: Path) -> None:
    generate_synthetic_dataset(tmp_path, num_subjects=18, positives_fraction=0.5, seed=4)
    datamodule = ChestXrayDataModule(manifest_path=tmp_path / "manifest.csv", batch_size=2, num_workers=0, image_size=64)
    datamodule.setup()
    batch = next(iter(datamodule.test_dataloader()))

    model = LightningBinaryClassifier(model_name="resnet18", pretrained=False, fine_tune_mode="head_only")
    image_tensor = batch["image"][0:1]
    cam = GradCAM(model, model.target_layer)
    try:
        heatmap = cam.generate(image_tensor)
    finally:
        cam.close()

    assert isinstance(heatmap, torch.Tensor)
    assert tuple(heatmap.shape) == (64, 64)


from pathlib import Path

import torch

from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.data.synthetic import generate_synthetic_dataset
from cxr_project.models.simclr import LightningSimCLR, nt_xent_loss


def test_simclr_training_step_is_finite(tmp_path: Path) -> None:
    generate_synthetic_dataset(tmp_path, num_subjects=18, positives_fraction=0.5, seed=5)
    datamodule = ChestXrayDataModule(
        manifest_path=tmp_path / "manifest.csv",
        batch_size=4,
        num_workers=0,
        image_size=64,
        task_mode="simclr",
    )
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    model = LightningSimCLR(
        model_name="resnet18",
        pretrained=False,
        learning_rate=1e-3,
        weight_decay=1e-4,
        projection_hidden_dim=64,
        projection_dim=32,
        temperature=0.2,
    )
    projection_one = model(batch["view1"])
    projection_two = model(batch["view2"])
    loss = nt_xent_loss(projection_one, projection_two, temperature=0.2)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)

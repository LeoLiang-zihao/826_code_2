from pathlib import Path

from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.data.synthetic import generate_synthetic_dataset


def test_datamodule_returns_expected_batch(tmp_path: Path) -> None:
    generate_synthetic_dataset(tmp_path, num_subjects=18, positives_fraction=0.5, seed=3)
    datamodule = ChestXrayDataModule(manifest_path=tmp_path / "manifest.csv", batch_size=4, num_workers=0, image_size=64)
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    assert tuple(batch["image"].shape[1:]) == (3, 64, 64)
    assert batch["label"].shape[0] == 4

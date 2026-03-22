from pathlib import Path

from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.data.synthetic import generate_synthetic_dataset


def test_simclr_datamodule_returns_two_views(tmp_path: Path) -> None:
    generate_synthetic_dataset(tmp_path, num_subjects=18, positives_fraction=0.5, seed=4)
    datamodule = ChestXrayDataModule(
        manifest_path=tmp_path / "manifest.csv",
        batch_size=4,
        num_workers=0,
        image_size=64,
        task_mode="simclr",
    )
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    assert tuple(batch["view1"].shape[1:]) == (3, 64, 64)
    assert tuple(batch["view2"].shape[1:]) == (3, 64, 64)

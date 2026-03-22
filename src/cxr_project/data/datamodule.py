from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from cxr_project.data.dataset import ChestXrayDataset, SimCLRDataset
from cxr_project.data.transforms import build_eval_transforms, build_simclr_transforms, build_train_transforms


def load_manifest_frame(manifest_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(manifest_path))


class ChestXrayDataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest_path: str | Path,
        batch_size: int = 8,
        num_workers: int = 0,
        image_size: int = 224,
        task_mode: str = "supervised",
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.task_mode = task_mode
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        frame = load_manifest_frame(self.manifest_path)
        train_frame = frame.loc[frame["split"] == "train"].copy()
        val_frame = frame.loc[frame["split"] == "val"].copy()
        test_frame = frame.loc[frame["split"] == "test"].copy()

        if self.task_mode == "simclr":
            self.train_dataset = SimCLRDataset(train_frame, transform=build_simclr_transforms(self.image_size))
        else:
            self.train_dataset = ChestXrayDataset(train_frame, transform=build_train_transforms(self.image_size))
        self.val_dataset = ChestXrayDataset(val_frame, transform=build_eval_transforms(self.image_size))
        self.test_dataset = ChestXrayDataset(test_frame, transform=build_eval_transforms(self.image_size))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

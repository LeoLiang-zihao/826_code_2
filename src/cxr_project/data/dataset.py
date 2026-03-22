from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": float(row["label"]),
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "split": str(row["split"]),
            "image_path": str(row["image_path"]),
        }


class SimCLRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        first_view = self.transform(image)
        second_view = self.transform(image)
        return {
            "view1": first_view,
            "view2": second_view,
            "view_1": first_view,
            "view_2": second_view,
            "label": float(row["label"]),
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "split": str(row["split"]),
            "image_path": str(row["image_path"]),
        }

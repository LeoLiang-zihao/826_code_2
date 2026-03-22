from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_image_path(image_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path | None:
    prefix = f"p{str(subject_id)[:2]}"
    patient_dir = f"p{subject_id}"
    study_dir = f"s{study_id}"
    for relative in [
        Path("files") / prefix / patient_dir / study_dir / f"{dicom_id}.jpg",
        Path(prefix) / patient_dir / study_dir / f"{dicom_id}.jpg",
    ]:
        candidate = (image_root / relative).resolve()
        if candidate.exists():
            return candidate
    return None


def make_patient_splits(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 826,
) -> pd.DataFrame:
    subjects = np.array(sorted(df["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = max(1, int(round(n_subjects * train_fraction)))
    n_val = max(1, int(round(n_subjects * val_fraction)))
    if n_subjects >= 3:
        n_train = min(n_train, n_subjects - 2)
        n_val = min(n_val, n_subjects - n_train - 1)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])

    splits = []
    for subject_id in df["subject_id"]:
        if subject_id in train_subjects:
            splits.append("train")
        elif subject_id in val_subjects:
            splits.append("val")
        else:
            splits.append("test")
    return df.assign(split=splits)


def build_mimic_manifest(
    labels_path: str | Path,
    metadata_path: str | Path,
    image_root: str | Path,
    pathology: str,
    negative_ratio: float = 3.0,
    view_position: str = "PA",
    seed: int = 826,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, compression="infer")
    metadata = pd.read_csv(metadata_path, compression="infer")

    if pathology not in labels.columns:
        raise ValueError(f"Pathology '{pathology}' not found in labels file.")

    merged = labels.merge(metadata, on=["subject_id", "study_id"], how="inner")
    filtered = merged.loc[merged[pathology].isin([0.0, 1.0])].copy()
    filtered = filtered.loc[filtered["ViewPosition"] == view_position].copy()
    filtered["label"] = filtered[pathology].astype(int)

    positives = filtered.loc[filtered["label"] == 1].copy()
    negatives = filtered.loc[filtered["label"] == 0].copy()
    max_negatives = int(len(positives) * negative_ratio)
    if len(positives) > 0 and len(negatives) > max_negatives:
        negatives = negatives.sample(n=max_negatives, random_state=seed, replace=False)

    prepared = pd.concat([positives, negatives], ignore_index=True).sample(frac=1.0, random_state=seed)
    prepared = prepared.reset_index(drop=True)

    root = Path(image_root)
    prepared["image_path"] = prepared.apply(
        lambda row: _resolve_image_path(root, int(row["subject_id"]), int(row["study_id"]), str(row["dicom_id"])),
        axis=1,
    )
    prepared = prepared.loc[prepared["image_path"].notna()].copy()
    if prepared.empty:
        raise ValueError("No rows remained after filtering and image path resolution.")

    prepared["image_path"] = prepared["image_path"].map(str)
    prepared["pathology"] = pathology
    prepared["view_position"] = view_position
    columns = ["subject_id", "study_id", "dicom_id", "image_path", "label", "pathology", "view_position"]
    return make_patient_splits(prepared[columns], seed=seed)


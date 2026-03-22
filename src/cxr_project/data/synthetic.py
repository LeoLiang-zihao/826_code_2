from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def _make_background(rng: np.random.Generator, image_size: int) -> Image.Image:
    noise = rng.normal(loc=115, scale=28, size=(image_size, image_size)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(noise)
    return image.filter(ImageFilter.GaussianBlur(radius=1.0))


def _draw_positive_signal(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    center_x = int(width * rng.uniform(0.38, 0.62))
    center_y = int(height * rng.uniform(0.34, 0.58))
    radius_x = int(width * rng.uniform(0.12, 0.18))
    radius_y = int(height * rng.uniform(0.10, 0.16))
    draw.ellipse(
        [
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        ],
        fill=int(rng.integers(180, 230)),
    )
    return image.filter(ImageFilter.GaussianBlur(radius=2.0))


def _draw_negative_structure(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(2):
        x0 = int(width * rng.uniform(0.2, 0.8))
        y0 = int(height * rng.uniform(0.2, 0.8))
        x1 = x0 + int(width * rng.uniform(-0.08, 0.08))
        y1 = y0 + int(height * rng.uniform(0.08, 0.18))
        draw.line((x0, y0, x1, y1), fill=int(rng.integers(120, 160)), width=2)
    return image.filter(ImageFilter.GaussianBlur(radius=1.2))


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_subjects: int = 36,
    positives_fraction: float = 0.5,
    image_size: int = 192,
    seed: int = 826,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(round(num_subjects * 0.67))
    n_val = int(round(num_subjects * 0.17))
    n_test = num_subjects - n_train - n_val
    split_sizes = {"train": n_train, "val": n_val, "test": n_test}

    rng = np.random.default_rng(seed)
    subject_ids = np.arange(10000000, 10000000 + num_subjects)
    positive_total = int(round(num_subjects * positives_fraction))
    positive_total = min(max(positive_total, 1), num_subjects - 1)
    negative_total = num_subjects - positive_total

    positive_ids = subject_ids[:positive_total].copy()
    negative_ids = subject_ids[positive_total:].copy()
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)

    positive_allocations: dict[str, int] = {}
    assigned_positives = 0
    for split, split_size in split_sizes.items():
        remaining_slots = len(split_sizes) - len(positive_allocations) - 1
        if split_size >= 2 and positive_total >= len(split_sizes):
            minimum = 1
        else:
            minimum = 0
        proposed = int(round(split_size * positives_fraction))
        proposed = max(minimum, proposed)
        upper_bound = positive_total - assigned_positives - remaining_slots
        positive_count = max(minimum, min(proposed, upper_bound, split_size))
        positive_allocations[split] = positive_count
        assigned_positives += positive_count

    if assigned_positives < positive_total:
        positive_allocations["train"] += positive_total - assigned_positives
    elif assigned_positives > positive_total:
        positive_allocations["train"] -= assigned_positives - positive_total

    records: list[dict[str, object]] = []
    positive_cursor = 0
    negative_cursor = 0
    sample_index = 0
    for split, split_size in split_sizes.items():
        positive_count = positive_allocations[split]
        negative_count = split_size - positive_count
        split_subjects = [(subject_id, 1) for subject_id in positive_ids[positive_cursor : positive_cursor + positive_count]]
        split_subjects += [(subject_id, 0) for subject_id in negative_ids[negative_cursor : negative_cursor + negative_count]]
        rng.shuffle(split_subjects)
        positive_cursor += positive_count
        negative_cursor += negative_count

        for subject_id, label in split_subjects:
            study_id = 50000000 + sample_index
            sample_index += 1
            dicom_id = f"synthetic-{study_id}"
            image = _make_background(rng, image_size)
            image = _draw_positive_signal(image, rng) if label == 1 else _draw_negative_structure(image, rng)
            image_path = image_dir / f"{dicom_id}.png"
            image.save(image_path)
            records.append(
                {
                    "subject_id": int(subject_id),
                    "study_id": int(study_id),
                    "dicom_id": dicom_id,
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "split": split,
                    "view_position": "PA",
                    "pathology": "SyntheticOpacity",
                }
            )

    manifest = pd.DataFrame.from_records(records)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest

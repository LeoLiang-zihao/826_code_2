from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from cxr_project.config import load_config
from cxr_project.data.dataset import ChestXrayDataset
from cxr_project.data.transforms import build_eval_transforms
from cxr_project.models.attribution import GradCAM, save_cam_figure
from cxr_project.utils.seed import seed_everything
from cxr_project.workflows import choose_device, ensure_manifest, load_classifier_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Grad-CAM attributions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-positive", type=int, default=5)
    parser.add_argument("--num-negative", type=int, default=5)
    return parser.parse_args()


def sample_examples(frame: pd.DataFrame, split: str, num_positive: int, num_negative: int, seed: int) -> pd.DataFrame:
    split_frame = frame.loc[frame["split"] == split].copy()
    positives = split_frame.loc[split_frame["label"] == 1].sample(n=min(num_positive, (split_frame["label"] == 1).sum()), random_state=seed)
    negatives = split_frame.loc[split_frame["label"] == 0].sample(n=min(num_negative, (split_frame["label"] == 0).sum()), random_state=seed)
    return pd.concat([positives, negatives], ignore_index=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    manifest_path = ensure_manifest(config)
    frame = pd.read_csv(manifest_path)
    sampled = sample_examples(frame, args.split, args.num_positive, args.num_negative, int(config["seed"]))

    dataset = ChestXrayDataset(sampled, transform=build_eval_transforms(int(config["data"]["image_size"])))
    model = load_classifier_checkpoint(args.checkpoint, config["model"])
    device = choose_device(config["trainer"]["accelerator"])
    model.to(device)
    model.eval()
    cam = GradCAM(model, model.target_layer)

    output_dir = Path(config["output_dir"]) / "attributions" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            probability = float(model.predict_proba(image_tensor).detach().cpu().item())
            heatmap = cam.generate(image_tensor)
            output_path = output_dir / f"{sample['dicom_id']}.png"
            save_cam_figure(
                original_image_path=sample["image_path"],
                normalized_tensor=sample["image"],
                cam=heatmap,
                probability=probability,
                label=int(sample["label"]),
                output_path=output_path,
            )
            records.append(
                {
                    "dicom_id": sample["dicom_id"],
                    "label": int(sample["label"]),
                    "probability": probability,
                    "figure_path": str(output_path),
                }
            )
    finally:
        cam.close()

    pd.DataFrame.from_records(records).to_csv(output_dir / "attribution_index.csv", index=False)


if __name__ == "__main__":
    main()

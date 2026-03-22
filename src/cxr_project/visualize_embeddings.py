from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_project.config import load_config
from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.evaluation import plot_embedding_scatter, tsne_project
from cxr_project.models.classifier import LightningBinaryClassifier
from cxr_project.models.simclr import LightningSimCLR
from cxr_project.utils.seed import seed_everything
from cxr_project.workflows import choose_device, ensure_manifest, load_classifier_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize encoder embeddings with t-SNE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-type", choices=["classifier", "simclr"], default="simclr")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    return parser.parse_args()


def extract_embeddings(model, dataloader, device: torch.device, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            features = model.encode(images).detach().cpu().numpy()
            embeddings.append(features)
            labels.append(batch["label"].numpy())
            if sum(chunk.shape[0] for chunk in embeddings) >= max_samples:
                break
    stacked_embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    stacked_labels = np.concatenate(labels, axis=0)[:max_samples]
    return stacked_embeddings, stacked_labels


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    ensure_manifest(config)

    datamodule = ChestXrayDataModule(**config["data"])
    datamodule.setup()

    if args.checkpoint_type == "classifier":
        model = load_classifier_checkpoint(args.checkpoint, config["model"])
    else:
        model = LightningSimCLR.load_from_checkpoint(args.checkpoint)

    device = choose_device(config["trainer"]["accelerator"])
    model.to(device)

    if args.split == "train":
        dataloader = datamodule.train_dataloader()
    elif args.split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    embeddings, labels = extract_embeddings(model, dataloader, device, args.max_samples)
    coords = tsne_project(embeddings, seed=int(config["seed"]))

    output_dir = Path(config["output_dir"]) / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels}).to_csv(output_dir / f"{args.split}_tsne.csv", index=False)
    plot_embedding_scatter(coords, labels, output_dir / f"{args.split}_tsne.png")


if __name__ == "__main__":
    main()

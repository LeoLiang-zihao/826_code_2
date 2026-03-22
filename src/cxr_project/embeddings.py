from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE


def extract_embeddings(encoder, dataloader, device: torch.device, max_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    embedding_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    collected = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            features = encoder(images).cpu().numpy()
            labels = batch["label"].cpu().numpy()

            remaining = max_samples - collected
            if remaining <= 0:
                break

            embedding_batches.append(features[:remaining])
            label_batches.append(labels[:remaining])
            collected += min(len(features), remaining)
            if collected >= max_samples:
                break

    embeddings = np.concatenate(embedding_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    return embeddings, labels


def save_tsne_plot(embeddings: np.ndarray, labels: np.ndarray, output_dir: str | Path, seed: int = 826) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = TSNE(n_components=2, init="random", learning_rate="auto", random_state=seed).fit_transform(embeddings)
    frame = pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1], "label": labels.astype(int)})
    frame.to_csv(output_dir / "embeddings.csv", index=False)

    figure, axis = plt.subplots(figsize=(6, 5))
    for label_value in sorted(frame["label"].unique()):
        subset = frame.loc[frame["label"] == label_value]
        axis.scatter(subset["x"], subset["y"], label=f"label={label_value}", alpha=0.8)
    axis.set_title("t-SNE of Encoder Embeddings")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "tsne.png", dpi=150)
    plt.close(figure)

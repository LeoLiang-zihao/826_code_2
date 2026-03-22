from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve


def bootstrap_auc_ap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 826,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    aps: list[float] = []
    n_samples = len(y_true)
    if len(np.unique(y_true)) < 2:
        return {"auroc_ci95": None, "average_precision_ci95": None}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        sample_true = y_true[indices]
        sample_score = y_score[indices]
        if len(np.unique(sample_true)) < 2:
            continue
        aurocs.append(float(roc_auc_score(sample_true, sample_score)))
        aps.append(float(average_precision_score(sample_true, sample_score)))

    def percentile_interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        low, high = np.percentile(values, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "auroc_ci95": percentile_interval(aurocs),
        "average_precision_ci95": percentile_interval(aps),
    }


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 0, seed: int = 826) -> dict[str, float | list[float] | None]:
    positive_rate = float(np.mean(y_true))
    metrics: dict[str, float | list[float] | None] = {
        "positive_rate": positive_rate,
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }
    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = float("nan")
        metrics["average_precision"] = positive_rate
        metrics["auroc_ci95"] = None
        metrics["average_precision_ci95"] = None
        return metrics

    metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    if n_bootstrap > 0:
        metrics.update(bootstrap_auc_ap_ci(y_true, y_score, n_bootstrap=n_bootstrap, seed=seed))
    else:
        metrics["auroc_ci95"] = None
        metrics["average_precision_ci95"] = None
    return metrics


def save_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_predictions(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def plot_curves(predictions: pd.DataFrame, output_path: str | Path) -> None:
    y_true = predictions["label"].to_numpy()
    y_score = predictions["probability"].to_numpy()
    if len(np.unique(y_true)) < 2:
        return

    roc_x, roc_y, _ = roc_curve(y_true, y_score)
    pr_x, pr_y, _ = precision_recall_curve(y_true, y_score)
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=5, strategy="uniform")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(roc_x, roc_y)
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")

    axes[1].plot(pr_x, pr_y)
    axes[1].set_title("Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    axes[2].plot(mean_pred, frac_pos, marker="o")
    axes[2].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[2].set_title("Calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction positives")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_training_curves(metrics_csv_path: str | Path, output_path: str | Path) -> None:
    frame = pd.read_csv(metrics_csv_path)
    figure, axis = plt.subplots(figsize=(6, 4))

    train_frame = pd.DataFrame(columns=["epoch", "train_loss_epoch"])
    if "train_loss_epoch" in frame.columns:
        train_frame = frame.loc[frame["train_loss_epoch"].notna(), ["epoch", "train_loss_epoch"]].drop_duplicates()

    val_frame = pd.DataFrame(columns=["epoch", "val_loss"])
    if "val_loss" in frame.columns:
        val_frame = frame.loc[frame["val_loss"].notna(), ["epoch", "val_loss"]].drop_duplicates()
    if not train_frame.empty:
        axis.plot(train_frame["epoch"], train_frame["train_loss_epoch"], marker="o", label="train_loss")
    if not val_frame.empty:
        axis.plot(val_frame["epoch"], val_frame["val_loss"], marker="o", label="val_loss")
    axis.set_title("Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    if not train_frame.empty or not val_frame.empty:
        axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def tsne_project(embeddings: np.ndarray, seed: int = 826) -> np.ndarray:
    if len(embeddings) < 3:
        coordinates = np.zeros((len(embeddings), 2), dtype=float)
        if len(embeddings) > 0:
            coordinates[:, 0] = np.arange(len(embeddings))
        return coordinates
    perplexity = max(2, min(30, len(embeddings) - 1))
    reducer = TSNE(n_components=2, init="random", random_state=seed, perplexity=perplexity)
    return reducer.fit_transform(embeddings)


def plot_embedding_scatter(coordinates: np.ndarray, labels: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="coolwarm", alpha=0.85)
    axis.set_title("t-SNE of Encoder Embeddings")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    legend = axis.legend(*scatter.legend_elements(), title="Label")
    axis.add_artist(legend)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

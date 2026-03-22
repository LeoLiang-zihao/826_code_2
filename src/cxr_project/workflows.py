from __future__ import annotations

from pathlib import Path
from typing import Iterable

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from cxr_project.data.synthetic import generate_synthetic_dataset
from cxr_project.evaluation import compute_binary_metrics, plot_curves, plot_training_curves, save_metrics, save_predictions
from cxr_project.models.classifier import LightningBinaryClassifier


def ensure_manifest(config: dict) -> Path:
    manifest_path = Path(config["data"]["manifest_path"])
    if manifest_path.exists():
        return manifest_path
    if not config["runtime"].get("create_manifest_if_missing", False):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    synthetic_config = config["runtime"]["synthetic"]
    generate_synthetic_dataset(
        output_dir=manifest_path.parent,
        num_subjects=int(synthetic_config["num_subjects"]),
        positives_fraction=float(synthetic_config["positives_fraction"]),
        image_size=int(config["data"]["image_size"]) * 2,
        seed=int(synthetic_config["seed"]),
    )
    return manifest_path


def build_csv_logger(output_dir: Path, name: str = "logs") -> CSVLogger:
    return CSVLogger(save_dir=str(output_dir), name=name)


def build_checkpoint_callback(output_dir: Path, monitor: str = "val_auroc", mode: str = "max") -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        save_top_k=1,
        monitor=monitor,
        mode=mode,
    )


def build_early_stopping(monitor: str = "val_auroc", mode: str = "max", patience: int = 5) -> EarlyStopping:
    return EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=True)


def build_trainer(config: dict, output_dir: Path, logger: CSVLogger, callbacks: Iterable) -> L.Trainer:
    return L.Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        max_epochs=config["trainer"]["max_epochs"],
        precision=config["trainer"]["precision"],
        deterministic=config["trainer"]["deterministic"],
        limit_train_batches=config["trainer"]["limit_train_batches"],
        limit_val_batches=config["trainer"].get("limit_val_batches", 1.0),
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        logger=logger,
        callbacks=list(callbacks),
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )


def collect_predictions(
    model: L.LightningModule,
    dataloader,
    device: torch.device,
    split: str,
    tta: bool = False,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)

            if tta:
                flipped = torch.flip(images, dims=[-1])
                logits_flip = model(flipped)
                probabilities = torch.sigmoid((logits + logits_flip) / 2).cpu().numpy()
            else:
                probabilities = torch.sigmoid(logits).cpu().numpy()

            labels = batch["label"].cpu().numpy()
            for index, probability in enumerate(probabilities):
                rows.append(
                    {
                        "subject_id": int(batch["subject_id"][index]),
                        "study_id": int(batch["study_id"][index]),
                        "dicom_id": str(batch["dicom_id"][index]),
                        "split": split,
                        "label": int(labels[index]),
                        "probability": float(probability),
                        "image_path": str(batch["image_path"][index]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    n_bootstrap: int = 0,
    seed: int = 826,
) -> dict[str, object]:
    metrics = compute_binary_metrics(
        predictions["label"].to_numpy(),
        predictions["probability"].to_numpy(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    save_predictions(predictions, output_dir / "predictions" / f"{split_name}_predictions.csv")
    save_metrics(metrics, output_dir / "metrics" / f"{split_name}_metrics.json")
    plot_curves(predictions, output_dir / "plots" / f"{split_name}_curves.png")
    return metrics


def load_classifier_checkpoint(checkpoint_path: str | Path, model_config: dict) -> LightningBinaryClassifier:
    return LightningBinaryClassifier.load_from_checkpoint(
        checkpoint_path,
        model_name=model_config["name"],
        pretrained=bool(model_config["pretrained"]),
        fine_tune_mode=model_config["fine_tune_mode"],
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        loss_type=model_config.get("loss_type", "bce"),
        focal_gamma=float(model_config.get("focal_gamma", 2.0)),
        focal_alpha=float(model_config.get("focal_alpha", 0.75)),
        pos_weight=float(model_config.get("pos_weight", 1.0)),
        label_smoothing=float(model_config.get("label_smoothing", 0.0)),
        backbone_lr_factor=float(model_config.get("backbone_lr_factor", 1.0)),
    )


def load_encoder_from_simclr_checkpoint(model: LightningBinaryClassifier, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        key.replace("encoder.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    model._apply_fine_tuning("head_only")


def choose_device(accelerator: str) -> torch.device:
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_loss_plot(logger: CSVLogger, output_dir: Path) -> None:
    metrics_csv = Path(logger.log_dir) / "metrics.csv"
    if metrics_csv.exists():
        plot_training_curves(metrics_csv, output_dir / "plots" / "loss_curves.png")

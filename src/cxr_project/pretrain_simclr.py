from __future__ import annotations

import argparse
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from cxr_project.config import load_config
from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.models.simclr import LightningSimCLR
from cxr_project.utils.seed import seed_everything
from cxr_project.workflows import build_csv_logger, build_trainer, ensure_manifest, save_loss_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a SimCLR encoder.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    ensure_manifest(config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    simclr_data_config = dict(config["data"])
    simclr_data_config["task_mode"] = "simclr"
    datamodule = ChestXrayDataModule(**simclr_data_config)
    model = LightningSimCLR(
        model_name=config["model"]["name"],
        pretrained=bool(config["model"]["pretrained"]),
        learning_rate=float(config["simclr"]["learning_rate"]),
        weight_decay=float(config["simclr"]["weight_decay"]),
        projection_hidden_dim=int(config["simclr"]["projection_hidden_dim"]),
        projection_dim=int(config["simclr"]["projection_dim"]),
        temperature=float(config["simclr"]["temperature"]),
    )

    logger = build_csv_logger(output_dir, name="simclr_logs")
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="simclr-best",
        save_top_k=1,
        monitor="train_loss_epoch",
        mode="min",
        save_last=True,
    )
    trainer = build_trainer(config, output_dir, logger, [checkpoint])
    trainer.fit(model, datamodule=datamodule)
    save_loss_plot(logger, output_dir)


if __name__ == "__main__":
    main()

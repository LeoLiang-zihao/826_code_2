from __future__ import annotations

import argparse
from pathlib import Path

from cxr_project.config import load_config
from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.evaluation import save_metrics
from cxr_project.models.classifier import LightningBinaryClassifier
from cxr_project.utils.seed import seed_everything
from cxr_project.workflows import (
    build_checkpoint_callback,
    build_csv_logger,
    build_early_stopping,
    build_trainer,
    choose_device,
    collect_predictions,
    ensure_manifest,
    load_classifier_checkpoint,
    load_encoder_from_simclr_checkpoint,
    save_loss_plot,
    summarize_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-encoder linear evaluation from a SimCLR checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--simclr-checkpoint", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    ensure_manifest(config)

    output_dir = Path(config["output_dir"]) / "linear_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = ChestXrayDataModule(**config["data"])
    model = LightningBinaryClassifier(
        model_name=config["model"]["name"],
        pretrained=False,
        fine_tune_mode="head_only",
        learning_rate=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    load_encoder_from_simclr_checkpoint(model, args.simclr_checkpoint)

    logger = build_csv_logger(output_dir, name="linear_eval_logs")
    checkpoint = build_checkpoint_callback(output_dir, monitor="val_auroc", mode="max")

    callbacks = [checkpoint]
    patience = config["trainer"].get("early_stopping_patience", 0)
    if patience > 0:
        callbacks.append(build_early_stopping(monitor="val_auroc", mode="max", patience=patience))

    trainer = build_trainer(config, output_dir, logger, callbacks)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    best_path = checkpoint.best_model_path or None
    if best_path:
        model = load_classifier_checkpoint(
            best_path,
            {
                **config["model"],
                "pretrained": False,
                "fine_tune_mode": "head_only",
            },
        )
    device = choose_device(config["trainer"]["accelerator"])
    model.to(device)
    datamodule.setup()

    use_tta = config.get("inference", {}).get("tta", False)
    val_predictions = collect_predictions(model, datamodule.val_dataloader(), device, "val", tta=use_tta)
    test_predictions = collect_predictions(model, datamodule.test_dataloader(), device, "test", tta=use_tta)
    metrics = {
        "val": summarize_predictions(val_predictions, output_dir, "val", n_bootstrap=args.bootstrap, seed=int(config["seed"])),
        "test": summarize_predictions(test_predictions, output_dir, "test", n_bootstrap=args.bootstrap, seed=int(config["seed"])),
    }
    save_metrics(metrics, output_dir / "metrics" / "summary.json")
    save_loss_plot(logger, output_dir)


if __name__ == "__main__":
    main()

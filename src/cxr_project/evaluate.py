from __future__ import annotations

import argparse
from pathlib import Path

from cxr_project.config import load_config
from cxr_project.data.datamodule import ChestXrayDataModule
from cxr_project.evaluation import save_metrics
from cxr_project.utils.seed import seed_everything
from cxr_project.workflows import (
    choose_device,
    collect_predictions,
    ensure_manifest,
    load_classifier_checkpoint,
    summarize_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    ensure_manifest(config)

    output_dir = Path(config["output_dir"]) / "evaluation"
    datamodule = ChestXrayDataModule(**config["data"])
    datamodule.setup()
    model = load_classifier_checkpoint(args.checkpoint, config["model"])
    device = choose_device(config["trainer"]["accelerator"])
    model.to(device)

    val_predictions = collect_predictions(model, datamodule.val_dataloader(), device, "val")
    test_predictions = collect_predictions(model, datamodule.test_dataloader(), device, "test")
    metrics = {
        "val": summarize_predictions(val_predictions, output_dir, "val", n_bootstrap=args.bootstrap, seed=int(config["seed"])),
        "test": summarize_predictions(test_predictions, output_dir, "test", n_bootstrap=args.bootstrap, seed=int(config["seed"])),
    }
    save_metrics(metrics, output_dir / "metrics" / "summary.json")


if __name__ == "__main__":
    main()


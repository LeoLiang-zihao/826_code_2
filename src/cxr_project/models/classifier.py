from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from cxr_project.models.backbones import build_backbone


class LightningBinaryClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        fine_tune_mode: str = "head_only",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        encoder_checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        bundle = build_backbone(model_name, pretrained)
        self.encoder = bundle.encoder
        self.encoder_stages = bundle.stages
        self.target_layer = bundle.target_layer
        self.head = nn.Linear(bundle.feature_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_auroc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_ap = BinaryAveragePrecision()
        if encoder_checkpoint_path is not None:
            self._load_encoder_checkpoint(encoder_checkpoint_path)
        self._apply_fine_tuning(fine_tune_mode)

    def _freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def _load_encoder_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key.removeprefix("encoder.")] = value
        if not encoder_state:
            raise ValueError(f"No encoder weights found in checkpoint: {checkpoint_path}")
        self.encoder.load_state_dict(encoder_state, strict=True)

    def _apply_fine_tuning(self, mode: str) -> None:
        self._freeze_encoder()
        for parameter in self.head.parameters():
            parameter.requires_grad = True

        if mode == "head_only":
            return
        if mode == "last1":
            stages_to_unfreeze = 1
        elif mode == "last2":
            stages_to_unfreeze = 2
        elif mode == "full":
            stages_to_unfreeze = len(self.encoder_stages)
        else:
            raise ValueError(f"Unsupported fine_tune_mode: {mode}")

        for stage in self.encoder_stages[:stages_to_unfreeze]:
            for parameter in stage.parameters():
                parameter.requires_grad = True

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode(images)
        return self.head(features).squeeze(-1)

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(images))

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"].float()
        logits = self(images)
        probabilities = torch.sigmoid(logits)
        loss = self.loss_fn(logits, labels)
        batch_size = int(labels.shape[0])

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"), batch_size=batch_size)
        if stage == "val":
            self.val_auroc.update(probabilities, labels.int())
            self.val_ap.update(probabilities, labels.int())
        elif stage == "test":
            self.test_auroc.update(probabilities, labels.int())
            self.test_ap.update(probabilities, labels.int())
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        if getattr(self.val_auroc, "_update_count", 0) > 0:
            self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
            self.log("val_ap", self.val_ap.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_ap.reset()

    def on_test_epoch_end(self) -> None:
        if getattr(self.test_auroc, "_update_count", 0) > 0:
            self.log("test_auroc", self.test_auroc.compute())
            self.log("test_ap", self.test_ap.compute())
        self.test_auroc.reset()
        self.test_ap.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.Adam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

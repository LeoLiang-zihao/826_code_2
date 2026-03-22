from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from cxr_project.models.backbones import build_backbone


class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class LightningBinaryClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        fine_tune_mode: str = "head_only",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        encoder_checkpoint_path: str | None = None,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
        backbone_lr_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        bundle = build_backbone(model_name, pretrained)
        self.encoder = bundle.encoder
        self.encoder_stages = bundle.stages
        self.target_layer = bundle.target_layer
        self.head = nn.Linear(bundle.feature_dim, 1)

        if loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]) if pos_weight != 1.0 else None
            )

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

        smoothing = float(self.hparams.label_smoothing)
        if smoothing > 0 and stage == "train":
            smooth_labels = labels * (1 - smoothing) + 0.5 * smoothing
        else:
            smooth_labels = labels

        loss = self.loss_fn(logits, smooth_labels)
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

    def configure_optimizers(self) -> dict:
        backbone_lr_factor = float(self.hparams.backbone_lr_factor)
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())

        if backbone_params and backbone_lr_factor != 1.0:
            param_groups = [
                {"params": backbone_params, "lr": self.hparams.learning_rate * backbone_lr_factor},
                {"params": head_params, "lr": self.hparams.learning_rate},
            ]
        else:
            param_groups = [{"params": backbone_params + head_params}]

        optimizer = torch.optim.AdamW(
            param_groups, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

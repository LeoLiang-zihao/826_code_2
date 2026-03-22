from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from cxr_project.models.backbones import build_backbone


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity = representations @ representations.T
    logits = similarity / temperature
    logits = logits.masked_fill(torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool), float("-inf"))

    labels = torch.arange(batch_size, device=logits.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    return F.cross_entropy(logits, labels)


class LightningSimCLR(L.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        projection_hidden_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        bundle = build_backbone(model_name, pretrained)
        self.encoder = bundle.encoder
        self.target_layer = bundle.target_layer
        self.projection_head = ProjectionHead(bundle.feature_dim, projection_hidden_dim, projection_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encode(images))

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        view1 = batch["view1"] if "view1" in batch else batch["view_1"]
        view2 = batch["view2"] if "view2" in batch else batch["view_2"]
        z1 = self(view1)
        z2 = self(view2)
        loss = nt_xent_loss(z1, z2, temperature=float(self.hparams.temperature))
        if getattr(self, "_trainer", None) is not None:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=int(view1.shape[0]))
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

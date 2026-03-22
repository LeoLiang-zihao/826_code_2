from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision import models


@dataclass
class BackboneBundle:
    encoder: nn.Module
    feature_dim: int
    stages: list[nn.Module]
    target_layer: nn.Module


def build_backbone(name: str, pretrained: bool) -> BackboneBundle:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        encoder = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        encoder = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    feature_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()
    stages = [encoder.layer4, encoder.layer3, encoder.layer2, encoder.layer1]
    return BackboneBundle(
        encoder=encoder,
        feature_dim=feature_dim,
        stages=stages,
        target_layer=encoder.layer4,
    )

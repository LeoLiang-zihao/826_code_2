from __future__ import annotations

import random

import lightning as L
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


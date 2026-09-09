from __future__ import annotations

import os
import random
from typing import Any, Dict
import numpy as np
import torch
import yaml

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    denom = int(mask.sum().item())
    if denom == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=-1)
    return float((pred == labels[mask]).float().mean().item())

def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable per-node predictive entropy.

    Do not clamp before negation: sum(p log p) is non-positive, while entropy is
    its negative. Clamping the non-positive term first makes the loss constant.
    """
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent.clamp_min(float(eps))

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import nn
from .curvature import CurvatureStats, median_shift_gate
from .utils import accuracy

@dataclass
class AdaptResult:
    model: nn.Module | None
    acc: float
    loss_last: float
    extra: Dict[str, float]

def select_adapt_mask(data, mode: str = "test") -> torch.Tensor:
    if mode == "test":
        return data.test_mask.bool()
    if mode == "all":
        return torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
    raise ValueError(f"Unknown adapt mask mode: {mode}")

def freeze_except_head(model: nn.Module, head_only: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad_(not head_only)
    if head_only:
        if not hasattr(model, "head"):
            raise AttributeError("adapt_head_only=True requires model.head")
        for p in model.head.parameters():
            p.requires_grad_(True)

def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

@torch.no_grad()
def evaluate(model: nn.Module, data, edge_index: torch.Tensor) -> Tuple[float, torch.Tensor]:
    model.eval()
    logits = model(data.x, edge_index)
    return accuracy(logits, data.y, data.test_mask), logits

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return accuracy(logits, labels, mask)

def get_source_outputs(model: nn.Module, data, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target-graph features, logits, and posterior from a frozen source model."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, "encode"):
            feats = model.encode(data.x, edge_index)
            logits = model.head(feats) if hasattr(model, "head") else model(data.x, edge_index)
        else:
            logits = model(data.x, edge_index)
            feats = logits
        probs = torch.softmax(logits, dim=-1)
    return feats.detach(), logits.detach(), probs.detach()

def _weighted_mean_by_sum_weights(values: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    """Gate-weighted average used in the primary interface.

    This is intentionally sum(w * loss) / sum(w), not mean(w * loss).
    """
    return (weights * values).sum() / weights.sum().clamp_min(float(eps))

@torch.no_grad()
def _distance_logits_to_prototypes(logits: torch.Tensor, prototypes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between node logits and class prototypes."""
    logits_n = torch.nn.functional.normalize(logits, p=2, dim=-1, eps=float(eps))
    proto_n = torch.nn.functional.normalize(prototypes, p=2, dim=-1, eps=float(eps))
    return logits_n @ proto_n.t()

def _cga_gate(src_curv, tgt_curv, device, alpha, eps, gate_mode="median_shift"):
    """Matched source-target curvature-change gate for the primary protocol."""
    if gate_mode not in ("median_shift", "node_shift"):
        raise ValueError("The primary evaluation uses the matched median-shift gate.")
    return median_shift_gate(
        src_curv.node_kappa.to(device), tgt_curv.node_kappa.to(device),
        alpha=alpha, eps=eps,
    ).detach()

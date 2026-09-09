from __future__ import annotations

from typing import Dict, List, Tuple
import torch

def parse_gate_power(name: str) -> float:
    # power_0p5 -> 0.5; power_2p0 -> 2.0
    raw = name.split("power_", 1)[1]
    return float(raw.replace("p", "."))

def transform_gate(gate: torch.Tensor, mode: str, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mode = str(mode)
    g = gate.clamp(0.0, 1.0)
    if mode in ("identity", "raw", "none"):
        return g.detach()
    if mode.startswith("power_"):
        gamma = parse_gate_power(mode)
        return g.clamp_min(float(eps)).pow(gamma).clamp(0.0, 1.0).detach()
    if mode.startswith("rank_"):
        # rank_20_80 means bottom 20% -> 0, top 20% -> 1, linear between.
        try:
            _, lo_s, hi_s = mode.split("_")
            lo_q = float(lo_s) / 100.0
            hi_q = float(hi_s) / 100.0
        except Exception as e:
            raise ValueError(f"Invalid rank gate transform: {mode}") from e
        gm = g[mask]
        if gm.numel() == 0:
            return g.detach()
        qlo = torch.quantile(gm, lo_q)
        qhi = torch.quantile(gm, hi_q)
        out = (g - qlo) / (qhi - qlo).clamp_min(float(eps))
        return out.clamp(0.0, 1.0).detach()
    raise ValueError(f"Unknown gate transform: {mode}")

def support_score_from_mode(conf: torch.Tensor, gate: torch.Tensor, mode: str, eps: float = 1e-8) -> torch.Tensor:
    mode = str(mode)
    g = gate.clamp(0.0, 1.0)
    if mode == "confidence":
        return conf
    if mode == "confidence_x_sqrt_gate":
        return conf * g.clamp_min(float(eps)).sqrt()
    if mode == "confidence_x_gate":
        return conf * g
    if mode == "confidence_x_gate2":
        return conf * g.pow(2.0)
    if mode == "gate_only":
        return g
    raise ValueError(f"Unknown support score mode: {mode}")

@torch.no_grad()
def build_prototypes_by_score(
    logits: torch.Tensor,
    probs: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    confidence_quantile: float,
    support_score_mode: str,
    gate_for_score: torch.Tensor | None = None,
    gate_for_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build class prototypes with optional gate-aware support selection and weighting.

    - gate_for_weight=None uses unweighted vanilla prototypes.
    - For CGA, the support score also supplies the prototype averaging weight.
    """
    device = logits.device
    conf, pred = probs.max(dim=-1)
    if gate_for_score is None:
        gate_for_score = torch.ones_like(conf)
    score = support_score_from_mode(conf, gate_for_score, support_score_mode, eps=eps)

    score_test = score[mask]
    if score_test.numel() == 0:
        raise ValueError("T3A support selection needs at least one node in the adaptation/test mask")
    threshold = torch.quantile(score_test, float(confidence_quantile))
    selected_mask = mask.bool() & (score >= threshold)
    if int(selected_mask.sum().item()) == 0:
        selected_mask = mask.bool()

    prototypes = torch.zeros((num_classes, logits.size(-1)), device=device, dtype=logits.dtype)
    global_proto = logits[mask].mean(dim=0)
    support_counts: List[float] = []
    support_weight_sums: List[float] = []

    for k in range(num_classes):
        class_mask = selected_mask & (pred == k)
        if int(class_mask.sum().item()) == 0:
            class_mask = mask & (pred == k)
        if int(class_mask.sum().item()) == 0:
            prototypes[k] = global_proto
            support_counts.append(0.0)
            support_weight_sums.append(0.0)
            continue

        feats = logits[class_mask]
        if gate_for_weight is None:
            w = torch.ones(feats.size(0), device=device, dtype=logits.dtype)
        else:
            w = score[class_mask].to(dtype=logits.dtype).clamp_min(float(eps))
        prototypes[k] = (w.unsqueeze(-1) * feats).sum(dim=0) / w.sum().clamp_min(float(eps))
        support_counts.append(float(class_mask.sum().item()))
        support_weight_sums.append(float(w.sum().item()))

    extra = {
        "support_threshold": float(threshold.item()),
        "selected_frac_all": float(selected_mask.float().mean().item()),
        "selected_frac_adapt": float((selected_mask & mask).float().sum().item() / max(1, int(mask.sum().item()))),
        "min_support": float(min(support_counts)) if support_counts else float("nan"),
        "max_support": float(max(support_counts)) if support_counts else float("nan"),
        "min_support_weight": float(min(support_weight_sums)) if support_weight_sums else float("nan"),
        "max_support_weight": float(max(support_weight_sums)) if support_weight_sums else float("nan"),
    }
    return prototypes, extra

from __future__ import annotations

from copy import deepcopy
import torch
import torch.nn.functional as F
from .tta import freeze_except_head, trainable_params, select_adapt_mask, _weighted_mean_by_sum_weights, evaluate
from .utils import entropy_from_logits

def run_tent_like(
    source_model,
    data,
    edge_index_tgt: torch.Tensor,
    p_src: torch.Tensor,
    gate: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
    adapt_head_only: bool,
    mask_mode: str,
    eps: float,
    variant: str,
    gamma_ent: float = 1.0,
    gamma_anchor: float = 1.0,
    lambda_anchor: float = 4.0,
    lr_scale: float = 1.0,
):
    """Run vanilla/gate-only/anchor-only/full TENT adaptation.

    variant:
      - vanilla: unweighted entropy, no source anchor
      - gate_only: gate-weighted entropy, no source anchor
      - anchor_only: unweighted entropy + source posterior anchor weighted by (1-g)^gamma_anchor
      - cga: gate^gamma_ent weighted entropy + source anchor weighted by (1-g)^gamma_anchor
    """
    device = data.x.device
    model = deepcopy(source_model)
    model.eval()
    freeze_except_head(model, adapt_head_only)
    params = trainable_params(model)
    if not params:
        raise RuntimeError("No trainable parameters for TENT-like adaptation")
    opt = torch.optim.Adam(params, lr=float(lr) * float(lr_scale), weight_decay=float(weight_decay))
    mask = select_adapt_mask(data, mask_mode).to(device)
    g = gate.clamp(0.0, 1.0).detach()

    if variant == "vanilla":
        w_ent = torch.ones_like(g)
        w_anchor = torch.zeros_like(g)
        lam = 0.0
    elif variant == "gate_only":
        w_ent = g.clamp_min(float(eps)).pow(float(gamma_ent))
        w_anchor = torch.zeros_like(g)
        lam = 0.0
    elif variant == "anchor_only":
        w_ent = torch.ones_like(g)
        w_anchor = (1.0 - g).clamp_min(0.0).pow(float(gamma_anchor))
        lam = float(lambda_anchor)
    elif variant == "cga":
        w_ent = g.clamp_min(float(eps)).pow(float(gamma_ent))
        w_anchor = (1.0 - g).clamp_min(0.0).pow(float(gamma_anchor))
        lam = float(lambda_anchor)
    else:
        raise ValueError(f"unknown TENT variant: {variant}")

    w_ent_m = w_ent[mask]
    w_anchor_m = w_anchor[mask]

    last_total = 0.0
    last_ent = 0.0
    last_anchor = 0.0
    with torch.no_grad():
        logits_before = model(data.x, edge_index_tgt).detach()
        pred_before = logits_before.argmax(dim=-1)
        ent_before = float(entropy_from_logits(logits_before, eps=eps)[data.test_mask].mean().item())

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(data.x, edge_index_tgt)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        entropy = -(p * logp).sum(dim=-1)
        loss_ent = _weighted_mean_by_sum_weights(entropy[mask], w_ent_m, eps)
        if lam > 0:
            kl_per_node = F.kl_div(logp, p_src, reduction="none").sum(dim=-1)
            loss_anchor = _weighted_mean_by_sum_weights(kl_per_node[mask], w_anchor_m, eps)
        else:
            loss_anchor = torch.zeros((), device=device)
        loss = loss_ent + lam * loss_anchor
        loss.backward()
        opt.step()
        last_total = float(loss.detach().item())
        last_ent = float(loss_ent.detach().item())
        last_anchor = float(loss_anchor.detach().item())

    acc, logits_after = evaluate(model, data, edge_index_tgt)
    with torch.no_grad():
        ent_after = float(entropy_from_logits(logits_after, eps=eps)[data.test_mask].mean().item())
        pred_after = logits_after.argmax(dim=-1)
        flip = float((pred_after[data.test_mask] != pred_before[data.test_mask]).float().mean().item())
    extra = {
        f"{variant}_loss_last": last_total,
        f"{variant}_entropy_last": last_ent,
        f"{variant}_anchor_last": last_anchor,
        f"{variant}_entropy_before_test": ent_before,
        f"{variant}_entropy_after_test": ent_after,
        f"{variant}_pred_flip_rate_test": flip,
    }
    return acc, extra

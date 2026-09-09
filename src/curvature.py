from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np
import torch
from torch_geometric.utils import to_undirected


@dataclass
class CurvatureStats:
    node_kappa: torch.Tensor
    mean_train: float
    std_train: float
    mean_all: float
    std_all: float


def adjacency_sets(edge_index: torch.Tensor, num_nodes: int) -> List[Set[int]]:
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    row = edge_index[0].detach().cpu().tolist()
    col = edge_index[1].detach().cpu().tolist()
    for u, v in zip(row, col):
        if u != v:
            adj[u].add(v)
    return adj


def node_forman_curvature(edge_index: torch.Tensor, num_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    """Degree-normalized Forman curvature used in the primary protocol."""
    adj = adjacency_sets(edge_index, num_nodes)
    deg = [len(s) for s in adj]
    sums = np.zeros(num_nodes, dtype=np.float64)
    counts = np.zeros(num_nodes, dtype=np.int64)

    for u in range(num_nodes):
        du = deg[u]
        if du == 0:
            continue
        for v in adj[u]:
            if u < v:
                dv = deg[v]
                if dv == 0:
                    continue
                if du <= dv:
                    common = sum((w in adj[v]) for w in adj[u])
                else:
                    common = sum((w in adj[u]) for w in adj[v])
                k = (4.0 - float(du) - float(dv) + 3.0 * float(common)) / np.sqrt(float(du * dv))
                sums[u] += k
                sums[v] += k
                counts[u] += 1
                counts[v] += 1

    out = np.zeros(num_nodes, dtype=np.float32)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    t = torch.from_numpy(out)
    return t.to(device) if device is not None else t


def curvature_stats(edge_index: torch.Tensor, num_nodes: int, train_mask: torch.Tensor, eps: float = 1e-8, device: torch.device | None = None) -> CurvatureStats:
    kappa = node_forman_curvature(edge_index, num_nodes, device=device)
    train_mask = train_mask.to(kappa.device).bool()
    vals = kappa[train_mask]
    if vals.numel() == 0:
        vals = kappa
    return CurvatureStats(
        node_kappa=kappa,
        mean_train=float(vals.mean().item()),
        std_train=float(vals.std(unbiased=False).clamp_min(eps).item()),
        mean_all=float(kappa.mean().item()),
        std_all=float(kappa.std(unbiased=False).item()),
    )


def median_shift_gate(kappa_src: torch.Tensor, kappa_tgt: torch.Tensor, alpha: float = 2.0, eps: float = 1e-8) -> torch.Tensor:
    """CGA gate used in the primary protocol: exp(-alpha * |k_tgt-k_src| / median_shift)."""
    shift = torch.abs(kappa_tgt - kappa_src)
    norm = torch.median(shift.detach().float()).clamp_min(float(eps))
    return torch.exp(-float(alpha) * shift / norm).clamp(0.0, 1.0)


def empirical_w1_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = torch.sort(a.detach().float().cpu())[0]
    bb = torch.sort(b.detach().float().cpu())[0]
    n = min(aa.numel(), bb.numel())
    if n == 0:
        return float("nan")
    if aa.numel() != bb.numel():
        q = torch.linspace(0.0, 1.0, n)
        aa = aa[(q * (aa.numel() - 1)).round().long()]
        bb = bb[(q * (bb.numel() - 1)).round().long()]
    return float(torch.mean(torch.abs(aa[:n] - bb[:n])).item())

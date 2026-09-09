from __future__ import annotations

import random
from typing import List, Set, Tuple

import torch
from torch_geometric.utils import coalesce, to_undirected


Pair = Tuple[int, int]


def _to_upper_undirected_pairs(edge_index: torch.Tensor, num_nodes: int) -> List[Pair]:
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    row = edge_index[0].detach().cpu().tolist()
    col = edge_index[1].detach().cpu().tolist()
    pairs: Set[Pair] = set()
    for u, v in zip(row, col):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    return sorted(pairs)


def random_rewire_edges(edge_index: torch.Tensor, num_nodes: int, severity: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Random edge rewiring protocol used in the primary protocol.

    The protocol is intentionally stricter than a generic random replacement:

    1. Convert source edges to an undirected graph and keep only upper-triangle
       unique edge pairs.
    2. Remove ``round(severity * |E|)`` original pairs.
    3. Add the same number of uniformly sampled non-self-loop, non-duplicate
       *new* pairs. A newly sampled pair is not allowed to be any original edge,
       including an edge that was just removed. This preserves the requested
       replacement severity instead of silently adding removed edges back.
    4. Return a bidirectional ``edge_index``.

    If the complement of the original graph does not contain enough candidate
    non-edges, a ``RuntimeError`` is raised rather than silently reducing the
    shift severity.
    """
    if not (0.0 <= severity <= 1.0):
        raise ValueError("severity must be in [0, 1]")

    pairs = _to_upper_undirected_pairs(edge_index, num_nodes)
    if len(pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if severity == 0.0:
        undirected = []
        for u, v in pairs:
            undirected.append((u, v))
            undirected.append((v, u))
        out = torch.tensor(undirected, dtype=torch.long).t().contiguous()
        return coalesce(out, num_nodes=num_nodes)

    rng = random.Random(int(seed))
    original_pairs: Set[Pair] = set(pairs)
    m = len(pairs)
    k = min(max(int(round(float(severity) * m)), 0), m)

    max_possible = num_nodes * (num_nodes - 1) // 2
    available_new_pairs = max_possible - len(original_pairs)
    if available_new_pairs < k:
        raise RuntimeError(
            "Cannot perform requested random rewiring: not enough non-original "
            f"candidate edges. requested_new={k}, available_new={available_new_pairs}, "
            f"num_nodes={num_nodes}, original_edges={m}"
        )

    removed = set(rng.sample(pairs, k))
    kept: Set[Pair] = set(pairs) - removed
    added: Set[Pair] = set()

    # Rejection sampling is adequate for the benchmark graphs. The explicit
    # upper bound prevents an accidental infinite loop on unexpectedly dense
    # graphs; dense impossible cases are caught by available_new_pairs above.
    attempts = 0
    max_attempts = 1000 * max(k, 1) + 10000
    while len(added) < k:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to sample enough new random edges without reusing original edges. "
                f"sampled={len(added)}, required={k}, attempts={attempts}"
            )
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pair = (a, b)
        if pair in original_pairs or pair in added or pair in kept:
            continue
        added.add(pair)

    final_pairs = kept | added
    if len(final_pairs) != m:
        raise RuntimeError(f"Rewired edge count mismatch: got {len(final_pairs)} pairs, expected {m}")

    edges = []
    for u, v in sorted(final_pairs):
        edges.append((u, v))
        edges.append((v, u))
    out = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return coalesce(out, num_nodes=num_nodes)

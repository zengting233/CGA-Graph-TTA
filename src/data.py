from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


PLANETOID = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
}

WEBKB = {
    "cornell": "Cornell",
    "texas": "Texas",
    "wisconsin": "Wisconsin",
}

WIKI = {
    "chameleon": "chameleon",
    "squirrel": "squirrel",
}


def _ensure_planetoid_raw_files(root_path: Path, case_name: str, lower: str) -> None:
    raw_dir = root_path / "Planetoid" / case_name / "raw"
    required = [
        f"ind.{lower}.x",
        f"ind.{lower}.tx",
        f"ind.{lower}.allx",
        f"ind.{lower}.y",
        f"ind.{lower}.ty",
        f"ind.{lower}.ally",
        f"ind.{lower}.graph",
        f"ind.{lower}.test.index",
    ]
    missing = [str(raw_dir / f) for f in required if not (raw_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Planetoid raw files. The strict CGA reproduction package "
            "requires preparation of the raw data. Missing:\n" + "\n".join(missing)
        )


def _candidate_raw_dirs(root_path: Path, group: str, case_name: str, lower: str):
    return [
        root_path / group / case_name / "raw",
        root_path / group / case_name / case_name / "raw",
        root_path / group / case_name / lower / "raw",
        root_path / group / case_name / case_name / "geom_gcn" / "raw",
        root_path / group / case_name / "geom_gcn" / "raw",
        root_path / group / lower / "raw",
        root_path / group / lower / lower / "raw",
        root_path / group / lower / "geom_gcn" / "raw",
    ]


def _find_geomgcn_raw_dir(root_path: Path, group: str, case_name: str, lower: str) -> Path:
    for d in _candidate_raw_dirs(root_path, group, case_name, lower):
        if (d / "out1_node_feature_label.txt").is_file() and (d / "out1_graph_edges.txt").is_file():
            return d

    candidates = "\n".join(str(x) for x in _candidate_raw_dirs(root_path, group, case_name, lower))
    raise FileNotFoundError(
        f"Cannot find Geom-GCN raw files for {case_name}.\n"
        f"Expected out1_node_feature_label.txt and out1_graph_edges.txt in one of:\n{candidates}"
    )


def _find_split_npz(raw_dir: Path, lower: str, split_id: int) -> Path:
    expected = raw_dir / f"{lower}_split_0.6_0.2_{split_id}.npz"
    if expected.is_file():
        return expected

    matches = sorted(raw_dir.glob(f"{lower}_split_0.6_0.2_*.npz"))
    if matches:
        available = "\n".join(str(x) for x in matches)
        raise FileNotFoundError(
            f"Required split file not found: {expected}\n"
            f"Available split files:\n{available}\n"
            f"The strict CGA protocol requires split_id={split_id}."
        )

    raise FileNotFoundError(
        f"Required Geom-GCN split file not found: {expected}\n"
        "Strict CGA reproduction requires Geom-GCN split_0 files and does not "
        "fallback to random 60/20/20 split."
    )


def _as_bool_mask(arr, num_nodes: int, name: str = "mask") -> torch.Tensor:
    """Convert Geom-GCN split arrays to a boolean node mask.

    bool or uint8/int 0/1 arrays of length num_nodes are masks, not index lists.
    Index-list interpretation is used only when the array is not a node-length
    binary mask.
    """
    arr = np.asarray(arr)

    # Support both [num_nodes, num_splits] and [num_splits, num_nodes].
    if arr.ndim == 2:
        if arr.shape[0] == num_nodes:
            arr = arr[:, 0]
        elif arr.shape[1] == num_nodes:
            arr = arr[0, :]
        else:
            raise ValueError(f"{name}: unsupported 2D mask shape {arr.shape} for num_nodes={num_nodes}")

    arr = np.asarray(arr).squeeze()

    if arr.ndim != 1:
        raise ValueError(f"{name}: expected 1D mask/index array after squeeze, got shape {arr.shape}")

    if arr.shape[0] == num_nodes:
        unique = np.unique(arr)
        if arr.dtype == np.bool_ or set(unique.tolist()).issubset({0, 1, False, True}):
            return torch.from_numpy(arr.astype(bool))

    if np.issubdtype(arr.dtype, np.integer):
        idx = torch.from_numpy(arr.astype("int64"))
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if idx.numel() == 0:
            return mask
        if int(idx.min()) < 0 or int(idx.max()) >= num_nodes:
            raise ValueError(
                f"{name}: index list contains out-of-range node id. "
                f"min={int(idx.min())}, max={int(idx.max())}, num_nodes={num_nodes}"
            )
        mask[idx] = True
        return mask

    raise ValueError(
        f"{name}: cannot convert array to boolean mask. "
        f"shape={arr.shape}, dtype={arr.dtype}, unique_head={np.unique(arr)[:10]}"
    )


def _preprocess_edges_for_tex(edge_index: torch.Tensor, num_nodes: int, group: str, lower: str) -> torch.Tensor:
    if group == "WikipediaNetwork":
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    elif group == "WebKB" and lower == "wisconsin":
        edge_index, _ = remove_self_loops(edge_index)

    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def _load_geomgcn_text_dataset(
    root_path: Path,
    group: str,
    case_name: str,
    lower: str,
    split_id: int,
    normalize_features: bool = True,
) -> Data:
    raw_dir = _find_geomgcn_raw_dir(root_path, group, case_name, lower)
    node_file = raw_dir / "out1_node_feature_label.txt"
    edge_file = raw_dir / "out1_graph_edges.txt"
    split_file = _find_split_npz(raw_dir, lower, split_id)

    features = {}
    labels = {}

    with node_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Bad node line in {node_file}: {line[:200]}")

        node_id = int(parts[0])
        feat = [float(x) for x in parts[1].split(",") if x != ""]
        label = int(parts[2])

        features[node_id] = feat
        labels[node_id] = label

    if not features:
        raise ValueError(f"No nodes parsed from {node_file}")

    num_nodes = max(features.keys()) + 1
    feat_dim = len(next(iter(features.values())))

    x = torch.zeros((num_nodes, feat_dim), dtype=torch.float)
    y = torch.full((num_nodes,), -1, dtype=torch.long)

    for node_id, feat in features.items():
        if len(feat) != feat_dim:
            raise ValueError(f"Inconsistent feature dim at node {node_id} in {node_file}")
        x[node_id] = torch.tensor(feat, dtype=torch.float)
        y[node_id] = int(labels[node_id])

    if (y < 0).any():
        missing = int((y < 0).sum())
        raise ValueError(f"{missing} nodes have no label in {node_file}")

    edges = []
    with edge_file.open("r", encoding="utf-8") as f:
        edge_lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in edge_lines[1:]:
        parts = line.replace(",", "\t").split("\t")
        parts = [p for p in parts if p != ""]
        if len(parts) < 2:
            raise ValueError(f"Bad edge line in {edge_file}: {line[:200]}")
        src = int(parts[0])
        dst = int(parts[1])
        edges.append([src, dst])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index = _preprocess_edges_for_tex(edge_index, num_nodes=num_nodes, group=group, lower=lower)

    split_obj = np.load(split_file, allow_pickle=True)
    required_keys = ["train_mask", "val_mask", "test_mask"]
    for key in required_keys:
        if key not in split_obj.files:
            raise KeyError(f"{split_file} missing required key: {key}. keys={split_obj.files}")

    train_mask = _as_bool_mask(split_obj["train_mask"], num_nodes, name=f"{lower}.train_mask")
    val_mask = _as_bool_mask(split_obj["val_mask"], num_nodes, name=f"{lower}.val_mask")
    test_mask = _as_bool_mask(split_obj["test_mask"], num_nodes, name=f"{lower}.test_mask")

    if int((train_mask & val_mask).sum()) or int((train_mask & test_mask).sum()) or int((val_mask & test_mask).sum()):
        raise ValueError(
            f"{lower}: train/val/test masks overlap. "
            f"overlaps: train&val={int((train_mask & val_mask).sum())}, "
            f"train&test={int((train_mask & test_mask).sum())}, "
            f"val&test={int((val_mask & test_mask).sum())}"
        )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    if normalize_features:
        data = T.NormalizeFeatures()(data)

    return data


def _normalize_planetoid_masks(data: Data) -> Data:
    for attr in ["train_mask", "val_mask", "test_mask"]:
        mask = getattr(data, attr)
        if mask.dim() == 2:
            mask = mask[:, 0]
        setattr(data, attr, mask.bool())
    return data


def load_dataset(
    name: str,
    root: str = "./data",
    split: str = "official",
    geom_gcn_split: int = 0,
    normalize_features: bool = True,
    to_undirected: bool | None = None,
    **kwargs,
) -> Tuple[Data, int, int]:
    key = name.lower()
    root_path = Path(root)

    if split != "official":
        raise ValueError(f"Unsupported split={split!r}. Strict CGA reproduction uses split='official' only.")

    if key in PLANETOID:
        case_name = PLANETOID[key]
        _ensure_planetoid_raw_files(root_path, case_name, key)

        transform = T.NormalizeFeatures() if normalize_features else None
        ds = Planetoid(root=str(root_path / "Planetoid"), name=case_name, split="public", transform=transform)
        data = _normalize_planetoid_masks(ds[0])
        return data, int(ds.num_features), int(ds.num_classes)

    if key in WEBKB:
        case_name = WEBKB[key]
        data = _load_geomgcn_text_dataset(
            root_path=root_path,
            group="WebKB",
            case_name=case_name,
            lower=key,
            split_id=int(geom_gcn_split),
            normalize_features=normalize_features,
        )
        return data, int(data.x.size(-1)), int(data.y.max().item() + 1)

    if key in WIKI:
        case_name = WIKI[key]
        data = _load_geomgcn_text_dataset(
            root_path=root_path,
            group="WikipediaNetwork",
            case_name=case_name,
            lower=key,
            split_id=int(geom_gcn_split),
            normalize_features=normalize_features,
        )
        return data, int(data.x.size(-1)), int(data.y.max().item() + 1)

    supported = list(PLANETOID.values()) + list(WEBKB.values()) + ["Chameleon", "Squirrel"]
    raise ValueError(f"Unknown dataset {name!r}. Supported: {supported}")

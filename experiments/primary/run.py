"""Run the fixed primary CGA evaluation; no hyperparameter sweep."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = ("Cora", "CiteSeer", "Cornell", "Texas", "Wisconsin", "Chameleon", "Squirrel")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=("t3a", "tent", "all"))
    parser.add_argument("--dataset", choices=DATASETS, default="Cora")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--all", action="store_true", help="Run all seven datasets and seeds 0, 1, 2.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations and print the execution plan without loading data or training.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_config(adapter):
    import yaml
    path = ROOT / "configs" / "primary" / (adapter + ".yaml")
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    expected = {
        "data": {"root": "./data", "datasets": list(DATASETS), "split": "official", "geom_gcn_split": 0, "normalize_features": True},
        "model": {"name": "GraphSAGE", "hidden_dim": 64, "dropout": 0.5, "activation": "elu"},
        "train": {"epochs": 200, "patience": 30, "optimizer": "AdamW", "lr": 0.01, "weight_decay": 0.0005},
        "shift": {"type": "random_rewire", "severity": 0.5},
        "adapt": {"mask": "test", "alpha": 2.0, "eps": 1e-8},
        "run": {"seeds": [0, 1, 2], "device": "auto"},
        "adapter": adapter.upper(),
    }
    expected[adapter] = {
        "t3a": {"gate_transform": "rank_20_80", "support_score": "confidence_x_gate2", "confidence_quantile": 0.7},
        "tent": {"gate_transform": "identity", "steps": 20, "lr": 0.0015, "weight_decay": 0.0, "adapt_head_only": True, "gamma_ent": 4.0, "gamma_anchor": 0.5, "lambda_anchor": 12.0, "lr_scale": 1.0},
    }[adapter]
    if cfg != expected:
        raise ValueError("Configuration differs from the fixed primary protocol: " + path.name)
    return cfg


def run_one(adapter, dataset, seed, cfg, args):
    import torch
    from src.curvature import curvature_stats
    from src.data import load_dataset
    from src.models import GraphSAGE
    from src.shift import random_rewire_edges
    from src.train import train_source
    from src.tta import (
        _cga_gate, _distance_logits_to_prototypes, accuracy_from_logits,
        evaluate, get_source_outputs, select_adapt_mask,
    )
    from src.t3a import build_prototypes_by_score, transform_gate
    from src.tent import run_tent_like
    from src.utils import get_device, set_seed

    set_seed(seed)
    device = get_device(args.device)
    data_cfg = cfg["data"]
    data, in_dim, num_classes = load_dataset(
        dataset, root=str(args.data_root.resolve()), split=data_cfg["split"],
        geom_gcn_split=data_cfg["geom_gcn_split"],
        normalize_features=data_cfg["normalize_features"],
    )
    data = data.to(device)
    edge_src = data.edge_index.to(device)
    severity = float(cfg["shift"]["severity"])
    edge_tgt = random_rewire_edges(
        edge_src.detach().cpu(), num_nodes=int(data.num_nodes), severity=severity, seed=seed,
    ).to(device)
    model = GraphSAGE(in_dim, cfg["model"]["hidden_dim"], num_classes, cfg["model"]["dropout"]).to(device)
    train_cfg = cfg["train"]
    trained = train_source(
        model, data, edge_src, epochs=train_cfg["epochs"], patience=train_cfg["patience"],
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
        optimizer_name=train_cfg["optimizer"], verbose=args.verbose,
    )
    source = trained.model
    source_acc, _ = evaluate(source, data, edge_tgt)
    src_curv = curvature_stats(edge_src, int(data.num_nodes), data.train_mask, device=device)
    tgt_curv = curvature_stats(edge_tgt, int(data.num_nodes), data.train_mask, device=device)
    alpha, eps = cfg["adapt"]["alpha"], cfg["adapt"]["eps"]
    mask = select_adapt_mask(data, cfg["adapt"]["mask"]).to(device)

    if adapter == "t3a":
        options = cfg["t3a"]
        with torch.no_grad():
            _, logits, p_src = get_source_outputs(source, data, edge_tgt)
            gate = transform_gate(
                _cga_gate(src_curv, tgt_curv, device, alpha=alpha, eps=eps),
                options["gate_transform"], mask=mask, eps=eps,
            )
            proto, _ = build_prototypes_by_score(
                logits, p_src, mask, num_classes, options["confidence_quantile"],
                "confidence", gate_for_score=None, gate_for_weight=None, eps=eps,
            )
            p_base = torch.softmax(_distance_logits_to_prototypes(logits, proto, eps=eps), dim=-1)
            proto_cga, _ = build_prototypes_by_score(
                logits, p_src, mask, num_classes, options["confidence_quantile"],
                options["support_score"], gate_for_score=gate, gate_for_weight=gate, eps=eps,
            )
            p_adapt = torch.softmax(_distance_logits_to_prototypes(logits, proto_cga, eps=eps), dim=-1)
            p_cga = gate.unsqueeze(-1) * p_adapt + (1.0 - gate).unsqueeze(-1) * p_src
            base_acc = accuracy_from_logits(torch.log(p_base.clamp_min(eps)), data.y, data.test_mask)
            cga_acc = accuracy_from_logits(torch.log(p_cga.clamp_min(eps)), data.y, data.test_mask)
    elif adapter == "tent":
        options = cfg["tent"]
        with torch.no_grad():
            source.eval()
            p_src = torch.softmax(source(data.x, edge_tgt).detach(), dim=-1).detach()
            gate = _cga_gate(src_curv, tgt_curv, device, alpha=alpha, eps=eps)
        common = dict(
            source_model=source, data=data, edge_index_tgt=edge_tgt, p_src=p_src,
            steps=options["steps"], lr=options["lr"], weight_decay=options["weight_decay"],
            adapt_head_only=options["adapt_head_only"], mask_mode=cfg["adapt"]["mask"], eps=eps,
        )
        base_acc, _ = run_tent_like(
            **common, gate=torch.ones_like(gate), variant="vanilla",
            gamma_ent=1.0, gamma_anchor=1.0, lambda_anchor=0.0, lr_scale=1.0,
        )
        cga_acc, _ = run_tent_like(
            **common, gate=gate, variant="cga", gamma_ent=options["gamma_ent"],
            gamma_anchor=options["gamma_anchor"], lambda_anchor=options["lambda_anchor"],
            lr_scale=options["lr_scale"],
        )
    return {
        "adapter": adapter.upper(), "dataset": dataset, "seed": seed,
        "backbone": "GraphSAGE", "severity": severity, "best_epoch": trained.best_epoch,
        "source_accuracy": source_acc, "adapter_accuracy": base_acc, "cga_accuracy": cga_acc,
        "adapter_delta_pp": 100.0 * (base_acc - source_acc),
        "cga_delta_pp": 100.0 * (cga_acc - source_acc),
        "swing_pp": 100.0 * (cga_acc - base_acc),
    }


def main(argv=None):
    args = parse_args(argv)
    adapters = ("t3a", "tent") if args.adapter == "all" else (args.adapter,)
    datasets = DATASETS if args.all else (args.dataset,)
    seeds = (0, 1, 2) if args.all else (args.seed,)
    plan = [(adapter, dataset, seed) for adapter in adapters for dataset in datasets for seed in seeds]
    configs = {adapter: load_config(adapter) for adapter in adapters}
    if args.dry_run:
        print(json.dumps({"runs": len(plan), "plan": plan, "data_root": str(args.data_root.resolve()),
                          "output_dir": str(args.output_dir.resolve())}, indent=2))
        return 0
    for adapter, dataset, seed in plan:
        path = args.output_dir / adapter / (dataset + "_seed" + str(seed) + ".csv")
        if path.exists():
            raise FileExistsError("Use a new output directory; existing result: " + str(path))
    for adapter, dataset, seed in plan:
        row = run_one(adapter, dataset, seed, configs[adapter], args)
        path = args.output_dir / adapter / (dataset + "_seed" + str(seed) + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

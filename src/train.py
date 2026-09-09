from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .utils import accuracy


@dataclass
class TrainResult:
    model: nn.Module
    best_val_acc: float
    best_epoch: int
    test_acc_at_best: float


def train_source(
    model: nn.Module,
    data,
    edge_index: torch.Tensor,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    optimizer_name: str = "AdamW",
    verbose: bool = False,
) -> TrainResult:
    if optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported source optimizer: {optimizer_name}")

    best_state = deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = -1
    best_test = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(data.x, edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_test = test_acc
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(f"epoch={epoch:03d} loss={loss.item():.4f} val={val_acc:.4f} test={test_acc:.4f}")

        if patience > 0 and stale >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(model=model, best_val_acc=best_val, best_epoch=best_epoch, test_acc_at_best=best_test)

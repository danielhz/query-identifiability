"""
Shared training loop for all predictor architectures.

All models expose the same interface:
    forward(x: Tensor[B, feature_dim]) -> logits: Tensor[B]

Usage:
    cfg = TrainConfig(epochs=300, lr=1e-3, patience=30, device="cuda")
    result = train(model, X_train, y_train, X_val, y_val, cfg)
    metrics = evaluate(model, X_test, y_test, device=cfg.device)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    patience: int = 30  # early stopping on val loss; disabled if no val set given
    device: str = "cpu"  # "cuda" on cluster


@dataclass
class TrainResult:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cfg: TrainConfig | None = None,
) -> TrainResult:
    """
    Train model with Adam + BCE loss.  Returns loss history and restores the
    best (lowest val-loss) checkpoint when a validation set is provided.
    """
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device)
    model = model.to(device)

    X_tr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    result = TrainResult()

    # Models with no learnable parameters (e.g. ClosureAwarePredictor on a
    # non-certified query) have nothing to train — return immediately.
    if not list(model.parameters()):
        return result

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_v = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        y_v = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state: dict | None = None
    wait = 0

    for epoch in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        result.train_loss.append(ep_loss / len(X_tr))

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = bce(model(X_v), y_v).item()
            result.val_loss.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                result.best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    result.stopped_early = True
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return result


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
) -> dict[str, float]:
    """Return raw accuracy, balanced accuracy, positive rate, and BCE loss.

    balanced_accuracy = (TPR + TNR) / 2.  This is the correct metric for the
    error-floor claim: a predictor that exploits majority-class bias achieves
    high raw accuracy but balanced_accuracy = 0.5, matching the theorem.
    Returns NaN for balanced_accuracy when one class is absent from y.
    """
    dev = torch.device(device)
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = model(Xt)
        loss = nn.BCEWithLogitsLoss()(logits, yt).item()
        preds = (logits > 0.0).float()
        acc = (preds == yt).float().mean().item()

    y_np = yt.cpu().numpy()
    p_np = preds.cpu().numpy()
    pos_mask = y_np == 1
    neg_mask = y_np == 0
    tpr = float(p_np[pos_mask].mean()) if pos_mask.any() else float("nan")
    tnr = float((1 - p_np[neg_mask]).mean()) if neg_mask.any() else float("nan")
    if pos_mask.any() and neg_mask.any():
        balanced_acc = (tpr + tnr) / 2.0
    else:
        balanced_acc = float("nan")

    return {
        "loss": loss,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "positive_rate": float(pos_mask.mean()),
    }

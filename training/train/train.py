"""Training loop: weighted BCE, early stop on validation macro-F1, checkpointing.

Status: needs torch + torch_geometric; py_compiled here, run on the Studio. Each
run is seeded and driven by one config; for every reported number we save the
checkpoint, the split index and the config alongside it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scgnn.common.seeds import set_seed
from scgnn.schema import N_FLAWS
from training.evaluate.metrics import macro_f1


def pos_weight_from_labels(Y: np.ndarray):
    """Per-class neg/pos ratio, so rare flaws are not drowned out."""
    import torch

    pos = Y.sum(axis=0).astype(float)
    neg = Y.shape[0] - pos
    w = neg / np.clip(pos, 1.0, None)
    return torch.tensor(w, dtype=torch.float)


def evaluate(model, loader, device) -> float:
    import torch

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for ast, cfg, y in loader:
            ast, cfg = ast.to(device), cfg.to(device)
            prob = torch.sigmoid(model(ast, cfg)).cpu().numpy()
            ps.append(prob); ys.append(y.numpy())
    y_true = (np.vstack(ys) >= 0.5).astype(int)
    y_pred = (np.vstack(ps) >= 0.5).astype(int)
    return macro_f1(y_true, y_pred)


def train_model(config: dict, train_loader, val_loader, train_labels: np.ndarray,
                out_dir: str) -> dict:
    """Train one model under ``config`` and save the best checkpoint by val macro-F1."""
    import torch
    from torch.optim import Adam

    from scgnn.models.dual_gnn import build_model

    set_seed(int(config.get("seed", 42)))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_labels(train_labels).to(device))
    opt = Adam(model.parameters(), lr=float(config.get("lr", 1e-3)),
               weight_decay=float(config.get("weight_decay", 1e-5)))

    best_f1, best_epoch, patience = -1.0, -1, int(config.get("patience", 15))
    history = []
    for epoch in range(int(config.get("epochs", 200))):
        model.train()
        running = 0.0
        for ast, cfg, y in train_loader:
            ast, cfg, y = ast.to(device), cfg.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(ast, cfg), y)
            loss.backward(); opt.step()
            running += float(loss)
        val_f1 = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running, "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), out / "best_model.pt")
        elif epoch - best_epoch >= patience:
            break

    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch,
            "checkpoint": str(out / "best_model.pt")}

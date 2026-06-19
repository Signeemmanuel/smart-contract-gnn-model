"""The dual-graph model: one encoder per view, concatenate, multi-label head.

Status: needs torch + torch_geometric; py_compiled here, run on the Studio.

Five independent logits (sigmoid at inference), never softmax: a contract may
carry several flaws at once. The class mixes in ``PyTorchModelHubMixin`` so it
gains ``save_pretrained`` / ``from_pretrained`` / ``push_to_hub`` and its config
is pushed to the Hub bundle automatically; the fitted PCA and feature config
ride alongside as extra files (see scripts/build_release_bundle.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scgnn.models.encoders import Encoder
from scgnn.schema import N_FLAWS

try:
    from huggingface_hub import PyTorchModelHubMixin
except Exception:  # the Hub mixin is optional at import time
    class PyTorchModelHubMixin:  # type: ignore
        pass


def _num_graphs(ast, cfg) -> int:
    """Authoritative number of graphs in the batch.

    Both views describe the SAME set of contracts, so they contain the same
    number of graphs even when one view's graph is empty (zero nodes). We take
    the count from whichever batch tensor reports the larger graph count, which
    is robust when the LAST contract in the batch has an empty CFG (its batch
    entries are absent, so cfg.batch.max() would under-count). PyG's Batch also
    exposes ``num_graphs``; prefer it when present.
    """
    n = 0
    for g in (ast, cfg):
        ng = getattr(g, "num_graphs", None)
        if ng is not None:
            n = max(n, int(ng))
        b = getattr(g, "batch", None)
        if b is not None and b.numel() > 0:
            n = max(n, int(b.max().item()) + 1)
    return n or 1


class DualGNN(nn.Module, PyTorchModelHubMixin):
    """AST encoder + CFG encoder -> concat -> small MLP head -> 5 logits."""

    def __init__(self, in_dim: int, hid: int = 128, conv: str = "sage",
                 n_classes: int = N_FLAWS, layers: int = 3, heads: int = 4,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.config = dict(in_dim=in_dim, hid=hid, conv=conv, n_classes=n_classes,
                           layers=layers, heads=heads, dropout=dropout)
        self.ast = Encoder(in_dim, hid, conv, layers, heads, dropout)
        self.cfg = Encoder(in_dim, hid, conv, layers, heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, n_classes),
        )

    def forward(self, ast, cfg) -> torch.Tensor:
        # Pass the authoritative graph count to BOTH encoders so an empty graph
        # in either view still yields a (zero) pooled row, keeping the two
        # branches the same length before concat. Without this, a contract with
        # an empty CFG makes the CFG branch one row short -> concat size mismatch.
        n = _num_graphs(ast, cfg)
        h = torch.cat(
            [self.ast(ast.x, ast.edge_index, ast.batch, size=n),
             self.cfg(cfg.x, cfg.edge_index, cfg.batch, size=n)],
            dim=1,
        )
        return self.head(h)  # raw logits, shape (batch, n_classes)

    @torch.no_grad()
    def predict_proba(self, ast, cfg) -> torch.Tensor:
        return torch.sigmoid(self.forward(ast, cfg))


def build_model(config: dict) -> "DualGNN":
    """Construct a model from a config dict (the same one saved in the bundle)."""
    keys = ("in_dim", "hid", "conv", "n_classes", "layers", "heads", "dropout")
    kwargs = {k: config[k] for k in keys if k in config}
    return DualGNN(**kwargs)
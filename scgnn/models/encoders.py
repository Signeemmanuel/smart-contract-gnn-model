"""Swappable single-stage GNN encoder (GCN / GraphSAGE / GAT).

Status: needs torch + torch_geometric; py_compiled here, run on the Studio.

Three message-passing layers then mean pooling, exactly as fixed in the spec.
The GAT variant keeps its first-layer attention so the explanation component can
reuse it as a cheap secondary signal (Phase 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

CONV = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, conv: str = "sage",
                 layers: int = 3, heads: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        if conv not in CONV:
            raise ValueError(f"unknown conv {conv!r}; expected one of {list(CONV)}")
        self.conv_name = conv
        Conv = CONV[conv]
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid] * layers
        for i in range(layers):
            if conv == "gat":
                # First layer is multi-head (concat); later layers single-head so
                # the output width stays `hid`. Attention is captured from layer 0.
                if i == 0:
                    self.convs.append(Conv(dims[i], hid // heads, heads=heads, dropout=dropout))
                else:
                    self.convs.append(Conv(hid, hid, heads=1, dropout=dropout))
            else:
                self.convs.append(Conv(dims[i], dims[i + 1]))
        self.drop = nn.Dropout(dropout)
        self.last_attention = None  # (edge_index, alpha) from the first GAT layer

    def forward(self, x, edge_index, batch, size: int | None = None):
        """Encode a (batched) graph and mean-pool to one vector per graph.

        ``size`` is the number of graphs in the batch. It MUST be passed when a
        graph in the batch may be EMPTY (zero nodes) — e.g. a contract whose CFG
        extraction fell back and produced no nodes. Without it, global_mean_pool
        infers the graph count from ``batch.max() + 1`` and silently drops any
        trailing empty graph, yielding one fewer pooled row than the sibling
        branch and breaking the downstream concat. With ``size`` given, empty
        graphs still get a (zero) pooled row, keeping both branches aligned.
        """
        self.last_attention = None
        for i, conv in enumerate(self.convs):
            if self.conv_name == "gat" and i == 0:
                x, att = conv(x, edge_index, return_attention_weights=True)
                self.last_attention = att  # (edge_index, alpha)
            else:
                x = conv(x, edge_index)
            x = self.drop(x.relu())
        return global_mean_pool(x, batch, size=size)
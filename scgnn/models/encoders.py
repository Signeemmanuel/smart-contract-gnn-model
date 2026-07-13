"""Swappable single-stage GNN encoder (GCN / GraphSAGE / GAT / GATv2).

Status: needs torch + torch_geometric; py_compiled here, run on the Studio.

Three message-passing layers then mean pooling, exactly as fixed in the spec.

v2 (Workstream D) adds **GATv2** (``GATv2Conv``). GAT's attention is *static*:
its scoring function ranks the neighbours of every node in the same order,
regardless of the query node, which Brody et al. (2022) show is a strictly
weaker attention. GATv2 makes the attention *dynamic* by applying the linear
layer after the non-linearity. GAT itself is retained so the v1 results remain
reproducible for the before/after table, but it is retired from the active model
matrix and from the ensemble.

Both attention variants expose the first layer's attention weights so the
explanation component can reuse them as a cheap secondary signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, GATv2Conv, GCNConv, SAGEConv, global_mean_pool,
)

CONV = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv, "gatv2": GATv2Conv}
ATTENTION_CONVS = ("gat", "gatv2")


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
            if conv in ATTENTION_CONVS:
                # First layer is multi-head with `hid` split across heads, so the
                # concatenated output width stays exactly `hid` (same parameter
                # budget as GCN/SAGE). Later layers are single-head. Attention is
                # captured from layer 0.
                if i == 0:
                    self.convs.append(Conv(dims[i], hid // heads, heads=heads,
                                           dropout=dropout))
                else:
                    self.convs.append(Conv(hid, hid, heads=1, dropout=dropout))
            else:
                self.convs.append(Conv(dims[i], dims[i + 1]))
        self.drop = nn.Dropout(dropout)
        self.last_attention = None  # (edge_index, alpha) from the first attention layer

    def forward(self, x, edge_index, batch, size: int | None = None):
        """Encode a (batched) graph and mean-pool to one vector per graph.

        ``size`` is the number of graphs in the batch. Pass it whenever a graph
        in the batch may be EMPTY (zero nodes) - e.g. a contract whose CFG
        extraction degraded. Without it, ``global_mean_pool`` infers the graph
        count from ``batch.max() + 1`` and silently drops a trailing empty graph,
        producing one fewer pooled row than the sibling branch and breaking the
        downstream concatenation.
        """
        self.last_attention = None
        for i, conv in enumerate(self.convs):
            if self.conv_name in ATTENTION_CONVS and i == 0:
                x, att = conv(x, edge_index, return_attention_weights=True)
                self.last_attention = att  # (edge_index, alpha)
            else:
                x = conv(x, edge_index)
            x = self.drop(x.relu())
        return global_mean_pool(x, batch, size=size)
"""Collate (ast, cfg, y) triples into two aligned PyG Batches plus a label tensor.

Status: needs torch + torch_geometric; py_compiled here.
"""

from __future__ import annotations


def collate_pairs(batch):
    import torch
    from torch_geometric.data import Batch

    asts, cfgs, ys = zip(*batch)
    return (Batch.from_data_list(list(asts)),
            Batch.from_data_list(list(cfgs)),
            torch.stack(list(ys)))

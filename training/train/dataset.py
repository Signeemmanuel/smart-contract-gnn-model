"""ContractPairDataset over precomputed graph records.

Status: needs torch + torch_geometric; py_compiled here. Each processed record
is a dict saved with ``torch.save``: ``{ast_x, ast_edge_index, cfg_x,
cfg_edge_index, y}``. An index JSON lists ``{"id", "path"}`` per contract, plus
the content hash so the firewall can audit the split.
"""

from __future__ import annotations

import json
from pathlib import Path


class ContractPairDataset:
    def __init__(self, index_path: str) -> None:
        self.records = json.loads(Path(index_path).read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        import torch
        from torch_geometric.data import Data

        rec = torch.load(self.records[i]["path"], weights_only=True)
        ast = Data(x=rec["ast_x"], edge_index=rec["ast_edge_index"])
        cfg = Data(x=rec["cfg_x"], edge_index=rec["cfg_edge_index"])
        return ast, cfg, rec["y"].float()

#!/usr/bin/env python3
"""Evaluate a checkpoint on the frozen Curated test split. Status: needs the stack."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-index", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from scgnn.models.dual_gnn import build_model
    from training.evaluate.metrics import per_flaw_and_macro
    from training.train.collate import collate_pairs
    from training.train.dataset import ContractPairDataset

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    ds = ContractPairDataset(args.test_index)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_pairs)
    ys, ps = [], []
    with torch.no_grad():
        for ast, cfg, y in loader:
            ps.append(torch.sigmoid(model(ast, cfg)).numpy()); ys.append(y.numpy())
    y_true = (np.vstack(ys) >= 0.5).astype(int)
    y_pred = (np.vstack(ps) >= 0.5).astype(int)
    print(json.dumps(per_flaw_and_macro(y_true, y_pred), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

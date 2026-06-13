#!/usr/bin/env python3
"""Train one architecture from a YAML config. Status: needs the training stack.

Resolves config inheritance (``extends``) and injects ``in_dim`` from the built
feature config, since input width is data-derived, not a hyper-parameter.
"""

from __future__ import annotations

import argparse

from scgnn.common.seeds import set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML experiment config.")
    ap.add_argument("--train-index", required=True)
    ap.add_argument("--val-index", required=True)
    ap.add_argument("--feature-config", default="data/processed/feature_config.json")
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()

    import numpy as np
    from torch.utils.data import DataLoader

    from scgnn.extraction.features import FeatureConfig
    from training.config import load_config
    from training.train.collate import collate_pairs
    from training.train.dataset import ContractPairDataset
    from training.train.train import train_model

    config = load_config(args.config)
    config["in_dim"] = FeatureConfig.from_json(args.feature_config).in_dim
    set_seed(int(config.get("seed", 42)))

    train_ds = ContractPairDataset(args.train_index)
    val_ds = ContractPairDataset(args.val_index)
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 32)),
                              shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("batch_size", 32)),
                            shuffle=False, collate_fn=collate_pairs)
    train_labels = np.vstack([train_ds[i][2].numpy() for i in range(len(train_ds))])
    print(train_model(config, train_loader, val_loader, train_labels, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

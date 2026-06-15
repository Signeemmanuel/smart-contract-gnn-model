#!/usr/bin/env python3
"""Evaluate a checkpoint on the frozen Curated test split. Status: needs the stack."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-index", required=True)
    ap.add_argument("--val-index", default=None,
                    help="Validation index. Required with --tune-threshold; the "
                         "threshold is chosen here and applied unchanged to test.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--feature-config", default="data/processed/feature_config.json")
    ap.add_argument("--tune-threshold", choices=["global", "per_class"], default=None,
                    help="Select the decision threshold on the validation set to "
                         "maximise macro-F1 (global) or each class's F1 (per_class), "
                         "then report test metrics with it. Omit to use the config "
                         "threshold. Never tunes on the test set.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed decision threshold to apply at test time (e.g. 0.5), "
                         "overriding the config value. Mutually exclusive with "
                         "--tune-threshold.")
    ap.add_argument("--out", default="data/processed/eval_metrics.json",
                    help="Output path for the metrics JSON. May be a full file path "
                         "(…/eval_metrics.json) or a directory (eval_metrics.json is "
                         "written inside it).")
    args = ap.parse_args()
    if args.tune_threshold and not args.val_index:
        ap.error("--tune-threshold requires --val-index (threshold is chosen on val).")
    if args.tune_threshold and args.threshold is not None:
        ap.error("use either --threshold (fixed) or --tune-threshold (val-selected), not both.")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from scgnn.extraction.features import FeatureConfig
    from scgnn.models.dual_gnn import build_model
    from training.config import load_config
    from training.evaluate.metrics import per_flaw_and_macro
    from training.train.collate import collate_pairs
    from training.train.dataset import ContractPairDataset

    config = load_config(args.config)
    config["in_dim"] = FeatureConfig.from_json(args.feature_config).in_dim
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    def collect(index_path):
        """Return (y_true (n,5) int, probs (n,5) float) for an index file."""
        loader = DataLoader(ContractPairDataset(index_path), batch_size=32,
                            shuffle=False, collate_fn=collate_pairs)
        ys, ps = [], []
        with torch.no_grad():
            for ast, cfg, y in loader:
                ps.append(torch.sigmoid(model(ast, cfg)).numpy())
                ys.append(y.numpy())
        return (np.vstack(ys) >= 0.5).astype(int), np.vstack(ps)

    GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

    def best_global_threshold(y, p):
        """Threshold maximising macro-F1 on (y, p)."""
        best_t, best_f1 = 0.5, -1.0
        for t in GRID:
            f1 = per_flaw_and_macro(y, (p >= t).astype(int))["macro"]["f1"]
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return best_t

    def best_per_class_thresholds(y, p):
        """Per-class threshold, each maximising that class's F1 on (y, p)."""
        from training.evaluate.metrics import per_flaw_and_macro as _m
        from scgnn.schema import FLAWS
        thrs = []
        for j, flaw in enumerate(FLAWS):
            best_t, best_f1 = 0.5, -1.0
            for t in GRID:
                yj = y[:, j:j + 1]
                pj = (p[:, j:j + 1] >= t).astype(int)
                f1 = _m(yj, pj)["macro"]["f1"]
                if f1 > best_f1:
                    best_t, best_f1 = float(t), f1
            thrs.append(best_t)
        return thrs

    # Decide the threshold(s) — on validation when tuning, else from config.
    config_thr = float(config.get("threshold", 0.5))
    if args.tune_threshold:
        yv, pv = collect(args.val_index)
        if args.tune_threshold == "global":
            chosen = best_global_threshold(yv, pv)
            thr_desc = {"mode": "global", "threshold": chosen, "selected_on": "val"}
        else:
            chosen = best_per_class_thresholds(yv, pv)
            thr_desc = {"mode": "per_class", "thresholds": chosen, "selected_on": "val"}
    elif args.threshold is not None:
        chosen = float(args.threshold)
        thr_desc = {"mode": "fixed", "threshold": chosen, "selected_on": "cli"}
    else:
        chosen = config_thr
        thr_desc = {"mode": "config", "threshold": config_thr, "selected_on": "config"}

    def apply_threshold(p):
        if isinstance(chosen, list):
            return (p >= np.asarray(chosen)[None, :]).astype(int)
        return (p >= chosen).astype(int)

    y_true, p_test = collect(args.test_index)
    y_pred = apply_threshold(p_test)
    metrics = per_flaw_and_macro(y_true, y_pred)
    metrics["threshold"] = thr_desc
    print(json.dumps(metrics, indent=2))

    from pathlib import Path
    out = Path(args.out)
    # Accept either a full file path or a directory. Treat as a directory only if
    # it already exists as one, or has no .json suffix; otherwise it's the file.
    if out.is_dir() or out.suffix.lower() != ".json":
        out = out / "eval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
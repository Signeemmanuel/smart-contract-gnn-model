#!/usr/bin/env python3
"""Re-evaluate Test B only, after the record repair. No retraining.

Checkpoints, training, validation and the val-tuned thresholds are untouched
and untainted by the repaired rows; only the expert test rows changed (they now
contain the actual expert contracts). This script re-runs INFERENCE over the
repaired test_b split for every trained run, re-applies each run's frozen
thresholds, recomputes the test_b metrics (with bootstrap CIs), rebuilds the
two pinned ensembles from the fresh member probabilities with their stored
thresholds, and updates ``results.json`` and ``test_b_probs.json`` in place.
A before/after delta table is printed so the correction is fully visible.

Usage
-----
    PYTHONPATH=. python scripts/reeval_testb.py --runs runs/v2 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scgnn.schema import FLAWS

ENSEMBLES = {"ensemble": ["gcn", "sage", "gatv2"],
             "ensemble_df": ["gcn_df", "sage_df", "gatv2_df"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from scgnn.models.dual_gnn import build_model
    from training.evaluate.metrics import ensemble_probs
    from training.train.collate import collate_pairs
    from training.train.dataset import ContractPairDataset

    # reuse the evaluation helpers from the training script (same definitions)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_v2 import evaluate_probs, predict_probs

    runs_root = Path(args.runs)
    results_path = runs_root / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    loaders: dict[str, DataLoader] = {}       # data_dir -> test_b loader (cached)

    def loader_for(data_dir: str) -> DataLoader:
        if data_dir not in loaders:
            ds = ContractPairDataset(str(Path(data_dir) / "test_b_index.json"))
            loaders[data_dir] = DataLoader(
                ds, batch_size=256, shuffle=False, collate_fn=collate_pairs,
                num_workers=args.num_workers,
                pin_memory=(args.num_workers > 0))
        return loaders[data_dir]

    fresh: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # run -> (y, probs)
    before = {m: results[m]["test_b"]["macro"]["f1"] for m in results}

    # ---- singles: fresh inference on the repaired split ----
    for run_dir in sorted(p for p in runs_root.iterdir()
                          if p.is_dir() and (p / "best_model.pt").exists()):
        tag = run_dir.name
        prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        cfg, data_dir = prov["config"], prov["data_dir"]
        model = build_model(cfg)
        model.load_state_dict(torch.load(run_dir / "best_model.pt",
                                         map_location=device, weights_only=True))
        model.to(device).eval()
        y, probs = predict_probs(model, loader_for(data_dir), device)
        fresh[tag] = (y, probs)
        thr = results[tag]["test_b"]["thresholds"]          # val-tuned, frozen
        results[tag]["test_b"] = evaluate_probs(y, probs, thr,
                                                n_boot=args.bootstrap, seed=42)
        print(f"{tag:<12} test B macro F1: {before[tag]:.3f} -> "
              f"{results[tag]['test_b']['macro']['f1']:.3f}")

    # ---- ensembles: pinned membership, stored thresholds ----
    for name, members in ENSEMBLES.items():
        if name not in results or any(m not in fresh for m in members):
            continue
        thr_path = runs_root / name / "thresholds.json"
        thr = (json.loads(thr_path.read_text(encoding="utf-8"))["thresholds"]
               if thr_path.exists() else results[name]["test_b"]["thresholds"])
        y = fresh[members[0]][0]
        p_mean = ensemble_probs([fresh[m][1] for m in members], policy="mean")
        res_b = evaluate_probs(y, p_mean, thr, n_boot=args.bootstrap, seed=42)
        p_max = ensemble_probs([fresh[m][1] for m in members], policy="max")
        res_b["max_policy_macro_f1"] = evaluate_probs(
            y, p_max, thr, n_boot=1, seed=42)["macro"]["f1"]
        results[name]["test_b"] = res_b
        print(f"{name:<12} test B macro F1: {before[name]:.3f} -> "
              f"{res_b['macro']['f1']:.3f}")

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- refreshed probabilities for the Durieux matrix ----
    tb = {tag: {"probs": probs.tolist(),
                "thresholds": results[tag]["test_b"]["thresholds"]}
          for tag, (y, probs) in fresh.items()}
    (runs_root / "test_b_probs.json").write_text(json.dumps(tb), encoding="utf-8")

    best = max(results, key=lambda m: results[m]["test_b"]["macro"]["f1"])
    print("\n" + "=" * 56)
    print(f"{'run':<16} {'before':>8} {'after':>8}")
    print("-" * 56)
    for m in sorted(results):
        print(f"{m:<16} {before[m]:>8.3f} "
              f"{results[m]['test_b']['macro']['f1']:>8.3f}")
    print("=" * 56)
    print(f"winner on the expert test set (B): {best} "
          f"({results[best]['test_b']['macro']['f1']:.3f})")
    print(f"\nupdated {results_path} and {runs_root / 'test_b_probs.json'}")
    print("Next: bash run_pipeline.sh durieux localise figures   "
          "(matrix + benchmark + artifacts regenerate from the corrected rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
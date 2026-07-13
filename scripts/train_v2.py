#!/usr/bin/env python3
"""v2 training + evaluation matrix (Workstreams D, E, F, G).

Trains the six single runs, builds the two ensembles, evaluates everything on
BOTH frozen test sets, and emits results.json ready for the artifact generator.

    | Run | Encoder   | Data-flow |
    |-----|-----------|-----------|
    | 1   | gcn       | no        |
    | 2   | gcn_df    | yes       |
    | 3   | sage      | no        |
    | 4   | sage_df   | yes       |
    | 5   | gatv2     | no        |
    | 6   | gatv2_df  | yes       |
    | E1  | ensemble      (1+3+5) |
    | E2  | ensemble_df   (2+4+6) |

Reproducibility (non-negotiable #2): every run writes its resolved config, the
git commit hash, and the installed package versions next to its checkpoint.

Protocol (identical to v1): Adam, weighted BCE with pos_weight recomputed on the
new class frequencies, early stopping on validation macro-F1, seed 42.

THRESHOLDS ARE TUNED ON VALIDATION ONLY and then applied frozen to both test
sets (non-negotiable #3). Nothing in this script ever fits on a test set.

Usage
-----
    PYTHONPATH=. python scripts/train_v2.py \
        --data-nodf data/processed_nodf --data-df data/processed_df \
        --out runs/v2 --seeds 42
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAWS

ENCODERS = ["gcn", "sage", "gatv2"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    out = {}
    for mod in ("torch", "torch_geometric", "numpy", "sklearn", "transformers"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    out["python"] = sys.version.split()[0]
    return out


def _loaders(data_dir: Path, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader
    from training.train.collate import collate_pairs
    from training.train.dataset import ContractPairDataset

    kw = dict(collate_fn=collate_pairs, num_workers=num_workers,
              pin_memory=(num_workers > 0), persistent_workers=(num_workers > 0))
    ds = {s: ContractPairDataset(str(data_dir / f"{s}_index.json"))
          for s in ("train", "val", "test_a", "test_b")}
    loaders = {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True, **kw),
        "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False, **kw),
        "test_a": DataLoader(ds["test_a"], batch_size=batch_size, shuffle=False, **kw),
        "test_b": DataLoader(ds["test_b"], batch_size=batch_size, shuffle=False, **kw),
    }
    train_labels = np.vstack([ds["train"][i][2].numpy() for i in range(len(ds["train"]))])
    return loaders, train_labels


def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, probs) for a whole split. No thresholding here."""
    import torch

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for ast, cfg, y in loader:
            ast, cfg = ast.to(device), cfg.to(device)
            ps.append(torch.sigmoid(model(ast, cfg)).cpu().numpy())
            ys.append(y.numpy())
    if not ys:
        z = np.zeros((0, len(FLAWS)))
        return z, z
    return (np.vstack(ys) >= 0.5).astype(int), np.vstack(ps)


def evaluate_probs(y_true, probs, thresholds, *, n_boot: int, seed: int) -> dict:
    from training.evaluate.metrics import (
        apply_thresholds, bootstrap_ci, confusion_per_class, full_metrics,
    )
    y_pred = apply_thresholds(probs, thresholds)
    m = full_metrics(y_true, y_pred)
    m["ci"] = bootstrap_ci(y_true, y_pred, n_resamples=n_boot, seed=seed)
    m["confusion"] = confusion_per_class(y_true, y_pred)
    m["thresholds"] = list(thresholds)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-nodf", required=True, help="Build WITHOUT data-flow edges.")
    ap.add_argument("--data-df", required=True, help="Build WITH data-flow edges.")
    ap.add_argument("--configs", default="configs", help="Directory of YAML configs.")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="One seed (42) by default; give 42 43 44 to repeat the matrix.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Bootstrap resamples for the 95%% CIs.")
    args = ap.parse_args()

    import torch

    from scgnn.extraction.features import FeatureConfig
    from training.config import load_config
    from training.evaluate.metrics import ensemble_probs, tune_thresholds
    from training.train.train import train_model

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provenance = {"git_hash": git_hash(), "packages": package_versions(),
                  "device": device, "seeds": args.seeds}
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2),
                                              encoding="utf-8")
    print("provenance:", json.dumps(provenance["packages"]))

    results: dict[str, dict] = {}
    # probs[(seed, arm)][run] -> {"val":(y,p), "test_a":(y,p), "test_b":(y,p)}
    cached: dict[tuple[int, str], dict[str, dict]] = {}

    for seed in args.seeds:
        for arm, data_dir in (("nodf", Path(args.data_nodf)),
                              ("df", Path(args.data_df))):
            loaders, train_labels = _loaders(data_dir, 256, args.num_workers)
            in_dim = FeatureConfig.from_json(
                str(data_dir / "feature_config.json")).in_dim
            cached[(seed, arm)] = {}

            for enc in ENCODERS:
                run = enc if arm == "nodf" else f"{enc}_df"
                tag = run if len(args.seeds) == 1 else f"{run}_s{seed}"
                run_dir = out_root / tag
                print(f"\n=== {tag}  (encoder={enc}, data-flow={arm=='df'}, "
                      f"seed={seed}, in_dim={in_dim}) ===", flush=True)

                cfg = load_config(str(Path(args.configs) / f"{enc}.yaml"))
                cfg["in_dim"] = in_dim
                cfg["conv"] = enc                    # gatv2 needs the conv name set
                cfg["seed"] = seed
                if args.epochs:
                    cfg["epochs"] = args.epochs
                if args.patience:
                    cfg["patience"] = args.patience
                if args.batch_size:
                    cfg["batch_size"] = args.batch_size
                set_seed(seed)

                info = train_model(cfg, loaders["train"], loaders["val"],
                                   train_labels, str(run_dir))
                print(" ", info)

                # reproducibility record next to the checkpoint (non-negotiable #2)
                (run_dir / "provenance.json").write_text(
                    json.dumps({**provenance, "seed": seed, "config": cfg,
                                "data_dir": str(data_dir), "with_data_flow": arm == "df"},
                               indent=2), encoding="utf-8")

                # predict once per split; thresholds come from VAL only
                from scgnn.models.dual_gnn import build_model
                model = build_model(cfg).to(device)
                model.load_state_dict(torch.load(run_dir / "best_model.pt",
                                                 map_location=device))
                probs = {s: predict_probs(model, loaders[s], device)
                         for s in ("val", "test_a", "test_b")}
                cached[(seed, arm)][run] = probs

                thr = tune_thresholds(*probs["val"])          # VAL ONLY
                res = {"val": evaluate_probs(*probs["val"], thr,
                                             n_boot=args.bootstrap, seed=seed)}
                for split in ("test_a", "test_b"):
                    res[split] = evaluate_probs(*probs[split], thr,
                                                n_boot=args.bootstrap, seed=seed)
                results[tag] = res
                print(f"  val {res['val']['macro']['f1']:.3f} | "
                      f"A {res['test_a']['macro']['f1']:.3f} | "
                      f"B {res['test_b']['macro']['f1']:.3f}")

            # ---- ensemble for this arm (Workstream E) ----
            runs = list(cached[(seed, arm)])
            if len(runs) >= 2:
                name = "ensemble" if arm == "nodf" else "ensemble_df"
                tag = name if len(args.seeds) == 1 else f"{name}_s{seed}"
                print(f"\n=== {tag}  (mean of {runs}) ===", flush=True)
                ens = {}
                for split in ("val", "test_a", "test_b"):
                    y = cached[(seed, arm)][runs[0]][split][0]
                    p = ensemble_probs([cached[(seed, arm)][r][split][1] for r in runs],
                                       policy="mean")
                    ens[split] = (y, p)
                thr = tune_thresholds(*ens["val"])           # VAL ONLY, as for singles
                res = {s: evaluate_probs(*ens[s], thr, n_boot=args.bootstrap, seed=seed)
                       for s in ("val", "test_a", "test_b")}
                # also record the max policy on the test sets, for the record
                for split in ("test_a", "test_b"):
                    pmax = ensemble_probs(
                        [cached[(seed, arm)][r][split][1] for r in runs], policy="max")
                    res[split]["max_policy_macro_f1"] = evaluate_probs(
                        ens[split][0], pmax, thr, n_boot=1, seed=seed)["macro"]["f1"]
                results[tag] = res
                (out_root / tag).mkdir(parents=True, exist_ok=True)
                (out_root / tag / "thresholds.json").write_text(
                    json.dumps({"thresholds": thr, "members": runs}, indent=2),
                    encoding="utf-8")
                print(f"  val {res['val']['macro']['f1']:.3f} | "
                      f"A {res['test_a']['macro']['f1']:.3f} | "
                      f"B {res['test_b']['macro']['f1']:.3f}")

    (out_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Test B probabilities per model, in the manifest's contract order, so
    # scripts/durieux_baseline.py can place the models beside the four tools in
    # one matrix. Thresholds travel with them (they were tuned on VAL).
    tb_probs = {}
    for seed in args.seeds:
        for arm in ("nodf", "df"):
            for run, probs in cached.get((seed, arm), {}).items():
                tag = run if len(args.seeds) == 1 else f"{run}_s{seed}"
                tb_probs[tag] = {
                    "probs": probs["test_b"][1].tolist(),
                    "thresholds": results[tag]["test_b"]["thresholds"],
                }
    (out_root / "test_b_probs.json").write_text(json.dumps(tb_probs), encoding="utf-8")
    print(f"test-B probabilities -> {out_root / 'test_b_probs.json'} "
          f"({len(tb_probs)} models, for the Durieux matrix)")

    best = max(results, key=lambda m: results[m]["test_b"]["macro"]["f1"])
    print("\n" + "=" * 64)
    print(f"{'run':<16} {'val':>7} {'test A':>8} {'test B':>8}")
    print("-" * 64)
    for m in sorted(results):
        r = results[m]
        print(f"{m:<16} {r['val']['macro']['f1']:>7.3f} "
              f"{r['test_a']['macro']['f1']:>8.3f} {r['test_b']['macro']['f1']:>8.3f}")
    print("=" * 64)
    print(f"best on the expert test set (B): {best} "
          f"({results[best]['test_b']['macro']['f1']:.3f})")
    print(f"\nresults -> {out_root / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
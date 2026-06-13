#!/usr/bin/env python3
"""Top-k line-localisation accuracy over the frozen Curated test split.

For every test contract that carries expert gold lines, this runs the real
`explain_lines` for the flaw(s) the contract is annotated with, then scores the
ranked predicted lines against the gold lines with the proposal's metric:

    accuracy@k = fraction of flawed contracts whose top-k predicted lines
                 include at least one expert-marked line.

GNNExplainer is stochastic (random mask initialisation), so a single run is not
reproducible by default. This script seeds every run, making one pass
deterministic, and `--repeats N` runs N seeded passes (pass r uses seed+r) and
reports `mean ± std` — the figure to cite. It reuses the build caches
(`records/<hash>.pt`, `raw/<hash>.json`); nothing is re-extracted or re-embedded.
Explanation runs on CPU by default (matches the API, and is reproducible).

Example (single reproducible run):
    PYTHONPATH=. python scripts/localise_eval.py \
        --checkpoint runs/sage/best_model.pt --config configs/sage.yaml

Example (citable mean ± std over 5 seeded runs):
    PYTHONPATH=. python scripts/localise_eval.py --repeats 5 \
        --checkpoint runs/sage/best_model.pt --config configs/sage.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KS = (1, 3, 5)


def _dedup_keep_order(lines: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a pass is reproducible."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_pass(model, targets, gold_map, processed: Path, device: str,
              k: int, epochs: int, verbose: bool):
    """One full pass over all localisation targets. Returns per-contract rows."""
    import torch
    from torch_geometric.data import Data

    from scgnn.explain.explainer import explain_lines
    from scgnn.extraction.graph_types import RawGraph
    from scgnn.schema import FLAWS, display_name

    rows: list[dict] = []
    total = len(targets)
    for n, entry in enumerate(targets, 1):
        cid = entry["id"]
        gold = sorted(set(gold_map[cid]))
        # filename stem is the content hash, shared by records/ and raw/;
        # rebuild from --processed so a path-prefix mismatch can't drop a contract
        h = Path(entry["path"]).stem
        rec_path = processed / "records" / f"{h}.pt"
        rawj = processed / "raw" / f"{h}.json"
        if not rec_path.exists() or not rawj.exists():
            if verbose:
                print(f"  [{n}/{total}] {cid}: SKIP (missing record/raw cache for {h})")
            continue
        rec = torch.load(rec_path, weights_only=True)
        d = json.loads(rawj.read_text(encoding="utf-8"))
        ast_raw = RawGraph.from_dict(d["ast"])
        cfg_raw = RawGraph.from_dict(d["cfg"])
        ast_data = Data(x=rec["ast_x"], edge_index=rec["ast_edge_index"]).to(device)
        cfg_data = Data(x=rec["cfg_x"], edge_index=rec["cfg_edge_index"]).to(device)

        positive = [j for j, v in enumerate(rec["y"].tolist()) if v >= 0.5]
        if not positive:
            continue

        merged: list[int] = []
        for j in positive:
            lines, _unmapped = explain_lines(
                model, ast_data, cfg_data, j,
                ast_raw.node_lines, cfg_raw.node_lines, k=k, epochs=epochs,
            )
            merged.extend(lines)
        pred = _dedup_keep_order(merged)
        rows.append({"id": cid, "flaws": [FLAWS[j] for j in positive],
                     "gold": gold, "pred": pred})
        if verbose:
            hit = "\u2713" if set(pred[:k]) & set(gold) else "\u00b7"
            flaws = "/".join(display_name(FLAWS[j]) for j in positive)
            print(f"  [{n}/{total}] {hit} {cid} [{flaws}]  gold={gold}  "
                  f"pred@{k}={pred[:k]}", flush=True)
    return rows


def _accuracy(rows: list[dict], ks=KS) -> dict[int, float]:
    """Accuracy@k via the canonical metric, to avoid a second definition."""
    from training.evaluate.localisation import top_k_localisation

    pred_lines = [r["pred"] for r in rows]
    gold_lines = [set(r["gold"]) for r in rows]
    return top_k_localisation(pred_lines, gold_lines, ks=ks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/sage/best_model.pt")
    ap.add_argument("--config", default="configs/sage.yaml")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--feature-config", default=None,
                    help="Defaults to <processed>/feature_config.json (used only for in_dim).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=5, help="Lines requested per branch / final.")
    ap.add_argument("--epochs", type=int, default=150, help="GNNExplainer steps per branch.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Seeded passes; reports mean +/- std when >1.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed; pass r uses seed+r.")
    ap.add_argument("--out", default=None,
                    help="Defaults to <processed>/localisation_report.json")
    args = ap.parse_args()

    import torch

    from scgnn.extraction.features import FeatureConfig
    from scgnn.models.dual_gnn import build_model
    from training.config import load_config

    processed = Path(args.processed)
    fc_path = args.feature_config or str(processed / "feature_config.json")
    feat_cfg = FeatureConfig.from_json(fc_path)

    config = load_config(args.config)
    config["in_dim"] = feat_cfg.in_dim
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.to(args.device).eval()
    print(f"loaded {config.get('conv')} model (in_dim={feat_cfg.in_dim}) on {args.device}")

    test_index = json.loads((processed / "test_index.json").read_text(encoding="utf-8"))
    gold_map = json.loads((processed / "curated_gold_lines.json").read_text(encoding="utf-8"))
    targets = [e for e in test_index if gold_map.get(e["id"])]
    print(f"{len(test_index)} test contracts; {len(targets)} carry gold lines "
          f"(localisation targets)")
    print(f"{args.repeats} seeded pass(es), base seed {args.seed}, "
          f"{args.epochs} explainer epochs\n")

    acc_samples: dict[int, list[float]] = {k: [] for k in KS}
    hit_counts: dict[str, dict[int, int]] = {}
    per_pass: list[dict] = []
    example_rows: list[dict] | None = None

    for r in range(args.repeats):
        seed = args.seed + r
        _seed_everything(seed)
        verbose = (r == 0)
        if args.repeats > 1 and verbose:
            print(f"--- pass 1/{args.repeats} (seed={seed}); per-contract detail ---")
        rows = _run_pass(model, targets, gold_map, processed, args.device,
                         args.k, args.epochs, verbose)
        acc = _accuracy(rows)
        per_pass.append({"seed": seed,
                         "accuracy_at_k": {str(k): round(acc[k], 4) for k in KS}})
        for k in KS:
            acc_samples[k].append(acc[k])
        for row in rows:
            hc = hit_counts.setdefault(row["id"], {k: 0 for k in KS})
            for k in KS:
                if set(row["pred"][:k]) & set(row["gold"]):
                    hc[k] += 1
        if example_rows is None:
            example_rows = rows
        print(f"pass {r + 1}/{args.repeats} (seed={seed}): "
              f"@1={acc[1]:.3f}  @3={acc[3]:.3f}  @5={acc[5]:.3f}", flush=True)

    n = len(example_rows) if example_rows else 0
    mean = {k: statistics.mean(acc_samples[k]) for k in KS}
    std = {k: (statistics.stdev(acc_samples[k]) if args.repeats > 1 else 0.0) for k in KS}

    print("\n=== localisation accuracy (Curated test split) ===")
    if args.repeats > 1:
        for k in KS:
            print(f"  accuracy@{k}: {mean[k]:.3f} \u00b1 {std[k]:.3f}   "
                  f"(mean of {args.repeats} seeded runs)")
    else:
        for k in KS:
            print(f"  accuracy@{k}: {mean[k]:.3f}   ({round(mean[k] * n)}/{n})")

    per_contract = []
    for row in example_rows or []:
        hc = hit_counts[row["id"]]
        per_contract.append({
            "id": row["id"], "flaws": row["flaws"], "gold": row["gold"],
            "hit_at_k_over_runs": {str(k): hc[k] for k in KS},
            "example_pred": row["pred"][:args.k],
        })

    report = {
        "n_localisation_targets": n,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "epochs": args.epochs,
        "device": args.device,
        "accuracy_at_k_mean": {str(k): round(mean[k], 4) for k in KS},
        "accuracy_at_k_std": {str(k): round(std[k], 4) for k in KS},
        "passes": per_pass,
        "per_contract": per_contract,
    }
    outp = args.out or str(processed / "localisation_report.json")
    Path(outp).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
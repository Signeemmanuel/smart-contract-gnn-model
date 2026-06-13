#!/usr/bin/env python3
"""Smoke-test line-level localisation on ONE contract, using local artefacts.

`scripts/explain.py` loads the model from the Hub; this loads it from a finished
run on disk, so you can exercise the GNNExplainer path before publishing. It
extracts the contract, prints the per-flaw probabilities, then runs the real
`explain_lines` for the top-predicted class (always) plus any class above
`--threshold` (and an optional forced `--class`), printing the ranked source
lines. The explainer is the riskiest module, so this is the run that proves it.

Example:
    PYTHONPATH=. python scripts/explain_local.py \
        data/raw/curated/dataset/reentrancy/simple_dao.sol \
        --checkpoint runs/sage/best_model.pt --config configs/sage.yaml
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("contract", help="Path to a .sol file to analyse.")
    ap.add_argument("--checkpoint", default="runs/sage/best_model.pt")
    ap.add_argument("--config", default="configs/sage.yaml")
    ap.add_argument("--feature-config", default="data/processed/feature_config.json")
    ap.add_argument("--pca", default="data/processed/pca.joblib")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; matches the API and is safest for GNNExplainer) or cuda.")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Also explain any class with probability >= this.")
    ap.add_argument("--class", dest="force_class", default=None,
                    help="Force-explain this flaw (code or index), regardless of probability.")
    ap.add_argument("--k", type=int, default=5, help="Top-k lines per branch.")
    ap.add_argument("--epochs", type=int, default=150, help="GNNExplainer optimisation steps.")
    args = ap.parse_args()

    import joblib
    import torch

    from scgnn.explain.explainer import explain_lines
    from scgnn.extraction.extract import extract_contract
    from scgnn.extraction.features import CodeBERTEmbedder, FeatureConfig, FeatureEncoder
    from scgnn.models.dual_gnn import build_model
    from scgnn.schema import FLAWS, FLAW_INDEX
    from training.config import load_config

    # --- build the model + encoder from local files (no Hub) ------------------
    config = load_config(args.config)
    feat_cfg = FeatureConfig.from_json(args.feature_config)
    config["in_dim"] = feat_cfg.in_dim
    model = build_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.to(args.device).eval()
    pca = joblib.load(args.pca)
    encoder = FeatureEncoder(feat_cfg, CodeBERTEmbedder(device=args.device), pca)
    print(f"loaded {config.get('conv')} model (in_dim={feat_cfg.in_dim}) on {args.device}")

    # --- extract this one contract -------------------------------------------
    print(f"extracting {args.contract} ...", flush=True)
    ast_raw, cfg_raw = extract_contract(args.contract)
    print(f"  ast: {ast_raw.n_nodes} nodes / {len(ast_raw.edges)} edges | "
          f"cfg: {cfg_raw.n_nodes} nodes / {len(cfg_raw.edges)} edges")
    ast_data = encoder.to_data(ast_raw).to(args.device)
    cfg_data = encoder.to_data(cfg_raw).to(args.device)

    # --- predicted probabilities for all five flaws --------------------------
    ast_b = ast_data.clone(); ast_b.batch = torch.zeros(ast_data.x.size(0), dtype=torch.long, device=args.device)
    cfg_b = cfg_data.clone(); cfg_b.batch = torch.zeros(cfg_data.x.size(0), dtype=torch.long, device=args.device)
    with torch.no_grad():
        proba = model.predict_proba(ast_b, cfg_b).squeeze(0).tolist()
    print("\nper-flaw probability:")
    for flaw, p in zip(FLAWS, proba):
        print(f"  {flaw:<16} {p:.3f}")

    # --- decide which classes to explain (always at least the top one) -------
    to_explain = {int(max(range(len(proba)), key=lambda i: proba[i]))}
    to_explain |= {i for i, p in enumerate(proba) if p >= args.threshold}
    if args.force_class is not None:
        idx = FLAW_INDEX[args.force_class] if args.force_class in FLAW_INDEX else int(args.force_class)
        to_explain.add(idx)

    print(f"\nexplaining classes: {sorted(FLAWS[i] for i in to_explain)}")
    for idx in sorted(to_explain):
        lines, unmapped = explain_lines(
            model, ast_data, cfg_data, idx,
            ast_raw.node_lines, cfg_raw.node_lines, k=args.k, epochs=args.epochs,
        )
        print(f"\n[{FLAWS[idx]}]  p={proba[idx]:.3f}")
        print(f"  ranked lines: {lines}")
        print(f"  unmapped nodes: {len(unmapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Assemble a versioned Hub release bundle from a finished run.

Status: needs torch + safetensors; run after training. Collects the weights
(converted to safetensors), the architecture config, the fitted PCA, the feature
config, and - new in v2 - a MANIFEST describing everything the API needs to load
and use the model correctly.

v2 manifest (non-negotiable #5)
-------------------------------
The deployed API selects a bundle BY VERSION, so v1's bundle must never be
overwritten. The manifest records what v1 left implicit and what v2 makes
explicit:

    bundle_version   v1 | v2
    architecture     gcn | sage | gatv2
    in_dim           data-derived input width (NOT a hyper-parameter: it changes
                     with the node-type vocabulary of the training corpus)
    edge_schema      control_flow  |  control_flow+data_flow
    thresholds       PER-CLASS decision thresholds, tuned on validation
    class_names      the five flaw codes, in canonical order
    training_set     size of the train split the model was fitted on
    git_hash         the commit that produced the weights
    created          UTC timestamp

The per-class thresholds are the important addition. v1 shipped a single scalar
threshold in config.json; v2 tunes one threshold per class on validation, and a
deployment that ignored them would silently apply the wrong decision rule.
``scgnn/inference.py`` reads ``thresholds`` from the manifest when present and
falls back to the v1 scalar otherwise, so both bundles keep working.

Usage
-----
    python scripts/build_release_bundle.py \
        --checkpoint runs/v2/sage_df/best_model.pt \
        --config runs/v2/sage_df/config.json \
        --feature-config data/processed_df/feature_config.json \
        --pca data/processed_df/pca.joblib \
        --results runs/v2/results.json --run-name sage_df \
        --train-index data/processed_df/train_index.json \
        --with-data-flow \
        --bundle-version v2 \
        --out release/sage_df_v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scgnn.schema import FLAWS


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best_model.pt from a run.")
    ap.add_argument("--config", required=True, help="config.json from the run.")
    ap.add_argument("--feature-config", required=True, help="feature_config.json.")
    ap.add_argument("--pca", required=True, help="pca.joblib.")
    ap.add_argument("--out", required=True, help="Bundle output folder.")
    ap.add_argument("--bundle-version", default="v2", help="v1 | v2. Never overwrite v1.")
    ap.add_argument("--results", default=None,
                    help="results.json, to read the run's validation-tuned thresholds.")
    ap.add_argument("--run-name", default=None,
                    help="Key into results.json (e.g. sage_df).")
    ap.add_argument("--train-index", default=None,
                    help="train_index.json, to record the training-set size.")
    ap.add_argument("--thresholds", type=float, nargs=5, default=None,
                    help="Per-class thresholds, if not taken from results.json.")
    df = ap.add_mutually_exclusive_group()
    df.add_argument("--with-data-flow", dest="with_data_flow", action="store_true")
    df.add_argument("--no-data-flow", dest="with_data_flow", action="store_false")
    ap.set_defaults(with_data_flow=None)
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and args.bundle_version == "v1":
        raise SystemExit(f"REFUSING to write a v1 bundle into a non-empty {out}: "
                         f"v1's bundle must stay untouched (non-negotiable #5).")
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    save_file(state, str(out / "model.safetensors"))
    shutil.copy(args.config, out / "config.json")
    shutil.copy(args.feature_config, out / "feature_config.json")
    shutil.copy(args.pca, out / "pca.joblib")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    feat = json.loads(Path(args.feature_config).read_text(encoding="utf-8"))

    # --- per-class thresholds (the v2 addition that must not be lost) ---
    thresholds = args.thresholds
    if thresholds is None and args.results and args.run_name:
        res = json.loads(Path(args.results).read_text(encoding="utf-8"))
        run = res.get(args.run_name, {})
        thresholds = run.get("val", {}).get("thresholds")
        if thresholds is None:
            thresholds = run.get("test_b", {}).get("thresholds")
    if thresholds is None:
        scalar = float(config.get("threshold", 0.5))
        thresholds = [scalar] * len(FLAWS)
        print(f"WARNING: no tuned thresholds found; falling back to the scalar "
              f"{scalar} for every class. The deployed model will NOT use the "
              f"validation-tuned per-class rule.")

    # --- training-set size ---
    train_size = None
    if args.train_index and Path(args.train_index).exists():
        train_size = len(json.loads(Path(args.train_index).read_text(encoding="utf-8")))

    # --- edge schema ---
    if args.with_data_flow is None:
        # infer from the run name if we can, else say so honestly
        wdf = bool(args.run_name and args.run_name.endswith("_df"))
    else:
        wdf = bool(args.with_data_flow)
    edge_schema = "control_flow+data_flow" if wdf else "control_flow"

    manifest = {
        "bundle_version": args.bundle_version,
        "architecture": config.get("conv"),
        "in_dim": feat.get("in_dim", config.get("in_dim")),
        "edge_schema": edge_schema,
        "thresholds": [float(t) for t in thresholds],
        "class_names": list(FLAWS),
        "training_set_size": train_size,
        "git_hash": git_hash(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "hid": config.get("hid"),
        "layers": config.get("layers"),
        "heads": config.get("heads"),
        "dropout": config.get("dropout"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"bundle ({args.bundle_version}) ready at {out}")
    print("files:", sorted(p.name for p in out.iterdir()))
    print("manifest:", json.dumps(manifest, indent=2))
    print(f"\nnow: python scripts/publish_model.py --bundle {out} "
          f"--repo-id <id> --version {args.bundle_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
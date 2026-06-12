#!/usr/bin/env python3
"""Assemble a Hub release bundle from a finished run.

Status: needs torch + safetensors; run on the Studio after training. Collects the
weights (converted to safetensors), the architecture config, the fitted PCA and
the feature config into one folder ready for scripts/publish_model.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="best_model.pt from a run.")
    ap.add_argument("--config", required=True, help="config.json from the run.")
    ap.add_argument("--feature-config", required=True, help="feature_config.json.")
    ap.add_argument("--pca", required=True, help="pca.joblib.")
    ap.add_argument("--out", required=True, help="Bundle output folder.")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    save_file(state, str(out / "model.safetensors"))
    shutil.copy(args.config, out / "config.json")
    shutil.copy(args.feature_config, out / "feature_config.json")
    shutil.copy(args.pca, out / "pca.joblib")
    print("bundle ready at", out)
    print("files:", sorted(p.name for p in out.iterdir()))
    print("now: python scripts/publish_model.py --bundle", out, "--repo-id <id> --version vX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

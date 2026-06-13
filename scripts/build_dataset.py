#!/usr/bin/env python3
"""Build the processed dataset (records + train/val/test indices + fitted artefacts).

Status: needs solc/Slither + CodeBERT + torch; run on the Studio. The split
planning is unit-tested; this wires it to the real extractor and embedder.

Example:
    PYTHONPATH=. python scripts/build_dataset.py \
        --wild-dir data/raw/wild --wild-labels data/processed/labels.parquet \
        --curated-dir data/raw/curated --out data/processed \
        --max-wild 500          # cap Wild for a fast end-to-end smoke run first
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAWS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wild-dir", required=True, help="Root of Wild .sol files.")
    ap.add_argument("--wild-labels", required=True, help="labels.parquet from scripts/label.py.")
    ap.add_argument("--curated-dir", required=True, help="smartbugs-curated checkout root.")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--max-wild", type=int, default=None, help="Cap Wild for a smoke run.")
    ap.add_argument("--extract-timeout", type=float, default=120,
                    help="Per-contract extraction time budget (s); a hung Slither/solc is "
                         "aborted and the contract skipped.")
    ap.add_argument("--embed-batch", type=int, default=128,
                    help="CodeBERT embedding batch size (raise to use more GPU).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--solc", default=None,
                    help="Force one solc binary for every contract; default resolves "
                         "the right compiler per pragma from installed solc-select binaries.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    import logging

    import pandas as pd

    # Slither logs IR-generation failures at ERROR for individual functions
    # (e.g. old `call.value()` sends); these are non-fatal noise for CFG/AST
    # extraction, so quiet them to keep the build output readable.
    for _name in ("Slither", "SlitherSolcParsing", "CryticCompile", "crytic_compile"):
        logging.getLogger(_name).setLevel(logging.CRITICAL)

    from scgnn.extraction.extract import extract_contract
    from scgnn.extraction.features import CodeBERTEmbedder
    from scgnn.extraction.solc import installed_solc_binaries
    from training.data.build import materialise, plan_splits

    binaries = installed_solc_binaries()
    if args.solc is None and not binaries:
        print("WARNING: no solc-select binaries discovered under ~/.solc-select/artifacts; "
              "extraction will fall back to PATH solc and likely fail on old contracts. "
              "Install them with e.g. 'solc-select install 0.4.26 0.5.17 0.6.12 0.7.6 0.8.19'.")
    elif args.solc is None:
        print("solc binaries by minor:", {k: Path(v).name for k, v in binaries.items()})
    from training.data.curated import load_curated

    wild_paths = {p.stem: str(p) for p in Path(args.wild_dir).rglob("*.sol")}
    df = pd.read_parquet(args.wild_labels)
    wild_labels = {str(r["contract"]): [int(r[f]) for f in FLAWS] for _, r in df.iterrows()}
    curated = load_curated(args.curated_dir)
    print(f"wild={len(wild_paths)} contracts, labelled={len(wild_labels)}, "
          f"curated={len(curated)}")

    plan = plan_splits(wild_paths, wild_labels, curated, val_frac=args.val_frac,
                       test_frac=args.test_frac, seed=args.seed, max_wild=args.max_wild)
    print("split plan:", plan.counts)

    embedder = CodeBERTEmbedder(device=args.device)
    extract_fn = lambda path: extract_contract(path, solc_binary=args.solc, binaries=binaries)
    report = materialise(plan, extract_fn, embedder, args.out,
                         embed_dim=args.embed_dim, seed=args.seed,
                         extract_timeout=args.extract_timeout, embed_batch=args.embed_batch)
    print("build report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

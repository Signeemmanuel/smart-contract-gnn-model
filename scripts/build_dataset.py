#!/usr/bin/env python3
"""Build the processed dataset (v2): records + four split indices + fitted artefacts.

Reads the FROZEN test manifests (scripts/freeze_testsets.py must have run) and
firewalls train/val against the union of both, so no test contract can be
trained on. Extraction adds data-flow edges to the CFG view; the ablation's
"without" arm reuses the SAME extraction cache with those edges stripped, so
extraction is paid for exactly once across both arms.

Order of operations:
    label_orchestrator -> label.py -> freeze_testsets.py -> [THIS] -> train_v2.py

Two builds are needed for the ablation, sharing one cache:

    # WITH data-flow edges (runs 2, 4, 6)
    PYTHONPATH=. python scripts/build_dataset.py \
        --wild-dir data/raw/wild --wild-labels data/processed/labels.parquet \
        --curated-dir data/raw/curated --testsets data/testsets \
        --out data/processed_df --cache data/extract_cache --with-data-flow

    # WITHOUT data-flow edges (runs 1, 3, 5) - reuses the same cache
    PYTHONPATH=. python scripts/build_dataset.py \
        --wild-dir data/raw/wild --wild-labels data/processed/labels.parquet \
        --curated-dir data/raw/curated --testsets data/testsets \
        --out data/processed_nodf --cache data/extract_cache --no-data-flow
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAWS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild-dir", required=True, help="Root of Wild .sol files.")
    ap.add_argument("--wild-labels", required=True,
                    help="labels.parquet from scripts/label.py (union rule).")
    ap.add_argument("--curated-dir", required=True,
                    help="smartbugs-curated checkout (for Test B gold lines).")
    ap.add_argument("--testsets", default="data/testsets",
                    help="Directory holding the frozen test_a.csv / test_b.csv.")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--cache", default=None,
                    help="Shared extraction cache dir. Point BOTH ablation builds at "
                         "the same cache so Slither runs once, not twice.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--max-wild", type=int, default=None, help="Cap Wild for a smoke run.")
    ap.add_argument("--extract-timeout", type=float, default=120)
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--pca-fit-sample", type=int, default=200_000,
                    help="Max snippets embedded to FIT the PCA (0 = all). Bounds the "
                         "fit's memory at full-corpus scale.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--solc", default=None, help="Force one solc binary for every contract.")
    ap.add_argument("--seed", type=int, default=42)

    df = ap.add_mutually_exclusive_group()
    df.add_argument("--with-data-flow", dest="with_data_flow", action="store_true",
                    help="Add data-flow (def-use) edges to the CFG view (default).")
    df.add_argument("--no-data-flow", dest="with_data_flow", action="store_false",
                    help="Ablation arm: strip data-flow edges from the cached graphs.")
    ap.set_defaults(with_data_flow=True)

    args = ap.parse_args()
    set_seed(args.seed)

    import json
    import logging

    import pandas as pd

    # Slither logs IR-generation failures at ERROR for individual functions; these
    # are non-fatal for AST/CFG extraction, so quiet them to keep the log readable.
    for name in ("Slither", "SlitherSolcParsing", "CryticCompile", "crytic_compile"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    from scgnn.extraction.extract import extract_contract
    from scgnn.extraction.features import CodeBERTEmbedder
    from scgnn.extraction.solc import installed_solc_binaries, installed_solc_full
    from training.data.build import materialise, plan_splits_v2
    from training.data.curated import load_curated

    ts = Path(args.testsets)
    mA, mB = ts / "test_a.csv", ts / "test_b.csv"
    for m in (mA, mB):
        if not m.exists():
            raise SystemExit(
                f"ERROR: frozen manifest {m} not found.\n"
                f"  Run scripts/freeze_testsets.py first: the test sets must be "
                f"frozen BEFORE the build reads them.")

    binaries = installed_solc_binaries()
    full_binaries = installed_solc_full()
    if args.solc is None and not binaries:
        print("WARNING: no solc-select binaries found; extraction will fall back to "
              "PATH solc and will fail on pinned contracts.")
    elif args.solc is None:
        print("solc by minor:", {k: Path(v).name for k, v in binaries.items()})

    # Wild pool + its union labels.
    wild_paths = {p.stem: str(p) for p in Path(args.wild_dir).rglob("*.sol")
                  if "__MACOSX" not in p.parts and not p.name.startswith("._")}
    dfl = pd.read_parquet(args.wild_labels)
    wild_labels = {str(r["contract"]): [int(r[f]) for f in FLAWS] for _, r in dfl.iterrows()}

    # Curated line annotations, carried into Test B for localisation.
    curated = load_curated(args.curated_dir)
    curated_lines = {cid: rec.get("lines", []) for cid, rec in curated.items()}

    print(f"wild={len(wild_paths)} .sol, labelled={len(wild_labels)}")

    plan = plan_splits_v2(
        wild_paths, wild_labels,
        test_a_manifest=mA, test_b_manifest=mB,
        curated_lines=curated_lines,
        val_frac=args.val_frac, seed=args.seed, max_wild=args.max_wild,
    )
    print("split plan:", plan.counts)
    print(f"data-flow edges: {'ON' if args.with_data_flow else 'OFF (ablation arm)'}")

    # Share one extraction cache across both ablation builds: materialise() reads
    # and writes out/raw, so we point that at the shared cache by symlink.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cache:
        cache = Path(args.cache)
        cache.mkdir(parents=True, exist_ok=True)
        raw = out / "raw"
        if raw.is_symlink() or raw.exists():
            if raw.is_symlink():
                raw.unlink()
            elif raw.is_dir() and not any(raw.iterdir()):
                raw.rmdir()
        if not raw.exists():
            raw.symlink_to(cache.resolve(), target_is_directory=True)
        print(f"extraction cache: {cache} (shared across ablation arms)")

    embedder = CodeBERTEmbedder(device=args.device)
    extract_fn = lambda p: extract_contract(
        p, solc_binary=args.solc, binaries=binaries, full_binaries=full_binaries,
        with_data_flow=True,      # ALWAYS extract with DF; the ablation strips it
    )

    report = materialise(
        plan, extract_fn, embedder, args.out,
        embed_dim=args.embed_dim, seed=args.seed,
        extract_timeout=args.extract_timeout, embed_batch=args.embed_batch,
        pca_fit_sample=args.pca_fit_sample,
        with_data_flow=args.with_data_flow,
    )
    print("build report:", json.dumps({k: v for k, v in report.items() if k != "failed"},
                                      indent=2))
    print(f"  ({len(report['failed'])} contracts failed extraction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

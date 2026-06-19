#!/usr/bin/env python3
"""Build the processed dataset (records + train/val/test indices + fitted artefacts).

Status: needs solc/Slither + CodeBERT + torch; run on the Studio. The split
planning is unit-tested; this wires it to the real extractor and embedder.

Training data is supplied via ``--wild-dir`` + ``--wild-labels`` (now DIVE: see
scripts/prepare_dive.py). The expert TEST pool is ``--curated-dir`` (SmartBugs
Curated) optionally augmented with extra expert sources in the same folder/
annotation format:
  * ``--bit-dir``  : the BIT smartcontract-benchmark (recommended; line-annotated,
                     adds real per-class test support incl. dos via gasless_send)
  * ``--swc-dir``  : an SWC-registry checkout with test_cases/<id>/*.sol (older
                     layout; the current SWC repo no longer ships .sol files)
All extra sources are de-duplicated against Curated (and each other) by content
hash, then flow into the same frozen, stratified, firewalled test split.

Example:
    PYTHONPATH=. python scripts/build_dataset.py \
        --wild-dir data/raw/dive_sources --wild-labels data/processed/dive_labels.parquet \
        --curated-dir data/raw/curated \
        --bit-dir data/raw/bit-benchmark \
        --out data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAWS


def _merge_extra_test(curated: dict, recs: dict, source_name: str,
                      content_hash_of_file) -> None:
    """Merge extra expert test contracts into ``curated`` in place, de-duplicated.

    Drops any contract whose content hash already appears in the pool (Curated or
    a previously-merged source) or whose cid collides. Prints a one-line summary.
    """
    pool_hashes = {content_hash_of_file(v["path"]) for v in curated.values()}
    added = dropped = 0
    for cid, rec in recs.items():
        h = content_hash_of_file(rec["path"])
        if h in pool_hashes or cid in curated:
            dropped += 1
            continue
        curated[cid] = rec
        pool_hashes.add(h)
        added += 1
    print(f"{source_name} merge: +{added} test contracts, {dropped} dropped as duplicates")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wild-dir", required=True, help="Root of Wild/DIVE .sol files.")
    ap.add_argument("--wild-labels", required=True, help="labels.parquet (label.py or prepare_dive.py).")
    ap.add_argument("--curated-dir", required=True, help="smartbugs-curated checkout root.")
    ap.add_argument("--bit-dir", default=None,
                    help="Optional BIT smartcontract-benchmark checkout; its labelled "
                         "contracts are merged into the expert TEST pool (de-duplicated).")
    ap.add_argument("--swc-dir", default=None,
                    help="Optional SWC-registry checkout (older test_cases/<id>/*.sol "
                         "layout) merged into the expert TEST pool (de-duplicated).")
    ap.add_argument("--bit-no-safe", action="store_true",
                    help="Exclude BIT's explicit safe-contract negatives.")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--expert-train-frac", type=float, default=0.0,
                    help="Fraction of the expert pool (Curated+BIT) to hold back for "
                         "TRAIN; default 0.0 sends the whole expert pool to TEST, where "
                         "its gold labels give credible per-class metrics. Raise only if "
                         "you deliberately want some expert contracts in training.")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--max-wild", type=int, default=None, help="Cap Wild for a smoke run.")
    ap.add_argument("--extract-timeout", type=float, default=120,
                    help="Per-contract extraction time budget (s); a hung Slither/solc is "
                         "aborted and the contract skipped.")
    ap.add_argument("--embed-batch", type=int, default=256,
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
    from scgnn.extraction.solc import installed_solc_binaries, installed_solc_full
    from training.data.build import materialise, plan_splits

    binaries = installed_solc_binaries()
    full_binaries = installed_solc_full()
    if args.solc is None and not binaries:
        print("WARNING: no solc-select binaries discovered under ~/.solc-select/artifacts; "
              "extraction will fall back to PATH solc and likely fail on old contracts. "
              "Install them with e.g. 'solc-select install 0.4.26 0.5.17 0.6.12 0.7.6 0.8.19'.")
    elif args.solc is None:
        print("solc binaries by minor:", {k: Path(v).name for k, v in binaries.items()})
        print("exact solc versions available:", sorted(full_binaries))

    from training.data.curated import content_hash_of_file, load_curated

    wild_paths = {p.stem: str(p) for p in Path(args.wild_dir).rglob("*.sol")}
    df = pd.read_parquet(args.wild_labels)
    wild_labels = {str(r["contract"]): [int(r[f]) for f in FLAWS] for _, r in df.iterrows()}
    curated = load_curated(args.curated_dir)

    # Merge extra expert sources into the TEST pool (same {cid:{path,y,lines}} shape).
    if args.bit_dir:
        from training.data.bit import load_bit
        bit = load_bit(args.bit_dir, include_safe=not args.bit_no_safe)
        _merge_extra_test(curated, bit, "BIT", content_hash_of_file)
    if args.swc_dir:
        from training.data.swc import load_swc
        swc = load_swc(args.swc_dir)
        _merge_extra_test(curated, swc, "SWC", content_hash_of_file)

    # Report the test-pool composition by class, so you can see dos is now covered.
    pool_counts = {f: sum(rec["y"][i] for rec in curated.values()) for i, f in enumerate(FLAWS)}
    print(f"wild={len(wild_paths)} contracts, labelled={len(wild_labels)}, "
          f"test-pool={len(curated)} (pre-split)")
    print("test-pool positives by class (pre-split):", pool_counts)

    # NOTE: we deliberately do NOT pass test_frac here. The expert pool (Curated +
    # BIT) now defaults to TEST inside plan_splits (expert_train_frac=0.0); passing
    # test_frac would override that and revert to the old train-heavy split.
    plan = plan_splits(wild_paths, wild_labels, curated, val_frac=args.val_frac,
                       expert_train_frac=args.expert_train_frac,
                       seed=args.seed, max_wild=args.max_wild)
    print("split plan:", plan.counts)

    embedder = CodeBERTEmbedder(device=args.device)
    extract_fn = lambda path: extract_contract(path, solc_binary=args.solc,
                                                binaries=binaries, full_binaries=full_binaries)
    report = materialise(plan, extract_fn, embedder, args.out,
                         embed_dim=args.embed_dim, seed=args.seed,
                         extract_timeout=args.extract_timeout, embed_batch=args.embed_batch)
    print("build report:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
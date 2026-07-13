#!/usr/bin/env python3
"""Freeze the two test-set manifests (Workstream B). Runs BETWEEN labelling and
the build, exactly once per dataset version.

    label_orchestrator -> label.py -> [THIS] -> build_dataset.py -> train_v2.py

Test Set A (tool-labelled, internal benchmark)
  Drawn from the freshly labelled Wild pool, rarest class first so scarce
  positives (dos) are captured before commoner classes exhaust the budget, with
  negatives at ~2 per positive. Every contract whose content hash collides with
  a Curated contract is excluded, so Test A can never contain Test B provenance.

Test Set B (expert, external benchmark)
  The SmartBugs Curated contracts via load_curated, with line annotations kept
  for localisation.

Both are written as immutable CSV manifests (contract_id, path, chash, 5 labels)
and COMMITTED to the repo. Non-negotiable #3: once frozen, they are never
regenerated for the same dataset version, and nothing tunes on them.

The script REFUSES to overwrite an existing manifest unless --force is given,
because silently redrawing a frozen benchmark would invalidate every comparison
made against it.

Usage
-----
    PYTHONPATH=. python scripts/freeze_testsets.py \
        --wild-dir data/raw/wild \
        --labels data/processed/labels.parquet \
        --curated-dir data/raw/curated \
        --out data/testsets \
        --target-per-class 100 --min-per-class 60 --neg-ratio 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAWS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild-dir", required=True, help="Root of the Wild .sol pool.")
    ap.add_argument("--labels", required=True,
                    help="labels.parquet produced by scripts/label.py (union rule).")
    ap.add_argument("--curated-dir", required=True, help="smartbugs-curated checkout.")
    ap.add_argument("--out", default="data/testsets", help="Where the manifests go.")
    ap.add_argument("--target-per-class", type=int, default=100,
                    help="Positives per class to aim for in Test A.")
    ap.add_argument("--min-per-class", type=int, default=60,
                    help="Reported as a shortfall if a class falls below this.")
    ap.add_argument("--neg-ratio", type=float, default=2.0,
                    help="Negatives per positive contract in Test A.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing manifests. Only for a deliberate new "
                         "dataset version: it invalidates every earlier comparison.")
    args = ap.parse_args()
    set_seed(args.seed)

    import pandas as pd

    from training.data.curated import load_curated
    from training.data.firewall import content_hash
    from training.data.testsets import (
        select_test_a, select_test_b, write_manifest,
    )

    out = Path(args.out)
    mA, mB = out / "test_a.csv", out / "test_b.csv"
    if (mA.exists() or mB.exists()) and not args.force:
        sys.exit(f"REFUSING to overwrite frozen manifests in {out}.\n"
                 f"  The test sets are frozen (non-negotiable #3): redrawing them "
                 f"invalidates every result already compared against them.\n"
                 f"  Pass --force only if you are deliberately creating a NEW "
                 f"dataset version.")

    # ---- Test B first: it defines the hashes Test A must avoid ----
    curated = load_curated(args.curated_dir)
    rows_b, report_b = select_test_b(curated)
    curated_hashes = {r["chash"] for r in rows_b}
    print(f"Test B (expert Curated): {report_b['contracts']} contracts")
    print(f"  positives per class: {report_b['positives_per_class']}")

    # ---- Test A: from the labelled Wild pool, excluding Curated provenance ----
    df = pd.read_parquet(args.labels)
    labels = {str(r["contract"]): [int(r[f]) for f in FLAWS] for _, r in df.iterrows()}
    paths = {p.stem: str(p) for p in Path(args.wild_dir).rglob("*.sol")
             if "__MACOSX" not in p.parts and not p.name.startswith("._")}
    print(f"\nlabelled Wild pool: {len(labels)} labelled, {len(paths)} .sol on disk")

    rows_a = select_test_a(
        labels, paths,
        exclude_hashes=curated_hashes,
        min_per_class=args.min_per_class,
        target_per_class=args.target_per_class,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    per_class_a = {f: sum(int(r[f]) for r in rows_a) for f in FLAWS}
    n_neg_a = sum(1 for r in rows_a if all(int(r[f]) == 0 for f in FLAWS))
    print(f"\nTest A (tool-labelled): {len(rows_a)} contracts "
          f"({len(rows_a) - n_neg_a} positive, {n_neg_a} negative)")
    print(f"  positives per class: {per_class_a}")

    # Honest reporting: say plainly which classes fell short of the target.
    short = {f: n for f, n in per_class_a.items() if n < args.min_per_class}
    if short:
        print(f"\n  NOTE: below the {args.min_per_class}-positive minimum: {short}")
        print("  These classes have limited statistical power on Test A and must be "
              "reported as such (the dos class in particular is rare in Wild and "
              "poorly covered by the four tools).")

    # ---- freeze ----
    write_manifest(rows_a, mA)
    write_manifest(rows_b, mB)
    summary = {
        "seed": args.seed,
        "test_a": {"contracts": len(rows_a), "positives_per_class": per_class_a,
                   "negatives": n_neg_a, "below_minimum": short,
                   "target_per_class": args.target_per_class},
        "test_b": report_b,
        "curated_hashes_excluded_from_test_a": len(curated_hashes),
    }
    (out / "testsets_summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(f"\nfrozen:\n  {mA}\n  {mB}\n  {out / 'testsets_summary.json'}")
    print("\nCOMMIT these manifests to the repo. They are now immutable inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

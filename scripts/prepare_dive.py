#!/usr/bin/env python3
"""Prepare the DIVE dataset for this project's build pipeline.

DIVE (Zenodo 10.5281/zenodo.18519253) ships:
  * source:  <dive_root>/raw/Raw/PRE/Source codes/<contractID>.sol
  * labels:  <dive_root>/labels/Labels/DIVE_Labels.csv
    (columns: contractID, Reentrancy, Access Control, Arithmetic,
     Unchecked Return Values, DoS, Bad Randomness, Front Running, Time manipulation)

This project keys contracts by file stem and consumes, from scripts/build_dataset.py:
  * a directory of .sol files (``--wild-dir``), and
  * a ``labels.parquet`` with columns ``contract`` + the five FLAWS
    (``--wild-labels``).

So this script does the conversion DIVE -> (source dir, labels.parquet), mapping
DIVE's eight DASP columns down to our five and DROPPING the other three
(bad randomness, front running, time manipulation) — a contract positive only
for a dropped class becomes an all-zero negative for our five, exactly as
training/data/curated.py treats out-of-scope categories.

It then prints per-class positive counts so they can be checked against DIVE's
published distribution (Reentrancy 11,400 / Access Control 16,723 /
Arithmetic 9,542 / Unchecked 5,911 / DoS 3,781). A close match confirms the
join is correct.

CRITICAL: the DIVE archive is a macOS zip, so it carries a parallel ``__MACOSX``
tree of ``._<name>.sol`` resource-fork stubs (≈ half the .sol paths found by a
naive glob). Those are junk and are skipped here (any path under ``__MACOSX`` or
any name starting with ``._``).

Usage
-----
    PYTHONPATH=. python scripts/prepare_dive.py \
        --dive-root data/raw/dive \
        --out-sources data/raw/dive_sources \
        --out-labels data/processed/dive_labels.parquet

Then build with DIVE in the Wild slot (unchanged build_dataset.py):
    PYTHONPATH=. python scripts/build_dataset.py \
        --wild-dir data/raw/dive_sources \
        --wild-labels data/processed/dive_labels.parquet \
        --curated-dir data/raw/curated \
        --out data/processed --device cuda
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# DIVE label column -> our flaw code. The three DIVE columns NOT listed here
# (Bad Randomness, Front Running, Time manipulation) are intentionally dropped.
DIVE_COLUMN_TO_FLAW = {
    "Reentrancy": "reentrancy",
    "Access Control": "access_control",
    "Arithmetic": "arithmetic",
    "Unchecked Return Values": "unchecked_calls",
    "DoS": "dos",
}
# Authoritative order (must equal scgnn.schema.FLAWS).
FLAWS = ["reentrancy", "access_control", "arithmetic", "unchecked_calls", "dos"]

# DIVE's published positive counts, for the sanity check.
DIVE_EXPECTED = {
    "reentrancy": 11400, "access_control": 16723, "arithmetic": 9542,
    "unchecked_calls": 5911, "dos": 3781,
}


def _is_junk(p: Path) -> bool:
    """True for macOS resource-fork stubs and anything under __MACOSX."""
    return "__MACOSX" in p.parts or p.name.startswith("._")


def find_source_dir(dive_root: Path) -> Path:
    """Locate the real 'Source codes' directory, ignoring the __MACOSX twin."""
    candidates = [
        d for d in dive_root.rglob("Source codes")
        if d.is_dir() and not _is_junk(d)
    ]
    if not candidates:
        # fall back: any dir holding real <int>.sol files
        for sol in dive_root.rglob("*.sol"):
            if not _is_junk(sol) and sol.stem.isdigit():
                return sol.parent
        sys.exit(f"ERROR: could not find DIVE 'Source codes' under {dive_root}. "
                 f"Pass --source-dir explicitly.")
    # prefer the shallowest (the real one, not a nested copy)
    return sorted(candidates, key=lambda d: len(d.parts))[0]


def find_labels_csv(dive_root: Path) -> Path:
    for c in dive_root.rglob("DIVE_Labels.csv"):
        if not _is_junk(c):
            return c
    sys.exit(f"ERROR: DIVE_Labels.csv not found under {dive_root}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dive-root", default="data/raw/dive",
                    help="Where fetch_dive.py extracted DIVE (holds labels/ and raw/).")
    ap.add_argument("--source-dir", default=None,
                    help="Override: the 'Source codes' dir of <contractID>.sol files.")
    ap.add_argument("--labels-csv", default=None,
                    help="Override: path to DIVE_Labels.csv.")
    ap.add_argument("--out-sources", default="data/raw/dive_sources",
                    help="Flat dir of .sol files to create (the --wild-dir for build).")
    ap.add_argument("--out-labels", default="data/processed/dive_labels.parquet",
                    help="labels.parquet to write (the --wild-labels for build).")
    ap.add_argument("--link", action="store_true",
                    help="Symlink sources instead of copying (saves disk; needs same FS).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N rows (for a smoke run).")
    args = ap.parse_args()

    import pandas as pd

    dive_root = Path(args.dive_root)
    src_dir = Path(args.source_dir) if args.source_dir else find_source_dir(dive_root)
    labels_csv = Path(args.labels_csv) if args.labels_csv else find_labels_csv(dive_root)
    print(f"source dir : {src_dir}")
    print(f"labels csv : {labels_csv}")

    out_sources = Path(args.out_sources); out_sources.mkdir(parents=True, exist_ok=True)
    out_labels = Path(args.out_labels); out_labels.parent.mkdir(parents=True, exist_ok=True)

    # Read labels, verify the expected DIVE columns are present.
    with labels_csv.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in DIVE_COLUMN_TO_FLAW if c not in header]
        if "contractID" not in header or missing:
            sys.exit(f"ERROR: unexpected DIVE_Labels.csv header.\n  got: {header}\n"
                     f"  missing: {(['contractID'] if 'contractID' not in header else []) + missing}")
        rows = list(reader)
    if args.limit:
        rows = rows[: args.limit]
    print(f"label rows  : {len(rows)}")

    records = []           # {contract, reentrancy, ..., dos}
    n_missing_src = 0
    n_dropped_only = 0     # positive only for a dropped class -> all-zero kept
    counts = {f: 0 for f in FLAWS}

    for row in rows:
        cid = str(row["contractID"]).strip()
        sol = src_dir / f"{cid}.sol"
        if not sol.exists() or _is_junk(sol):
            n_missing_src += 1
            continue

        y = {f: int(str(row.get(col, "0")).strip() or "0")
             for col, f in DIVE_COLUMN_TO_FLAW.items()}
        # provenance-tagged stem so DIVE ids never collide with other corpora.
        stem = f"dive_{cid}"
        dst = out_sources / f"{stem}.sol"
        if not dst.exists():
            if args.link:
                try:
                    dst.symlink_to(sol.resolve())
                except FileExistsError:
                    pass
            else:
                shutil.copyfile(sol, dst)

        rec = {"contract": stem}
        rec.update({f: y[f] for f in FLAWS})
        records.append(rec)
        for f in FLAWS:
            counts[f] += y[f]
        if sum(y.values()) == 0:
            # could be a true negative OR positive only for a dropped class;
            # we cannot tell from our 5 columns, which is fine — it's a negative for us.
            dropped = any(int(str(row.get(c, "0")).strip() or "0")
                          for c in ("Bad Randomness", "Front Running", "Time manipulation"))
            if dropped:
                n_dropped_only += 1

    if not records:
        sys.exit("ERROR: no DIVE contracts matched a source file. Check --source-dir "
                 "(the __MACOSX twin must be ignored) and that <contractID>.sol exist.")

    df = pd.DataFrame.from_records(records, columns=["contract", *FLAWS])
    df.to_parquet(out_labels, index=False)

    # ---- report + sanity check against DIVE's published distribution ----
    print(f"\nwrote {len(df)} contracts")
    print(f"  sources -> {out_sources}  ({'symlinked' if args.link else 'copied'})")
    print(f"  labels  -> {out_labels}")
    print(f"  source files missing for {n_missing_src} label rows (skipped)")
    print(f"  {n_dropped_only} contracts are negatives for us but positive only "
          f"for a dropped class")
    print("\nper-class positive counts (ours) vs DIVE published:")
    full = args.limit is None
    for f in FLAWS:
        exp = DIVE_EXPECTED[f]
        got = counts[f]
        flag = ""
        if full:
            # allow small slack for source files that failed to extract/exist
            ratio = got / exp if exp else 0
            flag = "  OK" if 0.9 <= ratio <= 1.02 else "  <-- CHECK (off from published)"
        print(f"  {f:<16} {got:>6}   (DIVE: {exp:>6}){flag}")
    if full and any(
        not (0.9 <= counts[f] / DIVE_EXPECTED[f] <= 1.02) for f in FLAWS
    ):
        print("\nWARNING: one or more classes differ markedly from DIVE's published "
              "counts. If many source files were missing the join may be wrong; "
              "otherwise small shortfalls are expected (a few missing/sources).")
    else:
        print("\nCounts align with DIVE's published distribution — join looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Regenerate the Test B SmartBugs results tree (data/sb_testb) after recovery.

The votes baseline scores each split from a SmartBugs results tree. The train/
val/Test A tree (data/sb_results) survived in the archive, but the Test B tree
(data/sb_testb) did not. This script rebuilds it: it rewrites the Test B
manifest so each row's ``path`` points at the RECOVERED .sol (verified by chash
during fetch), then hands off to the existing ``scripts/durieux_baseline.py run``
subcommand, which runs the four tools through SmartBugs exactly as the original
Durieux stage did. Nothing about the tool invocation changes, so the votes it
produces are directly comparable to the surviving train/val/Test A votes.

Only the contracts actually recovered are runnable; the manifest is filtered to
those, and the count is printed so the votes-on-Test-B row is reported over the
recovered subset honestly (coverage.json from fetch_sources.py has the details).

Usage
-----
    PYTHONPATH=. python scripts/rerun_testb_tools.py \
        --testsets data/testsets --recovered data/recovered \
        --results data/sb_testb --sb-cmd "python -m sb" --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from scgnn.schema import FLAWS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--recovered", default="data/recovered")
    ap.add_argument("--results", default="data/sb_testb")
    ap.add_argument("--sb-cmd", default="python -m sb")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    src_paths = json.loads(
        (Path(args.recovered) / "source_paths.json").read_text(encoding="utf-8"))

    # rewrite the Test B manifest to point at recovered .sol, keeping only rows
    # whose source we actually have.
    rows_in = list(csv.DictReader(open(Path(args.testsets) / "test_b.csv",
                                       encoding="utf-8")))
    kept, dropped = [], []
    for r in rows_in:
        p = src_paths.get(r["contract_id"])
        if p and Path(p).exists():
            r = dict(r); r["path"] = p
            kept.append(r)
        else:
            dropped.append(r["contract_id"])

    if not kept:
        sys.exit("no recovered Test B source available; run fetch_sources.py first.")

    manifest_rerun = Path(args.testsets) / "test_b_recovered.csv"
    cols = ["contract_id", "path", "chash", *FLAWS]
    with open(manifest_rerun, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow({k: r[k] for k in cols})
    print(f"Test B re-run manifest: {len(kept)} recovered contract(s) "
          f"({len(dropped)} unrecovered, excluded)"
          + (f"; e.g. {dropped[0]}" if dropped else ""))

    # hand off to the existing, tested durieux run path with the recovered manifest
    cmd = [sys.executable, "scripts/durieux_baseline.py", "run",
           "--testsets", str(args.testsets), "--results", args.results,
           "--sb-cmd", args.sb_cmd, "--workers", str(args.workers),
           "--timeout", str(args.timeout)]
    # durieux_baseline reads test_b.csv by name; temporarily swap in the recovered
    # manifest so the run covers exactly the recovered contracts.
    original = Path(args.testsets) / "test_b.csv"
    backup = Path(args.testsets) / "test_b.original.csv"
    original.replace(backup)
    manifest_rerun.replace(original)
    try:
        print("running four tools on the recovered Test B contracts ...")
        subprocess.check_call(cmd)
    finally:
        original.replace(manifest_rerun)   # restore recovered manifest name
        backup.replace(original)           # restore the frozen manifest
    print(f"\nTest B results tree -> {args.results}")
    print("Now run the votes baseline with --results-test-b "
          f"{args.results} to score votes on the recovered Test B subset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

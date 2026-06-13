#!/usr/bin/env python3
"""Weak-label a Wild subset with Slither alone (smoke-grade; no SmartBugs).

Writes ``data/processed/labels.parquet`` (a ``contract`` column plus the five
flaw columns), so build_dataset.py consumes it unchanged.

Compiles each contract with the solc matching its pragma — passed to Slither as
an explicit ``--solc <binary>`` discovered from solc-select, since the ``solc`` on
PATH may be a fixed compiler wrong for most contracts. Runs across a process pool
(``--jobs``), skips contracts no installed compiler can build, and stops once
``--limit`` are labelled.

Example:
    PYTHONPATH=. python scripts/label_slither.py --wild-dir data/raw/wild --limit 500 --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAW_INDEX, FLAWS
from training.labelling.run_slither import (
    choose_solc, installed_solc_binaries, installed_solc_by_minor, run_slither,
)


def _read_head(path: Path, n: int = 4096) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(n)
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wild-dir", required=True, help="Root of Wild .sol files.")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--limit", type=int, default=500, help="Target labelled contracts.")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--timeout", type=int, default=120, help="Per-contract Slither timeout (s).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    import pandas as pd

    # Prefer explicit solc binaries (robust to a wrong `solc` on PATH); fall back
    # to SOLC_VERSION if discovery finds nothing.
    binaries = installed_solc_binaries()
    if binaries:
        mode, selector = "binary", binaries
        print("solc binaries by minor:",
              {k: Path(v).name for k, v in binaries.items()})
    else:
        mode, selector = "version", installed_solc_by_minor()
        if selector:
            print("no solc-select binaries found; falling back to SOLC_VERSION:", selector)
        else:
            print("ERROR: no solc compilers discovered. Run e.g.\n"
                  "  solc-select install 0.4.26 0.5.17 0.6.12 0.7.6 0.8.19\n"
                  "and check ~/.solc-select/artifacts exists.")
            return 1

    files = sorted(Path(args.wild_dir).rglob("*.sol"))
    random.Random(args.seed).shuffle(files)

    # Keep only contracts an installed compiler can build; `sel` is a binary path
    # (binary mode) or a version string (version mode).
    candidates: list[tuple[str, str]] = []
    skipped_unsupported = 0
    for f in files:
        sel = choose_solc(_read_head(f), selector)
        if sel is None:
            skipped_unsupported += 1
            continue
        candidates.append((str(f), sel))
    print(f"{len(files)} contracts; {len(candidates)} buildable by an installed solc; "
          f"{skipped_unsupported} skipped (no matching solc)")

    rows: list[list] = []
    labelled = attempted = failed = 0
    cand = iter(candidates)

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        pending: dict = {}

        def submit_more(k: int) -> None:
            for _ in range(k):
                try:
                    path, sel = next(cand)
                except StopIteration:
                    return
                if mode == "binary":
                    fut = ex.submit(run_slither, path, sel, None, args.timeout)
                else:
                    fut = ex.submit(run_slither, path, None, sel, args.timeout)
                pending[fut] = path

        warned = False
        submit_more(args.jobs * 2)
        while pending and labelled < args.limit:
            done = next(as_completed(pending))
            path = pending.pop(done)
            attempted += 1
            try:
                flaws = done.result()
            except Exception:
                flaws = None
            if flaws is None:
                failed += 1
            else:
                y = [0] * len(FLAWS)
                for fl in flaws:
                    y[FLAW_INDEX[fl]] = 1
                rows.append([Path(path).stem] + y)
                labelled += 1
            if attempted % 25 == 0:
                print(f"  attempted {attempted}, labelled {labelled}, failed {failed}",
                      flush=True)
            if not warned and attempted >= 50 and labelled == 0:
                warned = True
                print("  !! 0 successes in the first 50 attempts. Test one contract directly:\n"
                      "       slither <one .sol> --solc <binary-from-the-map-above> --json -",
                      flush=True)
            submit_more(1)
        for fut in pending:
            fut.cancel()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["contract"] + FLAWS)
    df.to_parquet(out / "labels.parquet", index=False)
    freq = {fl: int(df[fl].sum()) for fl in FLAWS} if len(df) else {fl: 0 for fl in FLAWS}
    (out / "class_frequency.json").write_text(json.dumps(freq, indent=2), encoding="utf-8")

    print(f"\nlabelled {labelled} contracts (attempted {attempted}, "
          f"failures {failed}, unsupported-pragma {skipped_unsupported})")
    print("wild class frequency:", freq)
    print("note: arithmetic is expected ~0 (Slither has no arithmetic detector);"
          " Curated supplies arithmetic positives for the test split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

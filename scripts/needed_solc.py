#!/usr/bin/env python3
"""Report exact-pin solc versions a contract tree needs but doesn't have installed.

Caret/range pragmas (``^0.4.24``) are covered by the newest patch of their minor;
only EXACT pins (``pragma solidity 0.4.25;``) need that precise compiler. This
lists the missing ones and prints a ready ``solc-select install`` command.

    PYTHONPATH=. python scripts/needed_solc.py data/raw/curated
    PYTHONPATH=. python scripts/needed_solc.py data/raw/curated data/raw/wild
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from scgnn.extraction.solc import installed_solc_full, pragma_exact


def main(dirs: list[str]) -> int:
    have = set(installed_solc_full())
    pins: Counter[str] = Counter()
    for d in dirs:
        for f in Path(d).rglob("*.sol"):
            try:
                head = f.read_text(encoding="utf-8", errors="ignore")[:4096]
            except OSError:
                continue
            v = pragma_exact(head)
            if v:
                pins[v] += 1
    missing = sorted(v for v in pins if v not in have)
    print("installed exact versions:", sorted(have))
    print("exact pins found:", dict(sorted(pins.items())))
    if missing:
        print("\nMISSING exact versions:", missing)
        print("install with:\n  solc-select install " + " ".join(missing))
    else:
        print("\nAll exact-pin versions are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["data/raw/curated"]))

#!/usr/bin/env python3
"""Build auto-labels from SmartBugs tool outputs with Snorkel.

Status: needs snorkel + pandas + a SmartBugs results directory; run on the Studio.
Assumes SmartBugs has already produced normalised result files (one per
tool/contract). It parses them, builds per-flaw label matrices, fits one Snorkel
LabelModel per flaw, and writes labels + per-tool reliabilities + a class-
frequency table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scgnn.common.seeds import set_seed
from scgnn.schema import FLAWS
from training.labelling.run_tools import build_label_matrices, parse_smartbugs_result
from training.labelling.snorkel_label import label_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="SmartBugs results directory.")
    ap.add_argument("--out", default="data/processed", help="Output directory.")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    votes: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for path in Path(args.results).rglob("*.json"):
        contract, tool, flaws = parse_smartbugs_result(path)
        if tool:
            votes[contract][tool] = flaws
    contract_ids = sorted(votes)
    matrices = build_label_matrices(votes, contract_ids)

    Y, P, reliabilities = label_all(matrices, threshold=args.threshold, seed=args.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(Y, columns=FLAWS); df.insert(0, "contract", contract_ids)
    df.to_parquet(out / "labels.parquet", index=False)
    (out / "reliabilities.json").write_text(json.dumps(reliabilities, indent=2), encoding="utf-8")
    freq = {flaw: int(Y[:, j].sum()) for j, flaw in enumerate(FLAWS)}
    (out / "class_frequency.json").write_text(json.dumps(freq, indent=2), encoding="utf-8")
    print("labelled", len(contract_ids), "contracts; class frequency:", freq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Adapter for the Peculiar external learned baseline (Wu et al., ISSRE 2021).

Peculiar is the one published learned detector whose evaluation setting matches
this study: SmartBugs Wild filtered per Durieux et al., labels derived from
analysis-tool results. It is a single-class REENTRANCY model with its own
crucial-data-flow-graph preprocessing (tree-sitter-solidity plus its own solc
handling) and a dependency set pinned to an older torch/transformers than this
repo. We therefore do NOT import it: it runs in its OWN container, on CPU (Test B
is 141 contracts, so no GPU is needed), and writes a predictions file that this
adapter consumes. The two dependency worlds never meet.

Two deliberate scoping decisions, both to keep the comparison honest:

1. Test B only. Peculiar was trained on SmartBugs Wild filtered per Durieux, and
   this study's Test Set A is drawn from SmartBugs Wild, so Peculiar has very
   likely trained on contracts sitting in Test A. You cannot firewall someone
   else's training set, so a Test A comparison would be biased in Peculiar's
   favour. Test B (Curated) is much safer. Residual risk to DISCLOSE in one
   sentence: some Curated contracts also appear in Wild, so a clean-room claim
   is not available; the comparison is "as close to like-for-like as an external
   checkpoint allows", not "leakage-free".

2. Reentrancy column only, the other four MASKED (not zero). The adapter emits a
   (n_contracts, 5) matrix with only the reentrancy column populated and a
   companion mask ``["reentrancy"]`` so the evaluation scores that column alone.
   A zero-filled column would be scored as four all-negative predictions and
   would silently deflate every macro average; the mask makes the metrics skip
   them. This is what ``tests/test_masked_metrics.py`` protects.

Predictions file format (CSV, written by the Peculiar container):
    contract_id,reentrancy
    0x0000...,1
    parity_wallet_bug_1,0
    ...
The ``contract_id`` values MUST match the Test B manifest's ``contract_id``
column. A probability in [0,1] is accepted in place of a 0/1 label; the
evaluation thresholds it like any other baseline (Peculiar has no validation
split here, so a fixed 0.5 threshold is used and stated).

If the file is absent or unreadable, the adapter raises ``PeculiarUnavailable``;
the entrypoint catches it, prints the reason, and SKIPS the row. It never falls
back to Peculiar's published numbers: a borrowed number measured on a different
contract set is not a result on our test set.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scgnn.schema import FLAW_INDEX, FLAWS, N_FLAWS

PECULIAR_CLASS = "reentrancy"
PECULIAR_MASK = [PECULIAR_CLASS]


class PeculiarUnavailable(Exception):
    """Raised when the Peculiar predictions file is missing or malformed."""


def load_peculiar_predictions(predictions_csv: str | Path,
                              contract_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Read the Peculiar predictions file into a ``(n, 5)`` probability matrix.

    Only the reentrancy column is populated; the other four are left at 0 and are
    excluded from scoring by the returned mask, never counted. Returns
    ``(probs, mask)`` where ``mask == ["reentrancy"]``. Rows for contracts not in
    ``contract_ids`` are ignored; contracts absent from the file raise, because a
    silent 0 would be scored as a negative prediction and bias recall.
    """
    path = Path(predictions_csv)
    if not path.exists():
        raise PeculiarUnavailable(
            f"Peculiar predictions file not found: {path}. Run the Peculiar "
            f"container on the Test B contracts and write this CSV, or skip the "
            f"row. Published numbers are NOT a substitute.")
    try:
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:                       # malformed -> loud skip
        raise PeculiarUnavailable(f"cannot read {path}: {exc}") from exc

    if not rows or PECULIAR_CLASS not in rows[0]:
        raise PeculiarUnavailable(
            f"{path} has no '{PECULIAR_CLASS}' column; got "
            f"{list(rows[0].keys()) if rows else 'empty file'}.")

    by_id: dict[str, float] = {}
    for r in rows:
        cid = r.get("contract_id") or r.get("id") or r.get("contract")
        if cid is None:
            raise PeculiarUnavailable(f"{path} lacks a contract_id column.")
        try:
            by_id[cid] = float(r[PECULIAR_CLASS])
        except (TypeError, ValueError) as exc:
            raise PeculiarUnavailable(
                f"{path}: non-numeric {PECULIAR_CLASS} for {cid}: "
                f"{r[PECULIAR_CLASS]!r}") from exc

    missing = [c for c in contract_ids if c not in by_id]
    if missing:
        raise PeculiarUnavailable(
            f"{path} is missing {len(missing)} Test B contract(s) "
            f"(e.g. {missing[0]}). A partial file cannot be scored without "
            f"biasing recall; regenerate it over all Test B contracts.")

    probs = np.zeros((len(contract_ids), N_FLAWS), dtype=float)
    ri = FLAW_INDEX[PECULIAR_CLASS]
    for i, cid in enumerate(contract_ids):
        probs[i, ri] = by_id[cid]
    return probs, list(PECULIAR_MASK)

"""Static-analysis baselines on the Curated test set, mapped to the five flaws.

Status: needs SmartBugs 2.0 + Slither/Mythril; validate on the Studio. Runs the
baseline tools through SmartBugs on the SAME Curated test contracts and maps
their findings with the SAME DASP mapping, so the comparison is fair.
"""

from __future__ import annotations

import numpy as np

from scgnn.schema import FLAW_INDEX, FLAWS
from training.evaluate.metrics import per_flaw_and_macro
from training.labelling.run_tools import parse_smartbugs_result


def baseline_predictions(result_files_per_contract: dict[str, list[str]],
                         contract_ids: list[str], tool: str) -> np.ndarray:
    """Build a ``(n_contracts, 5)`` prediction matrix for one baseline tool."""
    Y = np.zeros((len(contract_ids), len(FLAWS)), dtype=int)
    for i, cid in enumerate(contract_ids):
        for path in result_files_per_contract.get(cid, []):
            _c, t, flaws = parse_smartbugs_result(path)
            if t != tool.lower():
                continue
            for flaw in flaws:
                Y[i, FLAW_INDEX[flaw]] = 1
    return Y


def score_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return per_flaw_and_macro(y_true, y_pred)

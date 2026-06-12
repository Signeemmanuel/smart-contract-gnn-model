"""The A/B/C labelling experiment on the frozen Curated test split.

Status: needs the training stack; py_compiled here. Implements the agreed
resolution of the firewall tension:

* freeze one stratified Curated test split up front (rare flaws represented);
* Condition A trains on the Curated remainder via stratified k-fold CV;
* Condition B trains on Wild only;
* Condition C trains on Wild + Curated remainder;
* all three are evaluated on the SAME frozen test split, so the headline model
  (expected to be C) is also the one carried to the dashboard.

This module wires the conditions; the caller supplies loaders built from the
firewall's indices.
"""

from __future__ import annotations

import numpy as np

from training.data.firewall import curated_remainder_folds


def run_condition_a(train_one_fold, Y_remainder: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Cross-validated Condition A over the Curated remainder.

    ``train_one_fold(train_idx, val_idx)`` trains on the remainder fold and
    returns its macro-F1 on the frozen test split. We average over folds.
    """
    folds = curated_remainder_folds(Y_remainder, n_splits=n_splits, seed=seed)
    scores = [train_one_fold(tr, va) for tr, va in folds]
    return {"macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)),
            "fold_scores": [float(s) for s in scores]}


def summarise(results: dict[str, dict]) -> dict:
    """Given {'A':..,'B':..,'C':..} score dicts, pick the headline condition."""
    headline = max(results, key=lambda c: results[c].get("macro_f1_mean",
                                                          results[c].get("macro_f1", 0.0)))
    return {"results": results, "headline_condition": headline}

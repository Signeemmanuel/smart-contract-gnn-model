"""Per-flaw and macro precision/recall/F1. Pure (scikit-learn); unit-tested."""

from __future__ import annotations

import numpy as np

from scgnn.schema import FLAWS


def per_flaw_and_macro(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return per-flaw and macro precision/recall/F1 for multi-label predictions.

    ``y_true``/``y_pred`` are ``(n_contracts, 5)`` binary arrays in canonical
    flaw order. ``zero_division=0`` so a flaw with no positives scores 0 rather
    than raising.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(FLAWS))), zero_division=0,
    )
    per_flaw = {
        FLAWS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(support[i])}
        for i in range(len(FLAWS))
    }
    macro = {
        "precision": float(np.mean(p)),
        "recall": float(np.mean(r)),
        "f1": float(np.mean(f)),
    }
    return {"per_flaw": per_flaw, "macro": macro}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return per_flaw_and_macro(y_true, y_pred)["macro"]["f1"]

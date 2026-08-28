"""Per-flaw and macro metrics, bootstrap CIs, confusion matrices, ensembling,
and the Durieux-style tool-vs-model comparison. Pure (numpy + scikit-learn);
unit-tested.

Workstreams E (ensemble) and G (evaluation protocol) live here so every number
the dissertation reports comes from one tested module.
"""

from __future__ import annotations

import numpy as np

from scgnn.schema import FLAWS

N = len(FLAWS)


# ------------------------------- core metrics -------------------------------

def per_flaw_and_macro(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-flaw and macro precision/recall/F1 for multi-label predictions.

    ``y_true``/``y_pred`` are ``(n_contracts, 5)`` binary arrays in canonical
    flaw order. ``zero_division=0`` so a flaw with no positives scores 0 rather
    than raising.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(N)), zero_division=0,
    )
    per_flaw = {
        FLAWS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(support[i])}
        for i in range(N)
    }
    macro = {"precision": float(np.mean(p)), "recall": float(np.mean(r)),
             "f1": float(np.mean(f))}
    return {"per_flaw": per_flaw, "macro": macro}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return per_flaw_and_macro(y_true, y_pred)["macro"]["f1"]


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Everything Workstream G asks for: per-class P/R/F1/accuracy, macro-F1,
    micro-F1, subset accuracy."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    base = per_flaw_and_macro(y_true, y_pred)
    for i, flaw in enumerate(FLAWS):
        acc = float((y_true[:, i] == y_pred[:, i]).mean()) if len(y_true) else 0.0
        base["per_flaw"][flaw]["accuracy"] = acc
    base["micro"] = {"f1": float(f1_score(y_true, y_pred, average="micro",
                                          zero_division=0))}
    # subset accuracy: the whole 5-label vector must be exactly right
    base["subset_accuracy"] = (float((y_true == y_pred).all(axis=1).mean())
                               if len(y_true) else 0.0)
    return base


def confusion_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class 2x2 confusion counts (TP/FP/FN/TN). Pure."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {}
    for i, flaw in enumerate(FLAWS):
        t, p = y_true[:, i], y_pred[:, i]
        out[flaw] = {
            "tp": int(((t == 1) & (p == 1)).sum()),
            "fp": int(((t == 0) & (p == 1)).sum()),
            "fn": int(((t == 1) & (p == 0)).sum()),
            "tn": int(((t == 0) & (p == 0)).sum()),
        }
    return out


# ------------------------------ bootstrap CIs ------------------------------

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, *, n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap 95% CIs for macro-F1 and per-class F1, resampling CONTRACTS.

    Resampling contracts (not labels) is the right unit: it reflects the
    uncertainty from having tested on this particular set of contracts. Returns
    ``{"macro_f1": {"lo","hi"}, "per_flaw": {flaw: {"lo","hi"}}}``. Pure.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        empty = {"lo": 0.0, "hi": 0.0}
        return {"macro_f1": empty, "per_flaw": {f: dict(empty) for f in FLAWS}}

    rng = np.random.default_rng(seed)
    macros = np.empty(n_resamples, dtype=float)
    per = np.empty((n_resamples, N), dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        _, _, f, _ = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], average=None, labels=list(range(N)),
            zero_division=0)
        per[b] = f
        macros[b] = f.mean()

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "macro_f1": {"lo": float(np.percentile(macros, lo_q)),
                     "hi": float(np.percentile(macros, hi_q))},
        "per_flaw": {FLAWS[i]: {"lo": float(np.percentile(per[:, i], lo_q)),
                                "hi": float(np.percentile(per[:, i], hi_q))}
                     for i in range(N)},
    }


# ------------------------------- thresholds --------------------------------

def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, *,
                    grid: np.ndarray | None = None) -> list[float]:
    """Per-class thresholds maximising F1 ON THE GIVEN SPLIT (call with VAL only).

    Never call this with a test set: thresholds are a fitted parameter, and
    fitting them on test would contaminate the frozen benchmark. Pure.
    """
    from sklearn.metrics import f1_score

    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    out = []
    for i in range(N):
        best_t, best_f = 0.5, -1.0
        for t in grid:
            f = f1_score(y_true[:, i], (probs[:, i] >= t).astype(int),
                         zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        out.append(best_t)
    return out


def apply_thresholds(probs: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Binarise per-class probabilities with per-class thresholds. Pure."""
    probs = np.asarray(probs)
    thr = np.asarray(thresholds).reshape(1, -1)
    return (probs >= thr).astype(int)


# -------------------------------- ensemble ---------------------------------

def ensemble_probs(prob_list: list[np.ndarray], policy: str = "mean") -> np.ndarray:
    """Combine per-class sigmoid probabilities of several models (Workstream E).

    ``mean`` (primary) averages; ``max`` takes the per-class maximum. Thresholds
    for the ensemble are then tuned on VALIDATION exactly as for a single model.
    Pure.
    """
    if not prob_list:
        raise ValueError("ensemble_probs needs at least one model's probabilities")
    stack = np.stack([np.asarray(p) for p in prob_list], axis=0)  # (m, n, 5)
    if policy == "mean":
        return stack.mean(axis=0)
    if policy == "max":
        return stack.max(axis=0)
    raise ValueError(f"unknown policy {policy!r}; expected 'mean' or 'max'")


# ------------------- Durieux-style tool-vs-model comparison -------------------

def tool_baseline_matrix(y_true: np.ndarray, tool_preds: dict[str, np.ndarray],
                         model_preds: dict[str, np.ndarray]) -> dict:
    """The Durieux-style matrix (Workstream G).

    ``tool_preds``: tool name -> (n,5) binary predictions (mapped findings).
    ``model_preds``: model name -> (n,5) binary predictions.
    Adds a ``union_of_tools`` row (a flaw is predicted if ANY tool reports it),
    which is exactly the labelling rule, so the table shows what the union
    oracle itself achieves against expert ground truth.

    Also computes the false-warning analogue: the fraction of flaw-FREE
    contracts that each row flags with at least one finding. Pure.
    """
    y_true = np.asarray(y_true)
    rows: dict[str, np.ndarray] = dict(tool_preds)
    if tool_preds:
        rows["union_of_tools"] = np.clip(
            np.sum([np.asarray(p) for p in tool_preds.values()], axis=0), 0, 1)
    rows.update(model_preds)

    clean = (y_true.sum(axis=1) == 0)          # contracts with no flaw at all
    out: dict[str, dict] = {}
    for name, pred in rows.items():
        pred = np.asarray(pred)
        m = full_metrics(y_true, pred)
        flagged_clean = (float(pred[clean].sum(axis=1).astype(bool).mean())
                         if clean.any() else 0.0)
        out[name] = {
            "per_flaw": m["per_flaw"],
            "macro": m["macro"],
            "false_warning_rate": flagged_clean,   # fraction of clean contracts flagged
        }
    return out
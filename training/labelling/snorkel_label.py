"""Per-flaw Snorkel LabelModel: denoise tool votes into probabilistic labels.

Status: needs snorkel; py_compiled here, run on the Studio. One LabelModel per
flaw (binary problem, cardinality 2, abstain = -1). We also surface each tool's
learned accuracy, which is the concrete evidence that the label model did
something a majority vote cannot.
"""

from __future__ import annotations

import numpy as np

from scgnn.schema import FLAWS
from training.labelling.run_tools import TOOLS

ABSTAIN = -1


def label_flaw(L: np.ndarray, threshold: float = 0.70, seed: int = 42):
    """Fit one LabelModel on ``L`` (n_contracts, n_tools) in {-1,0,1}.

    Returns ``(labels, probs, reliabilities)`` where ``labels`` are 0/1 at the
    threshold, ``probs`` are P(flaw present), and ``reliabilities`` maps each
    tool to its learned accuracy.
    """
    # Columns that actually carry signal for this flaw (a tool that abstains on
    # every contract — because it does not cover the flaw — contributes nothing).
    active = [j for j in range(L.shape[1]) if bool((L[:, j] != ABSTAIN).any())]

    if len(active) < 2:
        # Zero or one covering tool: there is nothing for a label model to
        # denoise, so take the single tool's vote directly (a label model on one
        # labelling function is degenerate and collapses to all-negative). This is
        # how arithmetic — detected only by Osiris — is labelled.
        if not active:
            labels = np.zeros(L.shape[0], dtype=int)
        else:
            labels = (L[:, active[0]] == 1).astype(int)
        probs = labels.astype(float)
        reliabilities = {TOOLS[j]: 1.0 for j in active}
        return labels, probs, reliabilities

    from snorkel.labeling.model import LabelModel

    L_active = L[:, active]                       # drop all-abstain columns
    lm = LabelModel(cardinality=2, verbose=False)
    lm.fit(L_train=L_active, n_epochs=500, log_freq=100, seed=seed)
    probs = lm.predict_proba(L_active)[:, 1]
    labels = (probs >= threshold).astype(int)
    try:
        weights = np.asarray(lm.get_weights()).ravel().tolist()
        reliabilities = {TOOLS[active[i]]: float(w) for i, w in enumerate(weights[: len(active)])}
    except Exception:
        reliabilities = {TOOLS[j]: float("nan") for j in active}
    return labels, probs, reliabilities


def _corroboration_reliability(L: np.ndarray, active: list[int]) -> dict[str, float]:
    """Per-tool corroboration rate: of the contracts a tool flags, the fraction
    also flagged by at least one other covering tool.

    A transparent, robust stand-in for a learned reliability when the votes are
    too low-overlap for a label model to estimate accuracies. ``nan`` when a tool
    is the only detector for the flaw (nothing to corroborate against).
    """
    fired = {j: (L[:, j] == 1) for j in active}
    rel: dict[str, float] = {}
    for j in active:
        n = int(fired[j].sum())
        if n == 0 or len(active) < 2:
            rel[TOOLS[j]] = float("nan")
            continue
        others = np.zeros(L.shape[0], dtype=bool)
        for k in active:
            if k != j:
                others |= fired[k]
        rel[TOOLS[j]] = float((fired[j] & others).sum()) / n
    return rel


def label_flaw_union(L: np.ndarray):
    """Label a flaw by the union of covering-tool detections: positive iff at
    least one tool that covers the flaw flagged it.

    Robust across overlap regimes (no collapse to the agreement core, no
    degenerate polarity flip), which a label model is not when the tools are
    heterogeneous positive detectors with low mutual coverage. Reliabilities are
    reported as per-tool corroboration rates for analysis.
    """
    active = [j for j in range(L.shape[1]) if bool((L[:, j] != ABSTAIN).any())]
    fired_any = np.zeros(L.shape[0], dtype=bool)
    for j in active:
        fired_any |= (L[:, j] == 1)
    labels = fired_any.astype(int)
    return labels, labels.astype(float), _corroboration_reliability(L, active)


def label_all(matrices: dict[str, np.ndarray], threshold: float = 0.70,
              seed: int = 42, method: str = "union"):
    """Run the chosen labeller for every flaw and stack into a multi-label target.

    ``method="union"`` (default, recommended) labels a flaw positive if any
    covering tool detects it — robust to the low mutual coverage of these tools.
    ``method="snorkel"`` uses the per-flaw LabelModel posterior; kept for the
    methodology comparison, but it collapses to the tool-agreement core on
    low-overlap classes (e.g. DoS), losing most positives.

    Returns ``(Y, probs, reliabilities)`` where ``Y`` is ``(n_contracts, 5)``.
    """
    n = next(iter(matrices.values())).shape[0]
    Y = np.zeros((n, len(FLAWS)), dtype=np.int8)
    P = np.zeros((n, len(FLAWS)), dtype=np.float32)
    reliabilities: dict[str, dict[str, float]] = {}
    for j, flaw in enumerate(FLAWS):
        if method == "union":
            labels, probs, rel = label_flaw_union(matrices[flaw])
        elif method == "snorkel":
            labels, probs, rel = label_flaw(matrices[flaw], threshold=threshold, seed=seed)
        else:
            raise ValueError(f"unknown method {method!r}; expected 'union' or 'snorkel'")
        Y[:, j] = labels
        P[:, j] = probs
        reliabilities[flaw] = rel
    return Y, P, reliabilities

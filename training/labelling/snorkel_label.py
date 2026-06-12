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
    from snorkel.labeling.model import LabelModel

    lm = LabelModel(cardinality=2, verbose=False)
    lm.fit(L_train=L, n_epochs=500, log_freq=100, seed=seed)
    probs = lm.predict_proba(L)[:, 1]
    labels = (probs >= threshold).astype(int)
    try:
        weights = np.asarray(lm.get_weights()).ravel().tolist()
        reliabilities = {TOOLS[i]: float(w) for i, w in enumerate(weights[: len(TOOLS)])}
    except Exception:
        reliabilities = {t: float("nan") for t in TOOLS}
    return labels, probs, reliabilities


def label_all(matrices: dict[str, np.ndarray], threshold: float = 0.70, seed: int = 42):
    """Run :func:`label_flaw` for every flaw and stack into a multi-label target.

    Returns ``(Y, probs, reliabilities)`` where ``Y`` is ``(n_contracts, 5)``.
    """
    n = next(iter(matrices.values())).shape[0]
    Y = np.zeros((n, len(FLAWS)), dtype=np.int8)
    P = np.zeros((n, len(FLAWS)), dtype=np.float32)
    reliabilities: dict[str, dict[str, float]] = {}
    for j, flaw in enumerate(FLAWS):
        labels, probs, rel = label_flaw(matrices[flaw], threshold=threshold, seed=seed)
        Y[:, j] = labels
        P[:, j] = probs
        reliabilities[flaw] = rel
    return Y, P, reliabilities

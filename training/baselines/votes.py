"""Predict-the-union baseline: learn the labels from the four tools' votes alone.

This is the baseline a sharp reviewer demands first. The GNN scores macro-F1
0.392 on Test B against a four-tool union worth 0.387; the obvious question is
how much of that comes from learning code structure versus from learning how
the four labelling tools behave. This baseline answers it directly: its ONLY
input is the four tools' per-class votes for a contract (a 20-dimensional
feature vector), with no source code, no graph and no embedding. Whatever it
scores is the part of the task that is explained by the tools' behaviour alone;
the gap between it and the GNN is the value the structural representation adds.

Feature construction reuses the labelling code verbatim so the votes here are
byte-identical to the votes that built the training labels:
``collect_votes`` walks a SmartBugs results tree and ``build_label_matrices``
turns it into one ``(n_contracts, n_tools)`` matrix per flaw with the encoding
-1 abstain / 0 negative / 1 positive. Abstention stays a distinct feature
value (-1); it is never collapsed into a negative vote, because "the tool did
not run or does not cover this flaw" is genuinely different information from
"the tool ran and found nothing", and conflating them is exactly the mistake
``tests/test_votes_baseline.py`` guards against.

A note for the write-up, not a defect: the TRAINING label is the union of these
same votes (positive iff any tool voted 1), so on tool-labelled data (Test A)
the target is nearly a deterministic function of the features and this baseline
will score very high. That is the point, not leakage: it measures how learnable
the union rule is. The informative comparison is on Test B, where the tool
votes meet independent expert truth and the union's own ceiling (0.387) applies.

Status: pure sklearn; trains in seconds on CPU. Needs a SmartBugs results tree
for the contracts in question (the same trees the labelling and Durieux stages
used), passed as --results-train and --results-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scgnn.schema import FLAWS, N_FLAWS
from training.labelling.run_tools import (
    TOOLS, build_label_matrices, collect_votes,
)

N_TOOLS = len(TOOLS)


def votes_feature_matrix(results_dir: str, contract_ids: list[str]) -> np.ndarray:
    """Assemble the ``(n_contracts, N_FLAWS * N_TOOLS)`` vote feature matrix.

    Column block ``f`` holds the four tools' votes for flaw ``f`` in TOOLS order,
    each in {-1 abstain, 0 negative, 1 positive}. The per-flaw matrices come
    straight from ``build_label_matrices`` so the encoding matches the labels.
    """
    votes = collect_votes(results_dir)
    per_flaw = build_label_matrices(votes, contract_ids)   # {flaw: (n, n_tools)}
    blocks = [per_flaw[f] for f in FLAWS]                   # canonical flaw order
    return np.concatenate(blocks, axis=1).astype(np.float64)


@dataclass
class VotesBaseline:
    """Per-class classifier over the tool-vote features.

    ``kind`` selects the per-class model: "logreg" (default; interpretable, the
    reported baseline) or "tree" (a shallow decision tree, offered because the
    union rule is a disjunction a tree can represent exactly, which is a useful
    sanity ceiling). One independent model per class, so a class the tools never
    positively vote for degrades to its majority label rather than erroring.
    """

    kind: str = "logreg"
    seed: int = 42
    models: list = None            # fitted per-class estimators
    fallback: list = None          # constant label where a class has one value

    def _make(self):
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=self.seed)
        if self.kind == "tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(max_depth=N_TOOLS, class_weight="balanced",
                                          random_state=self.seed)
        raise ValueError(f"unknown kind {self.kind!r}; expected 'logreg' or 'tree'")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "VotesBaseline":
        """Fit one classifier per class. ``X`` is (n, 20), ``Y`` is (n, 5)."""
        self.models = [None] * N_FLAWS
        self.fallback = [None] * N_FLAWS
        for i in range(N_FLAWS):
            yi = Y[:, i].astype(int)
            if len(np.unique(yi)) < 2:
                # only one label present in training for this class: a classifier
                # cannot be fitted, so predict that constant (the honest floor).
                self.fallback[i] = int(yi[0]) if len(yi) else 0
                continue
            clf = self._make()
            clf.fit(X, yi)
            self.models[i] = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-class positive-class probability, ``(n, 5)``, for thresholding."""
        n = X.shape[0]
        P = np.zeros((n, N_FLAWS), dtype=float)
        for i in range(N_FLAWS):
            if self.models[i] is None:
                P[:, i] = float(self.fallback[i] or 0)
                continue
            clf = self.models[i]
            classes = list(clf.classes_)
            proba = clf.predict_proba(X)
            P[:, i] = proba[:, classes.index(1)] if 1 in classes else 0.0
        return P

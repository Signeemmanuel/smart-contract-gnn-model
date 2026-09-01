"""Trivial floor baselines for the five-label task.

Their absence is noticeable on an imbalanced multi-label problem: a reader needs
to know what majority-voting and chance already achieve before any learned score
means anything. Three floors, each a ``(n_contracts, 5)`` predictor:

- majority: predict, per class, the majority label seen in TRAINING. On this
  data every class is minority-positive, so this is the all-negative predictor
  in disguise and scores macro-F1 0 (no true positives). Reported precisely
  because that zero is the number the learned models must clear.
- stratified: predict each class positive independently at random with the
  class's TRAINING positive rate (seeded). The chance baseline that respects
  class imbalance.
- all_positive: predict every class positive for every contract. Recall 1,
  precision the base rate; the upper-left floor that shows why precision matters.

All three fit only trivial statistics on the training split and are then frozen,
so they pass the firewall trivially (they never see contract content at all, only
label rates). Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scgnn.schema import N_FLAWS


@dataclass
class TrivialBaseline:
    kind: str = "majority"          # "majority" | "stratified" | "all_positive"
    seed: int = 42
    rates: np.ndarray = None        # per-class training positive rate
    majority: np.ndarray = None     # per-class majority label (0/1)

    def fit(self, Y: np.ndarray) -> "TrivialBaseline":
        Y = np.asarray(Y)
        self.rates = Y.mean(axis=0) if len(Y) else np.zeros(N_FLAWS)
        self.majority = (self.rates >= 0.5).astype(int)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        """Return an ``(n, 5)`` probability matrix; thresholding is applied by the
        caller exactly as for any other baseline, so the one metrics path is used."""
        if self.kind == "majority":
            return np.tile(self.majority.astype(float), (n, 1))
        if self.kind == "all_positive":
            return np.ones((n, N_FLAWS), dtype=float)
        if self.kind == "stratified":
            rng = np.random.default_rng(self.seed)
            # Emit the class positive rate as the "probability"; thresholding at
            # the tuned threshold reproduces stratified sampling in expectation,
            # and keeps every baseline on the identical predict_proba -> threshold
            # path. For a hard stratified draw use predict_hard below.
            return np.tile(self.rates.astype(float), (n, 1))
        raise ValueError(f"unknown kind {self.kind!r}")

    def predict_hard(self, n: int) -> np.ndarray:
        """A concrete stratified random draw (used only if a caller wants hard
        predictions rather than the threshold path). Seeded and reproducible."""
        if self.kind != "stratified":
            return (self.predict_proba(n) >= 0.5).astype(int)
        rng = np.random.default_rng(self.seed)
        draws = rng.random((n, N_FLAWS))
        return (draws < self.rates.reshape(1, -1)).astype(int)

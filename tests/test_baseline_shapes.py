"""Every baseline returns (n, 5) predictions in canonical FLAWS order."""
from __future__ import annotations

import numpy as np

from scgnn.schema import FLAWS
from training.baselines.trivial import TrivialBaseline
from training.baselines.votes import VotesBaseline, N_TOOLS


def test_votes_shape():
    rng = np.random.default_rng(0)
    X = rng.integers(-1, 2, size=(40, len(FLAWS) * N_TOOLS)).astype(float)
    Y = (rng.random((40, 5)) < 0.3).astype(int)
    P = VotesBaseline().fit(X, Y).predict_proba(X[:11])
    assert P.shape == (11, len(FLAWS))
    assert ((P >= 0) & (P <= 1)).all()


def test_trivial_shapes():
    rng = np.random.default_rng(0)
    Y = (rng.random((30, 5)) < 0.3).astype(int)
    for kind in ("majority", "stratified", "all_positive"):
        P = TrivialBaseline(kind=kind).fit(Y).predict_proba(7)
        assert P.shape == (7, len(FLAWS))


def test_peculiar_shape(tmp_path):
    from training.baselines.peculiar import load_peculiar_predictions
    csv = tmp_path / "p.csv"
    csv.write_text("contract_id,reentrancy\na,1\nb,0.5\n")
    probs, mask = load_peculiar_predictions(csv, ["a", "b"])
    assert probs.shape == (2, len(FLAWS))
    assert mask == ["reentrancy"]
    # only reentrancy populated
    assert probs[:, 1:].sum() == 0

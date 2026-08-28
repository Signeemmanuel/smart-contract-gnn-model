"""Abstention (-1) must stay distinct from a negative vote (0): a contract where
every tool abstains on a class is not scored as a negative for that class."""
from __future__ import annotations

import numpy as np

import training.baselines.votes as votes_mod
from training.baselines.votes import votes_feature_matrix, N_TOOLS
from scgnn.schema import FLAWS


def test_abstain_is_not_negative(monkeypatch):
    # c_abstain: no tool ran at all -> every feature must be -1, never 0.
    # c_ran: slither ran and reported reentrancy; osiris ran, no arithmetic.
    votes = {
        "c_abstain": {},
        "c_ran": {"slither": {"reentrancy"}, "osiris": set()},
    }
    monkeypatch.setattr(votes_mod, "collect_votes", lambda d: votes)
    X = votes_feature_matrix("ignored", ["c_abstain", "c_ran"])
    assert X.shape == (2, len(FLAWS) * N_TOOLS)

    # c_abstain row is entirely -1 (distinct from a negative 0)
    assert (X[0] == -1).all()

    # c_ran: slither(reentrancy) == 1; osiris(arithmetic) ran negative == 0 (NOT -1)
    re_block = X[1, 0:N_TOOLS]           # reentrancy block, TOOLS order
    arith_block = X[1, 2 * N_TOOLS:3 * N_TOOLS]
    assert re_block[0] == 1              # slither positive
    assert arith_block[3] == 0          # osiris ran, negative -> 0, not abstain
    # tools that did not run on c_ran stay -1 (e.g. mythril on reentrancy: mythril
    # does not cover reentrancy, so abstain)
    assert re_block[1] == -1


def test_all_abstain_class_not_a_negative(monkeypatch):
    # A class where every tool abstains for a contract must carry -1 across the
    # whole block, so the classifier can tell "unknown" from "known-negative".
    votes = {"c": {"slither": {"reentrancy"}}}  # only slither ran
    monkeypatch.setattr(votes_mod, "collect_votes", lambda d: votes)
    X = votes_feature_matrix("ignored", ["c"])
    # arithmetic: only osiris covers it, and osiris did not run -> all -1
    arith_block = X[0, 2 * N_TOOLS:3 * N_TOOLS]
    assert (arith_block == -1).all()

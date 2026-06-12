"""Bonus: metrics behave on tiny known cases, including the zero-support edge."""
import numpy as np
from training.evaluate.metrics import per_flaw_and_macro
from training.evaluate.localisation import top_k_localisation


def test_perfect_prediction_over_all_classes_scores_one():
    y = np.eye(5, dtype=int)  # every flaw class has exactly one positive
    out = per_flaw_and_macro(y, y)
    assert out["macro"]["f1"] == 1.0
    assert out["per_flaw"]["reentrancy"]["support"] == 1


def test_absent_class_scores_zero_by_design():
    # Only reentrancy is present and predicted; the other four classes have no
    # support and score 0 (zero_division=0), so macro-F1 is 1/5. This is the
    # conservative convention we report against on the Curated test set.
    y = np.array([[1, 0, 0, 0, 0]])
    out = per_flaw_and_macro(y, y)
    assert out["per_flaw"]["reentrancy"]["f1"] == 1.0
    assert out["per_flaw"]["dos"]["f1"] == 0.0
    assert out["macro"]["f1"] == 0.2


def test_topk_localisation():
    pred = [[10, 11, 12], [5, 99]]   # contract 2's top line (5) is in its gold set
    gold = [{12}, {5}]
    acc = top_k_localisation(pred, gold, ks=(1, 3))
    assert acc[1] == 0.5   # only contract 2 hits at k=1
    assert acc[3] == 1.0   # both hit within top-3

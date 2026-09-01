"""Masked columns must be EXCLUDED from averaging, never scored as zeros.

This is the test most likely to be got wrong, and getting it wrong would
silently corrupt the Peculiar comparison: a single-class baseline emits four
empty columns, and if those are averaged in as F1 0 the macro is meaningless.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from scgnn.schema import FLAWS
from training.evaluate.metrics import (bootstrap_ci, full_metrics,
                                       per_flaw_and_macro, resolve_mask)


def _reentrancy_only_case():
    # reentrancy column has real signal; arithmetic is all-true/pred-zero so it
    # would score F1 0 and drag any dense macro down.
    yt = np.zeros((10, 5), int)
    yp = np.zeros((10, 5), int)
    yt[:, 0] = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    yp[:, 0] = [1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    yt[:, 2] = 1
    return yt, yp


def test_masked_macro_equals_single_class_f1():
    yt, yp = _reentrancy_only_case()
    masked = full_metrics(yt, yp, mask=["reentrancy"])
    expected = f1_score(yt[:, 0], yp[:, 0])
    assert masked["macro"]["f1"] == pytest.approx(expected)


def test_masked_reports_only_masked_class():
    yt, yp = _reentrancy_only_case()
    masked = per_flaw_and_macro(yt, yp, mask=["reentrancy"])
    assert list(masked["per_flaw"]) == ["reentrancy"]
    assert masked["scored_classes"] == ["reentrancy"]


def test_masking_actually_changes_the_result():
    # The whole point: the dense macro is corrupted by the empty columns, the
    # masked macro is not. If these were equal, masking would be a no-op and the
    # comparison would be worthless.
    yt, yp = _reentrancy_only_case()
    dense = full_metrics(yt, yp)["macro"]["f1"]
    masked = full_metrics(yt, yp, mask=["reentrancy"])["macro"]["f1"]
    assert masked > dense


def test_default_none_is_unchanged():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 2, size=(30, 5))
    yp = rng.integers(0, 2, size=(30, 5))
    m = full_metrics(yt, yp)
    assert set(m["per_flaw"]) == set(FLAWS)
    assert "scored_classes" not in m
    manual = np.mean([m["per_flaw"][f]["f1"] for f in FLAWS])
    assert m["macro"]["f1"] == pytest.approx(manual)


def test_masked_bootstrap_only_masked_columns():
    yt, yp = _reentrancy_only_case()
    ci = bootstrap_ci(yt, yp, n_resamples=200, mask=["reentrancy"])
    assert list(ci["per_flaw"]) == ["reentrancy"]
    point = f1_score(yt[:, 0], yp[:, 0])
    assert ci["macro_f1"]["lo"] <= point <= ci["macro_f1"]["hi"] + 1e-9


def test_mask_name_and_index_equivalent():
    assert resolve_mask(["reentrancy"]) == resolve_mask([0]) == [0]
    assert resolve_mask(None) == [0, 1, 2, 3, 4]


def test_bad_mask_rejected():
    for bad in (["nope"], [], [9]):
        with pytest.raises(ValueError):
            resolve_mask(bad)

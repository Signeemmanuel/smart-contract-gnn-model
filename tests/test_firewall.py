"""Required test 3: the train/test firewall, de-dup and frozen split."""
import numpy as np
import pytest
from training.data.firewall import (
    content_hash, dedup_wild_against_curated, stratified_multilabel_split, assert_firewall,
)


def test_content_hash_normalises_whitespace():
    assert content_hash("contract C {\n  uint x;\n}") == content_hash("contract C {   uint x; }")
    assert content_hash("contract A {}") != content_hash("contract B {}")


def test_dedup_removes_overlaps_and_leaves_disjoint_set():
    curated = {"c1": "contract Dup {}"}
    wild = {"w1": "contract Dup {}\n", "w2": "contract Other {}"}  # w1 == c1 after normalisation
    kept, removed, n = dedup_wild_against_curated(wild, curated)
    assert n == 1 and removed == ["w1"] and kept == ["w2"]
    kept_hashes = {content_hash(wild[i]) for i in kept}
    curated_hashes = {content_hash(s) for s in curated.values()}
    assert kept_hashes.isdisjoint(curated_hashes)


def test_stratified_split_represents_rare_classes_in_test():
    # 40 contracts; flaw 2 (arithmetic) appears in only 4 of them.
    rng = np.random.default_rng(0)
    Y = (rng.random((40, 5)) < 0.5).astype(int)
    Y[:, 2] = 0
    Y[[0, 1, 2, 3], 2] = 1  # the only arithmetic positives
    train_idx, test_idx = stratified_multilabel_split(Y, test_frac=0.3, seed=1)
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == 40
    assert Y[test_idx, 2].sum() >= 1  # rare class is represented in test


def test_assert_firewall_detects_leak():
    assert_firewall({"a", "b"}, {"c", "d"})  # disjoint: ok
    with pytest.raises(AssertionError):
        assert_firewall({"a", "b", "c"}, {"c", "d"})  # 'c' leaked

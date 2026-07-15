"""Required test 3 (extended): train/test firewall, de-dup, frozen split, and
the TWO frozen test sets (Test A tool-labelled + Test B expert Curated).

Non-negotiable #4: the firewall must exclude, from train and val, (a) every
contract in Test Set A and (b) every Curated contract (Test B provenance). This
test FAILS the build on any leak and now covers Test A as well as Curated.

The canonical content hash is comment-stripped and whitespace-collapsed; the
comment-stripping tests below pin the string-literal-safe behaviour so a hash
change can never silently regress the firewall.
"""
import numpy as np
import pytest

from training.data.firewall import (
    content_hash, dedup_wild_against_curated, strip_comments,
    stratified_multilabel_split, assert_firewall,
)
from training.data.testsets import (
    select_test_a, select_test_b, write_manifest, read_manifest,
    manifest_hashes, firewall_hashes,
)


# ---------------------------------------------------------------- existing tests

def test_content_hash_normalises_whitespace():
    assert content_hash("contract C {\n  uint x;\n}") == content_hash("contract C {   uint x; }")
    assert content_hash("contract A {}") != content_hash("contract B {}")


def test_dedup_removes_overlaps_and_leaves_disjoint_set():
    curated = {"c1": "contract Dup {}"}
    wild = {"w1": "contract Dup {}\n", "w2": "contract Other {}"}
    kept, removed, n = dedup_wild_against_curated(wild, curated)
    assert n == 1 and removed == ["w1"] and kept == ["w2"]
    kept_hashes = {content_hash(wild[i]) for i in kept}
    curated_hashes = {content_hash(s) for s in curated.values()}
    assert kept_hashes.isdisjoint(curated_hashes)


def test_stratified_split_represents_rare_classes_in_test():
    rng = np.random.default_rng(0)
    Y = (rng.random((40, 5)) < 0.5).astype(int)
    Y[:, 2] = 0
    Y[[0, 1, 2, 3], 2] = 1
    train_idx, test_idx = stratified_multilabel_split(Y, test_frac=0.3, seed=1)
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == 40
    assert Y[test_idx, 2].sum() >= 1


def test_assert_firewall_detects_leak():
    assert_firewall({"a", "b"}, {"c", "d"})
    with pytest.raises(AssertionError):
        assert_firewall({"a", "b", "c"}, {"c", "d"})


# ----------------------------------------------- NEW: comment-stripped hashing

def test_content_hash_ignores_comments():
    """A comment-edited copy of a contract must hash equal to the original, so
    it can never evade the firewall or waste labelling compute."""
    base = "contract C { uint x; }"
    line = "contract C { uint x; } // audited 2026"
    block = "/* SPDX header */ contract C { uint x; }"
    inline = "contract C { /* width */ uint x; }"
    assert content_hash(base) == content_hash(line)
    assert content_hash(base) == content_hash(block)
    assert content_hash(base) == content_hash(inline)


def test_strip_comments_preserves_string_literals():
    """// inside a string is NOT a comment; two contracts differing only inside
    a string literal must hash differently."""
    a = 'contract C { string u = "https://a.io"; }'
    b = 'contract C { string u = "https:"; }'
    assert "https://a.io" in strip_comments(a)
    assert content_hash(a) != content_hash(b)
    # escaped quote inside a string does not end it early
    c = 'contract C { string s = "he said \\"hi\\" // ok"; }'
    assert '// ok' in strip_comments(c)


def test_strip_comments_keeps_token_boundaries():
    """A block comment between tokens must not fuse them (a/*x*/b -> a b)."""
    assert strip_comments("uint/*gap*/a;") == "uint a;"
    # and hashes reflect the preserved boundary: 'uint a' != 'uinta'
    assert content_hash("uint/*gap*/a;") == content_hash("uint a;")
    assert content_hash("uint/*gap*/a;") != content_hash("uinta;")


# ------------------------------------------------------- Test A + Test B firewall

@pytest.fixture
def mock_pool(tmp_path):
    """A Wild pool (some positives, some negatives, one dup-of-curated) and a
    Curated set, written to disk so content hashes are real."""
    def w(name, src):
        p = tmp_path / name
        p.write_text(src, encoding="utf-8")
        return str(p)

    cur_src = "contract Curated {}"
    curated = {"cur1": {"path": w("cur1.sol", cur_src), "y": [1, 0, 0, 0, 0], "lines": [5]}}
    paths = {
        "w_dup":   w("w_dup.sol",   "contract   Curated {}"),   # == cur1 (ws diff)
        "w_re":    w("w_re.sol",    "contract R1 {}"),
        "w_dos":   w("w_dos.sol",   "contract D1 {}"),
        "w_arith": w("w_arith.sol", "contract A1 {}"),
        "w_neg1":  w("w_neg1.sol",  "contract N1 {}"),
        "w_neg2":  w("w_neg2.sol",  "contract N2 {}"),
    }
    labels = {
        "w_dup":   [1, 0, 0, 0, 0],
        "w_re":    [1, 0, 0, 0, 0],
        "w_dos":   [0, 0, 0, 0, 1],
        "w_arith": [0, 0, 1, 0, 0],
        "w_neg1":  [0, 0, 0, 0, 0],
        "w_neg2":  [0, 0, 0, 0, 0],
    }
    return tmp_path, curated, paths, labels, cur_src


def test_test_a_excludes_curated_duplicates(mock_pool):
    """A Wild contract identical to a Curated one must NOT enter Test A."""
    _, curated, paths, labels, cur_src = mock_pool
    rows = select_test_a(labels, paths, exclude_hashes={content_hash(cur_src)},
                         min_per_class=1, target_per_class=100, seed=42)
    ids = {r["contract_id"] for r in rows}
    assert "w_dup" not in ids, "duplicate-of-Curated leaked into Test A"
    assert {"w_re", "w_dos", "w_arith"} <= ids, "rare-class positives missing from Test A"


def test_frozen_manifests_and_firewall_union(mock_pool, tmp_path):
    """Freezing both test sets and firewalling train against their union must
    exclude every Test A and Test B contract."""
    _, curated, paths, labels, cur_src = mock_pool
    rows_a = select_test_a(labels, paths, exclude_hashes={content_hash(cur_src)},
                           min_per_class=1, target_per_class=100, seed=42)
    rows_b, _ = select_test_b(curated)
    mA, mB = tmp_path / "test_a.csv", tmp_path / "test_b.csv"
    write_manifest(rows_a, mA)
    write_manifest(rows_b, mB)

    fw = firewall_hashes(mA, mB)
    # both a Test A contract and the Curated contract are firewalled
    assert content_hash("contract R1 {}") in fw
    assert content_hash(cur_src) in fw

    # simulate a train index that ACCIDENTALLY includes a Test A contract -> must fail
    leaked_train = {content_hash("contract R1 {}")}   # w_re, which is in Test A
    with pytest.raises(AssertionError):
        assert_firewall(leaked_train, fw)

    # a clean train index (disjoint) passes
    clean_train = {content_hash("contract Fresh {}")}
    assert_firewall(clean_train, fw)   # no raise


def test_manifest_roundtrip_preserves_contracts(mock_pool, tmp_path):
    _, curated, paths, labels, cur_src = mock_pool
    rows = select_test_a(labels, paths, exclude_hashes={content_hash(cur_src)},
                         min_per_class=1, target_per_class=100, seed=42)
    m = tmp_path / "a.csv"
    write_manifest(rows, m)
    back = read_manifest(m)
    assert {r["contract_id"] for r in back} == {r["contract_id"] for r in rows}
    assert manifest_hashes(m) == {r["chash"] for r in rows}
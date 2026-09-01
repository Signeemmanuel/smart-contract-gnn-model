"""Each training baseline's data must contain zero test content hashes, and a
deliberately injected leak must fail the check."""
from __future__ import annotations

import pytest

from training.data.firewall import assert_firewall, content_hash


def test_clean_split_passes():
    train = {content_hash(f"contract C{i} {{}}") for i in range(20)}
    test = {content_hash(f"contract T{i} {{}}") for i in range(5)}
    assert_firewall(train, test)  # disjoint -> no raise


def test_injected_leak_fails():
    shared = content_hash("contract Leak { uint x; }")
    train = {shared, content_hash("contract Other {}")}
    test = {shared}
    with pytest.raises(AssertionError):
        assert_firewall(train, test)


def test_comment_edited_copy_is_the_same_hash():
    # the firewall hashes comment-stripped source, so a commented copy of a test
    # contract is caught as a leak.
    original = "contract A { function f() public {} }"
    commented = "// header\ncontract A { function f() public {} } // trailing"
    assert content_hash(original) == content_hash(commented)
    with pytest.raises(AssertionError):
        assert_firewall({content_hash(commented)}, {content_hash(original)})

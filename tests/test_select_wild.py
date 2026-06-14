"""Pure selection logic for the Wild subset: signatures + balanced allocation."""
from scripts.select_wild_subset import detect_signals, balanced_select, CLASSES


def test_signals_detect_each_class():
    assert "reentrancy" in detect_signals("x = msg.sender.call.value(amount)();")
    assert "unchecked_calls" in detect_signals("bool ok = a.call(data);")
    assert "access_control" in detect_signals("function kill() { selfdestruct(owner); }")
    assert "access_control" in detect_signals("require(tx.origin == owner);")
    assert "dos" in detect_signals("for (uint i; i<n; i++){ payees[i].transfer(1); }")


def test_arithmetic_needs_old_pragma_and_ops():
    old = "pragma solidity 0.4.24;\ncontract C{ function f(uint a){ total = total + a; } }"
    new = "pragma solidity 0.8.19;\ncontract C{ function f(uint a){ total = total + a; } }"
    assert "arithmetic" in detect_signals(old)
    assert "arithmetic" not in detect_signals(new)   # 0.8+ has built-in checks
    nopragma = "contract C{ uint x; }"
    assert "arithmetic" not in detect_signals(nopragma)


def test_balanced_select_is_deterministic_and_capped():
    sig = {f"c{i}": set() for i in range(100)}
    sig["c0"] = {"arithmetic"}
    sig["c1"] = {"dos"}
    a = balanced_select(sig, n=10, seed=1)
    b = balanced_select(sig, n=10, seed=1)
    assert a == b                       # deterministic
    assert len(a) == 10                 # capped
    assert "c0" in a and "c1" in a      # rare-class candidates pulled in


def test_balanced_select_respects_availability():
    # only 3 ids exist but n=10 -> return the 3 available, no crash
    sig = {"a": {"reentrancy"}, "b": set(), "c": {"dos"}}
    out = balanced_select(sig, n=10, seed=0)
    assert sorted(out) == ["a", "b", "c"]


def test_quota_fractions_cover_all_classes():
    from scripts.select_wild_subset import QUOTA_FRAC
    assert set(QUOTA_FRAC) == set(CLASSES)


def test_is_collectable_requires_a_pragma():
    from scripts.select_wild_subset import is_collectable
    assert is_collectable("pragma solidity ^0.4.24;\ncontract C{}")
    assert is_collectable("// header\npragma solidity 0.8.19;\ncontract C{}")
    assert not is_collectable("contract C{ uint x; }")     # no pragma -> SmartBugs can't pick solc
    assert not is_collectable("")

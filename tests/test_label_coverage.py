"""Coverage-aware vote matrices + single-detector labelling fallback.

These guard the fix for the failure where tools that cannot detect a flaw voted
"absent" (0) and buried the one tool that could (e.g. Osiris on arithmetic).
"""
import numpy as np

from scgnn.schema import FLAWS, FLAW_INDEX
from training.labelling.map_dasp import TOOL_COVERAGE
from training.labelling.run_tools import TOOLS, build_label_matrices
from training.labelling.snorkel_label import label_flaw, ABSTAIN


def test_tool_coverage_values_are_canonical_and_arithmetic_is_osiris_only():
    for tool, flaws in TOOL_COVERAGE.items():
        assert flaws <= set(FLAWS), f"{tool} covers unknown flaw"
    covers_arith = {t for t, fs in TOOL_COVERAGE.items() if "arithmetic" in fs}
    assert covers_arith == {"osiris"}            # only Osiris is relied on for arithmetic
    # every tool we run has a declared coverage
    assert set(TOOL_COVERAGE) >= set(TOOLS)


def test_non_covering_tool_abstains_not_votes_absent():
    # Slither + Osiris both RAN. Slither found reentrancy (covers it); Osiris found
    # arithmetic (covers it). Each must ABSTAIN (-1) on the other's class, not vote 0.
    votes = {"c.sol": {"slither": {"reentrancy"}, "osiris": {"arithmetic"}}}
    mats = build_label_matrices(votes, ["c.sol"])
    js = TOOLS.index("slither")
    jo = TOOLS.index("osiris")
    # arithmetic: slither does not cover it -> abstain (NOT 0); osiris found it -> 1
    assert mats["arithmetic"][0, js] == ABSTAIN
    assert mats["arithmetic"][0, jo] == 1
    # reentrancy: slither found it -> 1; osiris does not cover it -> abstain (NOT 0)
    assert mats["reentrancy"][0, js] == 1
    assert mats["reentrancy"][0, jo] == ABSTAIN


def test_covering_tool_that_ran_but_didnt_find_votes_zero():
    # Slither ran and found only reentrancy -> it covers dos, so 'no dos' is a real 0.
    votes = {"c.sol": {"slither": {"reentrancy"}}}
    mats = build_label_matrices(votes, ["c.sol"])
    js = TOOLS.index("slither")
    assert mats["dos"][0, js] == 0               # covered + ran + not found -> 0
    assert mats["reentrancy"][0, js] == 1


def test_single_detector_flaw_labelled_directly_not_zeroed():
    # arithmetic-style matrix: only the Osiris column carries signal, the other
    # three abstain on every contract. The label must follow Osiris, not collapse.
    n = 6
    L = np.full((n, len(TOOLS)), ABSTAIN, dtype=np.int8)
    jo = TOOLS.index("osiris")
    L[:, jo] = np.array([1, 1, 0, 0, 1, 0])      # Osiris flags 3 of 6
    labels, probs, rel = label_flaw(L, threshold=0.70)
    assert labels.tolist() == [1, 1, 0, 0, 1, 0]  # follows Osiris exactly
    assert rel == {"osiris": 1.0}


def test_all_abstain_flaw_is_all_negative():
    n = 4
    L = np.full((n, len(TOOLS)), ABSTAIN, dtype=np.int8)
    labels, probs, rel = label_flaw(L)
    assert labels.tolist() == [0, 0, 0, 0]
    assert rel == {}


def test_union_labels_positive_if_any_covering_tool_fires():
    from training.labelling.snorkel_label import label_flaw_union
    n = 5
    L = np.full((n, len(TOOLS)), ABSTAIN, dtype=np.int8)
    js, jm = TOOLS.index("slither"), TOOLS.index("mythril")
    L[0, js] = 1                       # slither only
    L[1, jm] = 1                       # mythril only
    L[2, js] = 1; L[2, jm] = 1         # both
    L[3, js] = 0                       # ran, didn't fire
    # row 4: all abstain
    labels, probs, rel = label_flaw_union(L)
    assert labels.tolist() == [1, 1, 1, 0, 0]   # union, no collapse


def test_union_corroboration_reliability():
    from training.labelling.snorkel_label import label_flaw_union
    n = 4
    L = np.full((n, len(TOOLS)), ABSTAIN, dtype=np.int8)
    js, jm = TOOLS.index("slither"), TOOLS.index("mythril")
    # slither fires on 0,1,2 ; mythril fires on 2 only -> slither corroborated 1/3, mythril 1/1
    L[[0, 1, 2], js] = 1
    L[2, jm] = 1
    _, _, rel = label_flaw_union(L)
    assert abs(rel["slither"] - 1/3) < 1e-9
    assert abs(rel["mythril"] - 1.0) < 1e-9


def test_union_single_detector_has_nan_reliability():
    from training.labelling.snorkel_label import label_flaw_union
    import math
    n = 4
    L = np.full((n, len(TOOLS)), ABSTAIN, dtype=np.int8)
    jo = TOOLS.index("osiris")
    L[:, jo] = np.array([1, 1, 0, 0])
    labels, _, rel = label_flaw_union(L)
    assert labels.tolist() == [1, 1, 0, 0]
    assert math.isnan(rel["osiris"])     # nothing to corroborate against

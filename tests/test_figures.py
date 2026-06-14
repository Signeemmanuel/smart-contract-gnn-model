"""Pure data-shaping helpers behind the report figures."""
import numpy as np

from scgnn.schema import FLAWS
from scripts.make_figures import (
    reliabilities_to_matrix, cooccurrence_from_labels, coverage_matrix,
)


def test_reliabilities_to_matrix_orders_flaws_and_collects_tools():
    rel = {
        "reentrancy": {"slither": 0.9, "mythril": 0.8},
        "arithmetic": {"osiris": 0.95},
    }
    M, tools = reliabilities_to_matrix(rel, FLAWS)
    assert M.shape == (5, len(tools))
    assert set(tools) == {"slither", "mythril", "osiris"}
    # reentrancy row has slither/mythril filled, osiris NaN
    i = FLAWS.index("reentrancy")
    assert M[i, tools.index("slither")] == 0.9
    assert np.isnan(M[i, tools.index("osiris")])
    j = FLAWS.index("arithmetic")
    assert M[j, tools.index("osiris")] == 0.95


def test_cooccurrence_counts_shared_positives():
    # 3 contracts: c0 has reentrancy+arithmetic, c1 reentrancy, c2 dos
    Y = np.zeros((3, 5), int)
    Y[0, FLAWS.index("reentrancy")] = 1
    Y[0, FLAWS.index("arithmetic")] = 1
    Y[1, FLAWS.index("reentrancy")] = 1
    Y[2, FLAWS.index("dos")] = 1
    C = cooccurrence_from_labels(Y)
    r, a, d = (FLAWS.index(x) for x in ("reentrancy", "arithmetic", "dos"))
    assert C[r, r] == 2            # reentrancy total
    assert C[r, a] == 1 and C[a, r] == 1   # co-occur once, symmetric
    assert C[d, d] == 1 and C[d, r] == 0


def test_coverage_matrix_reads_positive_counts():
    summary = {
        "slither": {f: {"positive": 0, "ran": 10} for f in FLAWS},
        "osiris": {f: {"positive": 0, "ran": 10} for f in FLAWS},
    }
    summary["slither"]["reentrancy"]["positive"] = 7
    summary["osiris"]["arithmetic"]["positive"] = 5
    M = coverage_matrix(summary, FLAWS, ["slither", "osiris"])
    assert M[0, FLAWS.index("reentrancy")] == 7
    assert M[1, FLAWS.index("arithmetic")] == 5
    assert M[0, FLAWS.index("arithmetic")] == 0   # slither doesn't cover arithmetic

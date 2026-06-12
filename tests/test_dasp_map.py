"""Required test 1: the DASP mapping (silent breakage = mislabelled training data)."""
from scgnn.schema import FLAWS
from training.labelling.map_dasp import (
    SLITHER_MAP, MYTHRIL_SWC, SECURIFY_MAP, TOOL_MAPS, map_finding,
)


def test_known_findings_map_correctly():
    assert map_finding("slither", "reentrancy-eth") == "reentrancy"
    assert map_finding("mythril", "SWC-101") == "arithmetic"
    assert map_finding("mythril", "SWC-107") == "reentrancy"
    assert map_finding("securify", "UnrestrictedWrite") == "access_control"


def test_tool_name_is_case_insensitive():
    assert map_finding("SLITHER", "tx-origin") == "access_control"


def test_unknown_finding_returns_none():
    assert map_finding("slither", "not-a-detector") is None
    assert map_finding("unknown-tool", "SWC-107") is None


def test_every_mapped_value_is_a_canonical_flaw():
    for m in (SLITHER_MAP, MYTHRIL_SWC, SECURIFY_MAP):
        assert set(m.values()) <= set(FLAWS)


def test_all_five_flaws_are_reachable_across_tools():
    produced = set()
    for m in TOOL_MAPS.values():
        produced |= set(m.values())
    assert produced == set(FLAWS), f"unreachable flaws: {set(FLAWS) - produced}"

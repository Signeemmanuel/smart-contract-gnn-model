"""Slither detector -> flaw parsing (the smoke-labelling path)."""
from training.labelling.run_slither import parse_slither_detectors


def test_maps_known_detectors_and_ignores_unmapped():
    data = {"success": True, "results": {"detectors": [
        {"check": "reentrancy-eth"},
        {"check": "unchecked-lowlevel"},
        {"check": "tx-origin"},
        {"check": "some-detector-we-do-not-map"},
    ]}}
    assert parse_slither_detectors(data) == {"reentrancy", "unchecked_calls", "access_control"}


def test_empty_and_malformed_inputs():
    assert parse_slither_detectors({"results": {"detectors": []}}) == set()
    assert parse_slither_detectors({}) == set()
    assert parse_slither_detectors({"results": None}) == set()


def test_pragma_minor_extracts_major_minor():
    from training.labelling.run_slither import pragma_minor
    assert pragma_minor("pragma solidity ^0.4.24;") == "0.4"
    assert pragma_minor("pragma solidity >=0.5.0 <0.6.0;") == "0.5"
    assert pragma_minor("// c\npragma solidity 0.8.19;\ncontract X {}") == "0.8"
    assert pragma_minor("contract X {}") is None


def test_choose_solc_matches_installed_minor():
    from training.labelling.run_slither import choose_solc
    by_minor = {"0.4": "0.4.26", "0.8": "0.8.19"}
    assert choose_solc("pragma solidity ^0.4.21;", by_minor) == "0.4.26"
    assert choose_solc("pragma solidity ^0.7.0;", by_minor) is None   # not installed
    assert choose_solc("contract X {}", by_minor) is None             # no pragma


def test_mapped_detectors_are_slither_ids():
    from training.labelling.run_slither import MAPPED_DETECTORS
    from training.labelling.map_dasp import SLITHER_MAP
    assert set(MAPPED_DETECTORS) == set(SLITHER_MAP)
    assert "reentrancy-eth" in MAPPED_DETECTORS

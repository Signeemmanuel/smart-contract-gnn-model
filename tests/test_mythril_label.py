"""SmartBugs result.json (PARSER_OUTPUT) -> flaw parsing.

Fixtured on the real simple_dao runs: the Slither result.json is copied verbatim
from the Studio; the Mythril fixtures reflect its native SWC vocabulary.
"""
import json

from training.labelling.run_tools import (
    mythril_finding_ids,
    parse_mythril_issues,
    finding_identifiers,
    parser_output_flaws,
    collect_votes,
    build_label_matrices,
)

# Real Slither result.json findings for simple_dao (trimmed to the `name` field).
# Note: low-level-calls and solc-version are Informational and must NOT map.
SLITHER_SIMPLE_DAO = {
    "errors": [], "fails": [], "infos": [],
    "parser": {"id": "slither-0.11.3", "mode": "solidity", "version": "2025/09/14"},
    "findings": [
        {"name": "reentrancy-eth", "impact": "High", "line": 16},
        {"name": "solc-version", "impact": "Informational", "line": 7},
        {"name": "solc-version", "impact": "Informational"},
        {"name": "low-level-calls", "impact": "Informational", "line": 16},
    ],
}

# Mythril's native result.log issues (validated cross-check of the SWC vocabulary).
MYTHRIL_NATIVE = {
    "error": None, "success": True,
    "issues": [{"swc-id": "107"}, {"swc-id": "104"}, {"swc-id": "105"}],
}


def test_slither_result_json_maps_only_real_vulns():
    # reentrancy-eth -> reentrancy; informational low-level-calls/solc-version dropped
    assert parser_output_flaws("slither", SLITHER_SIMPLE_DAO) == {"reentrancy"}


def test_finding_identifiers_uses_name_and_swc_id():
    assert finding_identifiers({"name": "reentrancy-eth"}) == ["reentrancy-eth"]
    # Mythril-style: a title name plus a bare swc-id -> both offered, swc prefixed
    assert finding_identifiers({"name": "Unprotected Ether Withdrawal", "swc-id": "105"}) == \
        ["Unprotected Ether Withdrawal", "SWC-105"]
    assert finding_identifiers({"swc-id": "SWC-101"}) == ["SWC-101"]


def test_parser_output_maps_via_swc_id_when_name_is_a_title():
    data = {"findings": [
        {"name": "External Call To User-Supplied Address", "swc-id": "107"},
        {"name": "Unchecked return value from external call.", "swc-id": "104"},
    ]}
    assert parser_output_flaws("mythril", data) == {"reentrancy", "unchecked_calls"}


def test_missing_findings_is_abstain_not_clean():
    assert parser_output_flaws("securify", {"errors": ["Killed"]}) is None   # no findings key
    assert parser_output_flaws("slither", {"findings": []}) == set()         # ran, clean


def test_mythril_native_cross_check_still_holds():
    assert mythril_finding_ids(MYTHRIL_NATIVE) == ["SWC-107", "SWC-104", "SWC-105"]
    assert parse_mythril_issues(MYTHRIL_NATIVE) == {
        "reentrancy", "unchecked_calls", "access_control"}


def test_mythril_swc_recovered_from_title_text():
    # Mythril's parsed result.json `name` is a title with the SWC embedded; there
    # is no separate swc-id field. These are the exact titles seen on the Studio.
    findings = [{"name": t} for t in (
        "Unchecked return value from external call. (SWC 104)",
        "External Call To User-Supplied Address (SWC 107)",
        "Unprotected Ether Withdrawal (SWC 105)",
        "Multiple Calls in a Single Transaction (SWC 113)",
        "Write to an arbitrary storage location (SWC 124)",
        "Transaction Order Dependence (SWC 114)",   # out of scope -> must NOT map
        "Exception State (SWC 110)",                 # out of scope -> must NOT map
    )]
    assert parser_output_flaws("mythril", {"findings": findings}) == {
        "unchecked_calls", "reentrancy", "access_control", "dos"}


def test_finding_identifiers_extracts_embedded_swc():
    ids = finding_identifiers({"name": "Unchecked return value from external call. (SWC 104)"})
    assert "SWC-104" in ids


def test_collect_votes_reads_result_json_by_path(tmp_path):
    # slither contract dir
    sd = tmp_path / "slither-0.11.3" / "20260613_1011" / "simple_dao.sol"
    sd.mkdir(parents=True)
    (sd / "result.json").write_text(json.dumps(SLITHER_SIMPLE_DAO), encoding="utf-8")
    (sd / "smartbugs.json").write_text(json.dumps({"result": {"output": None}}), encoding="utf-8")
    # mythril contract dir (same contract, title names + swc-id)
    md = tmp_path / "mythril-0.24.8" / "20260613_1011" / "simple_dao.sol"
    md.mkdir(parents=True)
    (md / "result.json").write_text(json.dumps({"findings": [
        {"name": "x", "swc-id": "107"}, {"name": "y", "swc-id": "104"}]}), encoding="utf-8")

    votes = collect_votes(tmp_path)
    assert votes["simple_dao"]["slither"] == {"reentrancy"}
    assert votes["simple_dao"]["mythril"] == {"reentrancy", "unchecked_calls"}

    mats = build_label_matrices(votes, ["simple_dao"])
    # TOOLS = [slither, mythril, securify, osiris]; securify+osiris absent -> abstain (-1)
    assert mats["reentrancy"][0].tolist() == [1, 1, -1, -1]
    assert mats["unchecked_calls"][0].tolist() == [0, 1, -1, -1]
    assert mats["dos"][0].tolist() == [0, 0, -1, -1]


def test_osiris_arithmetic_mapping():
    from training.labelling.run_tools import parser_output_flaws
    data = {"findings": [
        {"name": "Overflow bugs"}, {"name": "Underflow bugs"},
        {"name": "Time dependency bug"},   # out of scope -> unmapped
    ]}
    assert parser_output_flaws("osiris", data) == {"arithmetic"}

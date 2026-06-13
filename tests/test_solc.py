"""Per-contract solc resolution (shared by extraction and the labeller)."""
from pathlib import Path

from scgnn.extraction.solc import (
    choose_solc, installed_solc_binaries, pragma_minor, solc_for_file,
)


def _make_artifacts(tmp_path: Path, versions: list[str]) -> Path:
    for v in versions:
        d = tmp_path / f"solc-{v}"
        d.mkdir()
        (d / f"solc-{v}").write_text("#!/bin/sh\n")
    return tmp_path


def test_pragma_minor_and_choose():
    assert pragma_minor("pragma solidity ^0.4.24;") == "0.4"
    assert pragma_minor(">=0.5.0 <0.6.0;") is None          # not a full pragma stmt
    assert pragma_minor("pragma solidity >=0.5.0 <0.6.0;") == "0.5"
    m = {"0.4": "/x/solc-0.4.26", "0.8": "/x/solc-0.8.19"}
    assert choose_solc("pragma solidity ^0.4.21;", m) == "/x/solc-0.4.26"
    assert choose_solc("pragma solidity ^0.7.0;", m) is None  # not installed


def test_installed_solc_binaries_discovers_highest_patch(tmp_path):
    base = _make_artifacts(tmp_path, ["0.4.26", "0.5.17", "0.8.19"])
    bins = installed_solc_binaries(base)
    assert set(bins) == {"0.4", "0.5", "0.8"}
    assert bins["0.4"].endswith("solc-0.4.26")


def test_installed_solc_binaries_empty_when_missing(tmp_path):
    assert installed_solc_binaries(tmp_path / "nope") == {}


def test_solc_for_file_resolves_and_skips(tmp_path):
    base = _make_artifacts(tmp_path, ["0.4.26", "0.5.17"])
    bins = installed_solc_binaries(base)
    good = tmp_path / "good.sol"
    good.write_text("pragma solidity ^0.4.23;\ncontract X {}\n")
    assert solc_for_file(good, bins).endswith("solc-0.4.26")
    bad = tmp_path / "bad.sol"
    bad.write_text("pragma solidity ^0.3.0;\n")        # no installed match
    assert solc_for_file(bad, bins) is None
    assert solc_for_file(tmp_path / "missing.sol", bins) is None  # unreadable

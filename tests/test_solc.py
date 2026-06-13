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


def test_pragma_exact_detects_pins_only():
    from scgnn.extraction.solc import pragma_exact
    assert pragma_exact("pragma solidity 0.4.25;") == "0.4.25"
    assert pragma_exact("pragma solidity =0.5.17;") == "0.5.17"
    assert pragma_exact("pragma solidity 0.4.9; /* comment */") == "0.4.9"
    assert pragma_exact("pragma solidity ^0.4.24;") is None      # caret -> not exact
    assert pragma_exact("pragma solidity >=0.4.22 <0.6.0;") is None
    assert pragma_exact("// no pragma here") is None


def test_installed_solc_full_and_exact_preference(tmp_path):
    from scgnn.extraction.solc import installed_solc_full, solc_for_file, installed_solc_binaries
    for v in ["0.4.24", "0.4.25", "0.4.26"]:
        d = tmp_path / f"solc-{v}"; d.mkdir(); (d / f"solc-{v}").write_text("#!/bin/sh\n")
    full = installed_solc_full(tmp_path)
    by_minor = installed_solc_binaries(tmp_path)            # -> {'0.4': '.../solc-0.4.26'}
    assert set(full) == {"0.4.24", "0.4.25", "0.4.26"}
    # exact pin wins over the minor's newest patch
    pinned = tmp_path / "pinned.sol"; pinned.write_text("pragma solidity 0.4.25;\n")
    assert solc_for_file(pinned, by_minor, full).endswith("solc-0.4.25")
    # caret falls back to newest patch of the minor
    caret = tmp_path / "caret.sol"; caret.write_text("pragma solidity ^0.4.0;\n")
    assert solc_for_file(caret, by_minor, full).endswith("solc-0.4.26")
    # exact pin with no installed match -> None (never a wrong patch)
    miss = tmp_path / "miss.sol"; miss.write_text("pragma solidity 0.4.9;\n")
    assert solc_for_file(miss, by_minor, full) is None


def test_fallback_cfg_is_valid_single_node(tmp_path):
    from scgnn.extraction.extract import _fallback_cfg
    f = tmp_path / "x.sol"; f.write_text("contract X { function f() public {} }")
    g = _fallback_cfg(str(f))
    assert g.view == "cfg" and g.n_nodes == 1 and g.edges == []
    indeg, outdeg = g.degrees()                  # must not raise on a lone node
    assert len(indeg) == 1 and len(outdeg) == 1

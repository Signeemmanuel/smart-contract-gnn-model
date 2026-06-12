"""Schema is the cross-repo contract; guard its invariants."""
import json
import pytest
from scgnn.schema import (
    FLAWS, FLAW_INDEX, N_FLAWS, FlawType, FlawResult, AnalysisResult, validate_flaw_code,
)


def test_canonical_order_is_fixed():
    assert FLAWS == ["reentrancy", "access_control", "arithmetic", "unchecked_calls", "dos"]
    assert N_FLAWS == 5
    assert [FLAW_INDEX[c] for c in FLAWS] == list(range(5))


def test_enum_is_string_valued():
    assert FlawType.REENTRANCY == "reentrancy"
    assert validate_flaw_code(FlawType.DOS) == "dos"


def test_lines_are_rank_preserving_deduped():
    fr = FlawResult(FlawType.REENTRANCY, 0.91, [42, 47, 53, 47])
    assert fr.lines == [42, 47, 53]


def test_roundtrip_is_lossless():
    res = AnalysisResult("contract C {}", [FlawResult("dos", 0.8, [1, 2])])
    assert AnalysisResult.from_dict(json.loads(res.to_json())).to_dict() == res.to_dict()


@pytest.mark.parametrize("bad", [
    lambda: FlawResult("nope", 0.5, []),
    lambda: FlawResult("reentrancy", 1.5, []),
    lambda: FlawResult("reentrancy", 0.5, [0]),
    lambda: validate_flaw_code("dao"),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises((ValueError, KeyError)):
        bad()

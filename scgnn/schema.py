"""Result schema for the SC-GNN flaw detector.

This module is the single source of truth for two things that must stay
identical across all three repositories:

1. the canonical set and *ordering* of the five flaw types, and
2. the JSON shape returned by the back-end ``/analyze`` endpoint.

``scgnn-api`` re-exports the structures defined here rather than redefining
them, and the model's five output logits are ordered to match :data:`FLAWS`.
Keeping that order in one place is what guarantees that logit ``i``, label
matrix column ``i``, the per-class positive weight ``i`` and ``FLAWS[i]`` all
refer to the same flaw. Get this wrong once and every downstream table is
silently mislabelled, so the order lives here and nowhere else.

The structures deliberately use the standard library only (``dataclasses`` and
``enum``), so installing ``scgnn`` does not force a particular validation
library onto the back end. ``scgnn-api`` is free to wrap :class:`AnalysisResult`
in its own FastAPI/Pydantic ``response_model``; the wire format is unchanged.

The wire contract is intentionally tiny::

    {"source": "<contract text>",
     "flaws": [{"type": "reentrancy", "confidence": 0.91, "lines": [42, 47, 53]}]}

The GAT first-layer attention signal (Phase 3) is produced for evaluation and
internal cross-checking; it is *not* part of this wire contract. If we later
decide to surface it in the dashboard, we extend the schema here, once, as a
deliberate cross-repository change rather than an ad-hoc field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FlawType(str, Enum):
    """The five flaw types, aligned to DASP categories.

    Inheriting from :class:`str` means a member serialises to its own string
    value (``json.dumps(FlawType.REENTRANCY)`` emits ``"reentrancy"``), so the
    enum is interchangeable with the raw code throughout the codebase.
    """

    REENTRANCY = "reentrancy"
    ACCESS_CONTROL = "access_control"
    ARITHMETIC = "arithmetic"
    UNCHECKED_CALLS = "unchecked_calls"
    DOS = "dos"


#: Canonical ordering of the flaw codes. This order is authoritative: the model
#: head emits one logit per entry in this exact order, and every label matrix,
#: positive-weight vector and metrics row is indexed by it. Do not reorder.
FLAWS: list[str] = [f.value for f in FlawType]

#: Reverse lookup: flaw code -> column index in ``range(N_FLAWS)``.
FLAW_INDEX: dict[str, int] = {code: i for i, code in enumerate(FLAWS)}

#: Number of flaw classes. Defined once; import it rather than hard-coding ``5``.
N_FLAWS: int = len(FLAWS)


def validate_flaw_code(code: "str | FlawType") -> str:
    """Return the canonical flaw code string if known, otherwise raise.

    Accepts either a raw string or a :class:`FlawType` member and always
    returns the plain code value (``"reentrancy"``). Used at the contract
    boundary so an unknown code fails loudly here rather than silently
    producing an out-of-range column index downstream. We read ``.value``
    explicitly because ``str(FlawType.REENTRANCY)`` is ``"FlawType.REENTRANCY"``,
    not the value.
    """
    key = code.value if isinstance(code, FlawType) else str(code)
    if key not in FLAW_INDEX:
        raise ValueError(f"unknown flaw code {code!r}; expected one of {FLAWS}")
    return key


@dataclass
class FlawResult:
    """A single detected flaw and the lines held responsible for it.

    Attributes:
        type: One of the five canonical flaw codes (see :data:`FLAWS`).
        confidence: The model's sigmoid probability for this flaw, in ``[0, 1]``.
        lines: Source line numbers implicated by the explanation component,
            ordered by influence (most influential first) and de-duplicated.
            This is a *ranked* list, not a numerically sorted one; callers must
            not re-sort it, as the order is the localisation result.
    """

    type: str
    confidence: float
    lines: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.type = validate_flaw_code(self.type)
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        # Order-preserving de-duplication: keep the first (highest-ranked)
        # occurrence of each line so the boundary guarantee in the docstring
        # holds regardless of what the caller passed in.
        cleaned: list[int] = []
        seen: set[int] = set()
        for raw in self.lines:
            ln = int(raw)
            if ln < 1:
                raise ValueError(f"line numbers are 1-based; got {ln!r}")
            if ln in seen:
                continue
            seen.add(ln)
            cleaned.append(ln)
        self.lines = cleaned

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "confidence": self.confidence, "lines": self.lines}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlawResult":
        return cls(
            type=d["type"],
            confidence=d["confidence"],
            lines=list(d.get("lines", [])),
        )


@dataclass
class AnalysisResult:
    """The full payload returned for one analysed contract.

    Only flaws the model flagged (confidence at or above the chosen threshold)
    should appear in ``flaws``; flaws below threshold are simply omitted.
    """

    source: str
    flaws: list[FlawResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "flaws": [f.to_dict() for f in self.flaws]}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisResult":
        return cls(
            source=d["source"],
            flaws=[FlawResult.from_dict(x) for x in d.get("flaws", [])],
        )

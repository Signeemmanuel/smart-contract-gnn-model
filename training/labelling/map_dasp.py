"""Map each tool's finding identifiers to the five canonical flaw codes.

This is the only domain knowledge injected by hand, and it is small and
auditable. The canonical codes and their order come from ``scgnn.schema`` so the
label-matrix columns line up with the model's logits. Pure module; unit-tested
(this is one of the three required tests).
"""

from __future__ import annotations

from scgnn.schema import FLAWS  # noqa: F401  (re-exported for callers)

# Slither detector ids -> flaw code.
SLITHER_MAP: dict[str, str] = {
    "reentrancy-eth": "reentrancy",
    "reentrancy-no-eth": "reentrancy",
    "reentrancy-benign": "reentrancy",
    "reentrancy-events": "reentrancy",
    "reentrancy-unlimited-gas": "reentrancy",
    "arbitrary-send": "access_control",
    "arbitrary-send-eth": "access_control",
    "suicidal": "access_control",
    "tx-origin": "access_control",
    "unprotected-upgrade": "access_control",
    "unchecked-lowlevel": "unchecked_calls",
    "unchecked-send": "unchecked_calls",
    "unchecked-transfer": "unchecked_calls",
    "calls-loop": "dos",
    "locked-ether": "dos",
    # arithmetic: Slither defers to the solc version (no dedicated detector since
    # 0.8 checks overflow); arithmetic votes come mainly from Mythril/Securify.
}

# Mythril SWC ids -> flaw code.
MYTHRIL_SWC: dict[str, str] = {
    "SWC-107": "reentrancy",
    "SWC-105": "access_control",
    "SWC-106": "access_control",
    "SWC-115": "access_control",   # tx.origin authorisation
    "SWC-112": "access_control",   # delegatecall to untrusted callee
    "SWC-124": "access_control",   # write to arbitrary storage location
    "SWC-101": "arithmetic",
    "SWC-104": "unchecked_calls",
    "SWC-113": "dos",
    "SWC-128": "dos",
}

# Securify pattern names -> flaw code.
SECURIFY_MAP: dict[str, str] = {
    "DAO": "reentrancy",
    "DAOConstantGas": "reentrancy",
    "UnrestrictedWrite": "access_control",
    "UnrestrictedSelfdestruct": "access_control",
    "TODReceiver": "access_control",
    "MissingInputValidation": "unchecked_calls",
    "UnhandledException": "unchecked_calls",
    "UnrestrictedEtherFlow": "unchecked_calls",
    "LockedEther": "dos",
}

# Osiris finding names -> flaw code. Osiris is the integer-bug specialist that
# fills the arithmetic gap none of Slither/Mythril/Securify cover. Names are the
# plain-English categories Osiris reports (validated on the curated arithmetic set).
OSIRIS_MAP: dict[str, str] = {
    "Overflow bugs": "arithmetic",
    "Underflow bugs": "arithmetic",
    "Division bugs": "arithmetic",
    "Modulo bugs": "arithmetic",
    "Truncation bugs": "arithmetic",
    "Signedness bugs": "arithmetic",
    # "Time dependency bug" is timestamp-dependence, out of the five -> unmapped.
}

TOOL_MAPS: dict[str, dict[str, str]] = {
    "slither": SLITHER_MAP,
    "mythril": MYTHRIL_SWC,
    "securify": SECURIFY_MAP,
    "osiris": OSIRIS_MAP,
}


def map_finding(tool: str, identifier: str) -> str | None:
    """Return the flaw code for a tool's finding id, or ``None`` if unmapped."""
    return TOOL_MAPS.get(tool.lower(), {}).get(identifier)

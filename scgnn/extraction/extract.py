"""Orchestrator: a Solidity file -> (AST RawGraph, CFG RawGraph).

Status: needs solc + Slither; validate on the Studio.

Compiler selection is per contract. If ``solc_binary`` is not given, we resolve
it from the contract's pragma against the installed solc-select binaries (so the
AST subprocess and Slither's CFG both use the right compiler instead of whatever
``solc`` is on PATH). Pass ``binaries`` to reuse a discovery done once by the
caller; pass ``solc_binary`` to force a specific compiler for every contract.
"""

from __future__ import annotations

from scgnn.extraction.graph_types import RawGraph
from scgnn.extraction.slither_ast import extract_ast
from scgnn.extraction.slither_cfg import extract_cfg
from scgnn.extraction.solc import installed_solc_binaries, solc_for_file


def extract_contract(path: str, solc_binary: str | None = None,
                     binaries: dict[str, str] | None = None) -> tuple[RawGraph, RawGraph]:
    """Extract both views for one contract, keeping them aligned by contract.

    Resolves the per-pragma solc binary when ``solc_binary`` is None; if no
    matching compiler is installed it falls back to PATH ``solc`` (which may fail
    on old contracts — the caller should skip and log such failures).
    """
    if solc_binary is None:
        if binaries is None:
            binaries = installed_solc_binaries()
        solc_binary = solc_for_file(path, binaries)
    ast = extract_ast(path, solc_binary=solc_binary or "solc")
    cfg = extract_cfg(path, solc_binary=solc_binary)
    return ast, cfg

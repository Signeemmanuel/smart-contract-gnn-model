"""Orchestrator: a Solidity file -> (AST RawGraph, CFG RawGraph).

Status: needs solc + Slither; validate on the Studio.

Compiler selection is per contract and pragma-aware: an exact pin
(``pragma solidity 0.4.25;``) uses that precise solc, otherwise the newest patch
of the pragma's minor. Pass ``binaries``/``full_binaries`` to reuse a discovery
done once by the caller, or ``solc_binary`` to force one compiler.

A few contracts compile (AST ok) but crash Slither's own SlitHIR generation
(upstream bugs on certain tuple-call or array-length patterns). With
``cfg_fallback`` (default True) such a contract is retained with its real AST and
a single-node placeholder CFG, rather than dropped — the event is logged so the
degraded CFG is visible. Set ``cfg_fallback=False`` to drop them instead.
"""

from __future__ import annotations

import logging

from scgnn.extraction.graph_types import RawGraph
from scgnn.extraction.slither_ast import extract_ast
from scgnn.extraction.slither_cfg import extract_cfg
from scgnn.extraction.solc import installed_solc_binaries, installed_solc_full, solc_for_file

_log = logging.getLogger(__name__)


def _fallback_cfg(path: str) -> RawGraph:
    """A minimal, valid one-node CFG for contracts Slither cannot analyse.

    Keeps the contract in the dataset (its AST view is intact) without an empty
    graph that would break mean-pooling. The single node carries the source so
    the embedding block still contributes something.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read(2000)
    except OSError:
        src = ""
    g = RawGraph(view="cfg")
    g.node_types = ["ENTRY_POINT"]
    g.snippets = [src]
    g.depths = [0]
    g.n_children = [0]
    g.edges = []
    g.node_lines = {0: [1]}
    return g


def extract_contract(path: str, solc_binary: str | None = None,
                     binaries: dict[str, str] | None = None,
                     full_binaries: dict[str, str] | None = None,
                     cfg_fallback: bool = True) -> tuple[RawGraph, RawGraph]:
    """Extract both views for one contract, keeping them aligned by contract.

    Resolves the per-pragma solc (exact pin preferred) when ``solc_binary`` is
    None; if no matching compiler is installed it falls back to PATH ``solc``
    (which will fail on a pinned contract — the caller skips and logs).
    """
    if solc_binary is None:
        if binaries is None:
            binaries = installed_solc_binaries()
        if full_binaries is None:
            full_binaries = installed_solc_full()
        solc_binary = solc_for_file(path, binaries, full_binaries)
    ast = extract_ast(path, solc_binary=solc_binary or "solc")
    try:
        cfg = extract_cfg(path, solc_binary=solc_binary)
    except Exception as exc:
        if not cfg_fallback:
            raise
        _log.warning("CFG fallback for %s (Slither failed: %s: %s)",
                     path, type(exc).__name__, exc)
        cfg = _fallback_cfg(path)
    return ast, cfg

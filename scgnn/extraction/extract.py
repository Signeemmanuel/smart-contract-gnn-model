"""Orchestrator: a Solidity file -> (AST RawGraph, CFG RawGraph).

Status: needs solc + Slither; validate on the Studio.

Compiler selection is per contract and pragma-aware: an exact pin
(``pragma solidity 0.4.25;``) uses that precise solc, otherwise the newest patch
of the pragma's minor. Pass ``binaries``/``full_binaries`` to reuse a discovery
done once by the caller, or ``solc_binary`` to force one compiler.

A few contracts compile (AST ok) but crash Slither's own SlitHIR generation
(upstream bugs on certain tuple-call or array-length patterns). With
``cfg_fallback`` (default True) such a contract is retained with its real AST and
a single-node placeholder CFG, rather than dropped - the event is logged and the
graph is flagged ``degraded=True`` so the condition is visible downstream.

v2 (Workstream C): ``with_data_flow`` (default True) adds def-use data-flow edges
to the CFG view, typed ``data_flow`` alongside the CFG's own ``control_flow``
edges. If the data-flow pass fails on a contract, extraction retries WITHOUT it
so the contract is kept with a control-flow-only CFG, flagged ``degraded``. The
AST view is never changed by this.
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
    the embedding block still contributes something. Flagged ``degraded``.
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
    g.edge_types = []
    g.node_lines = {0: [1]}
    g.degraded = True
    return g


def extract_contract(path: str, solc_binary: str | None = None,
                     binaries: dict[str, str] | None = None,
                     full_binaries: dict[str, str] | None = None,
                     cfg_fallback: bool = True,
                     with_data_flow: bool = True) -> tuple[RawGraph, RawGraph]:
    """Extract both views for one contract, keeping them aligned by contract.

    Resolves the per-pragma solc (exact pin preferred) when ``solc_binary`` is
    None; if no matching compiler is installed it falls back to PATH ``solc``
    (which will fail on a pinned contract - the caller skips and logs).

    ``with_data_flow`` adds def-use edges to the CFG. Three outcomes, all
    recorded so the dissertation can report per-contract success (Workstream C.4):

      * data-flow ok      -> cfg.n_data_flow_edges > 0, degraded False
      * data-flow failed  -> control-flow-only CFG, degraded True (logged)
      * Slither failed    -> single-node placeholder CFG, degraded True (logged)
    """
    if solc_binary is None:
        if binaries is None:
            binaries = installed_solc_binaries()
        if full_binaries is None:
            full_binaries = installed_solc_full()
        solc_binary = solc_for_file(path, binaries, full_binaries)

    ast = extract_ast(path, solc_binary=solc_binary or "solc")

    try:
        cfg = extract_cfg(path, solc_binary=solc_binary, with_data_flow=with_data_flow)
    except Exception as exc:
        # Distinguish "the data-flow pass broke" from "Slither cannot analyse this
        # contract at all": retry control-flow-only before giving up on the CFG,
        # so a data-flow hiccup never costs us a whole contract.
        if with_data_flow:
            try:
                cfg = extract_cfg(path, solc_binary=solc_binary, with_data_flow=False)
                cfg.degraded = True
                _log.warning("data-flow pass failed for %s (%s: %s); kept "
                             "control-flow-only CFG", path, type(exc).__name__, exc)
                return ast, cfg
            except Exception:
                pass                      # fall through to the placeholder below
        if not cfg_fallback:
            raise
        _log.warning("CFG fallback for %s (Slither failed: %s: %s)",
                     path, type(exc).__name__, exc)
        cfg = _fallback_cfg(path)
    return ast, cfg
"""CFG extraction to a :class:`RawGraph` using Slither.

Status: needs Slither (+ solc) and real contracts; validate on the Studio. The
assembler ``cfg_from_nodes`` is pure and unit-tested.

Slither exposes a per-function CFG through ``function.nodes``; each node carries
control-flow successors (``node.sons``) and a source mapping with line numbers.
We concatenate every function's nodes into one contract-level CFG and keep the
node-to-lines map.
"""

from __future__ import annotations

from typing import Any

from scgnn.extraction.graph_types import RawGraph


def cfg_from_nodes(collected: list[dict[str, Any]]) -> RawGraph:
    """Assemble a CFG :class:`RawGraph` from collected node records.

    Each record is ``{"type": str, "snippet": str, "lines": list[int],
    "sons": list[int]}`` where ``sons`` are indices into ``collected``. Pure and
    testable; ``extract_cfg`` produces these records from Slither.
    """
    g = RawGraph(view="cfg")
    for nid, rec in enumerate(collected):
        g.node_types.append(str(rec.get("type", "Unknown")))
        g.snippets.append(str(rec.get("snippet", "")))
        g.node_lines[nid] = list(rec.get("lines", []))
        g.depths.append(0)        # depth is not meaningful for a flat CFG
        g.n_children.append(len(rec.get("sons", [])))
    for nid, rec in enumerate(collected):
        for son in rec.get("sons", []):
            g.edges.append((nid, int(son)))
    return g


def extract_cfg(path: str, solc_binary: str | None = None) -> RawGraph:
    """Extract a contract-level CFG with Slither. Needs Slither + solc.

    ``solc_binary`` is passed straight to Slither's ``solc`` keyword (the binary
    location), so each contract compiles with the version matching its pragma
    rather than whatever ``solc`` happens to be on PATH.
    """
    from slither import Slither  # imported lazily so the package imports without Slither

    kwargs = {"solc": solc_binary} if solc_binary else {}
    sl = Slither(path, **kwargs)
    nodes: list[Any] = []
    index: dict[int, int] = {}
    for contract in sl.contracts:
        for fn in contract.functions:
            for node in fn.nodes:
                index[id(node)] = len(nodes)
                nodes.append(node)

    collected: list[dict[str, Any]] = []
    for node in nodes:
        lines = list(getattr(node.source_mapping, "lines", []) or [])
        sons = [index[id(s)] for s in node.sons if id(s) in index]
        collected.append(
            {
                "type": str(getattr(node, "type", "Node")),
                "snippet": str(getattr(node, "expression", "") or ""),
                "lines": lines,
                "sons": sons,
            }
        )
    return cfg_from_nodes(collected)

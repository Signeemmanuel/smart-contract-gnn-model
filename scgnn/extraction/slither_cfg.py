"""CFG extraction to a :class:`RawGraph` using Slither, with data-flow edges.

Status: needs Slither (+ solc) and real contracts; validate on the Studio. The
assemblers ``cfg_from_nodes`` and ``data_flow_edges`` are pure and unit-tested.

Slither exposes a per-function CFG through ``function.nodes``; each node carries
control-flow successors (``node.sons``) and a source mapping with line numbers.
We concatenate every function's nodes into one contract-level CFG and keep the
node-to-lines map.

v2 (Workstream C) additionally recovers DATA-FLOW edges from Slither's per-node
variable read/write sets: a def-use edge runs from the node that WRITES a
variable to each later node (in the same function) that READS it. Motivation:
the strongest published systems on this corpus (BugSweeper, Peculiar, ReVulDL)
all exploit data flow, and reentrancy in particular is a data-flow property (a
state variable read before an external call and written after it).

Edges are typed (``control_flow`` / ``data_flow``) so the mandatory ablation can
train with and without the data-flow edges on identical node features.
"""

from __future__ import annotations

from typing import Any

from scgnn.extraction.graph_types import CONTROL_FLOW, DATA_FLOW, RawGraph


def data_flow_edges(collected: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Def-use edges from per-node variable reads/writes. Pure; unit-tested.

    Each record may carry ``"writes"`` and ``"reads"``: the names of variables
    that node writes / reads, plus ``"fn"``, an id grouping nodes by function.
    For every variable, an edge is emitted from each writing node to every
    reading node that appears LATER in the same function's node order (Slither's
    node order follows control flow within a function, so 'later' approximates
    'reachable from').

    Emitting write -> read (rather than read -> write) means information flows in
    the direction the value does, which is what a message-passing GNN needs.
    Duplicate edges are collapsed; self-loops are dropped.
    """
    out: set[tuple[int, int]] = set()
    # last writers of each (function, variable), in node order
    writers: dict[tuple[Any, str], list[int]] = {}
    for nid, rec in enumerate(collected):
        fn = rec.get("fn")
        for var in rec.get("writes", []) or []:
            writers.setdefault((fn, str(var)), []).append(nid)

    for nid, rec in enumerate(collected):
        fn = rec.get("fn")
        for var in rec.get("reads", []) or []:
            for w in writers.get((fn, str(var)), []):
                if w < nid:                      # the write precedes this read
                    out.add((w, nid))
    return sorted(out)


def cfg_from_nodes(collected: list[dict[str, Any]], *,
                   with_data_flow: bool = True) -> RawGraph:
    """Assemble a CFG :class:`RawGraph` from collected node records.

    Each record is ``{"type": str, "snippet": str, "lines": list[int],
    "sons": list[int], "reads": list[str], "writes": list[str], "fn": Any}``
    where ``sons`` are indices into ``collected``. Control-flow edges come from
    ``sons``; data-flow edges from reads/writes when ``with_data_flow``. Pure and
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
            g.add_edge(nid, int(son), CONTROL_FLOW)

    if with_data_flow:
        for s, d in data_flow_edges(collected):
            # A pair may be BOTH a control-flow successor and a def-use edge (a
            # value defined on one line and used on the next). We keep both: in
            # the merged edge_index that is a parallel edge, which PyG handles
            # and which correctly strengthens a connection carrying real data
            # dependence. Dropping it would erase the very signal the data-flow
            # pass exists to add, and would leave the ablation with nothing to
            # measure on exactly the tightest def-use chains.
            g.add_edge(s, d, DATA_FLOW)
    return g


def _var_names(vars_) -> list[str]:
    """Names of a Slither variable collection, robust to missing ``name``."""
    names = []
    for v in vars_ or []:
        n = getattr(v, "name", None) or getattr(v, "canonical_name", None) or str(v)
        names.append(str(n))
    return names


def extract_cfg(path: str, solc_binary: str | None = None, *,
                with_data_flow: bool = True) -> RawGraph:
    """Extract a contract-level CFG with Slither. Needs Slither + solc.

    ``solc_binary`` is passed straight to Slither's ``solc`` keyword (the binary
    location), so each contract compiles with the version matching its pragma
    rather than whatever ``solc`` happens to be on PATH.

    When ``with_data_flow``, each node's variable reads/writes are collected from
    Slither and turned into def-use edges. If a node does not expose those sets
    (older Slither, or an odd node kind), it simply contributes no data-flow
    edges: the CFG degrades to control-flow only rather than failing.
    """
    from slither import Slither  # imported lazily so the package imports without Slither

    kwargs = {"solc": solc_binary} if solc_binary else {}
    sl = Slither(path, **kwargs)
    nodes: list[Any] = []
    owner: list[Any] = []            # the function each node belongs to
    index: dict[int, int] = {}
    for contract in sl.contracts:
        for fn in contract.functions:
            for node in fn.nodes:
                index[id(node)] = len(nodes)
                nodes.append(node)
                owner.append(id(fn))

    collected: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        lines = list(getattr(node.source_mapping, "lines", []) or [])
        sons = [index[id(s)] for s in node.sons if id(s) in index]
        rec: dict[str, Any] = {
            "type": str(getattr(node, "type", "Node")),
            "snippet": str(getattr(node, "expression", "") or ""),
            "lines": lines,
            "sons": sons,
            "fn": owner[i],
        }
        if with_data_flow:
            # Slither exposes read/written variables per node. Include state and
            # local variables; both matter (reentrancy is a state-variable
            # read/write ordering property).
            reads = list(getattr(node, "variables_read", []) or [])
            writes = list(getattr(node, "variables_written", []) or [])
            rec["reads"] = _var_names(reads)
            rec["writes"] = _var_names(writes)
        collected.append(rec)

    return cfg_from_nodes(collected, with_data_flow=with_data_flow)
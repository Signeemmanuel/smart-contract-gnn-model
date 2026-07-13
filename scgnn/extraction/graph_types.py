"""Shared, framework-free types for extracted graphs.

These structures are deliberately plain (no torch), so extraction can be tested
without a deep-learning stack and so the node-to-source-line map travels as an
ordinary Python dict rather than something that a PyG batch would silently drop.

v2 adds TYPED edges (Workstream C). ``edge_types[i]`` labels ``edges[i]`` as
either ``control_flow`` (the CFG's own successor edges, and every AST edge) or
``data_flow`` (a def-use / data-dependency edge recovered from Slither). Edges
stay a flat list so all existing consumers keep working unchanged; the type
labels are a parallel list, which lets the build either merge all edges (the
baseline variant used here) or filter the data-flow edges out for the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTROL_FLOW = "control_flow"
DATA_FLOW = "data_flow"
EDGE_TYPES = (CONTROL_FLOW, DATA_FLOW)


@dataclass
class RawGraph:
    """One extracted graph (AST or CFG) before feature encoding.

    Indices are positional: ``node_types[i]``, ``snippets[i]``, ``depths[i]`` and
    ``n_children[i]`` all describe node ``i``, and ``node_lines[i]`` is the list
    of 1-based source line numbers that node ``i`` covers. ``edges`` are
    directed ``(src, dst)`` index pairs, and ``edge_types[i]`` is the type of
    ``edges[i]`` (``control_flow`` by default, so a graph built without the v2
    data-flow pass behaves exactly as in v1).

    The ``node_lines`` map is the single most important artefact to preserve: it
    is what makes line-level localisation possible later, so every transformation
    must carry it through unchanged.
    """

    view: str  # "ast" or "cfg"
    node_types: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    n_children: list[int] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    node_lines: dict[int, list[int]] = field(default_factory=dict)
    edge_types: list[str] = field(default_factory=list)
    degraded: bool = False        # True when this graph came from a fallback path

    def __post_init__(self) -> None:
        # A graph built the v1 way (no edge_types) is all control-flow: fill in
        # the labels so `edge_types` is always aligned with `edges`.
        if len(self.edge_types) != len(self.edges):
            self.edge_types = [CONTROL_FLOW] * len(self.edges)

    @property
    def n_nodes(self) -> int:
        return len(self.node_types)

    @property
    def n_data_flow_edges(self) -> int:
        return sum(1 for t in self.edge_types if t == DATA_FLOW)

    def add_edge(self, src: int, dst: int, etype: str = CONTROL_FLOW) -> None:
        """Append one typed edge, keeping ``edges`` and ``edge_types`` aligned."""
        self.edges.append((int(src), int(dst)))
        self.edge_types.append(etype)

    def without_data_flow(self) -> "RawGraph":
        """A copy with data-flow edges removed: the ablation's 'without' arm.

        Node features are untouched; only DF edges are dropped, so any measured
        difference is attributable to the edges alone. Pure.
        """
        keep = [i for i, t in enumerate(self.edge_types) if t != DATA_FLOW]
        return RawGraph(
            view=self.view,
            node_types=list(self.node_types),
            snippets=list(self.snippets),
            depths=list(self.depths),
            n_children=list(self.n_children),
            edges=[self.edges[i] for i in keep],
            node_lines={k: list(v) for k, v in self.node_lines.items()},
            edge_types=[self.edge_types[i] for i in keep],
            degraded=self.degraded,
        )

    def degrees(self) -> tuple[list[int], list[int]]:
        """Return (in_degree, out_degree) per node, computed from ``edges``."""
        n = self.n_nodes
        indeg = [0] * n
        outdeg = [0] * n
        for s, d in self.edges:
            outdeg[s] += 1
            indeg[d] += 1
        return indeg, outdeg

    def to_dict(self) -> dict:
        """Plain-JSON form. ``node_lines`` int keys become strings (JSON rule)."""
        return {
            "view": self.view,
            "node_types": list(self.node_types),
            "snippets": list(self.snippets),
            "depths": list(self.depths),
            "n_children": list(self.n_children),
            "edges": [[int(s), int(d)] for s, d in self.edges],
            "node_lines": {str(k): list(v) for k, v in self.node_lines.items()},
            "edge_types": list(self.edge_types),
            "degraded": bool(self.degraded),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RawGraph":
        """Inverse of :meth:`to_dict`; restores int keys and tuple edges.

        Tolerates v1 caches (no ``edge_types``/``degraded`` keys): __post_init__
        fills edge_types with control_flow, so an old extraction cache still
        loads and behaves exactly as before.
        """
        return cls(
            view=d["view"],
            node_types=list(d["node_types"]),
            snippets=list(d["snippets"]),
            depths=list(d["depths"]),
            n_children=list(d["n_children"]),
            edges=[(int(s), int(dd)) for s, dd in d["edges"]],
            node_lines={int(k): list(v) for k, v in d["node_lines"].items()},
            edge_types=list(d.get("edge_types", [])),
            degraded=bool(d.get("degraded", False)),
        )


def byte_offset_to_line(source_bytes: bytes, offset: int) -> int:
    """Convert a solc byte offset into a 1-based line number.

    solc ``src`` fields are ``start:length:fileIndex`` where ``start`` is a
    *byte* offset into the UTF-8 source, so we count newlines over bytes, not
    characters, to stay correct in the presence of multi-byte characters.
    """
    if offset < 0:
        offset = 0
    offset = min(offset, len(source_bytes))
    return source_bytes[:offset].count(b"\n") + 1
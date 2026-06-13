"""Shared, framework-free types for extracted graphs.

These structures are deliberately plain (no torch), so extraction can be tested
without a deep-learning stack and so the node-to-source-line map travels as an
ordinary Python dict rather than something that a PyG batch would silently drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawGraph:
    """One extracted graph (AST or CFG) before feature encoding.

    Indices are positional: ``node_types[i]``, ``snippets[i]``, ``depths[i]`` and
    ``n_children[i]`` all describe node ``i``, and ``node_lines[i]`` is the list
    of 1-based source line numbers that node ``i`` covers. ``edges`` are
    directed ``(src, dst)`` index pairs.

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

    @property
    def n_nodes(self) -> int:
        return len(self.node_types)

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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RawGraph":
        """Inverse of :meth:`to_dict`; restores int keys and tuple edges."""
        return cls(
            view=d["view"],
            node_types=list(d["node_types"]),
            snippets=list(d["snippets"]),
            depths=list(d["depths"]),
            n_children=list(d["n_children"]),
            edges=[(int(s), int(dd)) for s, dd in d["edges"]],
            node_lines={int(k): list(v) for k, v in d["node_lines"].items()},
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

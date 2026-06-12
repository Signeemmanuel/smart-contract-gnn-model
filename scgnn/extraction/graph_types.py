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

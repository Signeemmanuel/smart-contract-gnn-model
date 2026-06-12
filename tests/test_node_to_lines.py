"""Required test 2: node-to-source-line conversion (the basis of localisation)."""
from scgnn.extraction.graph_types import byte_offset_to_line
from scgnn.extraction.slither_ast import ast_from_compact_json
from scgnn.explain.localise import rank_unique, nodes_to_lines


def test_byte_offset_handles_multibyte():
    src = "line1\nliné2\nline3"            # 'é' is two UTF-8 bytes
    b = src.encode("utf-8")
    assert byte_offset_to_line(b, 0) == 1
    assert byte_offset_to_line(b, src.encode("utf-8").index(b"line3")) == 3
    assert byte_offset_to_line(b, 10**9) == 3  # clamped to end


def test_ast_assembler_preserves_lines_and_edges():
    source = "AAAA\nBBBB\nCCCC"
    # root spans all; child1 -> "BBBB" (line 2), child2 -> "CCCC" (line 3)
    root = {
        "nodeType": "Root", "src": "0:14:0",
        "body": [
            {"nodeType": "A", "src": "5:4:0"},   # BBBB, line 2
            {"nodeType": "B", "src": "10:4:0"},  # CCCC, line 3
        ],
    }
    g = ast_from_compact_json(root, source)
    assert g.node_types[0] == "Root" and g.n_children[0] == 2 and g.depths[0] == 0
    assert g.depths[1] == 1
    assert g.node_lines[1] == [2]
    assert g.node_lines[2] == [3]
    assert (0, 1) in g.edges and (0, 2) in g.edges


def test_rank_unique_preserves_first_seen_order():
    assert rank_unique([5, 3, 5, 1, 3]) == [5, 3, 1]


def test_nodes_to_lines_logs_unmapped():
    node_lines = {0: [10, 11], 1: [], 2: [11, 12]}
    lines, unmapped = nodes_to_lines([0, 1, 2], node_lines)
    assert lines == [10, 11, 12]
    assert unmapped == [1]

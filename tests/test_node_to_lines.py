"""Required test 2: node-to-source-line conversion (the basis of localisation)."""
from scgnn.extraction.graph_types import byte_offset_to_line
from scgnn.extraction.slither_ast import ast_from_compact_json
from scgnn.explain.localise import (
    rank_unique,
    nodes_to_lines,
    normalise,
    line_scores_from_importance,
    merge_line_scores,
)


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


def test_normalise_scales_by_max_and_handles_degenerate():
    assert normalise([1.0, 2.0, 4.0]) == [0.25, 0.5, 1.0]
    assert normalise([0.0, 0.0]) == [0.0, 0.0]   # no positive mass
    assert normalise([]) == []


def test_line_scores_take_branch_max_per_line():
    # node 0 -> line 10 (imp 1.0 after norm), node 1 -> line 10 too (imp 0.5),
    # node 2 -> line 20 (imp 0.5). Line 10 keeps the larger (1.0).
    importance = [2.0, 1.0, 1.0]
    node_lines = {0: [10], 1: [10], 2: [20]}
    scores, unmapped = line_scores_from_importance(importance, node_lines)
    assert scores == {10: 1.0, 20: 0.5}
    assert unmapped == []


def test_line_scores_flag_influential_unmapped_node():
    importance = [5.0, 1.0]          # node 0 most influential but maps nowhere
    node_lines = {1: [7]}
    scores, unmapped = line_scores_from_importance(importance, node_lines)
    assert scores == {7: 0.2}
    assert unmapped == [0]


def test_merge_surfaces_strong_cfg_line_over_weak_ast_lines():
    # The bug this fixes: under concatenation, line 44 (strong in CFG) sat
    # behind every AST line. Merged by max, it now ranks first.
    ast_scores = {9: 0.6, 10: 0.55, 11: 0.5, 12: 0.45, 13: 0.4}
    cfg_scores = {44: 0.95, 43: 0.3}
    assert merge_line_scores(ast_scores, cfg_scores, k=5) == [44, 9, 10, 11, 12]


def test_merge_max_across_branches_and_tiebreak_by_line():
    a = {5: 1.0, 8: 0.5}
    b = {8: 0.9, 5: 0.2}      # line 8 -> max(0.5,0.9)=0.9; line 5 -> max(1.0,0.2)=1.0
    assert merge_line_scores(a, b, k=2) == [5, 8]
    # exact tie -> lower line number first
    assert merge_line_scores({3: 0.7, 1: 0.7}, k=2) == [1, 3]

"""RawGraph survives a JSON round-trip (used to cache extraction to disk)."""
import json
from scgnn.extraction.graph_types import RawGraph


def test_roundtrip_restores_int_keys_and_tuple_edges():
    g = RawGraph(view="ast", node_types=["A", "B"], snippets=["a", "b"],
                 depths=[0, 1], n_children=[1, 0], edges=[(0, 1)],
                 node_lines={0: [1, 2], 1: [3]})
    back = RawGraph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert back.view == "ast"
    assert back.node_types == g.node_types
    assert back.edges == [(0, 1)]                 # tuples, not lists
    assert back.node_lines == {0: [1, 2], 1: [3]} # int keys, not "0"/"1"
    assert back.degrees() == g.degrees()

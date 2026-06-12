"""Bonus: feature assembly (one-hot + structural + PCA) is correct and aligned."""
import numpy as np
from sklearn.decomposition import PCA
from scgnn.extraction.graph_types import RawGraph
from scgnn.extraction.features import FeatureConfig, FeatureEncoder


class StubEmbedder:
    DIM = 16
    def embed(self, snippet: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(snippet)) % (2**32))
        return rng.random(self.DIM).astype(np.float32)


def _graph():
    g = RawGraph(view="ast")
    g.node_types = ["FunctionDefinition", "Assignment", "WeirdUnknownType"]
    g.snippets = ["function f()", "x = 1", "??"]
    g.depths = [0, 1, 1]
    g.n_children = [2, 0, 0]
    g.edges = [(0, 1), (0, 2)]
    g.node_lines = {0: [1], 1: [2], 2: [3]}
    return g


def test_encode_array_shape_onehot_and_unknown_handling():
    cfg = FeatureConfig(node_types=["FunctionDefinition", "Assignment"], embed_dim=8)
    pca = PCA(n_components=8, random_state=0).fit(np.random.default_rng(0).random((20, 16)))
    enc = FeatureEncoder(cfg, StubEmbedder(), pca)
    x, edge_index = enc.encode_array(_graph())
    assert x.shape == (3, cfg.in_dim)           # 2 one-hot + 4 structural + 8 pca = 14
    assert x[0, 0] == 1.0 and x[1, 1] == 1.0     # one-hot for known types
    assert x[2, :2].sum() == 0.0                 # unknown type -> zero one-hot block
    assert enc.unknown_types == 1
    assert edge_index.shape == (2, 2)
    # structural block sits right after the one-hot block
    assert list(x[0, 2:6]) == [0.0, 2.0, 0.0, 2.0]  # depth, n_children, in_deg, out_deg

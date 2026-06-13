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


class StubBatchEmbedder(StubEmbedder):
    """Same per-snippet vectors as StubEmbedder, but exposes a batched path."""
    def embed_many(self, snippets, batch_size: int = 128):
        return (np.vstack([self.embed(s) for s in snippets]) if snippets
                else np.zeros((0, self.DIM), np.float32))


def test_encode_array_batched_matches_per_node():
    cfg = FeatureConfig(node_types=["FunctionDefinition", "Assignment"], embed_dim=8)
    train = np.random.default_rng(1).random((20, StubEmbedder.DIM))
    pca = PCA(n_components=8, random_state=0).fit(train)
    per_node = FeatureEncoder(cfg, StubEmbedder(), pca).encode_array(_graph())[0]
    batched = FeatureEncoder(cfg, StubBatchEmbedder(), pca).encode_array(_graph())[0]
    assert per_node.shape == batched.shape == (3, cfg.in_dim)
    assert np.allclose(per_node, batched)        # embed_many path == embed path


def test_encode_array_no_embed_block_when_pca_none():
    cfg = FeatureConfig(node_types=["FunctionDefinition", "Assignment"], embed_dim=8)
    x, _ = FeatureEncoder(cfg, StubBatchEmbedder(), pca=None).encode_array(_graph())
    assert x.shape == (3, len(cfg.node_types) + 4)   # one-hot + structural only

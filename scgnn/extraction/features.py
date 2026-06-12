"""Node feature encoding: one-hot type + structural + CodeBERT->PCA(64).

The fitted artefacts (the node-type vocabulary in :class:`FeatureConfig` and the
PCA) are produced on the TRAIN split only and shipped in the model bundle, so
inference encodes nodes identically to training. The numpy assembler
``FeatureEncoder.encode_array`` is unit-tested; ``to_data`` (the PyG wrapper) and
the CodeBERT embedder need torch/transformers and are validated on the Studio.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from scgnn.extraction.graph_types import RawGraph

STRUCTURAL = ["depth", "n_children", "in_degree", "out_degree"]


@dataclass
class FeatureConfig:
    """Everything needed to encode nodes the same way at train and serve time."""

    node_types: list[str]            # ordered one-hot vocabulary
    embed_dim: int = 64              # PCA target dimensionality
    structural: tuple[str, ...] = tuple(STRUCTURAL)

    @property
    def in_dim(self) -> int:
        return len(self.node_types) + len(self.structural) + self.embed_dim

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FeatureConfig":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(node_types=list(d["node_types"]),
                   embed_dim=int(d.get("embed_dim", 64)),
                   structural=tuple(d.get("structural", STRUCTURAL)))


class Embedder(Protocol):
    """Maps a code snippet to a fixed-length vector (e.g. CodeBERT, 768-d)."""

    def embed(self, snippet: str) -> np.ndarray: ...


class CodeBERTEmbedder:
    """CodeBERT sentence embedding with a per-snippet cache. Needs transformers.

    Status: not executed here (needs the model weights). Many AST/CFG nodes share
    identical snippets, so the hash cache avoids recomputing embeddings.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).eval().to(device)
        self.device = device
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, snippet: str) -> np.ndarray:
        key = hashlib.sha1(snippet.encode("utf-8")).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        torch = self._torch
        with torch.no_grad():
            ids = self.tok(snippet or " ", return_tensors="pt", truncation=True, max_length=64).to(self.device)
            vec = self.enc(**ids).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        self._cache[key] = vec
        return vec


class FeatureEncoder:
    """Turn a :class:`RawGraph` into model-ready features.

    ``pca`` is a fitted scikit-learn ``PCA`` (or any object with ``transform``)
    reducing the embedder output to ``config.embed_dim``. Pass ``pca=None`` to
    skip the embedding block (useful in tests).
    """

    def __init__(self, config: FeatureConfig, embedder: Embedder | None, pca=None) -> None:
        self.config = config
        self.embedder = embedder
        self.pca = pca
        self._type_index = {t: i for i, t in enumerate(config.node_types)}
        self.unknown_types = 0

    def _one_hot(self, node_type: str) -> np.ndarray:
        vec = np.zeros(len(self.config.node_types), dtype=np.float32)
        idx = self._type_index.get(node_type)
        if idx is None:
            self.unknown_types += 1  # unknown -> all-zero block; counted, not guessed
        else:
            vec[idx] = 1.0
        return vec

    def encode_array(self, raw: RawGraph) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, edge_index)`` as numpy arrays. Pure; unit-tested."""
        indeg, outdeg = raw.degrees()
        rows: list[np.ndarray] = []
        embeds: list[np.ndarray] | None = [] if (self.embedder and self.pca is not None) else None
        for i in range(raw.n_nodes):
            onehot = self._one_hot(raw.node_types[i])
            struct = np.array([raw.depths[i], raw.n_children[i], indeg[i], outdeg[i]], dtype=np.float32)
            parts = [onehot, struct]
            if embeds is not None:
                embeds.append(self.embedder.embed(raw.snippets[i]))
            rows.append(np.concatenate(parts))
        base = np.vstack(rows) if rows else np.zeros((0, len(self.config.node_types) + len(self.config.structural)), np.float32)
        if embeds is not None:
            reduced = self.pca.transform(np.vstack(embeds)).astype(np.float32) if embeds else \
                np.zeros((0, self.config.embed_dim), np.float32)
            base = np.hstack([base, reduced]) if base.shape[0] else \
                np.zeros((0, self.config.in_dim), np.float32)
        edge_index = (np.array(raw.edges, dtype=np.int64).T
                      if raw.edges else np.zeros((2, 0), dtype=np.int64))
        return base.astype(np.float32), edge_index

    def to_data(self, raw: RawGraph):
        """Wrap ``encode_array`` output in a PyG ``Data``. Needs torch/PyG."""
        import torch
        from torch_geometric.data import Data

        x, edge_index = self.encode_array(raw)
        return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index))


def fit_feature_config(raw_graphs: list[RawGraph], embed_dim: int = 64) -> FeatureConfig:
    """Collect the node-type vocabulary from TRAIN graphs only."""
    types: list[str] = []
    seen: set[str] = set()
    for g in raw_graphs:
        for t in g.node_types:
            if t not in seen:
                seen.add(t)
                types.append(t)
    return FeatureConfig(node_types=sorted(types), embed_dim=embed_dim)


def fit_pca(train_embeddings: np.ndarray, embed_dim: int = 64, seed: int = 42):
    """Fit a PCA on TRAIN-split embeddings only. Needs scikit-learn."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=embed_dim, random_state=seed)
    pca.fit(train_embeddings)
    return pca

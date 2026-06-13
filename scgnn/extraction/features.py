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

    def embed_many(self, snippets: list[str], batch_size: int = 128) -> np.ndarray:
        """Embed many snippets with batched forward passes; orders of magnitude
        faster than per-snippet calls on a GPU.

        Cache hits are served directly; only misses are tokenised and run, in
        ``batch_size`` chunks with attention-masked mean pooling — so each
        snippet's vector is identical to :meth:`embed` (which pools an unpadded
        sequence). Cache keys match :meth:`embed`, so the two paths interoperate.
        """
        n = len(snippets)
        if n == 0:
            return np.zeros((0, 768), np.float32)
        torch = self._torch
        out: list[np.ndarray | None] = [None] * n
        keys = [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in snippets]
        todo_idx: list[int] = []
        for i, k in enumerate(keys):
            hit = self._cache.get(k)
            if hit is not None:
                out[i] = hit
            else:
                todo_idx.append(i)
        for b in range(0, len(todo_idx), batch_size):
            batch = todo_idx[b:b + batch_size]
            texts = [snippets[i] or " " for i in batch]
            with torch.no_grad():
                ids = self.tok(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=64).to(self.device)
                hs = self.enc(**ids).last_hidden_state                 # [B, T, 768]
                mask = ids["attention_mask"].unsqueeze(-1).type_as(hs)  # [B, T, 1]
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                vecs = pooled.cpu().numpy()
            for j, i in enumerate(batch):
                v = vecs[j]
                out[i] = v
                self._cache[keys[i]] = v
        return np.vstack(out).astype(np.float32)


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
        """Return ``(x, edge_index)`` as numpy arrays. Pure; unit-tested.

        When the embedder exposes ``embed_many`` the snippet block is embedded in
        one batched call (much faster on a GPU); otherwise it falls back to
        per-node ``embed``. Either path yields the same result.
        """
        indeg, outdeg = raw.degrees()
        n = raw.n_nodes
        use_emb = self.embedder is not None and self.pca is not None
        base_cols = len(self.config.node_types) + len(self.config.structural)
        rows: list[np.ndarray] = []
        for i in range(n):
            onehot = self._one_hot(raw.node_types[i])
            struct = np.array([raw.depths[i], raw.n_children[i], indeg[i], outdeg[i]], dtype=np.float32)
            rows.append(np.concatenate([onehot, struct]))
        base = np.vstack(rows) if rows else np.zeros((0, base_cols), np.float32)
        if use_emb:
            if n:
                many = getattr(self.embedder, "embed_many", None)
                emb_mat = (many(raw.snippets) if many is not None
                           else np.vstack([self.embedder.embed(s) for s in raw.snippets]))
                reduced = self.pca.transform(emb_mat).astype(np.float32)
                base = np.hstack([base, reduced])
            else:
                base = np.zeros((0, self.config.in_dim), np.float32)
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

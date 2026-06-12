"""Inference entry point shared by training and the back end.

Status: needs the full stack (torch, PyG, Slither/solc, transformers) + a model
bundle; py_compiled here, run on the Studio. ``scgnn-api`` calls ``load_model``
once at start-up and ``analyze_source`` per request, so both repositories run
this identical code.

The bundle is pinned by an immutable commit revision on the Hugging Face Hub, so
the deployed service always pairs known weights with known code.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

from scgnn.schema import FLAWS, AnalysisResult, FlawResult


@dataclass
class LoadedModel:
    model: object
    encoder: object          # FeatureEncoder
    config: dict
    threshold: float


def load_model(repo_id: str, revision: str, device: str = "cpu",
               weights_name: str = "model.safetensors") -> LoadedModel:
    """Download a pinned bundle from the Hub and build a ready-to-serve model."""
    import joblib
    import torch
    from huggingface_hub import hf_hub_download

    from scgnn.extraction.features import CodeBERTEmbedder, FeatureConfig, FeatureEncoder
    from scgnn.models.dual_gnn import build_model

    def fetch(name: str) -> str:
        return hf_hub_download(repo_id=repo_id, filename=name, revision=revision)

    with open(fetch("config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    feat_cfg = FeatureConfig.from_json(fetch("feature_config.json"))
    pca = joblib.load(fetch("pca.joblib"))

    model = build_model(config)
    weights_path = fetch(weights_name)
    if weights_name.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights_path)
    else:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    encoder = FeatureEncoder(feat_cfg, CodeBERTEmbedder(device=device), pca)
    return LoadedModel(model=model, encoder=encoder, config=config,
                       threshold=float(config.get("threshold", 0.70)))


def analyze_source(loaded: LoadedModel, src: str, threshold: float | None = None) -> dict:
    """Analyse one Solidity source string and return the schema dict."""
    import torch

    from scgnn.explain.attention import attention_lines
    from scgnn.explain.explainer import explain_lines
    from scgnn.extraction.extract import extract_contract

    thr = loaded.threshold if threshold is None else threshold

    with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False, encoding="utf-8") as fh:
        fh.write(src)
        path = fh.name
    try:
        ast_raw, cfg_raw = extract_contract(path)
    finally:
        os.unlink(path)

    ast_data = loaded.encoder.to_data(ast_raw)
    cfg_data = loaded.encoder.to_data(cfg_raw)

    with torch.no_grad():
        ast_b = ast_data.clone(); ast_b.batch = torch.zeros(ast_data.x.size(0), dtype=torch.long)
        cfg_b = cfg_data.clone(); cfg_b.batch = torch.zeros(cfg_data.x.size(0), dtype=torch.long)
        proba = loaded.model.predict_proba(ast_b, cfg_b).squeeze(0).tolist()

    flaws: list[FlawResult] = []
    is_gat = loaded.config.get("conv") == "gat"
    for idx, p in enumerate(proba):
        if p < thr:
            continue
        lines, _unmapped = explain_lines(
            loaded.model, ast_data, cfg_data, idx, ast_raw.node_lines, cfg_raw.node_lines,
        )
        if is_gat and not lines:  # cheap fallback signal when GNNExplainer is empty
            lines, _ = attention_lines(loaded.model.ast, ast_raw.node_lines)
        flaws.append(FlawResult(type=FLAWS[idx], confidence=round(float(p), 4), lines=lines))

    return AnalysisResult(source=src, flaws=flaws).to_dict()

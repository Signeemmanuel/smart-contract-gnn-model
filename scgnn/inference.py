"""Inference entry point shared by training and the back end.

Status: needs the full stack (torch, PyG, Slither/solc, transformers) + a model
bundle; py_compiled here, run on the Studio. ``scgnn-api`` calls ``load_model``
once at start-up and ``analyze_source`` per request, so both repositories run
this identical code.

The bundle is pinned by an immutable commit revision on the Hugging Face Hub, so
the deployed service always pairs known weights with known code.

Bundle versions
---------------
v1 bundles carry a single scalar ``threshold`` in config.json. v2 bundles add a
``manifest.json`` with PER-CLASS thresholds tuned on validation, plus the edge
schema and provenance. This module reads the manifest when present and falls
back to the v1 scalar otherwise, so BOTH bundle versions keep working and the
API can select by version.

Applying one global threshold to a model whose thresholds were tuned per class
would silently use the wrong decision rule, so the per-class path is the one that
matters for a v2 deployment.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field

from scgnn.schema import FLAWS, AnalysisResult, FlawResult

ATTENTION_CONVS = ("gat", "gatv2")


@dataclass
class LoadedModel:
    model: object
    encoder: object                       # FeatureEncoder
    config: dict
    threshold: float                      # v1 scalar; kept for compatibility
    thresholds: list[float] = field(default_factory=list)   # v2 per-class
    manifest: dict = field(default_factory=dict)

    def threshold_for(self, idx: int) -> float:
        """The decision threshold for flaw ``idx``: per-class when the bundle
        provides them, else the single scalar."""
        if self.thresholds and idx < len(self.thresholds):
            return float(self.thresholds[idx])
        return float(self.threshold)


def load_model(repo_id: str, revision: str, device: str = "cpu",
               weights_name: str = "model.safetensors") -> LoadedModel:
    """Download a pinned bundle from the Hub and build a ready-to-serve model.

    Reads ``manifest.json`` if the bundle has one (v2) for the per-class
    thresholds and the edge schema; a v1 bundle without a manifest still loads,
    using its scalar threshold.
    """
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

    # v2 manifest is optional: a v1 bundle simply does not have one.
    manifest: dict = {}
    try:
        with open(fetch("manifest.json"), "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        manifest = {}

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

    thresholds = [float(t) for t in manifest.get("thresholds", [])]
    if thresholds and len(thresholds) != len(FLAWS):
        raise ValueError(
            f"bundle manifest has {len(thresholds)} thresholds but there are "
            f"{len(FLAWS)} classes; refusing to serve a mismatched decision rule.")

    return LoadedModel(
        model=model, encoder=encoder, config=config,
        threshold=float(config.get("threshold", 0.70)),
        thresholds=thresholds,
        manifest=manifest,
    )


def analyze_source(loaded: LoadedModel, src: str,
                   threshold: float | None = None) -> dict:
    """Analyse one Solidity source string and return the schema dict.

    ``threshold``, when given, OVERRIDES the bundle's rule for every class (a
    caller asking for one global cut-off). When it is None, each class uses its
    own tuned threshold from the v2 manifest, or the v1 scalar.
    """
    import torch

    from scgnn.explain.attention import attention_lines
    from scgnn.explain.explainer import explain_lines
    from scgnn.extraction.extract import extract_contract

    with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        path = fh.name
    try:
        # Extract with the edge schema the model was TRAINED on: a model fitted
        # with data-flow edges must be served graphs that have them, or its input
        # distribution silently shifts.
        with_df = loaded.manifest.get("edge_schema") == "control_flow+data_flow"
        try:
            ast_raw, cfg_raw = extract_contract(path, with_data_flow=with_df)
        except TypeError:
            # v1 extract_contract has no with_data_flow parameter.
            ast_raw, cfg_raw = extract_contract(path)
    finally:
        os.unlink(path)

    ast_data = loaded.encoder.to_data(ast_raw)
    cfg_data = loaded.encoder.to_data(cfg_raw)

    with torch.no_grad():
        ast_b = ast_data.clone()
        ast_b.batch = torch.zeros(ast_data.x.size(0), dtype=torch.long)
        cfg_b = cfg_data.clone()
        cfg_b.batch = torch.zeros(cfg_data.x.size(0), dtype=torch.long)
        proba = loaded.model.predict_proba(ast_b, cfg_b).squeeze(0).tolist()

    flaws: list[FlawResult] = []
    has_attention = loaded.config.get("conv") in ATTENTION_CONVS
    for idx, p in enumerate(proba):
        thr = loaded.threshold_for(idx) if threshold is None else float(threshold)
        if p < thr:
            continue
        lines, _unmapped = explain_lines(
            loaded.model, ast_data, cfg_data, idx,
            ast_raw.node_lines, cfg_raw.node_lines,
        )
        if has_attention and not lines:   # cheap fallback when GNNExplainer is empty
            lines, _ = attention_lines(loaded.model.ast, ast_raw.node_lines)
        flaws.append(FlawResult(type=FLAWS[idx], confidence=round(float(p), 4),
                                lines=lines))

    return AnalysisResult(source=src, flaws=flaws).to_dict()
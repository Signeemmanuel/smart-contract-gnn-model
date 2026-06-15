# scgnn-model — file manifest

Status legend:

- **[verified]** — executed here; covered by a passing unit test or run directly.
- **[needs-Studio]** — written faithfully to the implementation guide and the real
  library APIs, byte-compiled here (no syntax errors), but **not executed** because it
  needs a component unavailable in this environment (Slither/`solc`, SmartBugs,
  CodeBERT weights, Snorkel, torch/PyG, or a GPU). Smoke-test these first on the
  Lightning Studio.
- **[config/doc]** — not code.

The shipped/serving boundary: everything under `scgnn/` is published to `scgnn-api`;
everything under `training/` is excluded from the wheel (see `pyproject.toml`).

## Package — `scgnn/` (shipped to the back end)

| Path | Status | Purpose |
|------|--------|---------|
| `scgnn/__init__.py` | verified | Package exports; torch-free; `__version__`. |
| `scgnn/schema.py` | verified | Single source of truth for the five flaw codes, their order, and the `/analyze` wire contract. |
| `scgnn/common/__init__.py` | config/doc | Namespace. |
| `scgnn/common/seeds.py` | needs-Studio | `set_seed` (cuDNN deterministic; needs torch). |
| `scgnn/extraction/__init__.py` | verified | Re-exports `RawGraph`, `FeatureConfig`, `FeatureEncoder`. |
| `scgnn/extraction/graph_types.py` | verified | `RawGraph` + `byte_offset_to_line`; the node→lines carrier. |
| `scgnn/extraction/slither_ast.py` | mixed | `ast_from_compact_json` **[verified]**; `extract_ast` (solc) **[needs-Studio]**. |
| `scgnn/extraction/slither_cfg.py` | needs-Studio | CFG via Slither; pure assembler `cfg_from_nodes` is exercised indirectly. |
| `scgnn/extraction/features.py` | mixed | `FeatureConfig`/`FeatureEncoder.encode_array` **[verified]**; `CodeBERTEmbedder`/`to_data` **[needs-Studio]**. |
| `scgnn/extraction/extract.py` | needs-Studio | Orchestrates AST+CFG for one contract. |
| `scgnn/models/__init__.py` | needs-Studio | Model exports (imports torch). |
| `scgnn/models/encoders.py` | needs-Studio | Swappable GCN/SAGE/GAT encoder; GAT keeps attention. |
| `scgnn/models/dual_gnn.py` | needs-Studio | `DualGNN` (HF `PyTorchModelHubMixin`) + `build_model`. |
| `scgnn/explain/__init__.py` | verified | Re-exports localisation helpers. |
| `scgnn/explain/localise.py` | verified | `rank_unique`, `nodes_to_lines` (logs unmapped). |
| `scgnn/explain/attention.py` | needs-Studio | GAT first-layer attention → lines (fallback signal). |
| `scgnn/explain/explainer.py` | needs-Studio | Per-class binary GNNExplainer, per branch, union — the riskiest module. |
| `scgnn/inference.py` | needs-Studio | `load_model` (pinned Hub download) + `analyze_source`. |

## Training — `training/` (not shipped)

| Path | Status | Purpose |
|------|--------|---------|
| `training/__init__.py` | config/doc | Namespace. |
| `training/labelling/__init__.py` | config/doc | Namespace. |
| `training/labelling/map_dasp.py` | verified | Tool finding → flaw code mapping (required test 1). |
| `training/labelling/run_tools.py` | needs-Studio | Parse SmartBugs output → per-flaw vote matrices. |
| `training/labelling/snorkel_label.py` | needs-Studio | Per-flaw Snorkel `LabelModel`; reports tool reliabilities. |
| `training/data/__init__.py` | config/doc | Namespace. |
| `training/data/firewall.py` | verified | Hash de-dup, frozen stratified split, firewall guard (required test 3). |
| `training/train/__init__.py` | config/doc | Namespace. |
| `training/train/dataset.py` | needs-Studio | `ContractPairDataset` over precomputed records. |
| `training/train/collate.py` | needs-Studio | Collate two aligned PyG batches + labels. |
| `training/train/train.py` | needs-Studio | Weighted-BCE loop, early stop on val macro-F1, checkpoint. |
| `training/evaluate/__init__.py` | config/doc | Namespace. |
| `training/evaluate/metrics.py` | verified | Per-flaw + macro P/R/F1. |
| `training/evaluate/localisation.py` | verified | Top-k localisation accuracy. |
| `training/evaluate/baselines.py` | needs-Studio | Static-tool baselines on Curated, same DASP map. |
| `training/evaluate/abc_experiment.py` | mixed | `summarise` + CV wiring importable; runtime needs the training stack. |

## Scripts — `scripts/` (entry points)

| Path | Status | Purpose |
|------|--------|---------|
| `scripts/publish_model.py` | verified | Publish a bundle to the Hub; prints the immutable commit SHA. |
| `scripts/label.py` | needs-Studio | SmartBugs results → Snorkel labels + reliabilities. |
| `scripts/train.py` | needs-Studio | Train one architecture from a YAML config. |
| `scripts/evaluate.py` | needs-Studio | Score a checkpoint on the frozen Curated test split. |
| `scripts/explain.py` | needs-Studio | Analyse one `.sol` end to end. |
| `scripts/build_release_bundle.py` | needs-Studio | Assemble a Hub bundle from a finished run. |
| `scripts/make_figures.py` | needs-Studio | Render per-model report figures (PNG+PDF) + caption sheet. |
| `scripts/make_comparison_figure.py` | needs-Studio | Three-model detection comparison chart + CSV. |
| `scripts/localise_eval.py` | needs-Studio | Top-k localisation: GNNExplainer or GAT attention, with line tolerance. |
| `run_pipeline.sh` | config/doc | End-to-end orchestration: install → release, stage by stage. |
| `setup_gpu01.sh` | config/doc | One-shot gpu-01 environment install (preserves working torch). |

## Configs, tests, project files

| Path | Status | Purpose |
|------|--------|---------|
| `configs/base.yaml` | config/doc | Base hyper-parameters; logged with every run. |
| `configs/gcn.yaml`,`sage.yaml`,`gat.yaml` | config/doc | Architecture overrides on top of base. |
| `tests/test_schema.py` | verified | Schema contract invariants. |
| `tests/test_dasp_map.py` | verified | Required test 1. |
| `tests/test_node_to_lines.py` | verified | Required test 2. |
| `tests/test_firewall.py` | verified | Required test 3. |
| `tests/test_features.py` | verified | Feature assembly + unknown-type handling. |
| `tests/test_metrics.py` | verified | Metrics incl. the zero-support convention. |
| `pyproject.toml` | config/doc | Core inference deps vs `[training]`/`[dev]`; excludes `training*` from the wheel. |
| `README.md` | config/doc | Overview, install, workflow. |
| `LICENSE` | config/doc | Apache-2.0. |
| `.gitignore` | config/doc | Standard Python + data/runs ignores. |

## Verification run

`PYTHONPATH=. python -m pytest -q` → **81 passed** (after the four-tool labelling,
union labelling, coverage-aware abstention, threshold-tuning, tolerance and
attention-localisation additions). All other modules byte-compile cleanly
(`python -m compileall scgnn training scripts tests`).

## First things to run on the Studio

1. Confirm `torch.__version__` + `torch.version.cuda` so the exact-pin lockfile can be generated.
2. `extract_contract` on one real `.sol` (solc + Slither path).
3. One `DualGNN` forward pass on a tiny batch.
4. `explain_lines` on a single contract — the dual-graph branch handling is the part most likely to need iteration.
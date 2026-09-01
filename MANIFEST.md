# scgnn-model — file manifest

Status legend:

- **[verified]** — covered by a passing unit test or run directly.
- **[needs-runtime]** — written to the real library APIs and byte-compiles
  cleanly, but needs a component unavailable in a bare environment
  (Slither/`solc`, SmartBugs, CodeBERT weights, torch/PyG, or a GPU). Smoke-test
  these on a configured machine first.
- **[config/doc]** — not code.

The shipped/serving boundary: everything under `scgnn/` is published to
`scgnn-api`; everything under `training/` is excluded from the wheel (see
`pyproject.toml`).

## Package — `scgnn/` (shipped to the back end)

| Path | Status | Purpose |
|------|--------|---------|
| `scgnn/__init__.py` | verified | Package exports; torch-free; `__version__`. |
| `scgnn/schema.py` | verified | Single source of truth for the five flaw codes, their order, and the `/analyze` wire contract. |
| `scgnn/extraction/graph_types.py` | verified | `RawGraph` + line mapping; the node→lines carrier. |
| `scgnn/extraction/slither_ast.py` | mixed | Compact-JSON AST parse **[verified]**; `extract_ast` (solc) **[needs-runtime]**. |
| `scgnn/extraction/slither_cfg.py` | needs-runtime | CFG via Slither, with data-flow (def-use) edges. |
| `scgnn/extraction/features.py` | mixed | `FeatureConfig`/`FeatureEncoder.encode_array` **[verified]**; `CodeBERTEmbedder` **[needs-runtime]**. |
| `scgnn/extraction/extract.py` | needs-runtime | Orchestrates AST + CFG(+data-flow) for one contract. |
| `scgnn/models/encoders.py` | needs-runtime | Swappable encoder: GCN, GraphSAGE, GATv2, and the two-stage hybrid (SAGE stages then a GATv2 attention layer). Static-attention GAT is retired but retained for the v1 record. |
| `scgnn/models/dual_gnn.py` | needs-runtime | `DualGNN` (HF `PyTorchModelHubMixin`) + `build_model`; empty-CFG-safe pooling. |
| `scgnn/explain/localise.py` | verified | `rank_unique`, `nodes_to_lines` (logs unmapped). |
| `scgnn/explain/attention.py` | needs-runtime | GATv2 attention → lines (secondary signal). |
| `scgnn/explain/explainer.py` | needs-runtime | Per-class GNNExplainer, per branch, merged-max — `explain_lines`. |
| `scgnn/inference.py` | needs-runtime | `load_model` (pinned Hub download) + `analyze_source`. |

## Training & evaluation — `training/` (not shipped)

| Path | Status | Purpose |
|------|--------|---------|
| `training/labelling/map_dasp.py` | verified | Tool finding → flaw code mapping. |
| `training/labelling/run_tools.py` | needs-runtime | Parse SmartBugs output → per-flaw vote matrices; `collect_votes`, `build_label_matrices`. |
| `training/labelling/snorkel_label.py` | needs-runtime | Per-flaw Snorkel `LabelModel`; computes corroboration reliabilities (`reliabilities.json`); documents the κ=1 and low-corroboration bypasses (Paper A). |
| `training/data/firewall.py` | verified | Content-hash de-dup, frozen stratified split, firewall guard. |
| `training/data/build.py` | needs-runtime | Dataset build; per-item (split, cid) keys (fixes the cross-provenance id collision). |
| `training/data/testsets.py` | verified | Frozen manifest reader; firewall-hash helpers. |
| `training/train/train.py` | needs-runtime | Weighted-BCE loop, early stop on val macro-F1, checkpoint. |
| `training/evaluate/metrics.py` | verified | Per-flaw + macro P/R/F1, micro, subset accuracy, bootstrap CIs, ensembling, the Durieux matrix, and optional class masking for single-class external baselines. |
| `training/evaluate/localisation.py` | verified | Top-k localisation accuracy. |
| `training/baselines/votes.py` | verified | Predict-the-union baseline over the four tools' votes only (Paper A instrument). |
| `training/baselines/trivial.py` | verified | Majority / stratified / all-positive floors. |
| `training/baselines/sequence.py` | needs-runtime | CodeBERT flat-text baseline; sliding-window / truncation, truncation-rate metadata. |
| `training/baselines/peculiar.py` | verified | File-based adapter for the external Peculiar detector (reentrancy, expert set only, masked scoring). |

## Scripts — `scripts/` (entry points)

| Path | Status | Purpose |
|------|--------|---------|
| `scripts/download_data.py` | needs-runtime | Fetch SmartBugs Curated + Wild. |
| `scripts/select_wild_subset.py` | needs-runtime | Class-balanced Wild subset. |
| `scripts/label.py` | needs-runtime | SmartBugs results → union labels + reliabilities. |
| `scripts/label_orchestrator.py` | needs-runtime | Resumable four-tool labelling ledger. |
| `scripts/freeze_testsets.py` | needs-runtime | Freeze Test A (Wild) and Test B (Curated) manifests + firewall. |
| `scripts/build_dataset.py` | needs-runtime | Build the with- and without-data-flow arms. |
| `scripts/train_v2.py` | needs-runtime | The ten-configuration training + evaluation matrix (four encoders × ±data-flow + two ensembles); bootstrap CIs. |
| `scripts/train_baselines.py` | needs-runtime | The learned-baseline suite (votes, trivial, sequence, peculiar); same results schema. |
| `scripts/durieux_baseline.py` | needs-runtime | Four tools on Test B → the tool-vs-model matrix (T5 / F4.11). |
| `scripts/localise_eval.py` | needs-runtime | GNNExplainer top-k localisation over all single models, line tolerance. |
| `scripts/reeval_testb.py` | needs-runtime | Inference-only Test B re-evaluation after a record repair. |
| `scripts/repair_testb_records.py` | needs-runtime | Surgical repair of the six cross-provenance id-collision records. |
| `scripts/fetch_sources.py` | needs-runtime | Hash-verified re-fetch of contract source from public SmartBugs corpora. |
| `scripts/rerun_testb_tools.py` | needs-runtime | Re-run the four tools on recovered Test B source. |
| `scripts/make_figures.py` | needs-runtime | Report figures (PNG+PDF) + caption sheet. |
| `scripts/make_comparison_figure.py` | needs-runtime | Model-comparison chart + CSV. |
| `scripts/make_baseline_figures.py` | needs-runtime | Baseline benchmarking figures, read from `results.json` (no hardcoded values). |
| `scripts/build_release_bundle.py` | needs-runtime | Assemble a Hub bundle from a finished run. |
| `scripts/publish_model.py` | verified | Publish a bundle to the Hub; prints the immutable commit SHA. |
| `run_pipeline.sh` | config/doc | End-to-end v2 orchestration, stage by stage. |
| `setup_gpu01.sh` | config/doc | One-shot environment install (preserves a working torch build). |

## Configs, tests, project files

| Path | Status | Purpose |
|------|--------|---------|
| `configs/base.yaml` | config/doc | Base hyper-parameters; logged with every run. |
| `configs/gcn.yaml`, `sage.yaml`, `gatv2.yaml`, `hybrid.yaml` | config/doc | Encoder overrides on top of base. |
| `configs/gat.yaml` | config/doc | Retired static-attention GAT (v1 record only). |
| `configs/baseline_votes.yaml`, `baseline_trivial.yaml`, `baseline_sequence.yaml` | config/doc | Baseline overrides. |
| `tests/` | verified | Schema, DASP map, node-to-lines, firewall, features, metrics (incl. masked metrics), and baseline shape/firewall/votes/sequence/truncation tests. |
| `docs/BASELINES.md` | config/doc | What each learned baseline isolates and does not license. |
| `pyproject.toml` | config/doc | Inference deps vs `[training]`/`[dev]`; excludes `training*` from the wheel. |
| `README.md`, `LICENSE`, `.gitignore` | config/doc | Overview, licence, ignores. |

## Verification

`PYTHONPATH=. python -m pytest -q tests/` passes the full suite (schema, labelling,
firewall, metrics including masked metrics, and the learned-baseline tests). All
modules byte-compile cleanly (`python -m compileall scgnn training scripts tests`).
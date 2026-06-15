# scgnn-model

A dual-graph **Graph Neural Network** pipeline for detecting and localising
security flaws in Ethereum smart contracts. Each contract is turned into an
Abstract Syntax Tree (AST) and a Control-Flow Graph (CFG); two GNN encoders read
the graphs, and a multi-label head predicts five DASP flaw classes at once.
GNNExplainer then maps each prediction back to the responsible source lines.

This is **repository 1 of 3**. The FastAPI back end (`scgnn-api`) and the Vue.js
front end (`scgnn-web`) live in separate repositories. This repository owns
dataset preparation, weak-supervision labelling, the models, training, the
explanation component, evaluation, the figures, and the deployable release
bundle the back end consumes.

> **Reproducibility.** The default parameters in `configs/base.yaml` and in each
> script are the *settled recipe* that produced the reported results. Running the
> commands below with no extra flags reproduces them. The exact package versions
> are pinned in `requirements-lock.txt` (generated on the target machine).

---

## 1. What this produces

Two artefacts the other repositories consume:

- the importable **`scgnn`** package (inference-time code only), and
- a trained **release bundle** — `model.safetensors` + `config.json` +
  `feature_config.json` + `pca.joblib` — published as a versioned, pinned asset.

The five flaw classes (fixed order, DASP-aligned):

| Code label | Report name | DASP |
|------------|-------------|------|
| `reentrancy` | Reentrancy | 1 |
| `access_control` | Access Control | 2 |
| `arithmetic` | Integer Overflow/Underflow | 3 |
| `unchecked_calls` | Unchecked Low-Level Calls | 4 |
| `dos` | Denial of Service (DoS) | 5 |

Every analysis returns the schema defined once in `scgnn/schema.py`:

```json
{
  "source": "<contract text>",
  "flaws": [
    { "type": "reentrancy", "confidence": 0.91, "lines": [42, 47, 53] }
  ]
}
```

### Shipped vs. not shipped

- **`scgnn/`** — the installable package (graph extraction, feature encoding,
  the model, `analyze_source`, the schema). The back end installs exactly this.
- **`training/`** — labelling, training loops, evaluation. Part of this repo but
  **excluded from the built wheel**, so the back end cannot import it by accident.

---

## 2. Quick start (run the whole pipeline)

A single orchestration script runs every stage in order with the settled
defaults. Each stage depends on the previous one, so you can resume at any stage.

```bash
# the entire pipeline, install through release
bash run_pipeline.sh all

# or one stage at a time (resume-friendly)
bash run_pipeline.sh setup
bash run_pipeline.sh data select tools label build
bash run_pipeline.sh train evaluate localise figures release
```

Long stages (`tools`, `build`, `train`) should run under `tmux` so an SSH drop
doesn't kill them. Stage list and overridable paths (`REPO`, `SMARTBUGS`,
`SELECTED`, …) are documented at the top of `run_pipeline.sh`.

---

## 3. Requirements

- **Python 3.10+**, and a **CUDA GPU** for training/embedding (CPU works but is slow).
- **`torch` / `torch-geometric`** built for your CUDA image. The pin depends on
  the machine; `setup_gpu01.sh` installs everything *except* torch so it never
  disturbs a known-good GPU build, then freezes `requirements-lock.txt`.
- **Slither + `solc`** (via `solc-select`) — used on the host for graph
  extraction (every build and every inference call).
- **CodeBERT** weights (pulled by `transformers` on first use) — node embeddings.
- **SmartBugs** (Docker) — orchestrates the four labelling tools. Installed
  separately, per its own instructions, at `$SMARTBUGS`. **Only the `tools`
  stage needs it**; nothing else depends on it.

### Install

One-shot environment setup (preserves a working GPU torch build):

```bash
bash setup_gpu01.sh        # = run_pipeline.sh setup
```

Inference-only install (what `scgnn-api` pins — no training extras):

```bash
pip install "scgnn @ git+https://github.com/<you>/scgnn-model@v1.0.0"
```

---

## 4. The pipeline, stage by stage

Each stage maps to a `run_pipeline.sh` stage and to the script(s) it calls. Run
directly if you want to vary parameters.

### data — download datasets
SmartBugs **Curated** (~142 contracts, expert labels + gold lines → the frozen
test set) and **Wild** (~47k contracts → weak-label training pool).
```bash
python scripts/download_data.py
```

### select — class-balanced Wild subset
We never label all 47k. We pick a subset *dense in positives for every class*
(the earlier model's defect was empty arithmetic/dos classes, not size).
```bash
python scripts/select_wild_subset.py --wild-dir data/raw/wild --out ~/wild_subset
```

### tools — four-tool labelling (needs SmartBugs)
Slither, Mythril, Securify and Osiris run over the subset. Osiris supplies the
arithmetic votes the other three cannot produce.
```bash
cd $SMARTBUGS && PYTHONPATH=. python -m sb -t slither mythril securify osiris \
  -f '~/wild_subset/*.sol' --processes 6 --mem-limit 4g --timeout 300 \
  --json --continue-on-errors
python scripts/validate_labels.py --results $SMARTBUGS/results   # audit raw votes
```

### label — weak labels
Parses the tool results into per-flaw votes and combines them. The default is
**union** labelling (positive if any covering tool fires). A Snorkel `LabelModel`
pass is also produced for the methodology comparison; on these low-overlap
heterogeneous detectors the LabelModel collapses to the agreement core, which is
why union is the default. See `docs/multitool_labelling_runbook.md`.
```bash
python scripts/label.py --results $SMARTBUGS/results --out data/processed
python scripts/label.py --results $SMARTBUGS/results --out data/processed_snorkel --method snorkel
```

### build — dataset
Compiles each contract (`solc`), extracts AST + CFG (Slither), embeds nodes
(CodeBERT → PCA-64, **fit on train only**), and writes the frozen stratified
test split plus a content-hash firewall against the Wild pool.
```bash
python scripts/build_dataset.py --wild-dir data/raw/wild \
  --wild-labels data/processed/labels.parquet --curated-dir data/raw/curated \
  --out data/processed --device cuda
```

### train — three architectures
GCN, GraphSAGE, GAT, identical recipe (epochs/patience/batch from
`configs/base.yaml`), so any difference is attributable to the architecture.
```bash
for m in gcn sage gat; do
  python scripts/train.py --config configs/$m.yaml \
    --train-index data/processed/train_index.json \
    --val-index data/processed/val_index.json --out runs/${m}_multitool
done
```

### evaluate — Curated test
Scores each model on the frozen expert-labelled test split at the fixed decision
threshold (`0.5`, from config). `evaluate.py` also supports `--tune-threshold`
(selected on validation, applied to test) for the threshold-policy comparison.
```bash
for m in gcn sage gat; do
  python scripts/evaluate.py --config configs/$m.yaml \
    --checkpoint runs/${m}_multitool/best_model.pt \
    --test-index data/processed/test_index.json --out runs/${m}_multitool/eval_05.json
done
```

### localise — explanation
GNNExplainer on the **selected** model (GCN), at exact and ±1/±2 line tolerance;
plus GAT first-layer **attention** as the proposal's secondary signal.
```bash
python scripts/localise_eval.py --merge max --tolerance 0 \
  --checkpoint runs/gcn_multitool/best_model.pt --config configs/gcn.yaml
python scripts/localise_eval.py --method attention --merge max --tolerance 2 \
  --checkpoint runs/gat_multitool/best_model.pt --config configs/gat.yaml
```

### figures — report figures
Per-model figure sets plus the three-model comparison (the model-selection
chart). PNG (Word) + PDF (LaTeX), with a caption sheet.
```bash
python scripts/make_figures.py --processed data/processed \
  --run-dir runs/gcn_multitool --out reports/figures/gcn
python scripts/make_comparison_figure.py \
  --eval gcn=runs/gcn_multitool/eval_05.json \
  --eval sage=runs/sage_multitool_long/eval_05.json \
  --eval gat=runs/gat_multitool/eval_05.json --out reports/figures/comparison
```

### release — deployable bundle
Packages the selected model's weights (as `.safetensors`), architecture config,
feature config and the fitted PCA into one folder for `scgnn-api`.
```bash
python scripts/build_release_bundle.py \
  --checkpoint runs/gcn_multitool/best_model.pt --config runs/gcn_multitool/config.json \
  --feature-config data/processed/feature_config.json --pca data/processed/pca.joblib \
  --out release/gcn_v1
python scripts/publish_model.py --bundle release/gcn_v1 --repo-id <id> --version v1
```

---

## 5. Selected model & results

The three-architecture comparison on the frozen Curated test split selected
**GCN** (highest macro-F1, the only architecture with non-zero F1 on every
supported class; GAT converged prematurely and trailed). GCN is therefore the
model carried into localisation and the release bundle. See
`reports/figures/comparison/` for the comparison chart and CSV.

---

## 6. Deployment (handoff to `scgnn-api`)

The back end imports two functions from `scgnn/inference.py`:

- `load_model(repo_id, revision, device)` — once at start-up; downloads the
  pinned bundle and builds a ready-to-serve model.
- `analyze_source(loaded, src)` — per request; returns the schema dict above.

`analyze_source` runs the **full extraction pipeline** (`solc` + Slither +
CodeBERT + PCA + the GNN) on the incoming contract, so the API host needs the
same extraction stack as this repo — not just torch. Inference latency is
dominated by extraction, not the GNN forward pass.

---

## 7. Tests & licence

```bash
PYTHONPATH=. python -m pytest -q tests/
```

Apache-2.0. See [`LICENSE`](./LICENSE).
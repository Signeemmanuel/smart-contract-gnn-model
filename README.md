# scgnn-model

A dual-graph **Graph Neural Network** pipeline for detecting and localising
security flaws in Ethereum smart contracts. Each contract is turned into an
Abstract Syntax Tree (AST) and a Control-Flow Graph (CFG) enriched with
data-flow edges; two GNN encoders read the graphs, and a multi-label head
predicts five DASP flaw classes at once. GNNExplainer then maps each prediction
back to the responsible source lines.

This repository owns dataset preparation, weak-supervision labelling, the models,
the ten-configuration training and evaluation matrix, the learned-baseline suite,
the explanation component, evaluation, and the figures. A FastAPI back end
(`scgnn-api`) and a Vue.js front end (`scgnn-web`) consume the deployable release
bundle from separate repositories.

> **Reproducibility.** The default parameters in `configs/base.yaml` and in each
> script are the settled recipe that produced the reported results.
> `run_pipeline.sh` reproduces every stage in order. Exact package versions are
> pinned in `requirements-lock.txt` (generated on the target machine).

---

## Companion papers

Two papers are built on this repository. Both draw their numbers from the
artefacts produced here; neither restates the other.

- **Paper A — "Measuring the Ceiling on Tool-Derived Supervision."** What the
  union-of-tools labels are worth against expert ground truth. Introduces the
  coverage and corroboration diagnostics computed from the vote matrix, and the
  votes-only baseline that measures the labelling ceiling (macro-F1 0.391 on the
  expert test set, against a four-tool union oracle of 0.387).
- **Paper B — "A Controlled Study of Graph Representations and Architectures."**
  A ten-configuration comparison (four encoders, with and without data-flow
  edges, plus two ensembles) under one training protocol, with a paired-bootstrap
  analysis of the data-flow effect.

Cite the companion papers rather than this README for the experimental findings.

## Data availability

The labelled corpus, frozen test manifests, per-contract predictions, vote
matrices, learned tool reliabilities (`reliabilities.json`), and the analysis
code are archived with a DOI. See each paper's Data Availability section for the
exact contents and the reproducibility scope (which results are recomputable from
the released predictions and which require re-running a stage).

- Code DOI: _add on release_.
- Data DOI: _add on release_.

---

## 1. What this produces

- The importable **`scgnn`** package (inference-time code only), consumed by the
  back end.
- A trained **release bundle** — `model.safetensors` + `config.json` +
  `feature_config.json` + `pca.joblib` — published as a versioned, pinned asset.
- The **evaluation matrix** (`runs/v2/results.json`) and the reported tables and
  figures under `artifacts/v2/`.

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

- **`scgnn/`** — the installable package (graph extraction, feature encoding, the
  model, `analyze_source`, the schema). The back end installs exactly this.
- **`training/`** — labelling, training, evaluation, and the learned baselines.
  Part of this repo but **excluded from the built wheel**, so the back end cannot
  import it by accident.

---

## 2. Quick start (run the whole pipeline)

`run_pipeline.sh` runs every stage in order with the settled defaults. Each stage
depends on the previous one, so you can resume at any stage.

```bash
# one stage at a time (resume-friendly)
bash run_pipeline.sh setup
bash run_pipeline.sh data select tools label freeze build
bash run_pipeline.sh train durieux localise figures release
```

Long stages (`tools`, `build`, `train`) should run under `tmux` so an SSH drop
doesn't kill them. The stage list and overridable paths are documented at the top
of `run_pipeline.sh`. The labelling stage is resumable: re-run it and it picks up
where it left off.

---

## 3. Requirements

- **Python 3.10+**, and a **CUDA GPU** for training/embedding (CPU works but is slow).
- **`torch` / `torch-geometric`** built for your CUDA image. The pin depends on
  the machine; `setup_gpu01.sh` installs everything except torch so it never
  disturbs a known-good GPU build, then freezes `requirements-lock.txt`.
- **Slither + `solc`** (via `solc-select`) — used on the host for graph
  extraction (every build and every inference call).
- **CodeBERT** weights (pulled by `transformers` on first use) — node embeddings.
- **SmartBugs** (Docker) — orchestrates the four labelling tools. Installed
  separately, per its own instructions, at `$SMARTBUGS`. Only the `tools` and
  `durieux` stages need it; nothing else depends on it.

### Install

```bash
bash setup_gpu01.sh        # one-shot environment setup; preserves a working GPU torch build
```

Inference-only install (what `scgnn-api` pins — no training extras):

```bash
pip install "scgnn @ git+https://github.com/Signeemmanuel/smart-contract-gnn-model@<tag>"
```

---

## 4. The pipeline, stage by stage

Each stage maps to a `run_pipeline.sh` stage and to the script(s) it calls.

### data — download datasets
SmartBugs **Curated** (~142 contracts, expert labels + gold lines → the frozen
expert test set) and **Wild** (~47k contracts → weak-label training pool).
```bash
python scripts/download_data.py
```

### select — class-balanced Wild subset
Pick a subset dense in positives for every class (arithmetic and dos are the
rare classes Osiris and cross-tool coverage must reach).
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
```

### label — weak labels
Parses the tool results into per-flaw votes and combines them. The default is
**union** labelling (positive if any covering tool fires). A Snorkel `LabelModel`
pass is also produced for the methodology comparison; on these low-overlap
heterogeneous detectors the label model collapses on weakly corroborated classes,
which is why union is the default and is the subject of Paper A. Corroboration
reliabilities are written to `reliabilities.json`.
```bash
python scripts/label.py --results $SMARTBUGS/results --out data/processed
python scripts/label.py --results $SMARTBUGS/results --out data/processed_snorkel --method snorkel
```

### freeze — frozen test manifests
Freeze the tool-labelled (Test A, Wild) and expert-labelled (Test B, Curated)
test sets once, with a content-hash firewall against the training pool.
```bash
python scripts/freeze_testsets.py
```

### build — dataset (two arms)
Compiles each contract (`solc`), extracts AST + CFG (Slither), embeds nodes
(CodeBERT → PCA-64, fit on train only), and writes both arms: with data-flow
edges (`processed_df`) and without (`processed_nodf`).
```bash
python scripts/build_dataset.py --wild-dir data/raw/wild \
  --wild-labels data/processed/labels.parquet --curated-dir data/raw/curated \
  --out data/processed_df --with-data-flow --device cuda
python scripts/build_dataset.py --wild-dir data/raw/wild \
  --wild-labels data/processed/labels.parquet --curated-dir data/raw/curated \
  --out data/processed_nodf --device cuda
```

### train — the ten-configuration matrix
Four encoders (GCN, GraphSAGE, GATv2, and a two-stage hybrid), each with and
without data-flow edges, plus two probability-averaging ensembles (GCN + GraphSAGE
+ GATv2, per arm). One identical recipe, so any difference is attributable to the
encoder or the data-flow representation. Thresholds are tuned on validation only;
metrics carry 2,000-resample bootstrap intervals.
```bash
python scripts/train_v2.py \
  --data-df data/processed_df --data-nodf data/processed_nodf \
  --configs configs --out runs/v2 --seeds 42
```
The winning single configuration is **`sage_df`** (GraphSAGE with data-flow
edges): expert-set (Test B) macro-F1 0.392, tool-labelled (Test A) 0.786.

### durieux — tool-vs-model matrix (needs SmartBugs)
Runs the four tools on the expert test set and places them beside the models on
identical contracts, the exact analogue of Durieux et al. (2020).
```bash
python scripts/durieux_baseline.py run    --testsets data/testsets --results data/sb_testb
python scripts/durieux_baseline.py matrix --testsets data/testsets --results data/sb_testb \
  --model-probs runs/v2/test_b_probs.json --out artifacts/v2
```

### localise — explanation
GNNExplainer over all single models on the expert test set, at exact and ±1/±2
line tolerance, five seeded passes.
```bash
python scripts/localise_eval.py --runs-dir runs/v2 --k 10 --repeats 5
```

### figures — report figures
```bash
python scripts/make_figures.py --processed data/processed_df --run-dir runs/v2 --out artifacts/v2
python scripts/make_comparison_figure.py --results runs/v2/results.json --out artifacts/v2
```

### release — deployable bundle
Packages the selected model's weights, config, feature config and fitted PCA for
`scgnn-api`.
```bash
python scripts/build_release_bundle.py --checkpoint runs/v2/sage_df/best_model.pt ...
python scripts/publish_model.py --bundle release/sage_df_v2 --repo-id <id> --version v2
```

---

## 5. Learned baselines (the benchmarking suite)

Beyond the four analysis tools, the study benchmarks the model against learned
baselines run on the same frozen test sets under the same protocol, all scored
through the one metrics path and emitting the same `results.json` schema.

- **votes** — a per-class logistic regression over the four tools' votes only
  (no code, no graph). Measures how much of the score is explained by the tools'
  behaviour. Expert-set macro-F1 0.391, essentially the union ceiling.
- **trivial** — majority, stratified, and all-positive floors.
- **sequence** — CodeBERT fine-tuned on flat contract text, the control for the
  structure-over-text claim. Expert-set macro-F1 0.362; sage_df beats it on
  every split.
- **peculiar** — a file-based adapter for the external Peculiar detector
  (reentrancy only, expert test set only), scored on the reentrancy column alone.

```bash
python scripts/train_baselines.py \
  --data-nodf data/processed_nodf \
  --baselines votes,trivial,sequence,peculiar \
  --out runs/baselines --seeds 42
python scripts/make_baseline_figures.py \
  --baselines runs/baselines/results.json \
  --sequence runs/baselines_seq/results.json \
  --model runs/v2/results.json --out artifacts/v2/figures
```

See `docs/BASELINES.md` for what each baseline isolates and what it does not
license us to claim.

---

## 6. Selected model & results

The ten-configuration comparison on the frozen expert (Curated) test set selected
**`sage_df`** (GraphSAGE with data-flow edges): the highest expert-set macro-F1
(0.392) among the single configurations, and the model carried into localisation
and the release bundle. The static-attention GAT of the earlier study is retired
in favour of dynamic-attention GATv2. Data-flow edges are not uniformly helpful:
under a paired bootstrap they measurably improve GraphSAGE (+0.059) and measurably
degrade GCN (−0.052), while the effect on GATv2 and the hybrid is within noise.
Full tables and figures are under `artifacts/v2/`; the numbers are reported in the
companion papers.

---

## 7. Deployment (handoff to `scgnn-api`)

The back end imports two functions from `scgnn/inference.py`:

- `load_model(repo_id, revision, device)` — once at start-up; downloads the pinned
  bundle and builds a ready-to-serve model.
- `analyze_source(loaded, src)` — per request; returns the schema dict above.

`analyze_source` runs the full extraction pipeline (`solc` + Slither + CodeBERT +
PCA + the GNN) on the incoming contract, so the API host needs the same extraction
stack as this repo. Inference latency is dominated by extraction, not the GNN
forward pass.

---

## 8. Tests & licence

```bash
PYTHONPATH=. python -m pytest -q tests/
```

Apache-2.0. See [`LICENSE`](LICENSE).
# scgnn-model

A graph-neural-network pipeline for detecting security flaws in Ethereum smart
contracts. This is repository 1 of 3 (`scgnn-model`); the FastAPI back end
(`scgnn-api`) and the Vue.js front end (`scgnn-web`) live in separate
repositories and are out of scope here.

This repository owns dataset preparation, weak-supervision labelling, the GNN
models, training, the explanation component, and evaluation. It produces the two
artefacts the other repositories consume:

- the importable **`scgnn`** package (inference-time code only), and
- the trained model **checkpoint** (published as a tagged release asset, never
  committed to git).

## Shipped vs. not shipped

- **`scgnn/`** is the installable package: graph extraction, feature encoding,
  the model class, `analyze_source`, and the result schema. The back end
  installs this and runs the identical code that produced the weights.
- **`training/`** holds labelling, training loops and evaluation. It is part of
  this repository but is **not** included in the built wheel, so the back end
  cannot import it even by accident.

## Installation

Inference only (what `scgnn-api` pins):

```bash
pip install "scgnn @ git+https://github.com/<you>/scgnn-model@v0.1.0"
```

Full training environment (this repository, editable):

```bash
pip install -e ".[training,dev]"
```

> **Dependency pinning.** `pyproject.toml` declares version *floors* for
> resolvability. The exact, reproducible pins for any reported run are captured
> in a committed lockfile generated on the target machine, because the correct
> `torch` / `torch-geometric` build depends on its CUDA image. SmartBugs (which
> orchestrates Slither, Mythril and Securify for labelling) is installed
> separately per its own instructions and is a build-time dependency only.

## The result contract

Every analysis returns the following shape (defined once in `scgnn/schema.py`):

```json
{
  "source": "<contract text>",
  "flaws": [
    { "type": "reentrancy", "confidence": 0.91, "lines": [42, 47, 53] }
  ]
}
```

The five flaw types are fixed and DASP-aligned. The stable code label (used in
the API, the label columns and the model's output order) and the report name
(the proposal's wording) for each are:

| Code label | Report name | DASP |
|------------|-------------|------|
| `reentrancy` | Reentrancy | 1 |
| `access_control` | Access Control | 2 |
| `arithmetic` | Integer Overflow/Underflow | 3 |
| `unchecked_calls` | Unchecked Low-Level Calls | 4 |
| `dos` | Denial of Service (DoS) | 5 |

## Status

Under construction. Build order: scaffold → labelling → extraction & features →
models & training → A/B/C labelling experiment → explanation & inference → docs.

## Licence

Apache-2.0. See [`LICENSE`](./LICENSE).

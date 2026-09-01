# Learned baselines

Four analysis tools and their union are not a sufficient comparator set for a
graph neural network: without a learned baseline the headline claim reduces to
"a GNN beats rule-based static analysers," which is not a contribution. This
suite adds the missing learned comparators and scores every one through the
single tested metrics path (`training/evaluate/metrics.py`), emitting the same
`results.json` schema as `scripts/train_v2.py`, so the comparison figures and
tables consume it with no special casing.

The design principle: hold everything constant except the one thing being
tested, so each baseline isolates a specific question. All are scored on the
same frozen test sets as the model — Test A (tool-labelled Wild) and Test B
(expert Curated).

## Running the suite

```bash
python scripts/train_baselines.py \
    --data-nodf data/processed_nodf \
    --baselines votes,trivial,sequence,peculiar \
    --out runs/baselines --seeds 42
```

Multiple seeds for a paper (mean and standard deviation of macro-F1 per split):

```bash
python scripts/train_baselines.py \
    --data-nodf data/processed_nodf \
    --baselines votes,trivial \
    --out runs/baselines --seeds 42,43,44
```

CPU smoke of the whole path, no GPU, before renting a machine:

```bash
python scripts/train_baselines.py \
    --data-nodf data/processed_nodf --baselines sequence,votes,trivial \
    --out runs/baselines_smoke --seeds 42 \
    --seq-limit 50 --seq-epochs 1 --device cpu
```

Each baseline is independently skippable: a failure in one is recorded in
`skipped.json` and the others still run.

## What each baseline isolates, and what it does NOT license

### votes (`training/baselines/votes.py`)
The baseline a sharp reviewer demands first. Its only input is the four tools'
per-class votes for a contract (a 20-dimensional vector, four tools times five
classes, each in {-1 abstain, 0 negative, 1 positive}); no code, no graph, no
embedding. A per-class logistic regression predicts the labels. Whatever it
scores is the part of the task explained by the tools' behaviour alone; the gap
between it and the GNN is the value the structural representation adds beyond
re-learning the tools.

Abstention is kept as a distinct feature value (-1), never collapsed into a
negative (0): "the tool did not run or does not cover this flaw" is different
information from "the tool ran and found nothing."

Does NOT license a claim that the GNN "understands code" if the gap is small.
It licenses only "the GNN adds X macro-F1 over predicting from tool votes." On
tool-labelled data (Test A) the training target IS the union of these votes, so
this baseline scores near-perfectly there by construction; that measures how
learnable the union rule is, not leakage. The informative number is Test B,
where the votes meet independent expert truth and the union's own ceiling
applies. On the expert set this baseline reaches macro-F1 0.391, essentially the
four-tool union oracle (0.387).

### trivial (`training/baselines/trivial.py`)
The floor. Majority-per-label (all-negative on this imbalanced data, so
macro-F1 0), stratified-random at the training positive rate, and all-positive
(recall 1, precision the base rate). Every learned score must clear these to
mean anything. Does NOT license anything on its own; these exist so the learned
numbers are not quoted in a vacuum.

### sequence (`training/baselines/sequence.py`)
The control for the central structure-over-text claim. `microsoft/codebert-base`
fine-tuned on raw contract text under the SAME protocol as the GNN (same split,
union labels, class-weighted BCE, early stopping on validation macro-F1, seed
42, per-class thresholds tuned on validation only). The only variable that
changes is flat text versus dual AST + CFG(+data-flow) graphs.

The 512-token limit is handled explicitly, not silently. `sliding` (reported)
tiles the contract into overlapping windows and max-pools per-class logits;
`truncate` keeps only the first 512 tokens. The truncation rate (the fraction of
contracts exceeding 512 tokens) is recorded in `run_info.json` and belongs in
the results, because on this corpus almost all contracts exceed the limit. On
the expert set this baseline reaches macro-F1 0.362, below the graph model on
every split.

Does NOT license a "graphs beat text" claim from a single seed. Report multiple
seeds, and state the truncation rate so the reader knows the text model's
handicap.

### peculiar (`training/baselines/peculiar.py`)
An external learned detector on our contracts. Peculiar (Wu et al., ISSRE 2021)
is a single-class reentrancy model; the adapter scores the reentrancy column
alone, with the other four columns MASKED (excluded from averaging, not counted
as zeros — see `test_masked_metrics.py`). It runs in its OWN environment and
hands this repo a predictions CSV; the two dependency worlds never meet.

Two deliberate limits: expert test set (Test B) only, because Peculiar trained
on SmartBugs Wild and our Test A is drawn from Wild, so a Test A comparison would
be biased in its favour and cannot be firewalled; and reentrancy only, because
that is the class it predicts. Residual risk to disclose: some Curated contracts
also appear in Wild, so this is as close to like-for-like as an external
checkpoint allows, not a leakage-free comparison. If the predictions file is
absent, the row is skipped loudly; it never falls back to Peculiar's published
numbers.

## Firewall and thresholds (shared with train_v2)

Every baseline that trains (votes, sequence) asserts its training content hashes
are disjoint from the union of both test manifests via
`training.data.firewall.assert_firewall` before fitting. Thresholds are tuned on
validation only and applied frozen to both test sets. No baseline fits anything
on Test A or Test B. The external Peculiar adapter, being single-class, uses a
fixed, stated 0.5 threshold.

## Figures

`scripts/make_baseline_figures.py` reads every value from the `results.json`
files (nothing hardcoded) and emits the baseline benchmarking figures: the
baseline ladder on the expert test set, the structure-versus-text comparison
across splits, the token-length truncation pressure from `run_info`, and the
votes construct artifact (the union rule recoverable on tool labels but not on
expert truth).

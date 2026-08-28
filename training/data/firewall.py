"""Train/test firewall, Wild<->Curated de-duplication, and the frozen split.

Pure (numpy + scikit-learn); unit-tested (this is one of the three required
tests). Enforces in code what the proposal asks for in discipline:

* Wild and Curated are de-duplicated by content hash and confirmed disjoint, and
  the number of removed overlaps is reported.
* A single Curated test split is frozen up front, stratified so the rare flaws
  (arithmetic, DoS) are represented in the test set.
* Condition A trains only on the Curated *remainder*; the frozen test split is
  never trained on by any condition. ``assert_firewall`` fails loudly if a test
  hash ever appears in a training index.

The canonical content hash is COMMENT-STRIPPED and whitespace-collapsed: two
copies of a contract that differ only in comments or formatting hash equal, so
a comment-edited duplicate of a test contract can never reach train/val.
``scripts/label_orchestrator.py`` mirrors this function for pool dedup; keep
the two in lockstep.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_WS = re.compile(r"\s+")


def strip_comments(source: str) -> str:
    """Remove // and /* */ comments from Solidity source, respecting strings.

    A small character-level state machine rather than a regex, so ``//`` inside
    a string literal (``"https://example.com"``) is preserved and escaped
    quotes inside strings do not end the string early. Block comments are
    replaced by a single space so token boundaries survive
    (``a/*x*/b`` -> ``a b``, not ``ab``). Pure.
    """
    out: list[str] = []
    i, n = 0, len(source)
    in_line = in_block = False
    quote: str | None = None
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line:
            if c == "\n":
                in_line = False
                out.append(c)
            i += 1
            continue
        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                out.append(" ")
                i += 2
                continue
            i += 1
            continue
        if quote:
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def content_hash(source: str) -> str:
    """Comment-stripped, whitespace-normalised SHA-256 of contract source.

    Stripping comments and collapsing whitespace catches duplicates that differ
    only in formatting or annotation, which is common across scraped corpora,
    and closes the leak where a comment-edited copy of a frozen test contract
    would otherwise hash differently and slip into training.
    """
    normalised = _WS.sub(" ", strip_comments(source)).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def dedup_wild_against_curated(
    wild: dict[str, str], curated: dict[str, str]
) -> tuple[list[str], list[str], int]:
    """Remove Wild contracts whose content hash collides with any Curated one.

    ``wild``/``curated`` map contract id -> source. Returns
    ``(kept_wild_ids, removed_wild_ids, removed_count)``.
    """
    curated_hashes = {content_hash(s) for s in curated.values()}
    kept, removed = [], []
    for cid, src in wild.items():
        (removed if content_hash(src) in curated_hashes else kept).append(cid)
    return kept, removed, len(removed)


def stratified_multilabel_split(
    Y: np.ndarray, test_frac: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (train, test), guaranteeing rare-class presence in test.

    For every flaw with at least two positives, at least ``ceil(test_frac * pos)``
    positives are placed in the test set; the remainder is filled randomly to
    reach the target test size. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n, n_flaws = Y.shape
    target = max(1, int(round(test_frac * n)))
    test: set[int] = set()

    # Rarest flaws first, so scarce positives are allocated before the budget fills.
    order = np.argsort(Y.sum(axis=0))
    for j in order:
        pos = np.where(Y[:, j] == 1)[0]
        if pos.size < 2:
            continue
        need = max(1, int(np.ceil(test_frac * pos.size)))
        avail = [p for p in rng.permutation(pos) if p not in test]
        for p in avail[:need]:
            test.add(int(p))

    remaining = [i for i in range(n) if i not in test]
    rng.shuffle(remaining)
    while len(test) < target and remaining:
        test.add(remaining.pop())

    test_idx = np.array(sorted(test), dtype=int)
    train_idx = np.array([i for i in range(n) if i not in test], dtype=int)
    return train_idx, test_idx


def curated_remainder_folds(Y_remainder: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Stratified k-fold over the Curated remainder for Condition A.

    Stratifies on a 1-D key (the rarest present flaw per contract, else -1),
    falling back to plain KFold if a class is too small to stratify.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    rarity = Y_remainder.sum(axis=0)
    key = np.full(Y_remainder.shape[0], -1, dtype=int)
    for i in range(Y_remainder.shape[0]):
        present = np.where(Y_remainder[i] == 1)[0]
        if present.size:
            key[i] = int(present[np.argmin(rarity[present])])
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros_like(key), key))
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.zeros_like(key)))


def assert_firewall(train_hashes: set[str], test_hashes: set[str]) -> None:
    """Raise if any frozen test contract leaks into the training index."""
    leak = train_hashes & test_hashes
    if leak:
        raise AssertionError(
            f"firewall breach: {len(leak)} test contract(s) present in training "
            f"index (e.g. {next(iter(leak))})"
        )
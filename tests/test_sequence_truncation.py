"""The sliding-window path and the head-truncation path produce the same output
shape, and both report the truncation rate. A fake tokenizer keeps this a pure
unit test with no model download."""
from __future__ import annotations

import numpy as np
import pytest

from scgnn.schema import FLAWS
from training.baselines.sequence import SequenceConfig, tokenise_split


class FakeTokenizer:
    """Deterministic whitespace tokenizer: one integer id per word, id 1 = CLS."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 return_attention_mask=True):
        ids = [1] + [2 + (hash(w) % 100) for w in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _rows(tmp_path, lengths):
    rows = []
    for i, n_words in enumerate(lengths):
        p = tmp_path / f"c{i}.sol"
        p.write_text("word " * n_words)
        row = {"path": str(p)}
        for j, f in enumerate(FLAWS):
            row[f] = 1 if (i + j) % 2 == 0 else 0
        rows.append(row)
    return rows


def test_truncate_and_sliding_same_label_shape(tmp_path):
    # one short contract (< 512) and one long (> 512 tokens)
    rows = _rows(tmp_path, [10, 900])
    tok = FakeTokenizer()

    tr = tokenise_split(rows, tok, SequenceConfig(mode="truncate", max_tokens=512))
    sl = tokenise_split(rows, tok, SequenceConfig(mode="sliding", max_tokens=512,
                                                  window_stride=384))
    # labels have identical shape regardless of windowing
    assert tr.labels.shape == sl.labels.shape == (2, len(FLAWS))
    # each contract's window tensor is (n_windows, 512)
    for ts in (tr, sl):
        for ids in ts.input_ids:
            assert ids.shape[1] == 512
    # truncate keeps exactly one window per contract; sliding makes >1 for the
    # long contract
    assert all(ids.shape[0] == 1 for ids in tr.input_ids)
    assert sl.input_ids[1].shape[0] > 1        # long contract windowed
    assert sl.input_ids[0].shape[0] == 1       # short contract single window


def test_truncation_rate_reported(tmp_path):
    rows = _rows(tmp_path, [10, 900, 1200])   # two of three exceed 512
    tok = FakeTokenizer()
    ts = tokenise_split(rows, tok, SequenceConfig(mode="sliding", max_tokens=512))
    assert ts.n_total == 3
    assert ts.n_truncated == 2
    assert ts.truncation_rate == pytest.approx(2 / 3)


def test_limit_caps_contracts(tmp_path):
    rows = _rows(tmp_path, [10, 20, 30, 40])
    tok = FakeTokenizer()
    ts = tokenise_split(rows, tok, SequenceConfig(mode="truncate", limit=2))
    assert ts.labels.shape == (2, len(FLAWS))
    assert len(ts.input_ids) == 2

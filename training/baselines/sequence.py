"""CodeBERT sequence baseline: the flat-text control for the structure claim.

The dissertation's central claim is that a structure-preserving representation
(dual AST + CFG graphs) beats flat contract text. That claim is never tested on
this data unless a strong text model is trained under the SAME protocol, so this
module fine-tunes ``microsoft/codebert-base`` as a five-label multi-label
classifier on raw contract source. Everything except the representation is held
identical to ``scripts/train_v2.py``: the same training split, the same union
labels, class-weighted BCE with the same ``pos_weight`` construction, early
stopping on validation macro-F1, seed 42, per-class thresholds tuned on
validation only.

The 512-token limit is handled explicitly rather than silently, because a large
fraction of these contracts exceed it and pretending otherwise would misstate
the baseline:

- ``truncate`` (head): keep the first 512 tokens. Fast; the honest weak form.
- ``sliding`` (window): tile the contract into overlapping 512-token windows,
  run each, and MAX-POOL the per-class logits across windows, so a signal
  anywhere in the contract can fire the label. The reported form.

The truncation rate (fraction of contracts whose token length exceeds 512, i.e.
the fraction where windowing actually matters) is recorded in the run metadata
and belongs in the results discussion.

CPU smoke mode: pass a small ``--limit`` with ``--epochs 1 --device cpu`` to run
the whole path (tokenisation, both windowing modes, training step, thresholding,
results schema) on a laptop before renting a GPU. The model still loads, so the
smoke run confirms shapes and collation, not final accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scgnn.schema import FLAWS, N_FLAWS

MODEL_NAME = "microsoft/codebert-base"
MAX_TOKENS = 512


@dataclass
class SequenceConfig:
    mode: str = "sliding"              # "sliding" | "truncate"
    max_tokens: int = MAX_TOKENS
    window_stride: int = 384           # overlap = max_tokens - stride (128)
    max_windows: int = 8               # cap very long contracts (cost bound)
    epochs: int = 10
    lr: float = 2e-5
    batch_size: int = 8
    grad_accum: int = 1
    patience: int = 3
    seed: int = 42
    device: str = "cuda"
    limit: int | None = None           # smoke: cap contracts per split
    model_name: str = MODEL_NAME


@dataclass
class TokenisedSplit:
    input_ids: list          # list of (n_windows, T) tensors, one per contract
    attention: list
    labels: np.ndarray       # (n, 5)
    n_truncated: int         # contracts with > max_tokens tokens
    n_total: int

    @property
    def truncation_rate(self) -> float:
        return self.n_truncated / self.n_total if self.n_total else 0.0


def _read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def tokenise_split(rows: list[dict], tokenizer, cfg: SequenceConfig) -> TokenisedSplit:
    """Tokenise each contract into one or more windows per ``cfg.mode``.

    ``rows`` are manifest-style dicts with ``path`` and the five label columns.
    Returns per-contract lists of window tensors so a contract can carry a
    variable number of windows; the collate step pads within a batch.
    """
    import torch

    ids_all, att_all, labels = [], [], []
    n_trunc = 0
    rows = rows[: cfg.limit] if cfg.limit else rows
    for r in rows:
        text = _read_source(r["path"])
        enc = tokenizer(text, add_special_tokens=True, truncation=False,
                        return_attention_mask=True)
        ids = enc["input_ids"]
        if len(ids) > cfg.max_tokens:
            n_trunc += 1
        if cfg.mode == "truncate":
            windows = [ids[: cfg.max_tokens]]
        elif cfg.mode == "sliding":
            step = cfg.window_stride
            windows = [ids[i:i + cfg.max_tokens]
                       for i in range(0, max(1, len(ids)), step)][: cfg.max_windows]
            windows = [w for w in windows if w] or [ids[: cfg.max_tokens]]
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        w_ids, w_att = [], []
        pad = tokenizer.pad_token_id or 0
        for w in windows:
            attn = [1] * len(w)
            if len(w) < cfg.max_tokens:
                attn = attn + [0] * (cfg.max_tokens - len(w))
                w = w + [pad] * (cfg.max_tokens - len(w))
            w_ids.append(w)
            w_att.append(attn)
        ids_all.append(torch.tensor(w_ids, dtype=torch.long))
        att_all.append(torch.tensor(w_att, dtype=torch.long))
        labels.append([int(r[f]) for f in FLAWS])
    return TokenisedSplit(ids_all, att_all, np.array(labels, dtype=int),
                          n_trunc, len(rows))


class SequenceModel:
    """CodeBERT encoder + a 5-logit multi-label head, with max-pooled windows."""

    def __init__(self, cfg: SequenceConfig):
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        self.cfg = cfg
        self.device = cfg.device
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hid = self.encoder.config.hidden_size
        self.head = nn.Linear(hid, N_FLAWS)
        self.encoder.to(self.device)
        self.head.to(self.device)

    def parameters(self):
        import itertools
        return itertools.chain(self.encoder.parameters(), self.head.parameters())

    def _contract_logits(self, ids, att):
        """Max-pool per-class logits across a contract's windows -> (5,)."""
        import torch

        ids = ids.to(self.device)
        att = att.to(self.device)
        out = self.encoder(input_ids=ids, attention_mask=att)
        cls = out.last_hidden_state[:, 0, :]         # (n_windows, hid)
        logits = self.head(cls)                       # (n_windows, 5)
        return logits.max(dim=0).values               # (5,) max-pool over windows

    def forward_batch(self, ids_list, att_list):
        import torch
        return torch.stack([self._contract_logits(i, a)
                            for i, a in zip(ids_list, att_list)], dim=0)

    def train_mode(self):
        self.encoder.train(); self.head.train()

    def eval_mode(self):
        self.encoder.eval(); self.head.eval()

    def state_dict(self):
        return {"encoder": self.encoder.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"])
        self.head.load_state_dict(sd["head"])


def pos_weight_from_labels(Y: np.ndarray):
    """Same class-weighted BCE construction as scripts/train_v2.py: neg/pos ratio
    per class, clipped so a very rare class does not explode the loss."""
    import torch

    Y = np.asarray(Y)
    pos = Y.sum(axis=0).clip(min=1)
    neg = (len(Y) - Y.sum(axis=0)).clip(min=1)
    return torch.tensor((neg / pos).clip(max=50.0), dtype=torch.float32)


def predict_probs(model: SequenceModel, split: TokenisedSplit,
                  batch_size: int) -> np.ndarray:
    """Sigmoid probabilities ``(n, 5)`` over a tokenised split. No thresholding."""
    import torch

    model.eval_mode()
    out = []
    with torch.no_grad():
        for s in range(0, len(split.input_ids), batch_size):
            ids = split.input_ids[s:s + batch_size]
            att = split.attention[s:s + batch_size]
            logits = model.forward_batch(ids, att)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, N_FLAWS))


def train_sequence(model: SequenceModel, train: TokenisedSplit,
                   val: TokenisedSplit, cfg: SequenceConfig,
                   out_dir: str) -> dict:
    """Fine-tune with early stopping on validation macro-F1; checkpoint each
    epoch so a reclaimed box resumes. Returns run info (best epoch, val macro-F1,
    truncation rates)."""
    import torch
    from training.evaluate.metrics import macro_f1, tune_thresholds, apply_thresholds

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pw = pos_weight_from_labels(train.labels).to(cfg.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    ckpt = out / "best_model.pt"
    resume = out / "last_epoch.pt"
    start_epoch, best_f1, best_epoch, bad = 0, -1.0, 0, 0
    if resume.exists():
        state = torch.load(resume, map_location=cfg.device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_f1, best_epoch = state["best_f1"], state["best_epoch"]
        print(f"resuming sequence baseline from epoch {start_epoch}")

    n = len(train.input_ids)
    for epoch in range(start_epoch, cfg.epochs):
        model.train_mode()
        order = np.random.permutation(n)
        opt.zero_grad()
        for step, b in enumerate(range(0, n, cfg.batch_size)):
            idx = order[b:b + cfg.batch_size]
            ids = [train.input_ids[i] for i in idx]
            att = [train.attention[i] for i in idx]
            y = torch.tensor(train.labels[idx], dtype=torch.float32, device=cfg.device)
            logits = model.forward_batch(ids, att)
            loss = loss_fn(logits, y) / cfg.grad_accum
            loss.backward()
            if (step + 1) % cfg.grad_accum == 0:
                opt.step(); opt.zero_grad()

        val_probs = predict_probs(model, val, cfg.batch_size)
        thr = tune_thresholds(val.labels, val_probs)
        vf1 = macro_f1(val.labels, apply_thresholds(val_probs, thr))
        print(f"  epoch {epoch + 1}/{cfg.epochs}  val macro-F1 {vf1:.4f}", flush=True)

        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "best_f1": max(best_f1, vf1),
                    "best_epoch": best_epoch if vf1 <= best_f1 else epoch},
                   resume)
        if vf1 > best_f1:
            best_f1, best_epoch, bad = vf1, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  early stopping at epoch {epoch + 1}")
                break

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch,
            "mode": cfg.mode,
            "train_truncation_rate": round(train.truncation_rate, 4),
            "val_truncation_rate": round(val.truncation_rate, 4),
            "checkpoint": str(ckpt)}

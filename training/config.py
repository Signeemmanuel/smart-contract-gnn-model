"""YAML experiment-config loading with ``extends`` inheritance.

The architecture configs (``gcn.yaml`` etc.) say ``extends: base.yaml`` so a run
differs from the baseline only by what it overrides. Plain ``yaml.safe_load``
ignores that key, which would silently fall back to defaults — this resolves it.
Pure; unit-tested.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a config, resolving a relative ``extends:`` parent (child wins)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        base.update(cfg)   # child keys override the base
        return base
    return cfg

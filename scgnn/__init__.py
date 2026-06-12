"""scgnn: graph-neural-network smart-contract flaw detection (inference package).

This is the shippable, importable package consumed by ``scgnn-api``. It contains
only inference-time code: graph extraction, feature encoding, the model class,
``analyze_source`` and the result schema. Training-only code (labelling, training
loops, evaluation) lives in the repository's ``training/`` area and is excluded
from the built wheel.

Importing this package is deliberately torch-free: only the schema (standard
library) is pulled in here, so a tool that just needs the flaw constants does
not pay for importing PyTorch. The determinism helper lives at
``scgnn.common.seeds`` and is imported explicitly where needed.
"""

from __future__ import annotations

from scgnn.schema import (
    FLAW_INDEX,
    FLAWS,
    N_FLAWS,
    AnalysisResult,
    FlawResult,
    FlawType,
    validate_flaw_code,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FlawType",
    "FlawResult",
    "AnalysisResult",
    "FLAWS",
    "FLAW_INDEX",
    "N_FLAWS",
    "validate_flaw_code",
]

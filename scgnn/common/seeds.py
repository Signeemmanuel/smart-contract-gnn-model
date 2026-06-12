"""Global determinism helper.

Call :func:`set_seed` at the top of every entry point (labelling, training,
evaluation, inference) so that a run is reproducible from its config alone.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch and CUDA, and make cuDNN deterministic.

    Note:
        We set ``cudnn.deterministic = True`` and disable ``cudnn.benchmark``,
        which is the safe baseline. We deliberately do *not* call
        ``torch.use_deterministic_algorithms(True)``: several scatter and
        pooling kernels used by PyTorch Geometric have no deterministic
        implementation and would raise at runtime. If stricter guarantees are
        needed later, enable it with ``warn_only=True`` and audit the warnings
        rather than turning it on blindly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

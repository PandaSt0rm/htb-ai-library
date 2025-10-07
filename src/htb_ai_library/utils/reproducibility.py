"""
Reproducibility helpers for synchronizing random number generators.
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Any

import numpy as np
import torch


def set_reproducibility(seed: int = 1337) -> None:
    """
    Configure reproducible behavior across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int, optional
        Non-negative integer applied to all managed random number generators.
        Defaults to 1337.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``seed`` is negative.

    Examples
    --------
    >>> set_reproducibility(1234)
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Align Python and NumPy RNGs.
    random.seed(seed)
    np.random.seed(seed)

    # Align Torch RNGs (CPU and, when available, CUDA).
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # Keep existing user preference for workspace size if provided.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Disable algorithm heuristics that trade determinism for performance.
        if hasattr(torch.backends.cuda, "matmul") and hasattr(
            torch.backends.cuda.matmul, "allow_tf32"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_det_alg: Any = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_det_alg):
        try:
            use_det_alg(True, warn_only=True)
        except TypeError:
            use_det_alg(True)
        except RuntimeError as err:
            warnings.warn(
                f"Deterministic algorithms requested but unavailable: {err}",
                RuntimeWarning,
            )

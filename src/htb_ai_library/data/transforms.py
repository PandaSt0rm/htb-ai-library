"""
Common data transformation helpers.
"""

from __future__ import annotations

import torch

__all__ = ["cifar_normalize"]


def cifar_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Apply CIFAR-10 normalization to a tensor batch.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with shape ``(N, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Normalized tensor using CIFAR-10 channel statistics.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype, device=x.device)[None, :, None, None]
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / std

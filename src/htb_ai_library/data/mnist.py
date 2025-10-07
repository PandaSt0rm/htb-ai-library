"""
MNIST loading utilities and normalization helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    *,
    normalize: bool = False,
    seed: Optional[int] = 1337,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST data loaders with optional normalization and deterministic shuffling.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples per batch, by default 128.
    data_dir : str, optional
        Directory to store/load MNIST data, by default ``"./data"``.
    normalize : bool, optional
        If ``True``, applies MNIST normalization with ``mean=0.1307`` and ``std=0.3081``.
        Otherwise returns tensors in ``[0, 1]`` pixel space.
    seed : Optional[int], optional
        Random seed for reproducible shuffling; pass ``None`` to disable, by default 1337.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and test data loaders.
    """
    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_steps)

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader


def mnist_denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize MNIST images from standardized space back to ``[0, 1]`` pixel values.

    Parameters
    ----------
    x : torch.Tensor
        Normalized MNIST tensor.

    Returns
    -------
    torch.Tensor
        Denormalized tensor clipped to ``[0, 1]``.
    """
    mean = 0.1307
    std = 0.3081
    denorm = x * std + mean
    return torch.clamp(denorm, 0.0, 1.0)

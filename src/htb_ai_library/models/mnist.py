"""
Lightweight MNIST classifiers for experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SimpleCNN", "MNISTClassifierWithDropout"]


class SimpleCNN(nn.Module):
    """
    Compact convolutional network tailored to MNIST-sized inputs.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes, by default 10.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for the provided batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(N, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Logit tensor shaped ``(N, num_classes)``.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTClassifierWithDropout(nn.Module):
    """
    Dropout-regularized MNIST convolutional classifier.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for the provided batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(N, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Logit tensor shaped ``(N, num_classes)``.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

"""
Lightweight MNIST classifiers for experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SimpleLeNet", "SimpleCNN", "MNISTClassifierWithDropout"]


class SimpleLeNet(nn.Module):
    """LeNet-style CNN tailored to MNIST-sized grayscale inputs.

    The architecture stacks two 5x5 convolutional layers with Tanh activations
    and 2x2 max pooling ahead of a pair of fully connected layers. Call
    ``forward`` for logits-based workflows or ``forward_log_probs`` when
    log-softmax values are required.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.act = nn.Tanh()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden activations before the final linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Hidden representation shaped ``(batch, 500)``.
        """
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits prior to softmax normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Logits shaped ``(batch, 10)``.
        """
        features = self.forward_features(x)
        return self.fc2(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits for an input batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Logits shaped ``(batch, 10)``.
        """
        return self.forward_logits(x)

    def forward_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities via log-softmax for downstream uses.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities shaped ``(batch, 10)``.
        """
        logits = self.forward_logits(x)
        return F.log_softmax(logits, dim=1)



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

"""
Standardized model training and evaluation loops.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

__all__ = ["train_one_epoch", "train_model", "evaluate_accuracy"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train a model for a single epoch and return the average loss.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        count += x.size(0)

    return total_loss / max(count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str | torch.device,
    epochs: int = 5,
    learning_rate: float = 0.001,
) -> nn.Module:
    """
    Train a model for multiple epochs with periodic evaluation.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        accuracy = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}: Avg Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2f}%")

    return model


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device | str) -> float:
    """
    Compute classification accuracy on a dataset.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / max(total, 1)

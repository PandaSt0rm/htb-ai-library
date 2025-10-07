"""
Visualization routines for adversarial attack analysis.
"""

from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .styles import (
    apply_htb_axes_style,
    HTB_GREEN,
    NODE_BLACK,
    HACKER_GREY,
    WHITE,
    AZURE,
    NUGGET_YELLOW,
    MALWARE_RED,
)

__all__ = [
    "visualize_attack",
    "plot_attack_effectiveness",
    "visualize_perturbation_analysis",
]


def visualize_attack(
    model: nn.Module,
    image: torch.Tensor,
    true_label: int,
    adv_image: torch.Tensor,
    title: str,
    num_classes: int = 10,
) -> None:
    """
    Visualize a clean image, adversarial counterpart, and probability profiles.
    """
    model.eval()
    device = next(model.parameters()).device
    image_dev, adv_image_dev = image.to(device), adv_image.to(device)

    with torch.no_grad():
        clean_probs = F.softmax(model(image_dev.unsqueeze(0)), dim=1).squeeze()
        adv_probs = F.softmax(model(adv_image_dev.unsqueeze(0)), dim=1).squeeze()
        clean_pred = clean_probs.argmax().item()
        adv_pred = adv_probs.argmax().item()

    perturbation = (adv_image_dev - image_dev).cpu()

    fig = plt.figure(figsize=(16, 10), facecolor=NODE_BLACK)
    gs = plt.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    apply_htb_axes_style(ax1)
    ax1.imshow(image.cpu().permute(1, 2, 0).clamp(0, 1))
    ax1.set_title(f"Original: {clean_pred}", color=HTB_GREEN, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    apply_htb_axes_style(ax2)
    ax2.imshow(adv_image.cpu().permute(1, 2, 0).clamp(0, 1))
    title_color = MALWARE_RED if adv_pred != true_label else HTB_GREEN
    ax2.set_title(f"Adversarial: {adv_pred}", color=title_color, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    apply_htb_axes_style(ax3)
    pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
    ax3.imshow(pert_vis.permute(1, 2, 0))
    ax3.set_title("Perturbation", color=NUGGET_YELLOW, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, :])
    apply_htb_axes_style(ax4)
    x = np.arange(num_classes)
    width = 0.4
    ax4.bar(x - width / 2, clean_probs.cpu().numpy()[:num_classes], width, color=AZURE, label="Clean")
    ax4.bar(x + width / 2, adv_probs.cpu().numpy()[:num_classes], width, color=MALWARE_RED, label="Adversarial")
    ax4.set_xlabel("Class", color=WHITE)
    ax4.set_ylabel("Probability", color=WHITE)
    ax4.legend()
    ax4.set_title("Class Probabilities", color=HTB_GREEN, fontweight="bold")

    fig.suptitle(title, color=HTB_GREEN, fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_attack_effectiveness(attack_results) -> None:
    """
    Plot attack effectiveness versus effort for a tabular results object.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=NODE_BLACK)
    apply_htb_axes_style(ax)
    ax.plot(attack_results["num_words"], attack_results["evasion_rate"], marker="o", color=AZURE)
    ax.fill_between(attack_results["num_words"], 0, attack_results["evasion_rate"], alpha=0.2, color=AZURE)
    ax.set_xlabel("Effort (e.g., Words Added, Epsilon)", fontsize=12, color=HTB_GREEN)
    ax.set_ylabel("Evasion Rate (%)", fontsize=12, color=HTB_GREEN)
    ax.set_title("Attack Effectiveness", fontsize=14, color=HTB_GREEN, fontweight="bold")
    plt.tight_layout()
    plt.show()


def visualize_perturbation_analysis(results) -> None:
    """
    Visualize perturbation characteristics collected from attack runs.
    """
    if not results:
        raise ValueError("No perturbation results provided for visualization.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=NODE_BLACK)
    fig.suptitle("Perturbation Analysis", color=HTB_GREEN, fontsize=16, fontweight="bold")

    l2_norms = [r["l2_norm"] for r in results]
    iterations = [r["iterations"] for r in results]

    apply_htb_axes_style(axes[0])
    axes[0].hist(l2_norms, bins=15, color=HTB_GREEN, alpha=0.7)
    axes[0].set_title("L2 Norm Distribution", color=HTB_GREEN)
    axes[0].set_xlabel("L2 Norm", color=WHITE)
    axes[0].set_ylabel("Frequency", color=WHITE)

    apply_htb_axes_style(axes[1])
    axes[1].hist(iterations, bins=range(1, max(iterations) + 2), color=AZURE, alpha=0.7)
    axes[1].set_title("Iterations Required", color=HTB_GREEN)
    axes[1].set_xlabel("Iterations", color=WHITE)

    apply_htb_axes_style(axes[2])

    def _to_numpy(tensor_like):
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.detach().cpu().numpy()
        return np.asarray(tensor_like)

    l0_norms = [np.count_nonzero(_to_numpy(r["perturbation"])) for r in results]
    axes[2].hist(l0_norms, bins=15, color=NUGGET_YELLOW, alpha=0.7)
    axes[2].set_title("Sparsity (L0 Norm)", color=HTB_GREEN)
    axes[2].set_xlabel("Number of Pixels Modified", color=WHITE)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

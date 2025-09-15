
"""
Functions for evaluating attack effectiveness and model robustness.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def evaluate_attack_effectiveness(
    model: nn.Module,
    clean_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute accuracy, success rate, confidence shift, and perturbation norms.
    """
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean_images)
        adv_logits = model(adversarial_images)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)

        clean_correct = (clean_pred == true_labels)
        adv_correct = (adv_pred == true_labels)

        # Attack success is a flip from a correct prediction
        flipped = clean_correct & ~adv_correct

        conf_clean = clean_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        conf_adv = adv_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)

        # Perturbation norms
        perturbation = adversarial_images - clean_images
        l2 = perturbation.view(clean_images.size(0), -1).norm(p=2, dim=1)
        linf = perturbation.view(clean_images.size(0), -1).abs().max(dim=1)[0]

        clean_correct_total = clean_correct.float().sum()
        flipped_total = flipped.float().sum()
        attack_success_rate = (
            flipped_total.item() / clean_correct_total.item()
            if clean_correct_total.item() > 0
            else 0.0
        )

        return {
            "clean_accuracy": clean_correct.float().mean().item(),
            "adversarial_accuracy": adv_correct.float().mean().item(),
            "attack_success_rate": attack_success_rate,
            "avg_clean_confidence": conf_clean.mean().item(),
            "avg_adv_confidence": conf_adv.mean().item(),
            "avg_confidence_drop": (conf_clean - conf_adv).mean().item(),
            "avg_l2_perturbation": l2.mean().item(),
            "avg_linf_perturbation": linf.mean().item(),
        }

def analyze_model_confidence(
    model: nn.Module, 
    loader: DataLoader, 
    device: str, 
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Analyze model confidence distribution on clean examples.
    """
    model.eval()
    confidences, correct_confidences, incorrect_confidences = [], [], []

    processed = 0
    with torch.no_grad():
        for data, target in loader:
            if num_samples and processed >= num_samples:
                break

            data, target = data.to(device), target.to(device)
            probs = F.softmax(model(data), dim=1)
            confidence, predicted = probs.max(1)

            conf_list = confidence.detach().cpu().tolist()
            pred_list = predicted.detach().cpu().tolist()
            target_list = target.detach().cpu().tolist()

            for conf, pred, true in zip(conf_list, pred_list, target_list):
                if num_samples and processed >= num_samples:
                    break
                confidences.append(conf)
                if pred == true:
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)
                processed += 1

    return {
        'mean_confidence': np.mean(confidences) if confidences else 0,
        'mean_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
        'mean_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
    }

def print_attack_summary(results: Dict[str, float]):
    """
    Prints a formatted summary of attack evaluation results.
    """
    print("\n--- Attack Evaluation Summary ---")
    print(f"Clean Accuracy:           {results['clean_accuracy']:.2%}")
    print(f"Adversarial Accuracy:     {results['adversarial_accuracy']:.2%}")
    print(f"Attack Success Rate:      {results['attack_success_rate']:.2%}")
    print("-" * 35)
    print(f"Avg. Clean Confidence:    {results['avg_clean_confidence']:.2%}")
    print(f"Avg. Adversarial Confidence: {results['avg_adv_confidence']:.2%}")
    print(f"Avg. Confidence Drop:     {results['avg_confidence_drop']:.2%}")
    print("-" * 35)
    print(f"Avg. L2 Perturbation:     {results['avg_l2_perturbation']:.4f}")
    print(f"Avg. L-inf Perturbation:  {results['avg_linf_perturbation']:.4f}")
    print("-" * 35)

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from htb_ai_library.metrics import evaluate_attack_effectiveness, analyze_model_confidence


class ConstantZeroModel(nn.Module):
    def forward(self, inputs):
        return torch.zeros(inputs.size(0), 2, device=inputs.device)


class ConstantLogitModel(nn.Module):
    def forward(self, inputs):
        logits = torch.tensor([[0.0, 1.0]], device=inputs.device)
        return logits.expand(inputs.size(0), -1)


class InputSensitiveModel(nn.Module):
    def forward(self, inputs):
        mean = inputs.mean(dim=(1, 2, 3))
        return torch.stack([-mean, mean], dim=1)


def test_evaluate_attack_effectiveness_handles_zero_clean_accuracy():
    model = ConstantZeroModel()
    clean_images = torch.randn(4, 1, 2, 2)
    adversarial_images = torch.randn(4, 1, 2, 2)
    true_labels = torch.ones(4, dtype=torch.long)

    results = evaluate_attack_effectiveness(model, clean_images, adversarial_images, true_labels)

    assert results["clean_accuracy"] == pytest.approx(0.0)
    assert results["attack_success_rate"] == pytest.approx(0.0)


def test_evaluate_attack_effectiveness_computes_success_rate():
    model = InputSensitiveModel()
    clean = torch.ones(3, 1, 2, 2)
    adv = clean.clone()
    labels = torch.ones(3, dtype=torch.long)

    results = evaluate_attack_effectiveness(model, clean, adv, labels)
    assert results["clean_accuracy"] == pytest.approx(1.0)
    assert results["attack_success_rate"] == pytest.approx(0.0)

    adv[0] = -adv[0]
    flipped_results = evaluate_attack_effectiveness(model, clean, adv, labels)
    assert flipped_results["attack_success_rate"] == pytest.approx(1 / 3)


def test_analyze_model_confidence_respects_sample_limit():
    model = ConstantLogitModel()
    inputs = torch.randn(5, 1, 2, 2)
    targets = torch.tensor([1, 0, 1, 1, 0])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    summary = analyze_model_confidence(model, loader, device="cpu", num_samples=3)

    expected_confidence = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[1].item()
    assert summary["mean_confidence"] == pytest.approx(expected_confidence, abs=1e-6)
    assert summary["mean_correct_confidence"] == pytest.approx(expected_confidence, abs=1e-6)
    assert summary["mean_incorrect_confidence"] == pytest.approx(expected_confidence, abs=1e-6)


def test_print_attack_summary_outputs(capsys):
    summary = {
        "clean_accuracy": 0.9,
        "adversarial_accuracy": 0.5,
        "attack_success_rate": 0.4,
        "avg_clean_confidence": 0.8,
        "avg_adv_confidence": 0.3,
        "avg_confidence_drop": 0.5,
        "avg_l2_perturbation": 1.23,
        "avg_linf_perturbation": 0.12,
    }

    from htb_ai_library.metrics import print_attack_summary

    print_attack_summary(summary)

    captured = capsys.readouterr().out
    assert "Attack Evaluation Summary" in captured
    assert "Clean Accuracy" in captured

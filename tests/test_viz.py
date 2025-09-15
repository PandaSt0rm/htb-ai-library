import pandas as pd
import pytest
import torch
import matplotlib.pyplot as plt

from htb_ai_library.viz import (
    plot_attack_effectiveness,
    visualize_attack,
    visualize_perturbation_analysis,
)


class IdentityModel(torch.nn.Module):
    def forward(self, inputs):
        batch = inputs.view(inputs.size(0), -1)
        logits = torch.stack(
            [batch.sum(dim=1), torch.zeros(batch.size(0), device=inputs.device)],
            dim=1,
        )
        return logits

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1))


def test_visualize_perturbation_analysis_accepts_tensor_inputs(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    results = [
        {
            "l2_norm": 0.5,
            "iterations": 3,
            "perturbation": torch.randn(2, requires_grad=True),
        }
    ]

    if torch.cuda.is_available():
        results.append(
            {
                "l2_norm": 0.6,
                "iterations": 5,
                "perturbation": torch.randn(2, device="cuda"),
            }
        )
    else:
        results.append(
            {
                "l2_norm": 0.6,
                "iterations": 5,
                "perturbation": torch.randn(2),
            }
        )

    results.append(
        {
            "l2_norm": 0.7,
            "iterations": 2,
            "perturbation": [0.0, 1.0],
        }
    )

    visualize_perturbation_analysis(results)


def test_visualize_attack_handles_images(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    model = IdentityModel()
    image = torch.ones(3, 4, 4)
    adv_image = image * 0.5

    visualize_attack(model, image, true_label=0, adv_image=adv_image, title="demo", num_classes=2)


def test_plot_attack_effectiveness(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    df = pd.DataFrame({"num_words": [1, 2, 3], "evasion_rate": [10, 50, 90]})

    plot_attack_effectiveness(df)


def test_visualize_perturbation_analysis_rejects_empty(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    with pytest.raises(ValueError):
        visualize_perturbation_analysis([])

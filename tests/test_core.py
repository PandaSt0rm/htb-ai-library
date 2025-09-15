import os
import random

import numpy as np
import pytest
import torch
from torch import nn

from htb_ai_library.core import load_model, save_model, set_reproducibility


@pytest.mark.parametrize("filename", ["model.pth", os.path.join("nested", "model.pth")])
def test_save_model_creates_missing_directories(tmp_path, monkeypatch, filename):
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(4, 2)

    save_model(model, filename)

    assert (tmp_path / filename).exists()


def test_load_model_restores_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = nn.Linear(4, 2)
    target = nn.Linear(4, 2)
    for param in source.parameters():
        torch.nn.init.normal_(param)

    save_model(source, "weights.pth")
    loaded = load_model(target, "weights.pth", device="cpu")

    for p_loaded, p_source in zip(loaded.parameters(), source.parameters()):
        assert torch.allclose(p_loaded, p_source)


def test_set_reproducibility_aligns_rngs():
    set_reproducibility(1234)
    values_a = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )

    set_reproducibility(1234)
    values_b = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )

    assert values_a == values_b

    # restore default seed for other tests
    set_reproducibility()


def test_save_and_load_generic_object(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {"weights": [1, 2, 3], "bias": 0.1}

    save_model(payload, "model.pkl")
    loaded = load_model("model.pkl")

    assert loaded == payload

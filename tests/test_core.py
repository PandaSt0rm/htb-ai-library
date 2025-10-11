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


def test_get_color_returns_valid_hex():
    from htb_ai_library.utils import get_color

    color = get_color("HTB_GREEN")
    assert color == "#9fef00"
    assert color.startswith("#")


def test_get_color_case_insensitive():
    from htb_ai_library.utils import get_color

    assert get_color("htb_green") == get_color("HTB_GREEN")
    assert get_color("HtB_gReEn") == get_color("HTB_GREEN")


def test_get_color_raises_on_invalid_name():
    from htb_ai_library.utils import get_color

    with pytest.raises(KeyError, match="Unknown color"):
        get_color("INVALID_COLOR")


def test_get_color_palette_returns_mapping():
    from htb_ai_library.utils import get_color_palette

    palette = get_color_palette()
    assert "HTB_GREEN" in palette
    assert "NODE_BLACK" in palette
    assert palette["HTB_GREEN"] == "#9fef00"


def test_get_color_palette_is_read_only():
    from htb_ai_library.utils import get_color_palette

    palette = get_color_palette()
    with pytest.raises(TypeError):
        palette["NEW_COLOR"] = "#000000"

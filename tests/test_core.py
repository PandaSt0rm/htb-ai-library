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


def test_set_reproducibility_raises_on_negative_seed():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        set_reproducibility(-1)


def test_set_reproducibility_handles_deterministic_algorithms_errors(monkeypatch):
    # Test TypeError fallback (warn_only parameter not supported)
    def mock_use_deterministic_algorithms_type_error(mode, warn_only=None):
        if warn_only is not None:
            raise TypeError("unexpected keyword argument")
        # Don't actually enable deterministic algorithms in test
        pass

    monkeypatch.setattr(torch, "use_deterministic_algorithms", mock_use_deterministic_algorithms_type_error)
    set_reproducibility(42)  # Should handle TypeError gracefully

    # Test RuntimeError with warning
    def mock_use_deterministic_algorithms_runtime_error(mode, warn_only=None):
        raise RuntimeError("Deterministic algorithms not available")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", mock_use_deterministic_algorithms_runtime_error)

    with pytest.warns(RuntimeWarning, match="Deterministic algorithms requested but unavailable"):
        set_reproducibility(42)


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


def test_load_model_raises_when_filepath_missing_for_module():
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="filepath must be provided"):
        load_model(model)


def test_load_model_raises_when_device_missing_for_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(4, 2)
    save_model(model, "weights.pth")

    with pytest.raises(ValueError, match="device must be provided"):
        load_model(model, "weights.pth")


def test_load_model_handles_pytorch_loading_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model = nn.Linear(4, 2)

    # Create a corrupted file
    with open("corrupted.pth", "w") as f:
        f.write("not a valid pytorch file")

    result = load_model(model, "corrupted.pth", device="cpu")
    assert result is None

    captured = capsys.readouterr()
    assert "ERROR: Failed to load PyTorch model" in captured.out
    assert "Suggestion: Delete the file and retrain the model" in captured.out


def test_load_model_raises_type_error_with_extra_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {"data": [1, 2, 3]}
    save_model(payload, "model.pkl")

    with pytest.raises(TypeError, match="call load_model\\(filepath\\) without a model or device"):
        load_model("model.pkl", filepath="extra.pkl")

    with pytest.raises(TypeError, match="call load_model\\(filepath\\) without a model or device"):
        load_model("model.pkl", device="cpu")


def test_load_model_handles_file_not_found(capsys):
    result = load_model("nonexistent.pkl")
    assert result is None

    captured = capsys.readouterr()
    assert "ERROR: Model file not found: nonexistent.pkl" in captured.out


def test_load_model_warns_on_python_version_mismatch(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    import pickle
    import sys

    # Create a pickled object with different Python version metadata
    metadata = {
        "model": {"data": [1, 2, 3]},
        "python_version": "3.8.0",  # Different from current
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with pytest.warns(RuntimeWarning, match="Python version mismatch"):
        loaded = load_model("model.pkl")

    assert loaded == {"data": [1, 2, 3]}


def test_load_model_warns_on_torch_version_mismatch(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    import pickle
    import sys

    current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Create a pickled object with different PyTorch version metadata
    metadata = {
        "model": {"data": [1, 2, 3]},
        "python_version": current_python,
        "torch_version": "1.0.0",  # Different from current
        "numpy_version": np.__version__,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with pytest.warns(RuntimeWarning, match="PyTorch version mismatch"):
        loaded = load_model("model.pkl")

    assert loaded == {"data": [1, 2, 3]}


def test_load_model_handles_unpickling_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    import pickle

    # Create a file that will cause UnpicklingError
    with open("bad_pickle.pkl", "wb") as f:
        f.write(b"INVALID_PICKLE_DATA")

    result = load_model("bad_pickle.pkl")
    assert result is None

    captured = capsys.readouterr()
    assert "PICKLE DESERIALIZATION ERROR" in captured.out
    assert "RECOMMENDED ACTION" in captured.out


def test_load_model_handles_eof_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    # Create an empty file to trigger EOFError
    with open("empty.pkl", "wb") as f:
        pass

    result = load_model("empty.pkl")
    assert result is None

    captured = capsys.readouterr()
    assert "MODEL LOADING ERROR" in captured.out
    assert "RECOMMENDED ACTION" in captured.out


def test_load_model_handles_import_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    import pickle

    # Mock pickle.load to raise ImportError
    original_load = pickle.load

    def mock_pickle_load_import_error(f):
        raise ImportError("No module named 'nonexistent_module'")

    # First, create a valid pickle file
    payload = {"data": [1, 2, 3]}
    with open("import_error.pkl", "wb") as f:
        original_load
        pickle.dump(payload, f)

    # Then mock pickle.load to raise ImportError
    monkeypatch.setattr(pickle, "load", mock_pickle_load_import_error)

    result = load_model("import_error.pkl")
    assert result is None

    captured = capsys.readouterr()
    assert "MODEL LOADING ERROR" in captured.out


def test_load_model_handles_generic_exception(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    import pickle

    # Mock pickle.load to raise an unexpected exception
    original_load = pickle.load

    def mock_pickle_load(f):
        raise RuntimeError("Unexpected error during loading")

    monkeypatch.setattr(pickle, "load", mock_pickle_load)

    payload = {"data": [1, 2, 3]}
    with open("model.pkl", "wb") as f:
        original_load  # Keep reference
        pickle.dump(payload, f)

    # Now restore the mock
    monkeypatch.setattr(pickle, "load", mock_pickle_load)

    result = load_model("model.pkl")
    assert result is None

    captured = capsys.readouterr()
    assert "UNEXPECTED ERROR LOADING MODEL" in captured.out
    assert "Please report this issue" in captured.out


def test_load_model_loads_plain_pickle_without_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import pickle

    # Create a plain pickle without metadata wrapper
    plain_data = [1, 2, 3, 4, 5]
    with open("plain.pkl", "wb") as f:
        pickle.dump(plain_data, f)

    loaded = load_model("plain.pkl")
    assert loaded == plain_data

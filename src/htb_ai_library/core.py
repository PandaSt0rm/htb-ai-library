"""
Core utilities for the HTB Evasion library.
Includes reproducibility functions, model persistence, and color constants.
"""

import os
import pickle
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# --- Reproducibility ---

def set_reproducibility(seed: int = 1337) -> None:
    """
    Configure reproducible behavior across libraries.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- HTB Color Palette ---

HTB_GREEN = "#9fef00"
NODE_BLACK = "#141d2b"
HACKER_GREY = "#a4b1cd"
WHITE = "#ffffff"
AZURE = "#0086ff"
NUGGET_YELLOW = "#ffaf00"
MALWARE_RED = "#ff3e3e"
VIVID_PURPLE = "#9f00ff"
AQUAMARINE = "#2ee7b6"

# --- Model Persistence ---

def save_model(model: Any, filepath: str) -> None:
    """
    Save a model or arbitrary Python object to disk.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), filepath)
    else:
        with open(filepath, "wb") as handle:
            pickle.dump(model, handle)
    print(f"Model saved to {filepath}")

def load_model(
    model_or_filepath: Union[nn.Module, str],
    filepath: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Load model weights (for PyTorch modules) or arbitrary pickled objects.
    """
    if isinstance(model_or_filepath, nn.Module):
        if filepath is None:
            raise ValueError("filepath must be provided when loading into a PyTorch module")
        if device is None:
            raise ValueError("device must be provided when loading into a PyTorch module")
        state_dict = torch.load(filepath, map_location=device)
        model_or_filepath.load_state_dict(state_dict)
        model_or_filepath.to(device)
        model_or_filepath.eval()
        print(f"Model loaded from {filepath}")
        return model_or_filepath

    if filepath is not None or device is not None:
        raise TypeError(
            "When loading pickled objects, call load_model(filepath) without a model or device."
        )

    with open(model_or_filepath, "rb") as handle:
        obj = pickle.load(handle)
    print(f"Model loaded from {model_or_filepath}")
    return obj

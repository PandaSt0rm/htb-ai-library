"""
Core utilities for the HTB Evasion library.
Includes reproducibility functions, model persistence, and color constants.
"""

import os
import pickle
import random
import sys
import warnings
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# --- Reproducibility ---


def set_reproducibility(seed: int = 1337) -> None:
    """
    Configure reproducible behavior across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int, optional
        Non-negative integer applied to all managed random number generators.
        Defaults to 1337.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``seed`` is negative.

    Examples
    --------
    >>> set_reproducibility(1234)
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Align Python and NumPy RNGs.
    random.seed(seed)
    np.random.seed(seed)

    # Align Torch RNGs (CPU and, when available, CUDA).
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # Keep existing user preference for workspace size if provided.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Disable algorithm heuristics that trade determinism for performance.
        if hasattr(torch.backends.cuda, "matmul") and hasattr(
            torch.backends.cuda.matmul, "allow_tf32"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_det_alg = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_det_alg):
        try:
            use_det_alg(True, warn_only=True)
        except TypeError:
            use_det_alg(True)
        except RuntimeError as err:
            warnings.warn(
                f"Deterministic algorithms requested but unavailable: {err}",
                RuntimeWarning,
            )


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

_COLOR_PALETTE: dict[str, str] = {
    "HTB_GREEN": HTB_GREEN,
    "NODE_BLACK": NODE_BLACK,
    "HACKER_GREY": HACKER_GREY,
    "WHITE": WHITE,
    "AZURE": AZURE,
    "NUGGET_YELLOW": NUGGET_YELLOW,
    "MALWARE_RED": MALWARE_RED,
    "VIVID_PURPLE": VIVID_PURPLE,
    "AQUAMARINE": AQUAMARINE,
}

HTB_PALETTE: Mapping[str, str] = MappingProxyType(_COLOR_PALETTE)

def get_color(name: str) -> str:
    """
    Retrieve a Hack The Box brand color by name.

    Parameters
    ----------
    name : str
        Case-insensitive color identifier such as ``"HTB_GREEN"``.

    Returns
    -------
    str
        Hexadecimal color string including the leading hash.

    Raises
    ------
    KeyError
        If the requested color name does not exist in the palette.

    Examples
    --------
    >>> get_color("htb_green")
    '#9fef00'
    """
    normalized = name.upper()
    if normalized not in HTB_PALETTE:
        raise KeyError(f"Unknown color '{name}'. Available keys: {sorted(HTB_PALETTE)}")
    return HTB_PALETTE[normalized]

def get_color_palette() -> Mapping[str, str]:
    """
    Provide a read-only mapping of Hack The Box color names to hex strings.

    Returns
    -------
    Mapping[str, str]
        Read-only view keyed by uppercase color names.

    Examples
    --------
    >>> palette = get_color_palette()
    >>> palette["HTB_GREEN"]
    '#9fef00'
    """
    return HTB_PALETTE

# --- Model Persistence ---


def save_model(model: Any, filepath: str) -> None:
    """
    Save a model or arbitrary Python object to disk with version metadata.

    Args:
        model: Model or object to save
        filepath: Path where the model will be saved
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), filepath)
    else:
        # Wrap the model with version metadata for better compatibility checking
        metadata = {
            "model": model,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        with open(filepath, "wb") as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {filepath}")


def load_model(
    model_or_filepath: Union[nn.Module, str],
    filepath: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Load model weights (for PyTorch modules) or arbitrary pickled objects.

    This function handles version compatibility issues gracefully and provides
    informative error messages when loading fails.

    Args:
        model_or_filepath: Either a PyTorch module or filepath string
        filepath: Path to model file (required when first arg is a module)
        device: Device for PyTorch module loading

    Returns:
        Loaded model or object, or None if loading fails

    Raises:
        ValueError: If required parameters are missing
        TypeError: If parameters are used incorrectly
    """
    # Handle PyTorch module loading
    if isinstance(model_or_filepath, nn.Module):
        if filepath is None:
            raise ValueError(
                "filepath must be provided when loading into a PyTorch module"
            )
        if device is None:
            raise ValueError(
                "device must be provided when loading into a PyTorch module"
            )

        try:
            state_dict = torch.load(filepath, map_location=device, weights_only=False)
            model_or_filepath.load_state_dict(state_dict)
            model_or_filepath.to(device)
            model_or_filepath.eval()
            print(f"Model loaded from {filepath}")
            return model_or_filepath
        except Exception as e:
            print(f"\nERROR: Failed to load PyTorch model from {filepath}")
            print(f"Reason: {type(e).__name__}: {str(e)}")
            print(f"Suggestion: Delete the file and retrain the model.")
            return None

    # Handle pickled object loading
    if filepath is not None or device is not None:
        raise TypeError(
            "When loading pickled objects, call load_model(filepath) without a model or device."
        )

    # Check if file exists
    if not os.path.exists(model_or_filepath):
        print(f"\nERROR: Model file not found: {model_or_filepath}")
        return None

    # Try to load with comprehensive error handling
    try:
        with open(model_or_filepath, "rb") as handle:
            obj = pickle.load(handle)

        # Check if this is our new metadata format
        if isinstance(obj, dict) and "model" in obj:
            # Check version compatibility
            saved_python = obj.get("python_version", "unknown")
            saved_torch = obj.get("torch_version", "unknown")
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            current_torch = torch.__version__

            # Warn about version mismatches
            if saved_python != current_python:
                warnings.warn(
                    f"Python version mismatch: model saved with {saved_python}, "
                    f"loading with {current_python}. This may cause compatibility issues.",
                    RuntimeWarning,
                )

            if saved_torch != current_torch:
                warnings.warn(
                    f"PyTorch version mismatch: model saved with {saved_torch}, "
                    f"loading with {current_torch}. This may cause compatibility issues.",
                    RuntimeWarning,
                )

            print(f"Model loaded from {model_or_filepath}")
            return obj["model"]
        else:
            # Old format or plain object
            print(f"Model loaded from {model_or_filepath}")
            return obj

    except pickle.UnpicklingError as e:
        print(f"\n{'=' * 60}")
        print(f"PICKLE DESERIALIZATION ERROR")
        print(f"{'=' * 60}")
        print(f"Failed to load model from: {model_or_filepath}")
        print(f"Error: {str(e)}")
        print(f"\nPossible causes:")
        print(f"  1. File was saved with incompatible Python/PyTorch version")
        print(f"  2. File is corrupted or incomplete")
        print(f"  3. File format has changed between library versions")
        print(f"\nCurrent environment:")
        print(
            f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        print(f"  PyTorch: {torch.__version__}")
        print(f"\nRECOMMENDED ACTION:")
        print(f"  Delete the incompatible file and retrain the model:")
        print(f"  rm {model_or_filepath}")
        print(f"{'=' * 60}\n")
        return None

    except (EOFError, ValueError, ImportError, ModuleNotFoundError) as e:
        print(f"\n{'=' * 60}")
        print(f"MODEL LOADING ERROR")
        print(f"{'=' * 60}")
        print(f"Failed to load model from: {model_or_filepath}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print(f"\nRECOMMENDED ACTION:")
        print(f"  Delete the corrupted file and retrain the model:")
        print(f"  rm {model_or_filepath}")
        print(f"{'=' * 60}\n")
        return None

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"UNEXPECTED ERROR LOADING MODEL")
        print(f"{'=' * 60}")
        print(f"Failed to load model from: {model_or_filepath}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print(f"\nPlease report this issue with the full error message.")
        print(f"{'=' * 60}\n")
        return None

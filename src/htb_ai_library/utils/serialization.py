"""
Model persistence helpers with version metadata safeguards.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def save_model(model: Any, filepath: str) -> None:
    """
    Save a model or arbitrary Python object to disk with version metadata.

    Parameters
    ----------
    model : Any
        Model or object to save.
    filepath : str
        Path where the model will be saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), filepath)
    else:
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

    Parameters
    ----------
    model_or_filepath : Union[nn.Module, str]
        Either a PyTorch module or filepath string.
    filepath : Optional[str], default None
        Path to model file (required when first argument is a module).
    device : Optional[Union[str, torch.device]], default None
        Device for PyTorch module loading.

    Returns
    -------
    Any | None
        Loaded model or object, or ``None`` if loading fails.

    Raises
    ------
    ValueError
        If required parameters are missing.
    TypeError
        If parameters are used incorrectly.
    """
    if isinstance(model_or_filepath, nn.Module):
        if filepath is None:
            raise ValueError("filepath must be provided when loading into a PyTorch module")
        if device is None:
            raise ValueError("device must be provided when loading into a PyTorch module")

        try:
            state_dict = torch.load(filepath, map_location=device, weights_only=False)
            model_or_filepath.load_state_dict(state_dict)
            model_or_filepath.to(device)
            model_or_filepath.eval()
            print(f"Model loaded from {filepath}")
            return model_or_filepath
        except Exception as exc:  # noqa: BLE001
            print(f"\nERROR: Failed to load PyTorch model from {filepath}")
            print(f"Reason: {type(exc).__name__}: {exc}")
            print("Suggestion: Delete the file and retrain the model.")
            return None

    if filepath is not None or device is not None:
        raise TypeError(
            "When loading pickled objects, call load_model(filepath) without a model or device."
        )

    if not os.path.exists(model_or_filepath):
        print(f"\nERROR: Model file not found: {model_or_filepath}")
        return None

    try:
        with open(model_or_filepath, "rb") as handle:
            obj = pickle.load(handle)

        if isinstance(obj, dict) and "model" in obj:
            saved_python = obj.get("python_version", "unknown")
            saved_torch = obj.get("torch_version", "unknown")
            current_python = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            current_torch = torch.__version__

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

        print(f"Model loaded from {model_or_filepath}")
        return obj

    except pickle.UnpicklingError as exc:
        print(f"\n{'='*60}")
        print("PICKLE DESERIALIZATION ERROR")
        print(f"{'='*60}")
        print(f"Failed to load model from: {model_or_filepath}")
        print(f"Error: {exc}")
        print("\nPossible causes:")
        print("  1. File was saved with incompatible Python/PyTorch version")
        print("  2. File is corrupted or incomplete")
        print("  3. File format has changed between library versions")
        print("\nCurrent environment:")
        print(
            f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        print(f"  PyTorch: {torch.__version__}")
        print("\nRECOMMENDED ACTION:")
        print(f"  Delete the incompatible file and retrain the model:\n  rm {model_or_filepath}")
        print(f"{'='*60}\n")
        return None
    except (EOFError, ValueError, ImportError, ModuleNotFoundError) as exc:
        print(f"\n{'='*60}")
        print("MODEL LOADING ERROR")
        print(f"{'='*60}")
        print(f"Failed to load model from: {model_or_filepath}")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        print("\nRECOMMENDED ACTION:")
        print(f"  Delete the corrupted file and retrain the model:\n  rm {model_or_filepath}")
        print(f"{'='*60}\n")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"\n{'='*60}")
        print("UNEXPECTED ERROR LOADING MODEL")
        print(f"{'='*60}")
        print(f"Failed to load model from: {model_or_filepath}")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        print("\nPlease report this issue with the full error message.")
        return None

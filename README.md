# HTB AI Evasion Library

A focused toolkit used across Hack The Box’s AI evasion content. It bundles reproducibility helpers, dataset utilities, reference models, training loops, evaluation metrics, and Hack The Box–styled visualizations for building and analyzing adversarial ML experiments.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Quickstart](#quickstart)
- [Module Overview](#module-overview)
  - [Core Utilities](#core-utilities)
  - [Data Utilities](#data-utilities)
  - [Reference Models](#reference-models)
  - [Training Helpers](#training-helpers)
  - [Metrics](#metrics)
  - [Visualization](#visualization)
- [Documentation](#documentation)
- [Testing](#testing)
  - [Unit Tests](#unit-tests)
- [Dataset Management](#dataset-management)
- [Backwards Compatibility](#backwards-compatibility)
- [Contributing](#contributing)
- [License](#license)

Version: 0.2.1

## Getting Started

### Prerequisites

- Python 3.10+
- pip (or another PEP 517–compatible installer)
- PyTorch and TorchVision (install via the official instructions for your OS/CUDA)

### Installation

A new `venv` is recommended. First install PyTorch following the official guide for your platform, then install the library.

From PyPI (recommended for users of the library):

```bash
# Install PyTorch + TorchVision per https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install htb-ai-library
```

From source (recommended for contributors and for running tests/docs locally):

```bash
git clone https://github.com/PandaSt0rm/htb-ai-library.git
cd htb-ai-library

# Install PyTorch + TorchVision per https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install -e .
```

## Quickstart

```python
import torch

from htb_ai_library.utils import set_reproducibility
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import SimpleLeNet, SimpleCNN
from htb_ai_library.training import train_model
from htb_ai_library.evaluation import evaluate_attack_effectiveness
from htb_ai_library.visualization import use_htb_style, visualize_attack

# Apply HTB theme globally to all plots
use_htb_style()

# Reproducibility and device selection
set_reproducibility(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data + model
train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir="./data")
model = SimpleCNN().to(device)  # or SimpleLeNet() for a LeNet-style alternative

# Train for a couple of epochs
train_model(model, train_loader, test_loader, device=device, epochs=2)

# Evaluate an example attack (dummy perturbation shown for brevity)
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
adversarial = images + 0.01 * torch.sign(torch.randn_like(images))

results = evaluate_attack_effectiveness(model, images, adversarial, labels)
print(results)

# Visualize the difference for the first sample
visualize_attack(
    model,
    images[0].cpu(),
    labels[0].item(),
    adversarial[0].cpu(),
    title="Demo Attack",
    num_classes=10,
)
```

This snippet trains the reference MNIST CNN, computes attack metrics for a mock perturbation, and opens the Hack The Box–styled visualization.

Note: `get_mnist_loaders` returns MNIST tensors in `[0, 1]` by default. Pass `normalize=True` to apply the canonical `(mean=0.1307, std=0.3081)` normalization, and use `mnist_denormalize` when you need to revert normalized tensors for visualization.

## Module Overview

The library ships as focused subpackages. Prefer explicit imports from:
`htb_ai_library.utils`, `htb_ai_library.data`, `htb_ai_library.models`, `htb_ai_library.training`, `htb_ai_library.evaluation`, and `htb_ai_library.visualization`.
Legacy modules like `htb_ai_library.core`, `htb_ai_library.metrics`, and `htb_ai_library.viz` proxy to the new layout for backwards compatibility.

### Core Utilities

`htb_ai_library.utils`

- `set_reproducibility(seed=1337)` – align RNGs across `random`, NumPy, and PyTorch (CPU + CUDA). Also available via `htb_ai_library.core`.
- `save_model(model, filepath)` / `load_model(model_or_filepath, filepath=None, device=None)` – persistence helpers with version metadata safeguards and friendly error messages; compatible with all bundled architectures such as `SimpleLeNet`.
- HTB color constants (`HTB_GREEN`, `NODE_BLACK`, `MALWARE_RED`, `VIVID_PURPLE`, `AQUAMARINE`, etc.) plus `get_color(name)` and `get_color_palette()`.

### Data Utilities

`htb_ai_library.data`

- `get_mnist_loaders(batch_size=128, data_dir="./data", normalize=False, seed=1337)` – training/test loaders that leave MNIST pixels in `[0, 1]` by default; pass `normalize=True` to enable canonical normalization. Deterministic shuffling is used when `seed` is set.
- `download_sms_spam_dataset(data_dir="./data")` – fetches, preprocesses, and caches the UCI SMS Spam Collection as `sms_spam.csv`.
- `mnist_denormalize(x)` – invert MNIST normalization back to `[0, 1]` for visualization and certain metrics.
- `cifar_normalize(x)` – apply CIFAR‑10 mean/std normalization to a tensor batch.

### Reference Models

`htb_ai_library.models`

- `SimpleLeNet` – LeNet-style classifier with optional `forward_log_probs` for log-softmax outputs.
- `SimpleCNN` – fast MNIST‑ready CNN for demonstrations.
- `MNISTClassifierWithDropout` – dropout‑regularized MNIST classifier.
- `ResNetCIFAR` – lightweight ResNet‑18 style network for CIFAR‑10.

### Training Helpers

`htb_ai_library.training`

- `train_one_epoch` – single‑epoch loop with averaged loss reporting.
- `train_model` – multi‑epoch trainer with Adam optimizer and validation accuracy logging.
- `evaluate_accuracy` – dataset accuracy with safe handling of empty loaders.

### Metrics

`htb_ai_library.evaluation`

- `evaluate_attack_effectiveness` – clean/adversarial accuracy, attack success rate (safe against divide‑by‑zero), confidence deltas, and perturbation norms.
- `analyze_model_confidence` – batched confidence statistics for a configurable number of samples.
- `print_attack_summary` – formatted console report of the metrics dictionary.

### Visualization

`htb_ai_library.visualization`

- `use_htb_style` – apply Hack The Box theme globally to all matplotlib plots. Call once at the start of your script to automatically style all figures with HTB brand colors (dark backgrounds, HTB_GREEN titles, etc.).
- `visualize_attack` – side‑by‑side clean/adversarial comparison, perturbation heatmap, and probability histograms.
- `plot_attack_effectiveness` – effort vs. evasion‑rate plot for tabular attack results.
- `visualize_perturbation_analysis` – histograms of L2/L∞ norms and sparsity with CUDA‑safe tensor handling.

## Documentation

The repository ships with Sphinx docs that combine this README with an autogenerated API reference.

1. Install the optional docs dependencies:
   ```bash
   pip install -e .[docs]
   ```
   Or install them directly from `docs/requirements.txt`.
2. Build the HTML documentation:
   ```bash
   make -C docs html
   ```
   The rendered site is written to `docs/_build/html/index.html`. Pass additional options through `SPHINXOPTS`, e.g. `make -C docs html SPHINXOPTS="-W --keep-going"` to fail on warnings.
3. (Optional) Validate external links:
   ```bash
   make -C docs linkcheck
   ```

## Testing

### Unit Tests

A pytest suite lives in `tests/` and covers the public API with fast, CPU‑only tests. Run it after installing dev dependencies:

```bash
pytest
```

For coverage:

```bash
coverage run -m pytest
coverage report -m --include='src/htb_ai_library/*'
```

## Dataset Management

- MNIST caches under `./data/MNIST` by default.
- The SMS Spam Collection is written to `./data/sms_spam.csv` after downloading; intermediate archives are removed.
- To track datasets elsewhere, change `data_dir` arguments or update `.gitignore`.

## Backwards Compatibility

- `htb_ai_library.core`, `htb_ai_library.metrics`, and `htb_ai_library.viz` remain as compatibility shims that re‑export the new `utils`, `evaluation`, and `visualization` modules. New code should import from the subpackages directly.
- Saved models and generic Python objects use `save_model`/`load_model`. When loading generic objects, the loader warns on Python/PyTorch version mismatches and prints actionable messages for corrupted files.

## Contributing

Contributions are welcome. Please open an issue or pull request with a clear description of the change. If you add or modify functionality, include tests.

## License

This project is licensed under the [MIT License](LICENSE).

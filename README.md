# HTB AI Evasion Library

Comprehensive toolkit used throughout Hack The Box's AI evasion content. The library bundles reproducibility helpers, dataset utilities, reference models, training loops, evaluation metrics, and visualization helpers that support building, training, and analyzing adversarial machine-learning experiments.

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
- [Testing](#testing)
  - [Unit Tests](#unit-tests)
  - [Attack Smoke Tests](#attack-smoke-tests)
- [Dataset Management](#dataset-management)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

- Python **3.9+**
- pip (or another PEP 517 compatible installer)
- PyTorch and TorchVision (with GPU support if you have supported hardware)

### Installation

A new `venv` is recommended. Install directly from the repository:

```bash
pip install pytorch torchvision pandas numpy matplotlib
pip install git+https://github.com/PandaSt0rm/htb-ai-library
```

Alternatively, clone the project if you plan to modify the source or run the included tests:

```bash
git clone https://github.com/PandaSt0rm/htb-ai-library.git
cd htb-ai-library
pip install pytorch torchvision pandas numpy matplotlib
pip install -e .
```

## Quickstart

```python
from pathlib import Path

import torch
from htb_ai_library import (
    set_reproducibility,
    get_mnist_loaders,
    SimpleCNN,
    train_model,
    evaluate_attack_effectiveness,
    visualize_attack,
)

# Reproducibility and device selection
set_reproducibility(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data + model
train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir="./data")
model = SimpleCNN().to(device)

# Train for a couple of epochs
train_model(model, train_loader, test_loader, device=device, epochs=2)

# Evaluate an example attack (dummy perturbation shown for brevity)
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
adversarial = images + 0.01 * torch.sign(torch.randn_like(images))

results = evaluate_attack_effectiveness(model, images, adversarial, labels)
print(results)

# Visualize the difference for the first sample
visualize_attack(model, images[0].cpu(), labels[0].item(), adversarial[0].cpu(), "Demo Attack", num_classes=10)
```

This snippet trains the reference MNIST CNN, computes attack metrics for a mock perturbation, and opens the Hack The Box styled visualization.

Note: `get_mnist_loaders` returns MNIST tensors in `[0, 1]` by default. Pass `normalize=True` to apply the canonical `(mean=0.1307, std=0.3081)` normalization, and use `mnist_denormalize` when you need to revert normalized tensors for visualization.

## Module Overview

### Core Utilities

`htb_ai_library.core`

- `set_reproducibility(seed=1337)` – align RNGs across `random`, NumPy, and PyTorch (CPU + CUDA).
- `save_model(model, filepath)` / `load_model(model, filepath, device)` – persistence helpers tolerant of bare filenames.
- HTB color constants (`HTB_GREEN`, `NODE_BLACK`, `MALWARE_RED`, etc.) for branding plots and dashboards.

### Data Utilities

`htb_ai_library.data`

- `get_mnist_loaders(batch_size=128, data_dir="./data", normalize=False)` – training/test loaders that leave MNIST pixels in `[0, 1]` by default; pass `normalize=True` to enable canonical normalization.
- `download_sms_spam_dataset(data_dir="./data")` – fetches, preprocesses, and caches the UCI SMS Spam Collection as `sms_spam.csv`.
- `mnist_denormalize(tensor)` – invert MNIST normalization back to `[0, 1]` pixel space for visualization and metrics that expect original scales.
- `cifar_normalize(tensor)` – apply CIFAR-10 mean/std normalization to a tensor batch.

### Reference Models

`htb_ai_library.models`

- `SimpleCNN` – fast MNIST-ready CNN for demonstrations.
- `MNISTClassifierWithDropout` – dropout-enabled variant used in adversarial robustness chapters.
- `ResNetCIFAR` – lightweight ResNet-18 style network tailored for CIFAR-10 experiments.

### Training Helpers

`htb_ai_library.training`

- `train_one_epoch` – single-epoch loop with averaged loss reporting.
- `train_model` – multi-epoch trainer with Adam optimizer and validation accuracy logging.
- `evaluate_accuracy` – dataset accuracy calculation with safe handling of empty loaders.

### Metrics

`htb_ai_library.metrics`

- `evaluate_attack_effectiveness` – returns clean/adversarial accuracy, attack success rate (safe against divide-by-zero), confidence deltas, and perturbation norms.
- `analyze_model_confidence` – batched confidence statistics for a configurable number of samples.
- `print_attack_summary` – formatted console report of the metrics dictionary.

### Visualization

`htb_ai_library.viz`

- `visualize_attack` – side-by-side clean/adversarial comparison, perturbation heatmap, and probability histograms.
- `plot_attack_effectiveness` – effort vs. evasion-rate plot for tabular attack results.
- `visualize_perturbation_analysis` – histograms of L2/L∞ norms and sparsity with CUDA-safe tensor handling.

## Testing

### Unit Tests

A comprehensive pytest suite lives in `tests/` and covers every library module. Run it after installing dev dependencies:

```bash
pytest
```

For coverage insights:

```bash
coverage run -m pytest
coverage report -m --include='src/htb_ai_library/*'
```

## Dataset Management

- MNIST data is cached under `./data/MNIST` by default.
- The SMS Spam Collection is saved as `./data/sms_spam.csv` after downloading; intermediate archives are cleaned up automatically.
- If you wish to track datasets separately, adjust the `data_dir` arguments or update `.gitignore` rules accordingly.

## License

This project is licensed under the [MIT License](LICENSE).

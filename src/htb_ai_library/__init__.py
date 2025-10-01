"""
HTB Evasion Lib: A collection of common utilities for adversarial attacks.
"""

# core.py
from .core import (
    set_reproducibility,
    save_model,
    load_model,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE, 
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

# data.py
from .data import (
    get_mnist_loaders,
    download_sms_spam_dataset,
    mnist_denormalize,
    cifar_normalize
)

# models.py
from .models import (
    SimpleCNN,
    MNISTClassifierWithDropout,
    ResNetCIFAR
)

# training.py
from .training import (
    train_model,
    evaluate_accuracy
)

# metrics.py
from .metrics import (
    evaluate_attack_effectiveness,
    analyze_model_confidence,
    print_attack_summary
)

# viz.py
from .viz import (
    visualize_attack,
    plot_attack_effectiveness,
    visualize_perturbation_analysis
)

__all__ = [
    # core
    "set_reproducibility",
    "save_model",
    "load_model",
    "HTB_GREEN", "NODE_BLACK", "HACKER_GREY", "WHITE", "AZURE", 
    "NUGGET_YELLOW", "MALWARE_RED", "VIVID_PURPLE", "AQUAMARINE",
    # data
    "get_mnist_loaders",
    "download_sms_spam_dataset",
    "mnist_denormalize",
    "cifar_normalize",
    # models
    "SimpleCNN",
    "MNISTClassifierWithDropout",
    "ResNetCIFAR",
    # training
    "train_model",
    "evaluate_accuracy",
    # metrics
    "evaluate_attack_effectiveness",
    "analyze_model_confidence",
    "print_attack_summary",
    # viz
    "visualize_attack",
    "plot_attack_effectiveness",
    "visualize_perturbation_analysis",
]

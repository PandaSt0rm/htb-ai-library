"""
HTB Evasion Lib: A collection of common utilities for adversarial attacks.
"""

# core.py
from .core import (
    set_reproducibility,
    save_model,
    load_model,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE, 
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE,
    HTB_PALETTE, get_color, get_color_palette
)

# data subpackage
from .data import (
    get_mnist_loaders,
    download_sms_spam_dataset,
    mnist_denormalize,
    cifar_normalize
)

# models subpackage
from .models import (
    SimpleCNN,
    MNISTClassifierWithDropout,
    ResNetCIFAR
)

# training subpackage
from .training import (
    train_one_epoch,
    train_model,
    evaluate_accuracy
)

# evaluation subpackage
from .evaluation import (
    evaluate_attack_effectiveness,
    analyze_model_confidence,
    print_attack_summary
)

# visualization subpackage
from .visualization import (
    visualize_attack,
    plot_attack_effectiveness,
    visualize_perturbation_analysis,
    style_htb_axis,
    apply_htb_axes_style,
)

__all__ = [
    # core
    "set_reproducibility",
    "save_model",
    "load_model",
    "HTB_GREEN", "NODE_BLACK", "HACKER_GREY", "WHITE", "AZURE", 
    "NUGGET_YELLOW", "MALWARE_RED", "VIVID_PURPLE", "AQUAMARINE",
    "HTB_PALETTE", "get_color", "get_color_palette",
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
    "train_one_epoch",
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
    "style_htb_axis",
    "apply_htb_axes_style",
]

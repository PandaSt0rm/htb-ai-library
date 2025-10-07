"""
Model architectures organized by dataset and use case.
"""

from .mnist import SimpleCNN, MNISTClassifierWithDropout
from .cifar import ResNetCIFAR

__all__ = [
    "SimpleCNN",
    "MNISTClassifierWithDropout",
    "ResNetCIFAR",
]

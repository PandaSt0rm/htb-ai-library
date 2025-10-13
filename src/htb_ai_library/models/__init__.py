"""
Model architectures organized by dataset and use case.
"""

from .mnist import SimpleLeNet, SimpleCNN, MNISTClassifierWithDropout
from .cifar import ResNetCIFAR

__all__ = [
    "SimpleLeNet",
    "SimpleCNN",
    "MNISTClassifierWithDropout",
    "ResNetCIFAR",
]

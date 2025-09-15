import torch

from htb_ai_library.models import MNISTClassifierWithDropout, ResNetCIFAR, SimpleCNN


def test_simple_cnn_forward_shape():
    model = SimpleCNN()
    output = model(torch.randn(2, 1, 28, 28))
    assert output.shape == (2, 10)


def test_mnist_classifier_with_dropout():
    model = MNISTClassifierWithDropout()
    model.train()
    output = model(torch.randn(3, 1, 28, 28))
    assert output.shape == (3, 10)


def test_resnet_cifar_forward_shape():
    model = ResNetCIFAR()
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)

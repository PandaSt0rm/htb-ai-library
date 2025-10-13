import torch

from htb_ai_library.models import MNISTClassifierWithDropout, ResNetCIFAR, SimpleCNN, SimpleLeNet


def test_simple_lenet_forward_shape():
    model = SimpleLeNet()
    output = model(torch.randn(2, 1, 28, 28))
    assert output.shape == (2, 10)


def test_simple_lenet_log_probs_sum_to_one():
    model = SimpleLeNet()
    log_probs = model.forward_log_probs(torch.randn(3, 1, 28, 28))
    probs = log_probs.exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


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


def test_simple_lenet_serialization_roundtrip(tmp_path, monkeypatch):
    from htb_ai_library.utils import load_model, save_model

    monkeypatch.chdir(tmp_path)
    source = SimpleLeNet()
    for param in source.parameters():
        torch.nn.init.normal_(param)

    save_model(source, "lenet_state.pth")
    loaded = load_model(SimpleLeNet(), "lenet_state.pth", device="cpu")
    assert isinstance(loaded, SimpleLeNet)

    for p_loaded, p_source in zip(loaded.parameters(), source.parameters()):
        assert torch.allclose(p_loaded, p_source)

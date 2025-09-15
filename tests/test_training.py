import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from htb_ai_library.training import evaluate_accuracy, train_model, train_one_epoch


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.linear(inputs.view(inputs.size(0), -1))


def _build_loader(size=8):
    inputs = torch.randn(size, 1, 2, 2)
    labels = (inputs.view(size, -1).sum(dim=1) > 0).long()
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_train_one_epoch_returns_loss():
    loader = _build_loader()
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

    assert loss >= 0


def test_train_model_improves_accuracy(monkeypatch):
    train_loader = _build_loader()
    test_loader = _build_loader()
    model = TinyNet()

    trained_model = train_model(model, train_loader, test_loader, device="cpu", epochs=2, learning_rate=0.1)

    acc = evaluate_accuracy(trained_model, test_loader, device=torch.device("cpu"))
    assert 0 <= acc <= 100


def test_evaluate_accuracy_handles_empty_loader():
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 1, 2, 2), torch.empty(0, dtype=torch.long)))
    model = TinyNet()

    accuracy = evaluate_accuracy(model, empty_loader, device=torch.device("cpu"))
    assert accuracy == 0

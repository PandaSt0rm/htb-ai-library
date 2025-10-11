import io
import zipfile

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from htb_ai_library import data as data_module


class DummyMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, *_, transform=None, **__):
        self.transform = transform
        self.samples = [torch.ones(1, 28, 28) * idx for idx in range(4)]
        self.targets = torch.tensor([0, 1, 2, 3])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[index]


def test_get_mnist_loaders_uses_configured_batch(monkeypatch):
    from torchvision import transforms, datasets
    monkeypatch.setattr(transforms, "Compose", lambda steps: lambda x: x)
    monkeypatch.setattr(datasets, "MNIST", DummyMNISTDataset)

    train_loader, test_loader = data_module.get_mnist_loaders(batch_size=2, data_dir="ignored")

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader.dataset) == 4
    assert len(test_loader.dataset) == 4


def _fake_urlretrieve_factory(contents: bytes):
    def _fake_urlretrieve(url, filename):
        with open(filename, "wb") as fh:
            fh.write(contents)
        return filename, None

    return _fake_urlretrieve


def _build_sms_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("SMSSpamCollection", "ham\tHello\nspam\tBuy now\n")
        zf.writestr("readme", "dummy")
    return buffer.getvalue()


def test_download_sms_spam_dataset_creates_csv(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.chdir(tmp_path)
    zipped_bytes = _build_sms_zip()
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve_factory(zipped_bytes))

    df = data_module.download_sms_spam_dataset(data_dir=".")

    assert (tmp_path / "sms_spam.csv").exists()
    assert list(df.columns) == ["label", "message"]
    assert df.shape == (2, 2)
    assert not (tmp_path / "smsspamcollection.zip").exists()
    assert not (tmp_path / "SMSSpamCollection").exists()
    assert not (tmp_path / "readme").exists()


def test_download_sms_spam_dataset_uses_cache(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "sms_spam.csv"
    pd.DataFrame({"label": ["ham"], "message": ["hello"]}).to_csv(csv_path, index=False)

    calls = []

    def _tracking_urlretrieve(url, filename):
        calls.append(url)
        raise AssertionError("Should not be called when cache exists")

    monkeypatch.setattr(urllib.request, "urlretrieve", _tracking_urlretrieve)

    df = data_module.download_sms_spam_dataset(data_dir=".")

    assert df.shape[0] == 1
    assert not calls


def test_cifar_normalize_applies_stats():
    tensor = torch.ones(1, 3, 2, 2)
    normalized = data_module.cifar_normalize(tensor)

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    expected = ((torch.ones(3) - mean) / std).view(1, 3, 1, 1)

    assert torch.allclose(normalized, expected)


def test_mnist_denormalize_inverts_normalization():
    # Test denormalization with typical normalized values
    normalized = torch.tensor([[[[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]]])
    denormalized = data_module.mnist_denormalize(normalized)

    # Check that values are in [0, 1] range
    assert denormalized.min() >= 0.0
    assert denormalized.max() <= 1.0

    # Test specific values
    mean, std = 0.1307, 0.3081
    expected = torch.clamp(normalized * std + mean, 0.0, 1.0)
    assert torch.allclose(denormalized, expected)


def test_mnist_denormalize_handles_extreme_values():
    # Test clamping behavior with extreme values
    extreme = torch.tensor([[[[10.0, -10.0]]]])
    denormalized = data_module.mnist_denormalize(extreme)

    # Should be clamped to [0, 1]
    assert denormalized.min() >= 0.0
    assert denormalized.max() <= 1.0

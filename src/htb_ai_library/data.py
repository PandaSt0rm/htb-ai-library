"""
Data loading and preparation utilities.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = './data',
    *,
    normalize: bool = False,
    seed: int | None = 1337,
) -> tuple[DataLoader, DataLoader]:
    """Create MNIST data loaders with optional normalization and deterministic shuffling."""

    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_steps)

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, test_loader

def download_sms_spam_dataset(data_dir: str = './data') -> pd.DataFrame:
    """
    Download and prepare the SMS Spam Collection dataset from UCI ML Repository.
    """
    dataset_path = os.path.join(data_dir, 'sms_spam.csv')
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(dataset_path):
        print("Dataset already exists, loading...")
        return pd.read_csv(dataset_path)

    print("Downloading SMS Spam Collection dataset...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    zip_path = os.path.join(data_dir, 'smsspamcollection.zip')

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    df = pd.read_csv(os.path.join(data_dir, 'SMSSpamCollection'), sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].str.lower()
    df.to_csv(dataset_path, index=False)

    # Clean up
    os.remove(zip_path)
    if os.path.exists(os.path.join(data_dir, 'SMSSpamCollection')):
        os.remove(os.path.join(data_dir, 'SMSSpamCollection'))
    if os.path.exists(os.path.join(data_dir, 'readme')):
        os.remove(os.path.join(data_dir, 'readme'))

    print(f"Downloaded and prepared dataset with {len(df)} messages.")
    return df

def cifar_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Apply CIFAR-10 normalization to a tensor.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype, device=x.device)[None, :, None, None]
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / std

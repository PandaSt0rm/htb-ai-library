"""
Download and prepare the SMS Spam Collection dataset.
"""

from __future__ import annotations

import os
import zipfile
from typing import Optional

import pandas as pd

__all__ = ["download_sms_spam_dataset"]


def download_sms_spam_dataset(data_dir: str = "./data") -> pd.DataFrame:
    """
    Download and prepare the SMS Spam Collection dataset from the UCI repository.

    Parameters
    ----------
    data_dir : str, optional
        Directory where the dataset will be stored, by default ``"./data"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``label`` and ``message`` columns.
    """
    dataset_path = os.path.join(data_dir, "sms_spam.csv")
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(dataset_path):
        print("Dataset already exists, loading...")
        return pd.read_csv(dataset_path)

    print("Downloading SMS Spam Collection dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")

    import urllib.request  # local import to avoid hard dependency when unused

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    df = pd.read_csv(
        os.path.join(data_dir, "SMSSpamCollection"),
        sep="\t",
        header=None,
        names=["label", "message"],
    )
    df["label"] = df["label"].str.lower()
    df.to_csv(dataset_path, index=False)

    os.remove(zip_path)
    for leftover in ["SMSSpamCollection", "readme"]:
        leftover_path = os.path.join(data_dir, leftover)
        if os.path.exists(leftover_path):
            os.remove(leftover_path)

    print(f"Downloaded and prepared dataset with {len(df)} messages.")
    return df

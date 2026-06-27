from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_diabetes


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    features: pd.DataFrame
    target: pd.Series


def load_diabetes_dataset() -> DatasetBundle:
    """Load the sklearn Diabetes regression dataset into a reusable bundle."""
    dataset = load_diabetes(as_frame=True)

    features = dataset.data.copy()
    target = dataset.target.rename("target").copy()

    return DatasetBundle(
        dataset_name="diabetes",
        features=features,
        target=target,
    )

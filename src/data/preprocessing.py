from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load_diabetes import DatasetBundle


DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class TabularDataSplit:
    dataset_name: str
    feature_names: list[str]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    x_train_raw: np.ndarray
    x_val_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train_scaled: np.ndarray
    x_val_scaled: np.ndarray
    x_test_scaled: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def create_split_indices(
    num_samples: int,
    *,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitIndices:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0.")

    all_indices = np.arange(num_samples)
    train_indices, temp_indices = train_test_split(
        all_indices,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    val_fraction_of_temp = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_fraction_of_temp,
        random_state=random_state,
        shuffle=True,
    )

    return SplitIndices(
        train=np.sort(train_indices),
        val=np.sort(val_indices),
        test=np.sort(test_indices),
    )


def prepare_diabetes_tabular_data(
    dataset: DatasetBundle,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TabularDataSplit:
    features = dataset.features.copy()
    target = dataset.target.copy()

    split_indices = create_split_indices(
        len(features),
        random_state=random_state,
    )

    x_train_raw = features.iloc[split_indices.train].to_numpy(dtype=np.float64)
    x_val_raw = features.iloc[split_indices.val].to_numpy(dtype=np.float64)
    x_test_raw = features.iloc[split_indices.test].to_numpy(dtype=np.float64)

    y_train = target.iloc[split_indices.train].to_numpy(dtype=np.float64)
    y_val = target.iloc[split_indices.val].to_numpy(dtype=np.float64)
    y_test = target.iloc[split_indices.test].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    return TabularDataSplit(
        dataset_name=dataset.dataset_name,
        feature_names=list(features.columns),
        train_indices=split_indices.train,
        val_indices=split_indices.val,
        test_indices=split_indices.test,
        x_train_raw=x_train_raw,
        x_val_raw=x_val_raw,
        x_test_raw=x_test_raw,
        x_train_scaled=x_train_scaled,
        x_val_scaled=x_val_scaled,
        x_test_scaled=x_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

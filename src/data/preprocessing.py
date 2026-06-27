from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_diabetes import DatasetBundle
from src.data.load_parts import CASTINGS_CATEGORICAL_COLUMNS, CASTINGS_ID_COLUMN


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


def _build_feature_preprocessor(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _prepare_tabular_data(
    dataset: DatasetBundle,
    *,
    categorical_columns: list[str],
    drop_columns: list[str],
    random_state: int,
) -> TabularDataSplit:
    features = dataset.features.drop(columns=drop_columns, errors="ignore").copy()
    target = dataset.target.copy()

    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]
    categorical_columns = [
        column for column in categorical_columns if column in features.columns
    ]

    split_indices = create_split_indices(
        len(features),
        random_state=random_state,
    )

    x_train_frame = features.iloc[split_indices.train].copy()
    x_val_frame = features.iloc[split_indices.val].copy()
    x_test_frame = features.iloc[split_indices.test].copy()

    y_train = target.iloc[split_indices.train].to_numpy(dtype=np.float64)
    y_val = target.iloc[split_indices.val].to_numpy(dtype=np.float64)
    y_test = target.iloc[split_indices.test].to_numpy(dtype=np.float64)

    raw_preprocessor = _build_feature_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scale_numeric=False,
    )
    scaled_preprocessor = _build_feature_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scale_numeric=True,
    )

    x_train_raw = raw_preprocessor.fit_transform(x_train_frame)
    x_val_raw = raw_preprocessor.transform(x_val_frame)
    x_test_raw = raw_preprocessor.transform(x_test_frame)

    x_train_scaled = scaled_preprocessor.fit_transform(x_train_frame)
    x_val_scaled = scaled_preprocessor.transform(x_val_frame)
    x_test_scaled = scaled_preprocessor.transform(x_test_frame)

    feature_names = scaled_preprocessor.get_feature_names_out().tolist()

    return TabularDataSplit(
        dataset_name=dataset.dataset_name,
        feature_names=feature_names,
        train_indices=split_indices.train,
        val_indices=split_indices.val,
        test_indices=split_indices.test,
        x_train_raw=x_train_raw.astype(np.float64),
        x_val_raw=x_val_raw.astype(np.float64),
        x_test_raw=x_test_raw.astype(np.float64),
        x_train_scaled=x_train_scaled.astype(np.float64),
        x_val_scaled=x_val_scaled.astype(np.float64),
        x_test_scaled=x_test_scaled.astype(np.float64),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


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
    return _prepare_tabular_data(
        dataset,
        categorical_columns=[],
        drop_columns=[],
        random_state=random_state,
    )


def prepare_castings_tabular_data(
    dataset: DatasetBundle,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TabularDataSplit:
    return _prepare_tabular_data(
        dataset,
        categorical_columns=CASTINGS_CATEGORICAL_COLUMNS,
        drop_columns=[CASTINGS_ID_COLUMN],
        random_state=random_state,
    )

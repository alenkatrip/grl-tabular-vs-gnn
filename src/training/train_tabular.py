from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from src.data.preprocessing import TabularDataSplit
from src.models.tabular_baselines import TabularModelSpec


@dataclass(frozen=True)
class TabularModelRun:
    model_name: str
    validation_predictions: np.ndarray
    test_predictions: np.ndarray


def _select_features(
    data_split: TabularDataSplit,
    *,
    requires_scaled_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if requires_scaled_features:
        return (
            data_split.x_train_scaled,
            data_split.x_val_scaled,
            data_split.x_test_scaled,
        )

    return (
        data_split.x_train_raw,
        data_split.x_val_raw,
        data_split.x_test_raw,
    )


def train_and_predict_tabular_models(
    data_split: TabularDataSplit,
    model_specs: list[TabularModelSpec],
) -> list[TabularModelRun]:
    runs: list[TabularModelRun] = []

    for model_spec in model_specs:
        x_train, x_val, x_test = _select_features(
            data_split,
            requires_scaled_features=model_spec.requires_scaled_features,
        )
        estimator = deepcopy(model_spec.estimator)
        estimator.fit(x_train, data_split.y_train)

        runs.append(
            TabularModelRun(
                model_name=model_spec.name,
                validation_predictions=estimator.predict(x_val),
                test_predictions=estimator.predict(x_test),
            )
        )

    return runs

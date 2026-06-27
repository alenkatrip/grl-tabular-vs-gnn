from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocessing import TabularDataSplit
from src.training.train_tabular import TabularModelRun


@dataclass(frozen=True)
class RegressionMetrics:
    dataset: str
    split: str
    model: str
    mae: float
    rmse: float
    r2: float


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dataset_name: str,
    split_name: str,
    model_name: str,
) -> RegressionMetrics:
    return RegressionMetrics(
        dataset=dataset_name,
        split=split_name,
        model=model_name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def evaluate_tabular_runs(
    data_split: TabularDataSplit,
    runs: list[TabularModelRun],
) -> pd.DataFrame:
    metric_rows: list[RegressionMetrics] = []

    for run in runs:
        metric_rows.append(
            compute_regression_metrics(
                data_split.y_val,
                run.validation_predictions,
                dataset_name=data_split.dataset_name,
                split_name="validation",
                model_name=run.model_name,
            )
        )
        metric_rows.append(
            compute_regression_metrics(
                data_split.y_test,
                run.test_predictions,
                dataset_name=data_split.dataset_name,
                split_name="test",
                model_name=run.model_name,
            )
        )

    return pd.DataFrame(asdict(row) for row in metric_rows)

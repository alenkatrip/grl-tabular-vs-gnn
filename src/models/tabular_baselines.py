from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


@dataclass(frozen=True)
class TabularModelSpec:
    name: str
    estimator: object
    requires_scaled_features: bool


def build_tabular_model_specs(random_state: int = 42) -> list[TabularModelSpec]:
    return [
        TabularModelSpec(
            name="linear_regression",
            estimator=LinearRegression(),
            requires_scaled_features=True,
        ),
        TabularModelSpec(
            name="ridge",
            estimator=Ridge(alpha=1.0),
            requires_scaled_features=True,
        ),
        TabularModelSpec(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            requires_scaled_features=False,
        ),
        TabularModelSpec(
            name="mlp",
            estimator=MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                batch_size=32,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=random_state,
            ),
            requires_scaled_features=True,
        ),
    ]

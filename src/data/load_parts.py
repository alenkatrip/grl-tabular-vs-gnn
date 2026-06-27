from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_diabetes import DatasetBundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASTINGS_DATASET_PATH = PROJECT_ROOT / "data/raw/Castings_technical_dataset_csv.csv"
CASTINGS_TARGET_COLUMN = "Y"
CASTINGS_ID_COLUMN = "ID"
CASTINGS_CATEGORICAL_COLUMNS = ["Material", "Type of casting"]


def _strip_whitespace(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda column: column.str.strip() if column.dtype == object else column)


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .replace({"na": pd.NA, "NA": pd.NA, "": pd.NA})
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_castings_dataset(path: Path | None = None) -> DatasetBundle:
    """Load the castings CSV with locale-aware numeric parsing."""
    csv_path = path or CASTINGS_DATASET_PATH

    raw_frame = pd.read_csv(csv_path, sep=";", dtype=str, encoding="utf-8")
    raw_frame = _strip_whitespace(raw_frame)
    raw_frame = raw_frame.replace({"na": pd.NA, "NA": pd.NA, "": pd.NA})

    features = raw_frame.drop(columns=[CASTINGS_TARGET_COLUMN]).copy()
    target = _normalize_numeric_series(raw_frame[CASTINGS_TARGET_COLUMN]).rename("target")

    numeric_feature_columns = [
        column
        for column in features.columns
        if column not in CASTINGS_CATEGORICAL_COLUMNS
    ]

    for column in numeric_feature_columns:
        features[column] = _normalize_numeric_series(features[column])

    for column in CASTINGS_CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("string")

    return DatasetBundle(
        dataset_name="castings",
        features=features,
        target=target,
    )

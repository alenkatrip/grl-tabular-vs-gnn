from __future__ import annotations

from pathlib import Path

from src.data.load_parts import load_castings_dataset
from src.data.preprocessing import DEFAULT_RANDOM_STATE, prepare_castings_tabular_data
from src.models.tabular_baselines import build_tabular_model_specs
from src.training.evaluation import evaluate_tabular_runs
from src.training.train_tabular import train_and_predict_tabular_models


def run_castings_experiment(output_path: Path | None = None) -> Path:
    dataset = load_castings_dataset()
    data_split = prepare_castings_tabular_data(
        dataset,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model_specs = build_tabular_model_specs(random_state=DEFAULT_RANDOM_STATE)
    runs = train_and_predict_tabular_models(data_split, model_specs)
    results = evaluate_tabular_runs(data_split, runs)

    if output_path is None:
        output_path = Path("results/metrics/castings_tabular_results.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = run_castings_experiment()
    print(f"Saved metrics to {saved_path}")

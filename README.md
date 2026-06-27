# GRL: Tabular Models vs Graph Neural Networks

Mini-project for the TU Wien Graph Representation Learning course: comparing standard tabular regression models with graph-based models built on similarity graphs over tabular samples.

## Datasets

- `sklearn` Diabetes dataset
- Industrial parts pricing dataset for castings in `data/raw/Castings_technical_dataset_csv.csv`

## Models

- Linear Regression
- Ridge Regression
- Random Forest
- MLP
- GNN on a kNN similarity graph
- Graph Convolutional Network

## Evaluation

- MAE
- RMSE
- R²

## Run

Replicate the currently implemented tabular experiments with:

```bash
python -m src.experiments.run_diabetes_experiment
python -m src.experiments.run_parts_experiment
```

Exploration notebooks:

- `notebooks/01_diabetes_exploration.ipynb`
- `notebooks/02_castings_exploration.ipynb`

Implementation details and current project status are documented in `docs/`.

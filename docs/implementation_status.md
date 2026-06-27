## Implementation Status

## Current Scope

Implemented so far:

- Phase 1a: Diabetes tabular baseline pipeline
- Phase 1b: Castings tabular baseline pipeline

Not implemented yet:

- k-nearest-neighbor graph construction
- GNN training and evaluation
- Graph-vs-tabular comparison results

## Implemented Tabular Models

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- MLP Regressor

## Implemented Preprocessing

### Diabetes

- Shared train/validation/test split
- Train-only standardization for scaled-model inputs

### Castings

- Locale-aware CSV parsing
- Semicolon-separated input handling
- Comma decimal and period thousands normalization
- Literal `na` handling as missing data
- `ID` excluded from model features
- Numeric features: median imputation
- Categorical features: most-frequent imputation and one-hot encoding
- Shared train/validation/test split across all baselines

## Entry Points

Implemented experiment scripts:

- `python -m src.experiments.run_diabetes_experiment`
- `python -m src.experiments.run_parts_experiment`

Generated metrics files:

- `results/metrics/diabetes_tabular_results.csv`
- `results/metrics/castings_tabular_results.csv`

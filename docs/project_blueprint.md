# Title
From Tables to Graphs: Comparing Tabular Models and GNNs on Industrial Price Prediction and Diabetes Regression

## Main question
Does converting tabular data into a similarity graph improve prediction compared to standard tabular models?

## Datasets

1. Excel parts dataset for hydraulic cylinders
    * Rows: technical parts
    * Features: technical attributes
    * Target: price
    * Task: regression
2. Diabetes dataset
    * Use sklearn.datasets.load_diabetes
    * 442 samples, 10 numeric features
    * Target: disease progression
    * Task: regression


## GRL connection
The project compares row-wise tabular learning against graph-based message passing. In GRL terms, traditional graph/structured-data methods often use extracted features with standard ML, while GNNs learn representations by aggregating neighborhood information  . The graph construction is important because GNNs encode relational inductive bias: predictions depend not only on each row’s features, but also on neighboring similar rows .


## Models to compare

For both datasets:

Family | Model
Classical baseline | Linear Regression / Ridge
Strong tabular baseline | Random Forest Regressor
Neural tabular baseline | MLP regressor
Graph model | GCN or GraphSAGE regressor

Use the same train/validation/test split for all models.

## Graph construction

Each row becomes one node.

Build a **k-nearest-neighbor graph** from standardized tabular features:

* node = one part / one diabetes patient
* node features = original processed tabular features
* edge = similarity between rows
* try k = 5 as default
* optional ablation: k = 3, 5, 10

Important: build the kNN graph only from features, never from the target.

## Main experiments

### Experiment 1: Parts dataset

Compare:
* Ridge / Linear Regression
* Random Forest
* MLP
* GNN on kNN graph

Metrics:
* MAE
* RMSE
* R²

### Experiment 2: Diabetes dataset

Same models, same metrics.

### Optional ablation

Run GNN with:
* k = 3
* k = 5
* k = 10

This lets you discuss whether graph quality / graph density affects performance.

## Expected discussion

Possible conclusions:
* If GNN improves: neighboring similar rows carry useful information.
* If GNN does not improve: the constructed graph may not add information beyond tabular features.
* If GNN helps on one dataset but not the other: graph inductive bias is domain-dependent.
* If Random Forest wins: strong tabular models may still be preferable when relations are artificial.
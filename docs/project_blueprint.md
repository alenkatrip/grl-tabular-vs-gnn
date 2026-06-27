# From Tables to Graphs: Comparing Tabular Models and GNNs for Expected Price Estimation and Regression

## Main question
Does converting tabular data into a similarity graph improve prediction compared to standard tabular models?

For the industrial parts dataset, the project also asks whether predicted expected price can be interpreted as a data-driven fair-price estimate when only observed paid prices, but no expert fair-price labels, are available.

## Datasets

1. Excel parts dataset for hydraulic cylinders
    * Rows: technical parts
    * Features: technical attributes
    * Target: observed paid price
    * Task: regression
    * Interpretation: observed paid price is treated as a noisy proxy target, not as true expert fair price
2. Diabetes dataset
    * Use sklearn.datasets.load_diabetes
    * 442 samples, 10 numeric features
    * Target: disease progression
    * Task: regression
    * Role: public sanity-check dataset for validating the reusable pipeline

## Price-estimation framing

The parts-pricing dataset does not provide expert fair-price labels. It provides observed paid prices. Therefore, the project should avoid claiming to learn objective fair price directly.

Instead, the model learns expected observed price conditional on technical parameters and, for GNN models, similarity-graph neighborhood:

```text
predicted_price = f(technical_features, graph_neighborhood)
```

The predicted price can then be interpreted as a data-driven fair-price estimate, with the limitation that the training target is noisy. For the parts dataset, residuals can be analyzed as:

```text
residual = observed_price - predicted_price
```

Positive residuals may indicate items paid above the model-estimated expected price. Negative residuals may indicate items paid below the model-estimated expected price.

## GRL connection
The project compares row-wise tabular learning against graph-based message passing. In GRL terms, traditional graph/structured-data methods often use extracted features with standard ML, while GNNs learn representations by aggregating neighborhood information  . The graph construction is important because GNNs encode relational inductive bias: predictions depend not only on each row’s features, but also on neighboring similar rows .


## Models to compare

For both datasets:

| Family | Model |
|---|---|
| Classical baseline | Linear Regression / Ridge |
| Strong tabular baseline | Random Forest Regressor |
| Neural tabular baseline | MLP regressor |
| Graph model | GCN or GraphSAGE regressor |
| Optional extension | Self-supervised GNN embeddings + supervised regressor |

Use the same train/validation/test split for all models.

The optional self-supervised variant should only be added after the supervised tabular and supervised GNN pipelines are working.

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

Additional analysis:
* residual = observed paid price - predicted price
* absolute error per item
* examples of strong overprediction and underprediction

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
* For parts pricing: the model estimates expected observed price, not objective fair price.
* Residuals can suggest possible overpayment or underpayment, but should not be treated as definitive proof.
* Diabetes is useful for reproducibility and pipeline validation, while the parts dataset is the main domain contribution.
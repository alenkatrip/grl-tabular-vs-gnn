# From Tables to Graphs: Comparing Tabular Models and GNNs for Expected Price Estimation on Technical Parts Data

## Main question
Does converting tabular data into a similarity graph improve prediction compared to standard tabular models?

For the industrial parts dataset, the project also asks whether predicted expected price can be interpreted as a data-driven fair-price estimate when only observed paid prices, but no expert fair-price labels, are available.

## Project phases

The project is implemented incrementally:

| Phase | Scope | Status |
|---|---|---|
| Phase 1a | Public sanity-check regression pipeline on the sklearn Diabetes dataset | done |
| Phase 1b | Castings technical parts dataset with tabular regression baselines | current |
| Phase 2a | kNN similarity graph construction and graph exploration | next |
| Phase 2b | Supervised GNN node-level regression | planned |
| Phase 2c | Final ML vs GNN comparison | planned |
| Optional Phase 3 | Self-supervised graph embeddings plus supervised regressor | optional only if time remains |

The current implementation focus is Phase 1b: establishing reliable tabular baselines on the castings technical parts dataset. Graph construction and GNN regression should start only after this baseline is working.

## Datasets

### 1. Castings technical parts dataset

This is the main domain dataset.

- Rows: technical castings parts.
- Features: technical attributes such as volume, weight, process indicators, material, casting type, and related technical parameters.
- Target: observed paid price `Y`.
- Task: regression.
- Interpretation: observed paid price is treated as a noisy proxy target, not as true expert fair price.
- Current implementation phase: Phase 1b tabular baselines.

The raw file is local-only and should not be committed to GitHub.

### 2. Diabetes dataset

This is a public sanity-check dataset.

- Source: `sklearn.datasets.load_diabetes()`.
- 442 samples, 10 numeric features.
- Target: disease progression.
- Task: regression.
- Role: validates that the reusable preprocessing, training, evaluation, and result-saving pipeline works on a clean public dataset.

The Diabetes dataset is not the main domain contribution.

## Price-estimation framing

The parts-pricing dataset does not provide expert fair-price labels. It provides observed paid prices. Therefore, the project should avoid claiming to learn objective fair price directly.

Instead, the model learns expected observed price conditional on technical parameters. In the later GNN phase, the model additionally uses information from the similarity-graph neighborhood:

```text
tabular_model: predicted_price = f(technical_features)
gnn_model:     predicted_price = f(technical_features, graph_neighborhood)
```

The predicted price can then be interpreted as a data-driven fair-price estimate, with the limitation that the training target is noisy. For the parts dataset, residuals can be analyzed as:

```text
residual = observed_price - predicted_price
```

Positive residuals may indicate items paid above the model-estimated expected price. Negative residuals may indicate items paid below the model-estimated expected price. These residuals are diagnostic signals, not definitive proof of overpayment or underpayment.

## GRL connection

The project compares row-wise tabular learning against graph-based message passing.

In graph representation learning terms, the castings table is converted into an attributed graph:

- each row becomes a node,
- processed technical attributes become node features,
- k-nearest-neighbor similarity defines edges,
- the observed paid price is the node-level regression target.

This creates a supervised node-level regression task on a constructed similarity graph.

The graph is not naturally given by the domain. It is an imposed relational structure. The experiment therefore tests whether this imposed graph creates a useful relational inductive bias compared to standard tabular models.

In the GNN phase, predictions depend not only on a row's own features, but also on information aggregated from neighboring similar rows through message passing.


## Models to compare

### Phase 1: tabular baselines

For both datasets:

| Family | Model |
|---|---|
| Classical baseline | Linear Regression |
| Regularized classical baseline | Ridge Regression |
| Strong tabular baseline | Random Forest Regressor |
| Neural tabular baseline | MLP regressor |

### Phase 2: graph model

After Phase 1b is complete, add one simple GNN regressor:

| Family | Model |
|---|---|
| Graph model | GraphSAGE or GCN regressor |

Prefer one GNN architecture first. GraphSAGE is the preferred default; GCN is a reasonable alternative.

### Optional Phase 3

| Family | Model |
|---|---|
| Optional extension | Self-supervised graph embeddings + supervised regressor |

The optional self-supervised variant should only be added after the supervised tabular and supervised GNN pipelines are working.

Use the same train/validation/test split for all models.

## Graph construction

Graph construction belongs to Phase 2a, after the Phase 1b tabular baselines are working.

Each row becomes one node.

Build a k-nearest-neighbor graph from processed tabular features:

- node = one castings part or one diabetes patient,
- node features = processed tabular features,
- edge = similarity between rows,
- default `k = 5`,
- optional ablation: `k = 3`, `k = 5`, `k = 10`.

Important rules:

- Build the kNN graph only from input features.
- Never use the target column when constructing the graph.
- Never use the `ID` column as a feature.
- Use the same train/validation/test split as the tabular models.
- Make the graph undirected unless there is a clear reason not to.
- Convert edges into PyTorch Geometric `edge_index` format.

Because the graph is not naturally given, graph construction is itself an experimental design choice. The kNN graph should therefore be explored before training the GNN.

Graph validation should include:

- number of nodes and edges,
- degree distribution,
- connected components,
- isolated nodes,
- example nearest-neighbor neighborhoods,
- sensitivity to the choice of `k`.

## Main experiments

### Experiment 1: Castings technical parts dataset

Phase 1b compares tabular baselines:

- Linear Regression,
- Ridge Regression,
- Random Forest,
- MLP.

Phase 2 adds:

- GNN on the kNN similarity graph.

Metrics:

- MAE,
- RMSE,
- R².

Additional analysis:

- residual = observed paid price - predicted price,
- absolute error per item,
- examples of strong overprediction and underprediction,
- graph neighborhood examples after graph construction,
- graph statistics before GNN training.

### Experiment 2: Diabetes dataset

Same models and same metrics, but with a different role:

- use it as a public sanity-check dataset,
- use it to validate the reusable pipeline,
- do not present it as the main domain contribution.

### Optional k ablation

Run the GNN with:

- `k = 3`,
- `k = 5`,
- `k = 10`.

This lets the report discuss whether graph quality and graph density affect performance.

## Expected discussion

Possible conclusions:
* If GNN improves: neighboring similar rows carry useful information.
* If GNN does not improve: the constructed graph may not add information beyond tabular features.
* If GNN helps on one dataset but not the other: graph inductive bias is domain-dependent.
* If Random Forest wins: strong tabular models may still be preferable when relations are artificial.
* The GNN setup is transductive node regression: all nodes and graph edges are visible during message passing, but only training labels are used for optimization. This should be considered when comparing against tabular baselines.
* For parts pricing: the model estimates expected observed price, not objective fair price.
* Residuals can suggest possible overpayment or underpayment, but should not be treated as definitive proof.
* Graph neighborhoods should be inspected to check whether they are technically meaningful, not only whether they improve metrics.
* Diabetes is useful for reproducibility and pipeline validation, while the parts dataset is the main domain contribution.
[![DOI](https://zenodo.org/badge/897433892.svg)](https://doi.org/10.5281/zenodo.14261713)

# NSPPK
Neighbourhood Subgraph Pairwise Pathway Kernel 

NSPPK stands for **Neighborhood Subgraph Pairwise Pathway Kernel**. It is a graph encoding and feature extraction framework designed to capture both local (node-level) and structural (subgraph-level) information from graphs. The extracted features are then used as input for machine learning models, enabling tasks such as graph classification, regression, or clustering.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Then import it in Python with:

```python
from nsppk import NSPPK
```

## Demo Notebooks

The repository includes a few runnable notebooks that cover the main usage patterns:

- `notebooks/Demo.ipynb`: synthetic graph-sequence walkthrough for graph-level NSPPK behavior
- `notebooks/Importance Demo.ipynb`: synthetic two-class dataset showing node and edge importance with `NodeNSPPK` + `ImportanceNSPPK`
- `notebooks/Streaming Benchmark.ipynb`: source-loading and streaming examples
- `notebooks/Molecular Graph nbits Benchmark.ipynb`: predictive performance vs hashed feature dimension on molecular graphs
- `notebooks/Molecular Graph Speed Benchmark.ipynb`: runtime scaling on molecular graphs

## Loading And Streaming Graphs

NSPPK now exposes two source-ingestion helpers:

- `load_from(...)`: load graphs from a local path or remote URL and return a materialized `list` of `networkx.Graph`
- `stream_from(...)`: load graphs from a local path or remote URL, batch them, and yield transformed outputs from `transform(...)`

Both `NSPPK` and `NodeNSPPK` support the same interface:

```python
load_from(uri, type, reader=None, limit=None, random_state=None, verbose=False)
stream_from(uri, type, reader=None, limit=None, random_state=None, batch_size=128, verbose=False)
```

Built-in `type` values:

- `smiles`
- `sdf`
- `pyg_pt`

`load_from(...)` does not require a fitted estimator. `stream_from(...)` does, because it batches graphs and applies `transform(...)` lazily.

`limit` controls how much of the source is consumed:

- `None`: include all graphs
- integer `n >= 0`: take the first `n` graphs
- float `0 < p < 1`: sample each graph independently with probability `p`

`verbose=True` enables cumulative progress logs during loading or streaming. The log output includes:

- graphs seen and loaded so far
- cumulative parse / conversion errors so far
- elapsed time
- graphs per second
- average seconds per loaded graph

Examples:

```python
nsppk = NSPPK(radius=2, distance=3, connector=1, nbits=12, dense=False, parallel=False)

# Fit on a reproducible subset loaded from a remote SMILES source.
train_graphs = nsppk.load_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    limit=1000,
    random_state=42,
)
nsppk.fit(train_graphs)

# Stream transformed batches from the full source.
for X_batch in nsppk.stream_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    batch_size=256,
    verbose=True,
):
    print(X_batch.shape)

# Random Bernoulli subsample: keep each graph with probability 0.1.
sampled_graphs = nsppk.load_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    limit=0.1,
)

# Show cumulative progress logs while loading or streaming.
train_graphs = nsppk.load_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    limit=1000,
    verbose=True,
)
```

The `smiles` and `sdf` readers require RDKit. The `pyg_pt` reader requires both `torch` and `torch_geometric`. These remain optional dependencies and are only required when the corresponding reader is used.

The `smiles` reader is tolerant of common text formats:

- plain whitespace-delimited `.smi` files
- CSV-like first-column SMILES files
- header lines such as `smiles` or `smiles,name`

### Key Concepts of NSPPK:

1. **Node-Level and Subgraph-Level Information**:
   - NSPPK encodes information about the labels and structure of individual nodes and their neighborhoods.
   - It considers subgraph patterns around each node within a specified radius to build rich feature representations.

2. **Rooted Graph Hashing**:
   - A central part of NSPPK is the **rooted graph hash**, which summarizes the local subgraph around a node.
   - The hash is computed based on:
     - The label of the node itself.
     - The labels of its neighbors.
     - The labels of edges connecting these neighbors.
   - By recursively considering neighborhoods up to a certain **radius**, NSPPK captures multi-hop structural features.

3. **Pairwise Hashing**:
   - To capture interactions between different parts of the graph, NSPPK hashes pairs of subgraphs (rooted at different nodes) along with the shortest path distances between them.
   - This pairwise hashing ensures the kernel captures both local and relational graph structures.

4. **Flexible Parameters**:
   - **Radius**: Determines how far the neighborhood around a node is considered for hashing.
   - **Distance**: Specifies the maximum distance between nodes to be paired.
   - **Connector Thickness**: Adds complexity by considering paths of varying lengths between paired nodes.
   - **Bit Width (nbits)**: Limits the size of the hashed feature space, controlling computational efficiency.

5. **Feature Representation**:
   - NSPPK generates **sparse feature vectors** for nodes and graphs, encoding structural information in a compact format.
   - For nodes, NSPPK provides node-level feature matrices.
   - For graphs, it aggregates node features to generate graph-level representations.

6. **Attribute Handling**:
   - NSPPK can incorporate additional **node attributes** (e.g., numerical or categorical features) into the encoding process.
   - Attributes can be reduced in dimensionality using techniques like **Singular Value Decomposition (SVD)** or clustered into discrete labels using **KMeans** or **ExtraTreesClassifier**.

7. **Machine Learning Compatibility**:
   - NSPPK implements the **scikit-learn Transformer interface**, making it compatible with machine learning pipelines.
   - It supports tasks such as graph classification and regression by encoding input graphs into feature vectors usable by traditional ML models.

8. **Parallelism**:
   - NSPPK can encode graphs in parallel using multiprocessing, leveraging multiple CPU cores to handle large datasets efficiently.

**Applications of NSPPK:**

- **Graph Classification**: NSPPK can be used to generate feature vectors for graphs, enabling classification models to distinguish between different types of graphs based on their structural properties.
- **Node Classification**: With the `NodeNSPPK` class, NSPPK can generate feature vectors for individual nodes, facilitating tasks like node classification within a graph.
- **Graph Similarity**: By comparing the feature vectors generated by NSPPK, one can measure the similarity between graphs or subgraphs.

**Advantages of NSPPK:**

- **Captures Rich Structural Information**: By considering both node labels and their neighborhoods, NSPPK effectively captures the local and global structure of graphs.
- **Flexible Parameters**: The radius, distance, connector thickness, and number of bits can be tuned to capture different levels of detail and control the feature space size.
- **Scalable with Parallel Processing**: NSPPK can process multiple graphs in parallel, making it suitable for large datasets.

**Example Usage:**

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Instantiate the NSPPK transformer
nsppk_transformer = NSPPK(radius=2, distance=3, connector=1, nbits=12, dense=True, parallel=True)

# Create a pipeline with NSPPK and a classifier
pipeline = Pipeline([
    ('nsppk', nsppk_transformer),
    ('classifier', SVC())
])

# Fit the pipeline on training data (graphs and labels)
pipeline.fit(training_graphs, training_labels)

# Predict on new data
predictions = pipeline.predict(test_graphs)
```

## Node Importance

`ImportanceNSPPK` builds on `NodeNSPPK` to assign supervised importance scores to nodes and edges. It first learns node-level NSPPK features, then fits an ensemble classifier over graph labels and projects the learned feature importances back onto each node.

```python
from importance_nsppk import ImportanceNSPPK
from nsppk import NodeNSPPK

node_encoder = NodeNSPPK(radius=1, distance=4, connector=0, nbits=12, dense=True, parallel=False)
importance_model = ImportanceNSPPK(
    node_nsppk=node_encoder,
    importance_key="importance",
    n_iter=6,
    n_estimators=300,
    quantile=0.7,
    parallel=False,
)

importance_model.fit(graphs, labels)
importance_graphs = importance_model.transform(graphs)

# Per-node importance score
importance_graphs[0].nodes[0]["importance"]
```

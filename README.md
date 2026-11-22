# NSPPK

[**Documentation**](https://nsppk.readthedocs.io/en/latest/)

Neighbourhood Subgraph Pairwise Pathway Kernel (NSPPK) converts labeled NetworkX graphs into fixed-width graph or node feature vectors for use with scikit-learn models.

## Install

Python 3.10 or newer is required. From a checkout:

```bash
python -m pip install -e .
```

This installs the core dependencies. Optional source readers need extra packages:

- SMILES and SDF: `pip install rdkit`
- PyTorch Geometric files: install PyTorch for your platform, then `pip install torch_geometric`. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html) for compatible platform and accelerator commands.

These optional packages are not installed with NSPPK.

## Quick start

Create simple undirected NetworkX graphs. Node and edge labels are optional; missing labels are treated as `0` during encoding.

```python
import networkx as nx
from nsppk import NSPPK, NodeNSPPK


def labeled_graph(node_labels, edges):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node_id, {"label": label})
        for node_id, label in enumerate(node_labels)
    )
    graph.add_edges_from(
        (u, v, {"label": "bond"})
        for u, v in edges
    )
    return graph


graphs = [
    labeled_graph(["C", "O", "C"], [(0, 1), (1, 2)]),
    labeled_graph(["C", "N", "C"], [(0, 1), (1, 2)]),
    labeled_graph(["O", "C", "O"], [(0, 1), (1, 2)]),
]

encoder = NSPPK(nbits=8, dense=False, parallel=False)
X = encoder.fit_transform(graphs)
print(X.shape)  # (3, 256)
```

`NSPPK` returns one feature row per graph. With no continuous node attributes and the default graph-level mode, the width is `2 ** nbits`. `dense=False` returns a SciPy sparse matrix; the default `dense=True` returns a NumPy array.

For node-level vectors, use `NodeNSPPK`. Its output is a list with one node-by-feature matrix per input graph:

```python
node_encoder = NodeNSPPK(nbits=8, dense=False, parallel=False)
node_features = node_encoder.fit_transform(graphs)
print(node_features[0].shape)  # (number of nodes in graphs[0], 256)
```

Rows follow each graph’s `graph.nodes()` iteration order. See the [API reference](docs/API.md) for graph requirements, parameter meanings, and output shapes.

For supervised learning directly from graph lists, use `NSPPKClassifier` or `NSPPKRegressor`. Both accept any compatible scikit-learn estimator, with `"linear"` and `"rf"` shortcuts:

```python
from nsppk import NSPPKClassifier

classifier = NSPPKClassifier(
    estimator="linear",
    estimator_params={"C": 0.5},
    nbits=14,
    parallel=False,
)
classifier.fit(train_graphs, y_train)
predictions = classifier.predict(test_graphs)
```

The `"linear"` shortcut uses logistic regression for classification and Ridge regression for regression. The `"rf"` shortcut uses the corresponding random forest. See the [API reference](docs/API.md) for custom estimators and parameter options.

## Documentation

- [API and output reference](docs/API.md)
- [Graph loading and streaming](docs/IO.md)
- [Node attributes](docs/ATTRIBUTES_INTEGRATION.md)
- [Hashing procedure](docs/HASH.md)
- [Node and feature importance](docs/IMPORTANCE_NSPPK.md)
- [Development and releases](docs/DEVELOPMENT.md)
- [Changelog](CHANGELOG.md)

## Examples

Runnable notebooks are in `notebooks/`:

- `Demo.ipynb`: graph-level feature extraction
- `Importance Demo.ipynb`: node and edge importance on a synthetic dataset
- `Streaming Benchmark.ipynb`: loading and streaming examples
- `Molecular Graph Learning Curve.ipynb`, `Molecular Graph nbits Benchmark.ipynb`, and `Molecular Graph Speed Benchmark.ipynb`: molecular graph experiments
- `NSPPK Classifier Benchmark.ipynb`: compare the `linear` and `rf` supervised estimator shortcuts on HIV

Some notebooks require optional chemistry or machine-learning packages beyond NSPPK’s core dependencies. The graph readers and their optional requirements are documented in [IO.md](docs/IO.md).

## Citation

The repository’s Zenodo DOI is [10.5281/zenodo.14261713](https://doi.org/10.5281/zenodo.14261713).

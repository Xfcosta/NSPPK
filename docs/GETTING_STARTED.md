# Getting started

This guide takes you from installation to your first graph feature matrix.

## Install NSPPK

NSPPK requires Python 3.10 or newer. Install the current public version from
GitHub:

```bash
python -m pip install "git+https://github.com/fabriziocosta/NSPPK.git"
```

This installs NSPPK and its core dependencies. Optional readers for SMILES or
SDF files require RDKit; the PyTorch Geometric reader requires PyTorch and
PyTorch Geometric. See [Graph loading and streaming](IO.md) for those options.

To work on a local checkout instead, install it in editable mode from the
repository root:

```bash
python -m pip install -e ".[docs]"
```

## Encode a few graphs

Each input graph is one sample. NSPPK expects simple undirected
`networkx.Graph` objects. Node and edge labels can be any hashable values; if a
label is missing, NSPPK treats it as `0` during encoding.

```python
import networkx as nx
from nsppk import NSPPK


def make_graph(node_labels):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node_id, {"label": label})
        for node_id, label in enumerate(node_labels)
    )
    graph.add_edge(0, 1, label="bond")
    graph.add_edge(1, 2, label="bond")
    return graph


graphs = [
    make_graph(["C", "O", "C"]),
    make_graph(["C", "N", "C"]),
    make_graph(["O", "C", "O"]),
]

encoder = NSPPK(nbits=8, dense=False, parallel=False)
X = encoder.fit_transform(graphs)

print(X.shape)  # (3, 256)
```

The result has one row per graph. With `nbits=8`, the default structural
feature width is `2 ** 8`, so three graphs produce a `(3, 256)` matrix.
`dense=False` returns a SciPy CSR sparse matrix; leave it at the default
`True` for a NumPy array. Sparse output is often a good starting point when
feature width grows.

For the default discrete-label setup, fitting does not learn any state, so you
can also call `encoder.transform(graphs)` directly. Use `fit_transform` when
you want the familiar scikit-learn workflow or when you enable learned node
attribute preprocessing.

## Get one feature row per node

Use `NodeNSPPK` when you need node-level representations. The returned list has
one matrix per graph, with rows in that graph's `graph.nodes()` iteration
order:

```python
from nsppk import NodeNSPPK

node_encoder = NodeNSPPK(nbits=8, dense=False, parallel=False)
node_features = node_encoder.fit_transform(graphs)
print(node_features[0].shape)  # (3, 256)
```

## Use features in a classifier

`NSPPKClassifier` combines graph vectorization with a scikit-learn classifier.
Pass one target per graph:

```python
from nsppk import NSPPKClassifier

classifier = NSPPKClassifier(
    estimator="linear",
    nbits=8,
    parallel=False,
)
classifier.fit(train_graphs, train_targets)
predictions = classifier.predict(test_graphs)
```

The `"linear"` shortcut uses logistic regression. You can also pass a custom
scikit-learn classifier. See the [API reference](API.md) for classifier and
regressor options.

## Where to go next

- [API and representation reference](API.md): parameters, graph requirements,
  sparse and dense output, and feature shapes.
- [Graph loading and streaming](IO.md): built-in readers, optional packages,
  custom readers, and bounded batches.
- [Node attributes](ATTRIBUTES_INTEGRATION.md): continuous node features and
  learned preprocessing.
- [Node and feature importance](IMPORTANCE_NSPPK.md): supervised importance
  weights and graph explanations.

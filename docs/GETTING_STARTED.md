# Getting started

This guide takes you from installation to your first graph feature matrix.

## Install NSPPK

NSPPK requires Python 3.10 or newer. Install the current public version from
PyPI:

```bash
python -m pip install nsppk
```

This installs NSPPK and its core dependencies. Optional readers for SMILES or
SDF files require RDKit; the PyTorch Geometric reader requires PyTorch and
PyTorch Geometric. See [Graph loading and streaming](IO.md) for those options.

To develop NSPPK from a local checkout, install it in editable mode with the
documentation dependencies from the repository root:

```bash
python -m pip install -e ".[docs]"
```

## Encode a few graphs

Each input graph is one sample. NSPPK supports NetworkX `Graph`, `DiGraph`,
`MultiGraph`, and `MultiDiGraph` objects. Directed inputs use undirected
connectivity for distances and neighborhoods while retaining arc direction in
hashes. Parallel edges contribute individually. Node and edge labels can be
any hashable values; if a label is missing, NSPPK treats it as `0` during
encoding.

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

## Get one feature row per edge

Use `EdgeNSPPK` for per-edge representations. Each edge row combines the structural rows of its two endpoints, and rows align with `graph.edges()` order:

```python
from nsppk import EdgeNSPPK

edge_encoder = EdgeNSPPK(nbits=8, dense=False, parallel=False)
edge_features = edge_encoder.fit_transform(graphs)
print(edge_features[0].shape)  # (number_of_edges, 256)
```

See the [edge attributes guide](EDGE_ATTRIBUTES.md) for continuous edge features, cluster-derived edge labels, and graph-level edge blocks.

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

The `"linear"` shortcut uses one-pass SGD with logistic loss, configured for
fast fitting. This speed-first default may trade away predictive performance.
You can also pass a custom scikit-learn classifier. See the [API reference](API.md)
for classifier and regressor options.

For incremental learning, the default classifier uses an estimator that
supports `partial_fit`. Supply the complete class list on the first batch;
the first batch also initializes any learned attribute preprocessing:

```python
classifier = NSPPKClassifier(estimator="linear", nbits=8, parallel=False)
classifier.partial_fit(first_graphs, first_targets, classes=[0, 1])
classifier.partial_fit(next_graphs, next_targets)
```

To stream directly from a supported graph source, use
`classifier.partial_fit_from(..., label_extractor=..., classes=...)`. The
regressor can use `estimator="sgd"` for incremental updates; its default
`estimator="linear"` remains Ridge and does not support `partial_fit`.

## Where to go next

- [API and representation reference](API.md): parameters, graph requirements,
  sparse and dense output, and feature shapes.
- [Graph loading and streaming](IO.md): built-in readers, optional packages,
  custom readers, and bounded batches.
- [Node attributes](ATTRIBUTES_INTEGRATION.md): continuous node features and
  learned preprocessing.
- [Node and feature importance](IMPORTANCE_NSPPK.md): supervised importance
  weights and graph explanations.

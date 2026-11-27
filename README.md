[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14261713.svg)](https://doi.org/10.5281/zenodo.14261713)


<p align="center">
  <img src="https://raw.githubusercontent.com/fabriziocosta/NSPPK/main/assets/nsppk.png" alt="NSPPK graph logo" width="200">
</p>

# NSPPK


Neighbourhood Subgraph Pairwise Pathway Kernel (NSPPK) turns labeled NetworkX graphs into fixed-width feature vectors for graph-level prediction, node-level analysis, and scikit-learn workflows.

[**Read the documentation**](https://nsppk.readthedocs.io/en/latest/) · [Getting started](docs/GETTING_STARTED.md) · [API reference](docs/API.md)

## Install

NSPPK requires Python 3.10 or newer. Install the latest release from PyPI:

```bash
python -m pip install nsppk
```

## Your first graph features

Pass NSPPK a list of NetworkX `Graph`, `DiGraph`, `MultiGraph`, or `MultiDiGraph` objects. Each graph becomes one sample. For directed inputs, distances and neighborhoods use the undirected projection while hashes retain arc direction. Parallel edges and self-loops contribute individually. Node and edge labels can be any hashable values; a missing `label` is treated as `0`.

```python
import networkx as nx
from nsppk import NSPPK


def make_graph(node_labels):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node_id, {"label": label})
        for node_id, label in enumerate(node_labels)
    )
    graph.add_edges_from(
        (u, v, {"label": "bond"})
        for u, v in [(0, 1), (1, 2)]
    )
    return graph


graphs = [
    make_graph(["C", "O", "C"]),
    make_graph(["C", "N", "C"]),
    make_graph(["O", "C", "O"]),
]

encoder = NSPPK(nbits=8, dense=False, parallel=False)
X = encoder.fit_transform(graphs)
print(X.shape)  # (3, 256): one row per graph
```

With `nbits=8` and no continuous node attributes, this example produces 256 structural features per graph. `dense=False` returns a SciPy sparse matrix; the default `dense=True` returns a NumPy array. For the default discrete-label setup, you can also call `encoder.transform(graphs)` directly.

## Choose what you need

- **Graph features:** `NSPPK` returns one feature row per graph, as shown above.
- **Node features:** `NodeNSPPK` returns one node-by-feature matrix per graph. Rows follow each graph’s `graph.nodes()` iteration order.
- **Edge features:** `EdgeNSPPK` returns one edge-by-feature matrix per graph. Rows follow NetworkX edge iteration order; multigraph rows retain each keyed edge, and directed rows include the ordered arc pattern.
- **Prediction:** `NSPPKClassifier` and `NSPPKRegressor` wrap graph vectorization; `NodeNSPPKClassifier`/`NodeNSPPKRegressor` and `EdgeNSPPKClassifier`/`EdgeNSPPKRegressor` predict per node or edge. Incremental and streamed fitting are available when the selected estimator supports them.
- **Tuning:** Graph-, node-, and edge-level wrappers provide `.tune()` for encoder settings with optional Bayesian optimization. Install it with `python -m pip install 'nsppk[tuning]'`; see the [tuning guide](docs/TUNING.md).
- **Leaderboard comparisons:** `nsppk_leaderboard.check_leaderboard(...)` fetches a leaderboard table and compares a supplied score with its reported distribution; see [leaderboard comparisons](docs/LEADERBOARDS.md).

See the [getting-started guide](docs/GETTING_STARTED.md) for runnable examples of node features and prediction, and the [API reference](docs/API.md) for parameters and output details.

## Read graphs from files

The core install supports NetworkX graphs. Optional input formats require extra dependencies:

- SMILES and SDF: `python -m pip install rdkit`
- PyTorch Geometric files: install PyTorch for your platform, then `python -m pip install torch_geometric`. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html) for compatible platform and accelerator commands.

See [graph loading and streaming](docs/IO.md) for supported formats and examples. These optional packages are not installed with NSPPK.

## Documentation and examples

- [Getting started](docs/GETTING_STARTED.md)
- [API and output reference](docs/API.md)
- [Bayesian hyperparameter tuning](docs/TUNING.md)
- [Graph loading and streaming](docs/IO.md)
- [Node attributes](docs/ATTRIBUTES_INTEGRATION.md)
- [Hashing procedure](docs/HASH.md)
- [Node and feature importance](docs/IMPORTANCE_NSPPK.md)
- [Development and releases](docs/DEVELOPMENT.md)
- [Changelog](CHANGELOG.md)

Runnable notebooks are in `notebooks/`: `Demo.ipynb` covers graph features, `Graph Types Vectorization Demo.ipynb` compares multigraph, directed, and undirected inputs, `Importance Demo.ipynb` covers explanations, and `Streaming Benchmark.ipynb` covers loading and streaming. Other notebooks benchmark molecular graphs and supervised learning; some require optional packages. See [IO.md](docs/IO.md) for reader requirements.

## Develop from a checkout

From the repository root, install NSPPK in editable mode:

```bash
python -m pip install -e ".[docs]"
```

## Citation

The repository’s Zenodo DOI is [10.5281/zenodo.14261713](https://doi.org/10.5281/zenodo.14261713).

# I/O Interface

NSPPK exposes two source-ingestion helpers:

- `load_from(...)`: load graphs from a local path or remote URL and return a materialized `list` of `networkx.Graph`
- `stream_from(...)`: load graphs from a local path or remote URL, batch them, and yield transformed outputs from `transform(...)`

Both `NSPPK` and `NodeNSPPK` support the same interface:

```python
load_from(uri, type, reader=None, limit=None, random_state=None, verbose=False, balance=False, label_extractor=None, start_after_instance=0)
stream_from(uri, type, reader=None, limit=None, random_state=None, batch_size=128, verbose=False, label_extractor=None, start_after_instance=0)
```

Built-in `type` values:

- `smiles`
- `sdf`
- `pyg_pt`

`load_from(...)` does not require a fitted estimator. `stream_from(...)` requires a fitted estimator only when the encoder has to learn attribute-dependent preprocessing such as embedding or clustering. For the default discrete-label setup, `fit(...)` is a no-op and streaming works immediately.

When `label_extractor` is provided to `stream_from(...)`, each yielded item becomes `(X_batch, y_batch)` instead of `X_batch` alone.

## Selection Controls

`limit` controls how much of the source is consumed:

- `None`: include all graphs
- integer `n >= 0`: take the first `n` graphs
- float `0 < p < 1`: sample each graph independently with probability `p`

`start_after_instance` controls where loading starts:

- integer `k >= 0`: skip the first `k` materialized graphs before applying `limit`

## Balanced Loading

`balance=True` only applies to `load_from(...)`. It materializes the selected graphs first, then rebalances classes using `label_extractor(graph)`.

Balancing details:

- `limit=None`: return the largest class-balanced subset available
- `limit=<integer>`: return a balanced subset capped at that size
- `limit=<float>`: first apply Bernoulli sampling with probability `limit`, then rebalance the sampled subset
- if the requested size is not divisible by the number of classes, the remainder is filled from the leftover pool after equal per-class sampling
- if `label_extractor` is omitted, `load_from(...)` uses `graph.graph["name"]`

## Progress Logging

`verbose=True` enables cumulative progress logs during loading or streaming. The log output includes:

- graphs seen and loaded so far
- cumulative parse / conversion errors so far
- elapsed time
- graphs per second
- average seconds per loaded graph

## Examples

```python
from nsppk import NSPPK

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

# Stream supervised HIV batches directly.
for X_batch, y_batch in nsppk.stream_from(
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "smiles",
    batch_size=256,
    label_extractor=lambda graph: int(graph.graph["HIV_active"]),
):
    print(X_batch.shape, y_batch.shape)

# Random Bernoulli subsample: keep each graph with probability 0.1.
sampled_graphs = nsppk.load_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    limit=0.1,
)

# Balanced materialized subset using labels attached to each graph.
molecular_graphs = nsppk.load_from(
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "smiles",
    limit=8000,
    random_state=42,
    balance=True,
    label_extractor=lambda graph: int(graph.graph["HIV_active"]),
)

# Sample first, then rebalance the sampled subset.
sampled_balanced_graphs = nsppk.load_from(
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "smiles",
    limit=0.25,
    random_state=42,
    balance=True,
    label_extractor=lambda graph: int(graph.graph["HIV_active"]),
)

# Reserve the first N graphs for one split and stream everything after that.
test_graphs = nsppk.load_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    limit=1000,
)
for X_batch in nsppk.stream_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    start_after_instance=1000,
    batch_size=256,
):
    print(X_batch.shape)

# Show cumulative progress logs while loading or streaming.
train_graphs = nsppk.load_from(
    "https://cache.docking.org/2D/AA/AAAA.smi",
    "smiles",
    limit=1000,
    verbose=True,
)
```

## Reader Notes

The `smiles` and `sdf` readers require RDKit. The `pyg_pt` reader requires both `torch` and `torch_geometric`. These remain optional dependencies and are only required when the corresponding reader is used.

The `smiles` reader is tolerant of common text formats:

- plain whitespace-delimited `.smi` files
- CSV-like first-column SMILES files
- header lines such as `smiles` or `smiles,name`

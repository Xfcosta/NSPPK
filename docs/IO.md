# Graph loading and streaming

NSPPK supports local paths and HTTP(S) URLs through `load_from(...)` and `stream_from(...)`. Both `NSPPK` and `NodeNSPPK` expose these methods.

## Built-in readers

The required `type` argument selects a reader:

| `type` | Input | Conversion |
| --- | --- | --- |
| `smiles` | Plain-text SMILES, whitespace-delimited records, or CSV with SMILES in the first column | RDKit creates an undirected graph. Node labels encode atom number, charge, chirality, total hydrogens, hybridization, and aromaticity. Edge labels encode bond type, stereo, conjugation, and ring membership. CSV columns after SMILES are stored as graph metadata. |
| `sdf` | SDF file | RDKit creates an undirected graph with the same atom and bond labels. The record name is stored as `graph.graph["name"]`; other SDF properties are not currently copied. |
| `pyg_pt` | A PyTorch Geometric `Data`, `Batch`, or iterable of `Data` objects saved with PyTorch | Converts to undirected NetworkX graphs. Node `label` is used when available, otherwise `x`; edge `label` is used when available, otherwise `edge_attr`. Tensor values are converted to hashable Python values. |

Reader names are case-insensitive. Molecular records that RDKit cannot parse are skipped. With `verbose=True`, the progress summary includes the count of those skipped records.

SMILES CSV headers are recognized when the first field is `smiles` (case-insensitive). Header columns after the first are copied into `graph.graph`. For headerless whitespace records, the second field is stored as the graph name.

The `smiles` and `sdf` readers require RDKit. The official [RDKit installation guide](https://rdkit.org/docs/Install.html) lists supported installation methods; for a typical pip environment:

```bash
python -m pip install rdkit
```

The `pyg_pt` reader requires PyTorch and PyTorch Geometric. Install PyTorch for your platform first, then install PyG; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html). Only load `.pt` files from trusted sources because PyTorch deserialization uses pickle-based data.

These packages are optional and are not installed with NSPPK.

## Custom readers

Pass a callable that accepts `uri` and yields NetworkX graphs. When `reader` is provided, it overrides the built-in reader selected by `type`.

```python
import networkx as nx
from nsppk import NSPPK


def reader(_uri):
    graph = nx.Graph()
    graph.add_node("a", label="C")
    yield graph


encoder = NSPPK(parallel=False)
graphs = encoder.load_from("unused", "custom", reader=reader)
```

Use simple undirected `networkx.Graph` objects with ordinary single-edge semantics. See [API.md](API.md) for graph and label expectations.

## `load_from(...)`

`load_from(uri, type, reader=None, limit=None, random_state=None, verbose=False, balance=False, label_extractor=None, start_after_instance=0)` returns a materialized list of graphs. It does not require a fitted encoder.

- `limit=None`: consume all graphs after the starting offset.
- Integer `n >= 0`: without balancing, take the first `n` graphs after the offset.
- Float `0 < p < 1`: independently keep each graph with probability `p`; the returned count is not fixed.
- `start_after_instance=k`: skip the first `k` successfully parsed graphs, then apply the limit.
- `random_state`: seed fractional sampling and balancing selection. An integer seed makes repeated calls reproducible.
- `verbose=True`: print cumulative loading counts and rate information to standard error.

### Balanced loading

`balance=True` is available only for `load_from(...)`. It first materializes the candidate pool, extracts labels, and selects a roughly equal number from each class. The default label extractor reads `graph.graph["name"]`; provide `label_extractor` when labels are stored elsewhere.

- With `limit=None`, it returns the largest equal-per-class subset available.
- With an integer limit, it scans the candidate pool and returns a balanced subset capped at that size; the integer is not a cap on how many source records are read.
- With a fractional limit, Bernoulli sampling happens first, then the selected pool is balanced.
- If the requested size is not divisible by the number of classes, remaining slots are filled from the leftover candidates. Final class counts can therefore differ by up to that remainder.

After loading, these attributes describe the candidate pool before balancing:

- `loaded_balance_candidate_size_`: candidate count
- `loaded_label_counts_`: count per label
- `loaded_class_weight_`: weight per label, computed as `candidate_count / (number_of_classes * count_for_label)`

Balancing scans and stores the full candidate pool, so it can use substantial memory on large sources. Use `stream_from(...)` when you need bounded batches and do not need balanced selection.

## `stream_from(...)`

`stream_from(uri, type, reader=None, limit=None, random_state=None, batch_size=128, verbose=False, label_extractor=None, start_after_instance=0)` reads graphs in batches and transforms each batch.

- `batch_size` must be a positive integer.
- For `NSPPK`, each yielded feature batch is a dense NumPy array or a SciPy CSR matrix according to `dense`.
- For `NodeNSPPK`, each yielded feature batch is a list of per-graph node matrices.
- If `label_extractor` is provided, each item is `(X_batch, y_batch)`, where `y_batch` is a NumPy array of labels extracted from the corresponding graphs.
- Streaming does not support `balance=True`.
- For the default label-only encoder configuration, fitting is not required. Fit first when learned attribute embedding or clustering is configured.

Example with a local source:

```python
from nsppk import NSPPK

encoder = NSPPK(nbits=8, dense=False, parallel=False)
fit_graphs = encoder.load_from("molecules.smi", "smiles", limit=100)
encoder.fit(fit_graphs)

for X_batch in encoder.stream_from(
    "molecules.smi",
    "smiles",
    start_after_instance=100,
    batch_size=32,
    verbose=True,
):
    print(X_batch.shape)
```

A labeled stream can extract a target from graph metadata:

```python
for X_batch, y_batch in encoder.stream_from(
    "molecules.csv",
    "smiles",
    batch_size=32,
    label_extractor=lambda graph: int(graph.graph["active"]),
):
    print(X_batch.shape, y_batch.shape)
```

For exact feature shapes and dense/sparse behavior, see [API.md](API.md).

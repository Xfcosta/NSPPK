# Edge attributes and edge-level features

NSPPK represents continuous edge attributes in two related ways:

1. **Edge-level rows:** `EdgeNSPPK` returns one vector for each edge.
2. **Graph-level features:** `NSPPK` weights the structural edge rows by their continuous edge attributes and concatenates that block with the node block.

Edge attributes can also be clustered into discrete labels. Those labels enter the existing structural hash pipeline through the edge `label` field.

For a runnable example, see [Edge NSPPK Demo](../notebooks/Edge%20NSPPK%20Demo.ipynb).

## Input contract

Pass `networkx.Graph`, `DiGraph`, `MultiGraph`, or `MultiDiGraph` objects. Every edge must have a numeric vector at `edge_attribute_key`, and the vectors must have the same width throughout the graphs being encoded. Edge rows and edge attributes follow NetworkX edge iteration order; for multigraphs, each keyed edge receives its own row.

Directed graphs use the underlying undirected graph for distances, neighborhoods, and shortest-path unions. Structural local patterns retain each arc's direction. Parallel edges, including self-loops, contribute individually and retain their labels and attributes. Existing discrete edge labels remain supported independently of continuous attributes.

## How an edge gets a structural row

Let `X_N` be the structural node feature matrix, with one row per node and `F = 2 ** nbits` columns. For an undirected edge `e = (u, v)`, NSPPK defines its structural row by adding the rows for its endpoints:

```text
X_E[e] = X_N[u] + X_N[v]
```

The structural node matrix contains NSPPK's rooted, root-pair, and connector/path features, plus simplex fragment features when enabled. Summing both endpoint rows gives the edge a symmetric representation that reuses these structural features and caches. If a feature occurs in both endpoint rows, its value is added from both sides. A directed edge also adds a feature hashed from its ordered source label, edge label, and target label.

## `EdgeNSPPK`: one row per edge

Use `EdgeNSPPK` when the downstream task needs an edge-level matrix:

```python
from nsppk import EdgeNSPPK

edge_encoder = EdgeNSPPK(
    radius=1,
    distance=3,
    nbits=10,
    edge_attribute_key="bond_features",
    dense=False,
)
edge_matrices = edge_encoder.fit_transform(graphs)
```

The result is a list with one matrix per graph. Matrix row `i` corresponds to the `i`th edge in `graph.edges()` iteration order. With no `edge_attribute_key`, the matrix has shape `(E, F)`. With edge attributes of width `Q`, it has shape `(E, F + 2Q)` and columns in this order:

1. the `Q` raw (or embedded) edge attribute values
2. `Q` edge-attribute/structure interaction values
3. the `F` structural edge values from `X_E`

The interaction block follows the node interaction construction. Given edge attribute matrix `B` with shape `(E, Q)`, the implementation computes:

```text
C = cube_root(abs(B.T @ X_E @ X_E.T).T)
```

Thus `C` has shape `(E, Q)`. `dense=False` returns CSR matrices; `dense=True` returns NumPy arrays. A graph with no edges has zero rows. If an edge attribute width has been established during fitting, those rows retain the corresponding `(0, F + 2Q)` shape.

## Graph-level edge block in `NSPPK`

Set `edge_attribute_key` on `NSPPK` to include continuous edge attributes in each graph vector:

```python
from nsppk import NSPPK

encoder = NSPPK(
    node_attribute_key="atom_features",  # optional
    edge_attribute_key="bond_features",
    nbits=10,
    dense=False,
)
X = encoder.fit_transform(graphs)
```

The graph-level edge block is the attribute-weighted sum of structural edge rows:

```text
G_E = B.T @ X_E
```

With `Q` edge channels, `G_E` has shape `(Q, F)` and is flattened to `Q * F` columns. NSPPK concatenates this after the node block. If `node_attribute_key` is unset, the node block remains the existing unweighted sum of structural node rows, with width `F`. If node attributes of width `P` are enabled, the node block remains the existing node-attribute-weighted aggregation, with width `P * F`. Therefore, graph-level widths are:

| Configuration | Width |
| --- | ---: |
| No node or edge attributes | `F` |
| Edge attributes only | `F + Q * F` |
| Node attributes only | `P * F` |
| Both node and edge attributes | `(P + Q) * F` |

When `use_node_kernel=True`, NSPPK retains its existing node-kernel block and appends the same `Q * F` edge block before the existing magnitude transform.

## Learned edge preprocessing

Edge preprocessing has independent settings because edge vectors can differ in width and meaning from node vectors:

- `edge_attribute_dim`: fit a `TruncatedSVD` reducer on edge vectors and use this many components.
- `edge_attribute_alphabet_size`: cluster the edge vectors and train a classifier to predict those cluster IDs for later graphs. The built-in `KMeans` uses `random_state=0`.

Both options require `edge_attribute_key` and a prior `fit` on graphs containing edges. When both options are set, SVD is applied before clustering. At transform time, NSPPK copies each graph, applies the fitted reducer, and then predicts the edge labels. Predicted labels replace existing edge `label` values in those working copies; the input graphs remain unchanged. The continuous edge vectors also remain active in the edge feature block.

If only continuous edge attributes are needed, no learned preprocessing is required. Calling `fit` is still useful when later transforms may contain graphs with no edges: fitting on at least one graph with edge attributes records `Q`, allowing empty graphs to produce fixed-width zero edge blocks.

## How this relates to existing edge labels

Discrete edge labels are already part of structural hashes, including neighboring-node and optional simplex fragment hashes. Continuous edge attributes do not change the hash IDs by themselves; they change the numeric edge block through `B.T @ X_E`. Enabling `edge_attribute_alphabet_size` adds a learned path into the hashes by replacing edge labels with predicted cluster IDs in the working graph.

`edge_attribute_key` was accepted by earlier NSPPK versions but did not affect vectorization. Starting in 2.0.0, setting it changes graph-vector width and values by adding the graph-level edge block. Leave it unset to preserve node-only graph feature layouts.

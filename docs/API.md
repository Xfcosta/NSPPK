# API and representation reference

This page describes the public encoder contract. For data loading, see [IO.md](IO.md); for algorithm details, see [HASH.md](HASH.md).

## Input graphs

Pass a list of simple, undirected `networkx.Graph` objects to `fit`, `transform`, or `fit_transform`. A graph must have hashable node IDs. Each node and edge may have a discrete `label` attribute. If a label is missing, the structural encoder uses `0` for that node or edge in its working copy; it does not add those labels to the input graph.

For continuous node features, set `node_attribute_key` to the attribute name. Every node in every graph must have a numeric vector at that key, and vectors must have the same width. Attribute preprocessing options require this key.

`DiGraph` direction and parallel-edge semantics are not represented by the current hash construction. Use ordinary undirected `Graph` inputs.

## Encoders

### `NSPPK`

`NSPPK` returns graph-level feature rows.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `radius` | `1` | Maximum rooted neighborhood radius. Must be a non-negative integer. |
| `distance` | `3` | Maximum shortest-path distance between paired roots. Must be a non-negative integer. |
| `connector` | `0` | Connector thickness used to summarize shortest-path unions; `0` disables connector summaries. Must be a non-negative integer. |
| `nbits` | `10` | Hashed structural width is `2 ** nbits`. Accepted range is 2 through 30. |
| `degree_threshold` | `None` | For nodes above this degree, restrict rooted expansion to radius zero. Must be `None` or a non-negative integer. |
| `dense` | `True` | Return a dense NumPy array when true; return a SciPy CSR matrix when false. |
| `parallel` | `True` | Allow graph-level multiprocessing when the input contains more graphs than the machine’s reported CPU count. Smaller batches run in the current process. |
| `weight_key` | `None` | Node attribute used to multiply that node’s structural feature row. Values should be numeric. |
| `node_attribute_key` | `None` | Node attribute containing a same-width numeric vector on every node. Enables continuous attribute-structure interactions. |
| `attribute_dim` | `None` | If set, fit TruncatedSVD on the node vectors and use this many components. Requires `node_attribute_key`; must be positive. |
| `attribute_alphabet_size` | `None` | If set, cluster node vectors and learn a classifier that predicts cluster labels for new nodes. Requires `node_attribute_key`; must be positive. Built-in KMeans uses `random_state=0`. |
| `edge_attribute_key` | `None` | Accepted for compatibility, but currently does not affect vectorization. Use discrete edge `label` values for structural edge features. |
| `use_node_kernel` | `False` | Use the node-graph interaction representation instead of the standard graph sum. This can greatly increase output width. |
| `sigma` | `None` | If set to a finite positive number, weight contributions by `exp(-(distance ** 2) / sigma ** 2)`. |
| `use_edges_as_features` | `True` | Include distance-tagged edge-triplet features when true. |

The constructor also accepts the aliases `r`, `d`, and `c` for `radius`, `distance`, and `connector`. Prefer the long names.

### `NodeNSPPK`

`NodeNSPPK` uses the same structural parameters as `NSPPK` except `edge_attribute_key` and `use_node_kernel`. It returns a list of per-graph matrices. Matrix row `i` corresponds to the `i`th node in that graph’s iteration order.

With no continuous node attributes, each matrix has width `2 ** nbits`. With `node_attribute_key` enabled, the matrix also contains attribute and interaction columns. Use the graph’s node iteration order when aligning output rows with node labels.

### `ImportanceNSPPK`

`ImportanceNSPPK(node_nsppk, importance_key='att', n_iter=10, n_estimators=100, quantile=0.5, parallel=True, normalize=True)` learns global feature weights from graph-level targets, then projects them to nodes.

- `node_nsppk`: a `NodeNSPPK` encoder
- `importance_key`: node and edge attribute name for output scores
- `n_iter`: positive number of stratified train/test resamples
- `n_estimators`: positive number of ExtraTrees estimators per resample
- `quantile`: importance cutoff quantile in `[0, 1]`
- `parallel`: use all available cores for ExtraTrees when true
- `normalize`: normalize node scores independently within each graph

Call `fit(graphs, targets)` before `transform`. `targets` must be a one-dimensional class label per graph; there must be at least two classes and at least two graphs per class. The 70/30 split must be large enough to retain every class in both subsets. Edge scores are the product of the two endpoint node scores; they are a heuristic, not a separately learned edge attribution.

## Output shapes

Let `G` be the number of graphs, `N_g` the number of nodes in graph `g`, `D = 2 ** nbits` the structural width, and `P` the width of the node attribute vector after any embedding.

| Mode | `dense=False` | `dense=True` |
| --- | --- | --- |
| `NSPPK`, no attributes, standard graph mode | CSR matrix, `(G, D)` | NumPy array, `(G, D)` |
| `NSPPK` with node attributes, standard graph mode | CSR matrix, `(G, P × D)` | NumPy array, `(G, P × D)` |
| `NSPPK(use_node_kernel=True)`, no attributes | CSR matrix, `(G, D × D)` | NumPy array, `(G, D × D)` |
| `NodeNSPPK`, no attributes | list of `G` CSR matrices, each `(N_g, D)` | list of `G` NumPy arrays, each `(N_g, D)` |

For node features with continuous attributes, each graph’s `NodeNSPPK` matrix has width `D + 2P`. The attribute-bearing node-kernel mode has width `2P × D`. These layouts can be large; sparse output reduces storage for structural feature matrices, while dense output may require substantial memory. Attribute interaction and node-kernel calculations also densify intermediate matrices.

The structural matrix reserves columns 0 and 1 for a constant and node degree. Attribute columns are prepended when continuous attributes are enabled, so those reserved-column meanings apply to the structural block.

## Fit and preprocessing

For the default discrete-label setup, `fit` does not learn anything and may be omitted before transformation. Fit first when using `attribute_dim` or `attribute_alphabet_size`. Learned preprocessing is fitted on the nodes in the supplied training graphs and reused by later transforms.

The encoders copy graphs before injecting fallback labels, embedded attributes, or predicted labels. The caller’s graph objects are not modified by these transformations.

## Useful methods

- `fit(graphs, targets=None)`: fit learned attribute preprocessing, when configured.
- `transform(graphs)`: encode a list of graphs using the fitted preprocessing state.
- `fit_transform(graphs, targets=None)`: fit and encode in one call; inherited from scikit-learn’s transformer interface.
- `get_params()` and `set_params(...)`: inspect or update encoder settings. Constructor aliases `r`, `d`, and `c` map to their long-form parameter names.
- `load_from(...)` and `stream_from(...)`: read graph sources; see [IO.md](IO.md).

`ImportanceNSPPK` adds `build_feature_archive(graphs)` and `get_top_features(top_k=10, min_importance=0.0)`. Build the archive before requesting top features. The archive retains every observed occurrence, so it can be much larger than the encoded feature matrix.

The plotting helpers `plot_graphs(...)` and `plot_graph_importance(...)` are in [src/utils.py](../src/utils.py). For importance plots, set `importance_key` to the same attribute name used by `ImportanceNSPPK`.

## Hash collisions

A final feature ID is a bounded hash, not a unique motif name. Different raw signatures can map to the same feature ID. Smaller `nbits` increases collision likelihood; the feature archive can report distinct signatures sharing an ID.

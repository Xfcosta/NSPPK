# ImportanceNSPPK

This document explains how [ImportanceNSPPK](/home/fabrizio/code/NSPPK/src/importance_nsppk.py) works and how it relates to `NodeNSPPK`.

`ImportanceNSPPK` is not a separate graph kernel. It is a post hoc attribution layer built on top of node-level NSPPK features.

Its job is:

1. compute node-level NSPPK features for each graph
2. train an ensemble model to estimate which hashed NSPPK features matter for a supervised target
3. project those feature importances back onto graph nodes and edges
4. optionally recover representative subgraph occurrences for the most important feature ids

## High-Level Idea

`NodeNSPPK` produces, for each graph, a matrix:

- rows: graph nodes
- columns: NSPPK feature ids

Each row says which hashed rooted / paired / edge-triplet patterns are active around one node.

`ImportanceNSPPK` learns a global importance weight per NSPPK feature column. Then, for each node, it sums the weights of the active features in that node row. That sum becomes the node importance score.

So the attribution pipeline is:

```text
graph
-> node-level hashed NSPPK features
-> supervised feature importance over NSPPK columns
-> node scores
-> edge scores
```

## Constructor Parameters

From [src/importance_nsppk.py](/home/fabrizio/code/NSPPK/src/importance_nsppk.py), the main parameters are:

- `node_nsppk`: a fitted or fit-capable `NodeNSPPK` instance
- `importance_key`: node/edge attribute name where scores are stored
- `n_iter`: number of repeated train/test splits used to stabilize feature importances
- `n_estimators`: number of trees in each `ExtraTreesClassifier`
- `quantile`: threshold for zeroing out low-importance features
- `parallel`: whether to let `ExtraTreesClassifier` use all cores
- `normalize`: whether to normalize node scores graph-by-graph to `[0, 1]`

## 1. Fit Phase

The main fit logic is [ImportanceNSPPK.fit(...)](/home/fabrizio/code/NSPPK/src/importance_nsppk.py#L38).

### Step 1.1: Compute node-level NSPPK features

First:

```python
node_feature_mtx_list = self.node_nsppk.fit_transform(graphs)
```

For each graph, `NodeNSPPK` returns a node-feature matrix.

If graph `g` has:

- `N_g` nodes
- `F` NSPPK feature columns

then:

- `node_feature_mtx_list[g].shape == (N_g, F)`

### Step 1.2: Collapse node features to one graph vector

`ImportanceNSPPK` then sums each graph’s node rows:

```python
X = np.vstack([node_feature_mtx.sum(axis=0) for node_feature_mtx in node_feature_mtx_list])
```

So the supervised learner is trained on graph-level vectors, not node-level labels.

If there are `G` graphs, then:

- `X.shape == (G, F)`

This means each column in `X` corresponds to one hashed NSPPK feature id.

### Step 1.3: Repeated supervised feature-importance estimation

For `n_iter` rounds:

1. split the graph dataset into train and test subsets
2. fit an `ExtraTreesClassifier`
3. store `clf.feature_importances_`

In code:

```python
train_X, test_X, train_targets, test_targets = train_test_split(
    X, targets, train_size=0.7, random_state=it + 1
)
clf = ExtraTreesClassifier(...).fit(train_X, train_targets)
feature_importances.append(clf.feature_importances_)
```

The test split is used only to vary the train subset across iterations. The current implementation does not score on `test_X`; it uses repeated random subsetting as a stability device.

### Step 1.4: Aggregate importance across runs

After all runs:

```python
feature_importances_mtx = np.vstack(feature_importances)
mean_importances = np.mean(feature_importances_mtx, axis=0)
std_importances = np.std(feature_importances_mtx, axis=0)
```

So each NSPPK feature column gets:

- a mean importance across runs
- a standard deviation across runs

### Step 1.5: Penalize unstable features

The code computes:

```python
adjusted_importances = mean_importances - std_importances
adjusted_importances[adjusted_importances < 0] = 0
```

This removes features that are either:

- weak on average
- or unstable across repeated train/test splits

So the effective score is a conservative estimate of importance.

### Step 1.6: Quantile thresholding

Next:

```python
threshold = np.quantile(adjusted_importances, self.quantile)
adjusted_importances[adjusted_importances < threshold] = 0
```

This keeps only the top part of the importance distribution.

Examples:

- `quantile=0.0`: keep everything nonnegative
- `quantile=0.5`: keep roughly the top half
- `quantile=0.7`: keep only the top 30 percent by adjusted importance

### Step 1.7: Normalize the global feature-importance vector

Finally:

```python
max_importance = np.max(adjusted_importances)
self.feature_importance_vector = adjusted_importances / max_importance
```

unless the vector is all zeros, in which case it is kept as zero.

This produces one normalized weight per NSPPK feature column:

- `self.feature_importance_vector.shape == (F,)`

This is the core learned object used later for attribution.

## 2. Transform Phase

The projection back to graph nodes happens in [ImportanceNSPPK.transform(...)](/home/fabrizio/code/NSPPK/src/importance_nsppk.py#L132).

### Step 2.1: Recompute node-level feature matrices

First:

```python
node_feature_mtx_list = self.node_nsppk.transform(graphs)
```

Again, each graph gets a node-by-feature matrix.

### Step 2.2: Score each node by weighted feature sum

For one graph:

```python
weights = node_feature_mtx * self.feature_importance_vector
weights = np.sum(weights, axis=1)
```

Interpretation:

- each active NSPPK feature contributes its learned global importance
- the node score is the sum over all feature contributions in that node’s row

So a node becomes important if it activates many globally important NSPPK features, or a few very strong ones.

### Step 2.3: Optional graph-local normalization

If `normalize=True`, node scores are divided by the maximum node score in that graph:

```python
max_weight = np.max(weights) if np.max(weights) > 0 else 1
normalized_weights = weights / max_weight
```

This means:

- the highest-scoring node in each graph gets score `1`
- all others are scaled relative to it

If `normalize=False`, raw weighted sums are kept.

### Step 2.4: Write node importance values

For each node:

```python
out_graph.nodes[node_idx][self.importance_key] = weight
```

So if `importance_key="importance"`, each node gets:

```python
graph.nodes[u]["importance"]
```

### Step 2.5: Derive edge importance values

Edges are scored multiplicatively:

```python
out_graph.edges[u, v][self.importance_key] = (
    out_graph.nodes[u][self.importance_key] *
    out_graph.nodes[v][self.importance_key]
)
```

So an edge is important only if both endpoint nodes are important.

This is a simple heuristic rather than a separately learned edge attribution model.

### Step 2.6: Attach raw feature vectors to nodes and edges

The transform also stores:

```python
out_graph.nodes[u]["node_feature"] = node_feature
out_graph.edges[u, v]["edge_feature"] = (
    out_graph.nodes[u]["node_feature"] +
    out_graph.nodes[v]["node_feature"]
)
```

This is useful for inspection and downstream visualization.

## 3. Feature Archive

`ImportanceNSPPK` can also recover representative subgraphs for important feature ids.

This part is handled by:

- [build_feature_archive(...)](/home/fabrizio/code/NSPPK/src/importance_nsppk.py#L72)
- [get_top_features(...)](/home/fabrizio/code/NSPPK/src/importance_nsppk.py#L104)

### Step 3.1: Replay the NSPPK feature extraction path

`build_feature_archive(...)` calls:

```python
get_feature_archive(...)
```

for each prepared graph.

This replays the structural hashing path and stores, for each final feature id:

- `feature_id`
- `raw_signature`
- `feature_kind`
- `source_graph_index`
- `root_node`
- `target_node`
- `source_node_ids`
- `source_edge_ids`
- `subgraph`

So instead of only knowing that “feature 517 is important”, you can inspect concrete occurrences of that feature in actual graphs.

### Step 3.2: Handle hash collisions explicitly

Because NSPPK uses bounded hashed feature ids, collisions can happen.

The archive therefore tracks:

- total occurrence count per feature id
- number of distinct `raw_signature` values per feature id

This is summarized in:

```python
self.feature_archive_summary_[feature_id] = {
    "occurrence_count": ...,
    "distinct_signature_count": ...,
}
```

If `distinct_signature_count > 1`, that feature bucket contains collisions.

### Step 3.3: Retrieve top features

`get_top_features(...)` ranks features by learned importance and returns entries such as:

- `feature_id`
- `importance`
- `occurrence_count`
- `distinct_signature_count`
- `representative_graph`
- `representative_occurrence`
- `source_graph_index`
- `occurrences`

This gives a bridge from:

- abstract hashed feature ids

to:

- actual representative subgraphs in actual training graphs

## 4. Attribute-Aware Graph Preparation

If the `NodeNSPPK` instance uses real-valued node attributes, `ImportanceNSPPK` mirrors the same preprocessing before archive construction.

That logic lives in [_prepare_graphs_for_node_nsppk(...)](/home/fabrizio/code/NSPPK/src/importance_nsppk.py#L29).

It applies:

- `embed_attributes(...)` if `attribute_dim` is active
- `set_discrete_labels(...)` if `attribute_alphabet_size` is active

So the archive reflects the same transformed feature space used by the node encoder.

## 5. Practical Interpretation

The meaning of a high node importance score is:

- this node activates NSPPK features that help discriminate the supervised labels

The meaning of a high feature importance score is:

- this hashed NSPPK feature column is consistently useful to the `ExtraTreesClassifier` across repeated train/test splits

The meaning of a high-ranked archived subgraph is:

- this is one representative occurrence of a structurally important NSPPK feature bucket

## 6. Important Caveats

### It is feature importance, not causal attribution

The method is based on tree-model feature importances over graph-level NSPPK vectors. It does not prove causal importance of a node or motif.

### It is global first, local second

The feature weights are learned globally across the dataset. Node scores are then derived by applying those global weights locally.

### Hash collisions can mix structures

One feature id can correspond to multiple raw signatures when `nbits` is small. This is why the archive tracks `distinct_signature_count`.

### The train/test split loop is for stability, not evaluation

`ImportanceNSPPK.fit(...)` repeatedly resamples the training set to stabilize the feature-importance vector. It is not a benchmark or a calibrated validation pipeline.

### Edge scores are heuristic

Edge importance is just the product of endpoint node importances.

## 7. Minimal Usage Pattern

The common workflow is:

```python
from nsppk import NodeNSPPK
from importance_nsppk import ImportanceNSPPK

node_encoder = NodeNSPPK(
    radius=1,
    distance=4,
    connector=1,
    nbits=12,
    dense=True,
    parallel=False,
)

importance_model = ImportanceNSPPK(
    node_nsppk=node_encoder,
    importance_key="importance",
    n_iter=3,
    n_estimators=300,
    quantile=0.7,
    parallel=False,
    normalize=True,
)

importance_model.fit(graphs, targets)
importance_graphs = importance_model.transform(graphs)
importance_model.build_feature_archive(graphs)
top_features = importance_model.get_top_features(top_k=10)
```

After this:

- `importance_graphs[i].nodes[u]["importance"]` is the node attribution score
- `importance_graphs[i].edges[u, v]["importance"]` is the derived edge score
- `top_features` contains representative important subgraphs

## 8. Mental Model

If you want one compact summary:

1. `NodeNSPPK` turns each node into a hashed structural feature vector
2. `ImportanceNSPPK` learns which NSPPK columns matter for the graph labels
3. node importance is the weighted sum of the important NSPPK columns active at that node
4. the archive lets you map important feature ids back to concrete subgraph occurrences

# Attributes Integration

This document explains how NSPPK integrates real-valued node attributes into the hashed structural representation.

The implementation lives mainly in [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py).

There are two different integration paths:

1. attribute preprocessing before hashing
2. attribute-structure interaction features during vectorization

These paths can be used separately or together, depending on `node_attribute_key`, `attribute_dim`, and `attribute_alphabet_size`.

## Overview

The relevant parameters are:

- `node_attribute_key`: the node field containing the real-valued attribute vector
- `attribute_dim`: optional reduced dimension for the attribute vector
- `attribute_alphabet_size`: optional number of discrete attribute clusters

The high-level flow is:

1. collect node attribute vectors from all fit graphs
2. optionally fit a dimensionality reducer
3. optionally fit a clustering model and a classifier for discrete labels
4. at transform time, optionally replace node attributes by embeddings
5. optionally overwrite node labels with predicted discrete attribute labels
6. build structural hashed features
7. if `node_attribute_key` is still present, combine the real-valued attribute matrix with the structural node-feature matrix

## 1. Raw Attribute Storage

NSPPK expects each node to carry a real-valued vector under `graph.nodes[node_idx][node_attribute_key]`.

The helper [get_node_attribute_matrix(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L463) extracts those node vectors and stacks them into a matrix:

```python
attribute_mtx = np.vstack([
    original_graph.nodes[node_idx][node_attribute_key]
    for node_idx in original_graph.nodes()
])
```

So if a graph has `N` nodes and attribute dimension `D`, then:

- `attribute_mtx.shape == (N, D)`

## 2. Fitting The Attribute Pipeline

During [BaseNSPPK.fit(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L821), NSPPK first concatenates attributes from all training graphs:

```python
attribute_mtx = np.vstack([
    graph.nodes[node_idx][self.node_attribute_key]
    for graph in graphs
    for node_idx in graph.nodes()
])
```

This gives one global training matrix over all nodes in all fit graphs.

### 2.1 Optional dimensionality reduction

If `attribute_dim` is set, NSPPK builds a `TruncatedSVD` instance in [NSPPK.__init__(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L986) and fits it on the stacked node attributes:

```python
self.embedder.fit(attribute_mtx)
attribute_mtx = self.embedder.transform(attribute_mtx)
```

After this step, the attribute vectors live in a reduced latent space.

### 2.2 Optional discretization

If `attribute_alphabet_size` is set, NSPPK builds:

- a `KMeans` clustering model
- an `ExtraTreesClassifier`

The logic is:

1. cluster the attribute vectors with `KMeans`
2. treat the cluster assignments as pseudo-labels
3. fit `ExtraTreesClassifier` to predict those pseudo-labels from attribute vectors

In code:

```python
targets = self.clustering_predictor.fit_predict(attribute_mtx)
self.classifier.fit(attribute_mtx, targets)
```

So the clustering model defines the attribute alphabet, and the classifier becomes the reusable mapping from attribute vector to discrete label.

## 3. Transform-Time Attribute Embedding

During [BaseNSPPK.transform(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L900), if `attribute_dim` is active, NSPPK first calls [embed_attributes(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L853).

For each graph:

1. stack node attributes into a matrix
2. apply the fitted embedder
3. copy the graph
4. store the original attribute vector under `original_<node_attribute_key>`
5. replace `node_attribute_key` with the embedded vector

In code terms:

```python
embeddings = self.embedder.transform(attribute_mtx)
out_graph.nodes[node_idx]['original_' + self.node_attribute_key] = old_attribute
out_graph.nodes[node_idx][self.node_attribute_key] = embedding
```

This means later stages will see the embedded attribute vector rather than the original one.

## 4. Transform-Time Discrete Label Injection

If `attribute_alphabet_size` is active, NSPPK next calls [set_discrete_labels(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L880).

For each graph:

1. stack the current node attributes
2. predict a discrete class for each node with the fitted classifier
3. copy the graph
4. write the predicted class into `graph.nodes[node_idx]['label']`

In code:

```python
labels = self.classifier.predict(attribute_mtx)
out_graph.nodes[node_idx]['label'] = label
```

This is important:

- these predicted labels become the node labels used by the hashing pipeline
- after this step, rooted hashes and paired hashes are built from attribute-derived discrete labels, not just from the original symbolic labels

So this path integrates real-valued attributes directly into the structural hashing stage.

## 5. Structural Hashing Still Runs Normally

After optional embedding and optional discrete relabeling, NSPPK continues with the usual structural encoder:

- node label hashing
- node hash construction
- rooted subgraph hashing
- paired rooted subgraph hashing
- optional edge-triplet features

That process is described in [docs/HASH.md](/home/fabrizio/code/NSPPK/docs/HASH.md).

If `attribute_alphabet_size` is active, the node labels used there may already be derived from the real-valued attribute vectors.

## 6. Attribute-Structure Interaction Features

A second integration path happens later in the vectorizer functions:

- [get_node_vectors(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L535)
- [get_node_graph_vectors(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L568)
- [get_graph_vector(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L595)

This path uses the real-valued node attribute matrix together with the structural node-feature matrix.

### 6.1 Structural node vectors

First NSPPK computes the structural node-feature matrix:

```python
node_vectors_mtx = get_structural_node_vectors(...)
```

If a graph has:

- `N` nodes
- `F = 2 ** nbits` structural columns

then:

- `node_vectors_mtx.shape == (N, F)`

### 6.2 Crossing attributes with structural features

If `node_attribute_key` is not `None`, [get_node_vectors(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L535) does:

```python
attribute_mtx = get_node_attribute_matrix(original_graph, node_attribute_key)
feature_node_vectors_mtx = node_vectors_mtx.todense().A
feature_node_vectors_mtx = attribute_mtx.T.dot(feature_node_vectors_mtx).dot(feature_node_vectors_mtx.T).T
feature_node_vectors_mtx = np.power(np.abs(feature_node_vectors_mtx), 1/3)
feature_node_vectors_mtx = csr_matrix(feature_node_vectors_mtx)
```

Conceptually:

1. start from node-level structural features
2. aggregate them through the real-valued attribute channels
3. project that back to node space
4. compress magnitude with a cube-root transform

This produces an interaction matrix that mixes:

- continuous attribute values
- hashed structural node patterns

The code then concatenates:

- the raw attribute matrix
- the interaction matrix
- optionally the original structural node matrix

So the final node vector matrix is:

```python
hstack([attribute_mtx, interaction_features, structural_features])
```

or, when `add_structural_node_information=False`:

```python
hstack([attribute_mtx, interaction_features])
```

## 7. Graph-Level Attribute Integration

For graph-level vectors, [get_graph_vector(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L595) uses a simpler aggregation when `node_attribute_key` is present:

```python
attribute_mtx = get_node_attribute_matrix(original_graph, node_attribute_key)
node_vectors_mtx = node_vectors_mtx.todense().A
vector_ = attribute_mtx.T.dot(node_vectors_mtx)
graph_vector = vector_.reshape(1, -1)
```

This means the graph representation is formed by aggregating structural node features through the attribute channels.

Without `node_attribute_key`, the graph vector is just:

```python
graph_vector = node_vectors_mtx.sum(axis=0)
```

So with real-valued attributes enabled, the graph representation changes from:

- simple sum over structural node features

to:

- attribute-weighted aggregation of structural node features

## 8. Node-Graph Kernel Variant

The node-kernel path in [get_node_graph_vectors(...)](/home/fabrizio/code/NSPPK/src/nsppk.py#L568) also uses the attribute-enriched node vectors:

```python
attribute_node_vectors_mtx = get_node_vectors(..., add_structural_node_information=False)
vector_ = attribute_node_vectors_mtx.T.dot(node_vectors_mtx)
vector = np.power(np.abs(vector_), 1/2)
```

So in this variant, the graph-level object is built from a bilinear interaction between:

- attribute-enriched node vectors
- structural node vectors

followed by a square-root magnitude compression.

## 9. Putting The Pieces Together

There are three distinct ways real-valued node attributes influence the final representation.

### Path A: embedding only

If you set:

- `node_attribute_key`
- `attribute_dim`

then NSPPK:

1. reduces the attribute vectors with SVD
2. uses the embedded vectors in later attribute-structure interaction computations

This changes the continuous attribute side, but it does not by itself change node labels.

### Path B: discretization into labels

If you set:

- `node_attribute_key`
- `attribute_alphabet_size`

then NSPPK:

1. learns an attribute-derived discrete alphabet
2. predicts one discrete label per node
3. injects those labels into `graph.nodes[node]['label']`
4. hashes subgraphs using those predicted labels

This directly changes the hashed structural representation.

### Path C: continuous attribute interactions

If you set:

- `node_attribute_key`

then the vectorizers use the attribute matrix to build cross-features with the structural node-feature matrix.

This changes the final numeric representation even if you do not discretize attributes into labels.

## 10. Most Important Practical Consequence

In NSPPK, real-valued node attributes are not integrated in only one place.

They can affect the model in two fundamentally different ways:

- as continuous values that weight or cross with hashed structural features
- as learned discrete labels that alter the hashing itself

So when both `node_attribute_key` and `attribute_alphabet_size` are active, the attributes influence:

1. which hashed structural patterns are produced
2. how those structural patterns are numerically aggregated

## 11. Caveats

Some implementation details are worth keeping in mind:

- `embed_attributes(...)` and `set_discrete_labels(...)` copy graphs before modification
- `get_node_vectors(...)` and `get_graph_vector(...)` densify the structural node matrix before certain attribute interactions
- the interaction transforms use magnitude compression:
  - cube root in `get_node_vectors(...)`
  - square root in `get_node_graph_vectors(...)`
- edge attributes are currently much less integrated than node attributes; the real-valued attribute integration described here is primarily about `node_attribute_key`

## 12. Minimal Mental Model

If you want one compact summary, it is this:

1. real-valued node attributes are optionally reduced and/or discretized
2. discretized attributes can become the node labels used by the hashing pipeline
3. continuous attributes are also crossed with the structural node-feature matrix to produce additional numeric features

That is how NSPPK combines real-valued node descriptors with hashed subgraph structure.

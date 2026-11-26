# Hashing Procedure

This document describes the hashing path implemented in [src/hash.py](../src/hash.py) and consumed by [src/nsppk.py](../src/nsppk.py).

The goal is to explain how NSPPK turns a labeled local subgraph pattern into a final integer feature id.

## Overview

There are four layers:

1. primitive hashes of node labels, edge labels, and tuples of values
2. local node hashes that summarize one node and its incident labeled edges
3. rooted subgraph hashes that summarize a ball around a root node up to radius `r`
4. final feature hashes that combine two rooted subgraphs, their graph distance, and optionally connector information

Optional simplex features pair fragments from endpoint neighborhoods and add fragments from expanded shortest-path unions.

## 1. Primitive Hash

The base hash function is `_hash(value)` in [src/hash.py](../src/hash.py).

It does:

1. convert `value` to `str(value)`
2. encode that string as bytes
3. compute a 4-byte BLAKE2s digest
4. convert the digest directly to an unsigned 32-bit integer

In code terms:

```python
digest = hashlib.blake2s(str(value).encode(), digest_size=4).digest()
hash_int = int.from_bytes(digest, "big")
```

The hash is deterministic for a fixed input. It returns the full unsigned 32-bit result without a separate mask or hexadecimal conversion. Every structural object is ultimately hashed through the string form of a Python value or tuple, so values such as `1` and `"1"` still share the same input representation.

## 2. Helper Combinators

The next helpers define how multiple hashed objects are merged.

### `hash_sequence(iterable)`

Defined in [src/hash.py](../src/hash.py).

- converts the input to a tuple
- preserves order
- applies `_hash(tuple(iterable))`

Use this when order matters.

### `hash_set(iterable)`

Defined in [src/hash.py](../src/hash.py).

- sorts the input
- converts it to a tuple
- applies `_hash(tuple(sorted_iterable))`

Use this when order should not matter.

This is how NSPPK gets permutation invariance for neighbor sets and isodistance sets.

## 3. Node Label Hash

During structural preprocessing in [_prepare_structural_graph(...)](../src/nsppk.py), each node gets:

```python
node_label_hash = _hash(graph.nodes[node_idx]['label'])
```

This is the hash of the raw discrete node label only.

## 4. Extended Node Hash

The next level is `node_hash(node_idx, graph)` in [src/hash.py](../src/hash.py).

For an undirected simple graph, the existing calculation for node `u` is:

1. `uh = _hash(label(u))`
2. for each neighbor `v`, compute
   - `_hash(label(v))`
   - `_hash(label(u,v))`
   - `_hash((_hash(label(v)), _hash(label(u,v))))`
3. collect those per-neighbor edge codes into a multiset-like list
4. combine them with `hash_set(...)`
5. combine the center-node label hash and the neighbor summary with `_hash((uh, nh))`

In compact form:

```python
uh = _hash(label(u))
edges_h = [
    _hash((_hash(label(v)), _hash(label(u, v))))
    for v in neighbors(u)
]
nh = hash_set(edges_h)
ext_node_h = _hash((uh, nh))
```

This produces one hash per node that depends on:

- the node label
- the labels of neighboring nodes
- the labels of incident edges

but not on neighbor ordering.

For directed graphs, each incident edge pattern also records whether the edge is incoming or outgoing relative to `u`. For multigraphs, every keyed edge contributes its own pattern, so parallel-edge multiplicity and labels affect the local hash. The distance layers described below use the undirected projection of a directed graph.

## 5. Edge Triplet Hash

For each undirected edge `(u, v)`, `edge_triplet_hash(u, v, graph)` computes:

```python
hash_set([
    _hash(label(u)),
    _hash(label(u, v)),
    _hash(label(v)),
])
```

This is symmetric in `u` and `v` because `hash_set(...)` sorts before hashing. Directed edges instead hash the ordered source label, edge label, and target label. Each keyed parallel edge is hashed separately.

These values are precomputed once in `precompute_edge_triplet_hashes(graph)`.

## 6. Rooted Subgraph Hashes

The rooted subgraph logic lives in `rooted_graph_hashes(...)` in [src/hash.py](../src/hash.py).

Take a root node `u` and a maximum radius `R`.

### Step 6.1: Gather nodes by distance

Using BFS distances from `u`, the code groups nodes by their distance from the root:

- distance `0`: just the root
- distance `1`: immediate neighbors
- distance `2`: nodes two hops away
- and so on up to `R`

### Step 6.2: Hash each isodistance layer

For each distance layer:

- use the node-label hash for the root layer (`dist == 0`)
- use the extended node hash for all non-root layers (`dist > 0`)

Then combine the whole layer with `hash_set(...)`.

So the code builds:

```python
iso_distance_codes_list = [
    hash_set([
        node_label_hashes[curr_node_idx] if dist == 0 else node_hashes[curr_node_idx]
        for curr_node_idx in node_idxs
    ])
    for dist, node_idxs in sorted(dist_to_node_idxs_dict.items())
]
```

This gives one hash per distance shell.

### Step 6.3: Build cumulative rooted hashes

For radius `1`, hash the shell codes from distance `0` to `1`.

For radius `2`, hash the shell codes from distance `0` to `2`.

And so on.

That is:

```python
h_list = [
    hash_sequence(iso_distance_codes_list[:i])
    for i in range(1, len(iso_distance_codes_list) + 1)
]
```

So `h_list[r]` is the rooted subgraph hash for the ball of radius `r` around `u`.

Important detail:

- shell order matters, so `hash_sequence(...)` is used here
- within one shell, node order does not matter, so `hash_set(...)` is used there

## 7. Structural Preprocessing Cache

Before feature extraction, [_prepare_structural_graph(...)](../src/nsppk.py) computes and stores:

- `node_label_hashes`
- `node_hashes`
- `edge_triplet_hashes`
- `rooted_graph_hashes`

This means feature extraction reuses hashes instead of recomputing them inside the inner loop.

## 8. From Rooted Hashes To Final Feature Signatures

Feature generation happens in [_process_node_features(...)](../src/nsppk.py).

There are two main feature families.

### A. Paired rooted subgraph features

For a root node `u` and another node `v` at graph distance `dist`:

1. choose a rooted radius `radius_i` around `u`
2. choose a rooted radius `radius_j` around `v`
3. read the cached rooted hashes:
   - `code_i = rooted_hash(u, radius_i)`
   - `code_j = rooted_hash(v, radius_j)`

If `connector == 0`, the raw signature is:

```python
raw_signature = hash_sequence([code_i, dist, code_j])
```

If `connector > 0`, NSPPK also summarizes the union of shortest paths between `u` and `v`.

### B. Simplex fragment features

When `use_simplex_features=True`, NSPPK enumerates the dimensions selected by `simplex_feature_order` in each induced endpoint neighborhood. A scalar `n` selects all dimensions from 0 through `n`; an inclusive tuple `(min, max)` selects only those dimensions. A dimension-`k` simplex contains `k + 1` distinct vertices. Clique membership is determined from the graph's traversal projection; for directed graphs this is the underlying undirected projection.

A singleton's structural simplex hash is its own node-label hash. For a clique with two or more vertices, NSPPK reuses each edge's cached triplet hash and hashes the collection with `hash_set(...)`:

```python
simplex_hash = hash_set(edge_triplet_hashes_in_clique)
```

An undirected edge triplet hash keeps endpoint labels unordered while retaining the edge label's role:

```python
edge_triplet_hash = hash_sequence((
    hash_set((node_label_hash_u, node_label_hash_v)),
    edge_label_hash,
))
```

For a directed edge, source and target order is retained:

```python
edge_triplet_hash = hash_sequence((
    "directed-edge", source_label_hash, edge_label_hash, target_label_hash,
))
```

`hash_set(...)` sorts its inputs but retains duplicate values. Thus parallel edges do not create extra simplices, but their edge hashes contribute with their multiplicity to a clique's structural hash.

At distance zero, NSPPK emits a unary feature per simplex in the root's selected neighborhood. Its context is the root's node-label hash and the selected radius:

```python
unary_signature = hash_sequence((node_label_hash_u, radius, simplex_hash))
```

At positive distance, NSPPK enumerates cliques in both endpoint neighborhoods for each selected radius combination, then takes their union by clique vertex set. A clique present in both neighborhoods is emitted once for that combination. Distinct cliques with the same structural hash still each contribute. The feature context uses the unordered pair of root label hashes and the distance:

```python
pair_context_hash = hash_set((node_label_hash_u, node_label_hash_v))
pair_signature = hash_sequence((pair_context_hash, distance, simplex_hash))
```

This removes the Cartesian product of endpoint fragment lists. The same unordered union and root-pair context applies when the roots are visited in reverse order. Distances still use the undirected traversal projection for directed graphs.

For each connector thickness, a simplex in the expanded shortest-path-union region is contextualized by the unordered pair of root label hashes, the root distance, and the connector thickness:

```python
path_union_signature = hash_sequence((
    hash_set((node_label_hash_u, node_label_hash_v)),
    distance,
    connector_thickness,
    simplex_hash,
))
```

Fragments are cached by induced node set and selected dimension range on the prepared graph, so repeated neighborhoods or path regions reuse the same enumeration. The legacy `simplex_feature_max_order` parameter name remains accepted as an alias.

That connector summary is built in `_connector_provenance(...)` as:

```python
union_signature = hash_set([
    hash_sequence([node_distance_from_u, rooted_hash(node, connect - 1)])
    for node in union_of_shortest_paths
])
```

Then the final raw signature becomes:

```python
raw_signature = hash_sequence([code_i, dist, code_j, union_of_shortest_paths_code])
```

Finally it is projected to a feature id:

```python
paired_code = hash_value(raw_signature, nbits=nbits)
```

## 9. Final Projection Into Feature Space

The last step is `hash_value(value, nbits)` in [src/hash.py](../src/hash.py):

```python
max_index = 2 ** nbits
h = masked_hash_value(value, max_index - 3)
h += 2
```

This does two things:

1. reduce the raw signature hash into the target feature-space width `2 ** nbits`
2. reserve columns `0` and `1`

The encoder accepts `nbits` from 2 through 30. The primitive hash returns 32 bits; final feature projection supports the configured width through 30 bits and reserves columns `0` and `1`.

So final structural feature ids always lie in:

- `2 .. (2 ** nbits - 1)`

and columns:

- `0`: constant bias feature
- `1`: node degree

are inserted separately in `get_structural_node_vectors(...)`.

## 10. Worked Shape Of One Final Subgraph Hash

For a paired rooted feature without connector, the pipeline is:

1. hash raw node labels and edge labels
2. build node hashes for nodes in the graph
3. build rooted hashes for `u` and `v`
4. choose one `(radius_i, radius_j, dist)` combination
5. form

```python
raw_signature = hash_sequence([rooted_hash(u, radius_i), dist, rooted_hash(v, radius_j)])
```

6. project with

```python
feature_id = hash_value(raw_signature, nbits)
```

For a paired rooted feature with connector thickness `c > 0`, add one more term:

```python
raw_signature = hash_sequence([
    rooted_hash(u, radius_i),
    dist,
    rooted_hash(v, radius_j),
    connector_summary_hash,
])
```

Simplex features use the signatures above and are then mapped into the configured feature space with `hash_value(signature, nbits)`. This replaces the previous node-permutation canonicalization and changes feature IDs for simplex-enabled encoders. The revised undirected edge triplet hash is also used in connector summaries for undirected multigraphs, so those feature IDs can change as well. Models using either affected representation must be retrained.

When the root distance is zero, endpoint neighborhoods are the same region, so NSPPK emits each simplex fragment once as a unary feature rather than pairing it with itself:

```python
raw_signature = hash_sequence((root_label_hash, radius, simplex_hash))
```

At distance zero the root is also the singleton's node, so its label hash appears both as root context and as the singleton structural hash. The radius selects the neighborhood used to enumerate the fragment.

## 11. Invariance Properties

The implementation gets its invariances from the choice between `hash_set(...)` and `hash_sequence(...)`.

- permutation invariance inside neighbor collections and shell collections comes from `hash_set(...)`
- dependence on shell order, radius growth, and tuple position comes from `hash_sequence(...)`
- simplex neighborhood-pair features use a union of both endpoint fragment sets and are symmetric under endpoint swaps
- rooted NSPPK features are generally ordered as `[left_subgraph, distance, right_subgraph, ...]`

## 12. Important Caveat

The final feature id is not a canonical subgraph identifier. It is a hashed projection into a bounded feature space.

That means:

- different raw signatures can collide after `hash_value(..., nbits)`
- smaller `nbits` increases collision rate
- hashing starts from Python's string representation of each label; use stable, ordinary scalar labels when reproducible hashes across machines or custom label classes matter
- the `raw_signature` values stored by the feature archive in [src/nsppk.py](../src/nsppk.py) are more informative than the final projected feature ids

If you want to inspect actual provenance for a feature id, use the archive path rather than treating the final projected id as uniquely identifying one structure.

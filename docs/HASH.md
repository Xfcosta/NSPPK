# Hashing Procedure

This document describes the hashing path implemented in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py) and consumed by [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py).

The goal is to explain how NSPPK turns a labeled local subgraph pattern into a final integer feature id.

## Overview

There are four layers:

1. primitive hashes of node labels, edge labels, and tuples of values
2. local node hashes that summarize one node and its incident labeled edges
3. rooted subgraph hashes that summarize a ball around a root node up to radius `r`
4. final feature hashes that combine two rooted subgraphs, their graph distance, and optionally connector information

There is also an edge-triplet feature path that hashes a distance plus one labeled edge neighborhood.

## 1. Primitive Hash

The base hash function is `_hash(value)` in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py).

It does:

1. convert `value` to `str(value)`
2. encode that string as bytes
3. compute `sha256`
4. convert the digest to an integer
5. keep only the low 30 bits

In code terms:

```python
value_str = str(value)
value_bytes = value_str.encode()
sha256_hash = hashlib.sha256(value_bytes).hexdigest()
hash_int = int(sha256_hash, 16) & (2**30 - 1)
```

So every structural object is ultimately hashed through the string form of a Python value or tuple.

## 2. Helper Combinators

The next helpers define how multiple hashed objects are merged.

### `hash_sequence(iterable)`

Defined in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py).

- converts the input to a tuple
- preserves order
- applies `_hash(tuple(iterable))`

Use this when order matters.

### `hash_set(iterable)`

Defined in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py).

- sorts the input
- converts it to a tuple
- applies `_hash(tuple(sorted_iterable))`

Use this when order should not matter.

This is how NSPPK gets permutation invariance for neighbor sets and isodistance sets.

## 3. Node Label Hash

During structural preprocessing in [_prepare_structural_graph(...)](/home/fabrizio/code/NSPPK/src/nsppk.py), each node gets:

```python
node_label_hash = _hash(graph.nodes[node_idx]['label'])
```

This is the hash of the raw discrete node label only.

## 4. Extended Node Hash

The next level is `node_hash(node_idx, graph)` in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py).

For one node `u`, it computes:

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

## 5. Edge Triplet Hash

For each edge `(u, v)`, `edge_triplet_hash(u, v, graph)` computes:

```python
hash_set([
    _hash(label(u)),
    _hash(label(u, v)),
    _hash(label(v)),
])
```

This is symmetric in `u` and `v` because `hash_set(...)` sorts before hashing.

These values are precomputed once in `precompute_edge_triplet_hashes(graph)`.

## 6. Rooted Subgraph Hashes

The rooted subgraph logic lives in `rooted_graph_hashes(...)` in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py).

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

Before feature extraction, [_prepare_structural_graph(...)](/home/fabrizio/code/NSPPK/src/nsppk.py) computes and stores:

- `node_label_hashes`
- `node_hashes`
- `edge_triplet_hashes`
- `rooted_graph_hashes`

This means feature extraction reuses hashes instead of recomputing them inside the inner loop.

## 8. From Rooted Hashes To Final Feature Signatures

Feature generation happens in [_process_node_features(...)](/home/fabrizio/code/NSPPK/src/nsppk.py).

There are two main feature families.

### A. Edge-triplet features

For a root node `u`, a reachable target node `t`, and one neighbor `n` of `t`:

1. compute graph distance `dist(u, t)`
2. read the precomputed edge-triplet hash for edge `(t, n)`
3. build the raw signature:

```python
raw_signature = hash_sequence([dist, triplet_hash])
```

4. map that raw signature to a feature id:

```python
distance_triplet_hash = hash_value(raw_signature, nbits=nbits)
```

This is an edge-centered structural feature anchored at the root node through the distance term.

### B. Paired rooted subgraph features

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

The last step is `hash_value(value, nbits)` in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py):

```python
max_index = 2 ** nbits
h = masked_hash_value(value, max_index - 3)
h += 2
```

This does two things:

1. reduce the raw signature hash into the target feature-space width `2 ** nbits`
2. reserve columns `0` and `1`

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

For an edge-triplet feature:

```python
raw_signature = hash_sequence([dist, edge_triplet_hash])
feature_id = hash_value(raw_signature, nbits)
```

## 11. Invariance Properties

The implementation gets its invariances from the choice between `hash_set(...)` and `hash_sequence(...)`.

- permutation invariance inside neighbor collections and shell collections comes from `hash_set(...)`
- dependence on shell order, radius growth, and tuple position comes from `hash_sequence(...)`
- edge triplets are symmetric because they use `hash_set(...)`
- paired rooted features are generally ordered as `[left_subgraph, distance, right_subgraph, ...]`

## 12. Important Caveat

The final feature id is not a canonical subgraph identifier. It is a hashed projection into a bounded feature space.

That means:

- different raw signatures can collide after `hash_value(..., nbits)`
- smaller `nbits` increases collision rate
- the `raw_signature` values stored by the feature archive in [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py) are more informative than the final projected feature ids

If you want to inspect actual provenance for a feature id, use the archive path rather than treating the final projected id as uniquely identifying one structure.

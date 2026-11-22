# TODO

## Performance work status

This board separates completed optimizations from work that remains. Historical benchmark results are evidence for the tested workloads, not a current profile of every dataset or configuration.

## Solved

- [x] **Use BLAKE2s for the structural hash.** [`src/hash.py`](src/hash.py) hashes the existing string representation to a four-byte digest and converts it directly to a 32-bit integer. On the 14-bit HIV comparison, a regularized logistic model with average-precision tuning gave similar average precision to SHA-256: 0.6230 vs 0.6232 across five paired 70/30 splits. ROC-AUC and balanced accuracy were also close. Three full transforms had median times of 14.46 s for BLAKE2s and 15.61 s for SHA-256. Hash outputs differ from version 1.2.1.

- [x] **Remove shortest-path enumeration from the normal feature path.** [`_prepare_structural_graph`](src/nsppk.py) builds predecessor data before [`_process_node_features`](src/nsppk.py) runs. [`_shortest_path_edge_union`](src/nsppk.py) walks and memoizes the predecessor DAG to collect shortest-path unions. `nx.all_shortest_paths()` remains only as a fallback when that helper receives a graph without the internal BFS cache; normal encoder calls prepare the cache first.

- [x] **Reuse bounded BFS results during feature extraction.** The per-graph cache supplies distance layers to rooted hashes and feature extraction, and supplies neighborhood balls to `_node_ball()`.

- [x] **Use one BFS traversal per source.** `_build_bfs_cache()` now obtains distances and predecessor lists together from `nx.predecessor(..., return_seen=True)`. Radius zero is handled directly because NetworkX treats `cutoff=0` as unbounded in this API.

- [x] **Build structural node matrices directly as CSR.** `get_structural_node_vectors()` accumulates row, column, and data entries, then constructs one CSR matrix rather than allocating a LIL matrix per node.

- [x] **Reduce avoidable allocations in hash-heavy loops.** Feature signatures now use tuple literals instead of temporary lists, set hashes consume generators where possible, rooted distance signatures reuse tuple prefixes, and three-value edge triplets use a sorting network instead of allocating and sorting a list. These changes preserve the tuple-based hash representation.

## Open priorities

- [ ] **Measure current bottlenecks.** Profile hashing, BFS and predecessor-union work, sparse assembly, graph copying, and multiprocessing overhead on representative graph sets using the current BLAKE2s implementation.

- [ ] **Reduce graph copying.** `_prepare_structural_graph()` still copies each input graph. Explore a compact internal representation or safe temporary-state strategy while preserving the caller's graph objects.

- [ ] **Measure remaining `hash_set()` sorting cost.** General variable-length unordered collections still require sorting for permutation-invariant hashes. Profile the remaining calls before considering a broader canonicalization change, and recheck predictive quality if hash semantics change.

- [ ] **Measure the `connector` quality/runtime tradeoff.** Compare `connector=0` with positive values on the same train/test splits and report runtime with predictive metrics.

- [ ] **Compare parallel execution strategies.** Benchmark serial encoding, current graph-subset multiprocessing, and a persistent worker pool. Include pool startup and graph serialization costs.

## Longer-term options

- [ ] Prototype compact adjacency arrays, then compare traversal and end-to-end performance with NetworkX.
- [ ] If NetworkX remains a bottleneck, evaluate `rustworkx` and `igraph` on the same workload.
- [ ] If profiling still shows Python traversal dominates, prototype a compiled core for bounded BFS, predecessor-union extraction, and feature accumulation using Numba, Cython, Rust, or C++.

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

- [x] **Measure current bottlenecks.** `benchmarks/profile_current_bottlenecks.py` profiles the current BLAKE2s path on deterministic HIV graph samples (`radius=1`, `distance=4`, `connector=1`, `nbits=14`). On 64 graphs (median 24 nodes, max 68), cProfile counted 857k `_hash()` calls; its self time was 0.444 s in a 2.410 s profiled run. `_process_node_features()` took 2.232 s cumulative, led by `_connector_provenance()` at 0.969 s; `_induced_edge_ids()` used 0.507 s and `_shortest_path_edge_union()` 0.105 s. In one unprofiled instrumented transform, graph copy took 0.0027 s, BFS cache 0.0104 s, CSR constructors 0.0290 s, Counter plus CSR row assembly 0.0114 s, and final `vstack` 0.0002 s. On 128 graphs, 18-worker pool startup/close was 0.143 s and dill input chunks totaled 186 KiB. Serial / parallel transform medians were 0.213 / 0.213 s at 18 graphs, 0.224 / 0.141 s at 19, 0.398 / 0.144 s at 36, 0.937 / 0.250 s at 72, and 1.668 / 0.251 s at 128. Profiling points toward connector provenance and hash/sorting work; BFS and graph copying are small on these molecules. cProfile cumulative values overlap, and the timing conclusions apply to this HIV workload and machine.

- [ ] **Reduce graph copying.** `_prepare_structural_graph()` still copies each input graph. Explore a compact internal representation or safe temporary-state strategy while preserving the caller's graph objects.

- [ ] **Measure remaining `hash_set()` sorting cost.** General variable-length unordered collections still require sorting for permutation-invariant hashes. Profile the remaining calls before considering a broader canonicalization change, and recheck predictive quality if hash semantics change.

- [ ] **Measure the `connector` quality/runtime tradeoff.** Compare `connector=0` with positive values on the same train/test splits and report runtime with predictive metrics.

- [x] **Compare parallel execution strategies.** On cached HIV NetworkX graphs (`radius=1`, `distance=4`, `connector=1`, `nbits=14`, 18 workers), the initial four-repeat medians for serial / per-call pool / persistent pool were 0.979 / 0.263 / 0.165 s at 128 graphs, 2.403 / 0.482 / 0.296 s at 256, and 5.431 / 0.793 / 0.831 s at 512. An interleaved six-repeat recheck measured 0.265 / 0.134 s (per-call / persistent) at 128 and 1.069 / 0.769 s at 512; the latter had substantial run-to-run variance. Outputs matched exactly. Pool startup was 38–51 ms; serialized graph chunks were 117 KiB at 128 graphs and 575 KiB at 512, and graph transfer plus sparse-result collection were included in pool timings. Current multiprocessing beat serial for tested batches. A persistent pool reduced median time in the interleaved recheck, but variance and its lifecycle/API cost do not justify changing the normal one-shot transform path. Dispatch-boundary medians (8 interleaved repeats) were serial 0.166 s at 18 graphs, then parallel-requested 0.151 / 0.150 / 0.152 / 0.152 s at 19 / 24 / 32 / 48 graphs. No parallel API change made.

## Longer-term options

- [x] **Prototype compact CSR adjacency and compare it with NetworkX.** `benchmarks/compact_adjacency_prototype.py` converts each graph to a SciPy CSR matrix, computes bounded unweighted distances, and reconstructs all shortest-path predecessors required by connector features. On a deterministic sample of 64 cached HIV graphs (median 24 nodes, maximum 68; `radius=1`, `distance=4`, `connector=1`, `nbits=14`), BFS cache medians were 0.0092 s for NetworkX and 0.0756 s for CSR. Full structural node-feature extraction was 0.8553 s vs 0.9237 s. Distances, predecessor sets, neighborhood balls, and output matrices matched exactly. Traversal-only Barabasi-Albert cases (`m=2`) also favored NetworkX at 250/500/1000 nodes: 0.0315/0.1024/0.3152 s vs CSR's 0.0519/0.1510/0.4721 s. For these tested workloads, conversion and reconstruction erased the expected traversal gain, so do not adopt this adapter. Revisit only with a broader array-native feature path that avoids rebuilding Python dictionaries, and a demonstrated target workload where NetworkX is still the bottleneck.
- [ ] If NetworkX remains a bottleneck, evaluate `rustworkx` and `igraph` on the same workload.
- [ ] If profiling still shows Python traversal dominates, prototype a compiled core for bounded BFS, predecessor-union extraction, and feature accumulation using Numba, Cython, Rust, or C++.

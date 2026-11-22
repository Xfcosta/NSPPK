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

- [x] **Amortize connector provenance and archive-only occurrence work.** Connector signatures and provenance are now computed once per source/target/depth instead of inside the rooted-radius cross product. Normal feature transforms no longer build occurrence node/edge sets or shortest-path edge unions they never use; connector-depth-1 avoids redundant radius-zero ball expansions, and induced archive edges are visited once per undirected edge. Removed a dead sort-and-build that was immediately overwritten. `hash_value()` now applies its mask inline. On an interleaved 64-graph HIV benchmark (`radius=1`, `distance=4`, `connector=1`, `nbits=14`), median structural extraction fell from 0.8792 s to 0.3722 s (2.36x faster). All 64 sparse feature matrices matched exactly; connector depths 0–2, weighted/unweighted vectors, and a 1,888-occurrence archive fixture also matched the prior implementation.

- [x] **Retune the graph-level multiprocessing threshold after reducing per-graph cost.** On this 18-worker machine, parallel startup/close costs about 0.14–0.16 s. Interleaved serial/parallel medians were 0.118/0.139 s at 19 graphs, 0.157/0.141 s at 24, 0.177/0.139 s at 28, 0.240/0.144 s at 36, 0.293/0.142 s at 48, and 0.382/0.142 s at 64. The graph-level encoder now waits for roughly 1.5 graphs per worker before starting a pool, avoiding the measured 19-graph regression while retaining larger-batch gains.

## Open priorities

- [x] **Measure current bottlenecks.** `benchmarks/profile_current_bottlenecks.py` profiles the current BLAKE2s path on deterministic HIV graph samples (`radius=1`, `distance=4`, `connector=1`, `nbits=14`). Before the connector changes, a 64-graph profile counted 857k `_hash()` calls, 89.7k `hash_set()` calls, and 1.88m `sorted()` calls. Afterward, those counts fell to 554k, 25.9k, and 79.7k. The current profile puts `_hash()` at 0.382 s cumulative and `_connector_provenance()` at 0.124 s; BFS cache construction is 0.025 s, graph copies 0.006 s, and instrumented CSR constructors 0.029 s. cProfile cumulative values overlap. The results describe this HIV sample and machine; the remaining profile points toward hash and feature-pair computation rather than graph copying or BFS.

- [ ] **Reduce graph copying.** `_prepare_structural_graph()` still copies each input graph. Explore a compact internal representation or safe temporary-state strategy while preserving the caller's graph objects.

- [ ] **Measure remaining `hash_set()` sorting cost.** General variable-length unordered collections still require sorting for permutation-invariant hashes. Profile the remaining calls before considering a broader canonicalization change, and recheck predictive quality if hash semantics change.

- [ ] **Measure the `connector` quality/runtime tradeoff.** Compare `connector=0` with positive values on the same train/test splits and report runtime with predictive metrics.

- [x] **Compare parallel execution strategies.** On cached HIV NetworkX graphs (`radius=1`, `distance=4`, `connector=1`, `nbits=14`, 18 workers), the initial four-repeat medians for serial / per-call pool / persistent pool were 0.979 / 0.263 / 0.165 s at 128 graphs, 2.403 / 0.482 / 0.296 s at 256, and 5.431 / 0.793 / 0.831 s at 512. An interleaved six-repeat recheck measured 0.265 / 0.134 s (per-call / persistent) at 128 and 1.069 / 0.769 s at 512; the latter had substantial run-to-run variance. Outputs matched exactly. Pool startup was 38–51 ms; serialized graph chunks were 117 KiB at 128 graphs and 575 KiB at 512, and graph transfer plus sparse-result collection were included in pool timings. Current multiprocessing beat serial for tested batches. A persistent pool reduced median time in the interleaved recheck, but variance and its lifecycle/API cost do not justify changing the normal one-shot transform path. Dispatch-boundary medians (8 interleaved repeats) were serial 0.166 s at 18 graphs, then parallel-requested 0.151 / 0.150 / 0.152 / 0.152 s at 19 / 24 / 32 / 48 graphs. No parallel API change made.

## Longer-term options

- [x] **Prototype compact CSR adjacency and compare it with NetworkX.** `benchmarks/compact_adjacency_prototype.py` converts each graph to a SciPy CSR matrix, computes bounded unweighted distances, and reconstructs all shortest-path predecessors required by connector features. On a deterministic sample of 64 cached HIV graphs (median 24 nodes, maximum 68; `radius=1`, `distance=4`, `connector=1`, `nbits=14`), BFS cache medians were 0.0092 s for NetworkX and 0.0756 s for CSR. Full structural node-feature extraction was 0.8553 s vs 0.9237 s. Distances, predecessor sets, neighborhood balls, and output matrices matched exactly. Traversal-only Barabasi-Albert cases (`m=2`) also favored NetworkX at 250/500/1000 nodes: 0.0315/0.1024/0.3152 s vs CSR's 0.0519/0.1510/0.4721 s. For these tested workloads, conversion and reconstruction erased the expected traversal gain, so do not adopt this adapter. Revisit only with a broader array-native feature path that avoids rebuilding Python dictionaries, and a demonstrated target workload where NetworkX is still the bottleneck.
- [ ] If NetworkX remains a bottleneck, evaluate `rustworkx` and `igraph` on the same workload.
- [ ] If profiling still shows Python traversal dominates, prototype a compiled core for bounded BFS, predecessor-union extraction, and feature accumulation using Numba, Cython, Rust, or C++.

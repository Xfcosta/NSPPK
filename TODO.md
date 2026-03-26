# TODO

## Speed Up Graph Vectorization

This project is currently dominated by Python-side graph feature extraction work in [nsppk.py](/home/fabrizio/code/NSPPK/nsppk.py), not by the classifier. The main hotspots are repeated shortest-path traversals, expensive hashing, NetworkX object overhead, and inefficient sparse matrix construction.

## Highest-Value Immediate Changes

- Replace cryptographic hashing in `_hash()` with a much cheaper non-cryptographic integer hash.
  - Current code uses `hashlib.sha256` for every hashed object.
  - Feature hashing does not require cryptographic security.
  - Likely one of the easiest large wins.
  - Evaluated and rejected for the current benchmark: replacing `sha256` with a cheaper structural hash hurt learning-curve quality.
  - Final accepted state keeps the original `sha256`-based `_hash()` in [src/hash.py](/home/fabrizio/code/NSPPK/src/hash.py).

- Remove `nx.all_shortest_paths()` from the inner loop in `_process_node_features()`.
  - Current code recomputes all shortest paths for many source/target pairs.
  - This is likely the single most expensive graph-algorithm call in the encoder.
  - Precompute predecessor maps from one BFS per source node and derive shortest-path unions from those instead.
  - Done: replaced repeated `nx.all_shortest_paths()` calls with cached predecessor maps and derived shortest-path unions in [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py).

- Reuse BFS results instead of recomputing them.
  - `rooted_graph_hashes()` runs `nx.single_source_shortest_path_length`.
  - `_process_node_features()` runs another `nx.single_source_shortest_path_length`.
  - Cache bounded BFS results per source node once per graph and reuse them in both places.
  - Done: added a per-graph bounded BFS cache in [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py) and rewired `rooted_graph_hashes()`, `_process_node_features()`, and `_node_ball()` to reuse it.

- Build CSR matrices directly instead of creating many `lil_matrix` objects.
  - `items_to_sparse_histogram()` and `weighted_sparse_histogram()` create a sparse matrix per node using LIL format.
  - Instead accumulate `rows`, `cols`, `data` arrays and build one CSR matrix per graph.
  - This should reduce allocator overhead and Python-side sparse mutation cost.
  - Done: `get_structural_node_vectors()` now accumulates row entries directly and builds one CSR matrix per graph in [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py).

- Avoid copying the entire graph in `get_structural_node_vectors()`.
  - `graph = original_graph.copy()` adds overhead for every graph.
  - Either cache temporary attributes in place and clean them up, or build a compact internal representation once.
  - Partial: reduced repeated NetworkX node/edge/neighbor lookups by introducing a graph-local structural cache in [src/nsppk.py](/home/fabrizio/code/NSPPK/src/nsppk.py), but `original_graph.copy()` is still in place.

## Medium-Sized Refactors

- Convert NetworkX graphs into a compact internal adjacency representation before feature extraction.
  - Suggested representation:
    - `indptr`
    - `indices`
    - node label array
    - edge label array
    - degree array
  - This will reduce repeated dictionary lookups like `graph.nodes[...]` and `graph.edges[...]`.

- Precompute and cache more graph-local quantities.
  - Node label hashes
  - Node hashes
  - Edge triplet hashes
  - Rooted graph hashes
  - Distance layers per source node
  - Predecessor maps per source node if `connector > 0`

- Reduce repeated sorting during hashing.
  - `hash_set()` sorts inputs repeatedly.
  - If the calling code already has a stable order, avoid re-sorting.
  - Consider canonical integer encodings that do not require repeated tuple and sort creation.

- Reduce Python object creation in inner loops.
  - Avoid building many temporary tuples/lists inside nested loops.
  - Reuse buffers where possible.
  - Replace repeated `hash_sequence([a, b, c])` patterns with cheaper integer-mixing utilities.

- Review whether `connector > 0` is worth the runtime cost.
  - Connector handling is much more expensive than the rest of the feature extraction.
  - If benchmark quality is similar with `connector = 0`, that may be the most effective runtime/quality tradeoff.

## Parallelism Improvements

- Revisit multiprocessing strategy in `paired_*_vector_encoder`.
  - Current code parallelizes across graph subsets with `multiprocessing_on_dill`.
  - This may pay substantial serialization cost for many small graphs.
  - Benchmark:
    - serial
    - graph-level multiprocessing
    - source-node-level parallelism within large graphs

- Use a persistent worker pool across repeated benchmark runs if possible.
  - Current notebook sweeps `nbits` across many runs.
  - Recreating pools repeatedly can add avoidable overhead.

- Consider joblib or standard multiprocessing if `multiprocessing_on_dill` is not required.
  - Dill-based serialization is flexible but can be slower.

## Better Backends Than NetworkX

- Evaluate `rustworkx` as the first replacement candidate.
  - Likely the best balance of Python usability and lower graph traversal overhead.
  - Good fit for many small graph traversals.

- Evaluate `igraph` as another strong candidate.
  - Compact internal representation.
  - Fast BFS / shortest path implementations.

- Evaluate `NetworKit` if larger-scale graph algorithm performance becomes important.

- Consider `graph-tool` only if the environment can tolerate a heavier installation story.

## Array-Based / Compiled Rewrite Path

- Build a custom graph backend using adjacency arrays and compiled traversal kernels.
  - Store all graphs in compact arrays in memory.
  - Run bounded BFS directly over arrays.
  - Parallelize across graphs or across source nodes.
  - This is the closest practical equivalent to “NumPy for graphs” for this workload.

- If performance remains critical, move the feature extractor core to:
  - Numba
  - Cython
  - Rust extension
  - C++ extension

- Best candidate for compiled implementation:
  - bounded BFS
  - predecessor-map construction
  - shortest-path union extraction
  - feature code accumulation

## Benchmarking / Profiling Tasks

- Add profiling around:
  - `_hash()`
  - `rooted_graph_hashes()`
  - `_process_node_features()`
  - sparse histogram construction
  - graph-level parallel encoding

- Measure runtime split by:
  - hashing
  - BFS / shortest path traversal
  - sparse matrix construction
  - multiprocessing overhead

- Benchmark representative settings:
  - `connector = 0` vs `connector > 0`
  - `use_edges_as_features = True` vs `False`
  - serial vs parallel
  - current NetworkX backend vs rustworkx / igraph prototype

## Suggested Implementation Order

1. Replace SHA-256 hashing with a cheap integer hash. Rejected for the benchmark; original `sha256` hash kept.
2. Cache BFS results per source node. Done.
3. Eliminate `nx.all_shortest_paths()` from the inner loop. Done.
4. Build CSR matrices directly. Done.
5. Reduce graph copying and repeated NetworkX lookups. Partially done: repeated lookups reduced via cache, graph copying still remains.
6. Benchmark again.
7. If still too slow, prototype a `rustworkx` or adjacency-array backend.

"""Profile the current NSPPK path on deterministic HIV graph samples.

Run from the repository root:

    PYTHONPATH=src python benchmarks/profile_current_bottlenecks.py

This is a diagnostic benchmark, not a test. It profiles the current BLAKE2s
implementation with connector features enabled and compares one-shot serial
and multiprocessing transforms.
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import io
import pickle
import pstats
import statistics
import time
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import scipy
import networkx as nx

import nsppk


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "HIV_networkx_graphs.pkl"
PARAMS = {
    "radius": 1,
    "distance": 4,
    "connector": 1,
    "nbits": 14,
    "dense": False,
}


def load_graphs(count: int, seed: int):
    with DATA_PATH.open("rb") as stream:
        dataset = pickle.load(stream)
    graphs = dataset["graphs"]
    indices = np.sort(
        np.random.default_rng(seed).choice(
            len(graphs), size=min(count, len(graphs)), replace=False
        )
    )
    return [graphs[index] for index in indices], len(graphs)


def transform(graphs, parallel):
    return nsppk.NSPPK(parallel=parallel, **PARAMS).fit_transform(graphs)


def profile_serial(graphs):
    profiler = cProfile.Profile()
    profiler.runcall(transform, graphs, False)
    stats = pstats.Stats(profiler)
    selected = defaultdict(lambda: [0, 0.0, 0.0])
    wanted = {
        "_hash",
        "hash_set",
        "hash_sequence",
        "node_hash",
        "rooted_graph_hashes",
        "edge_triplet_hash",
        "precompute_edge_triplet_hashes",
        "_build_bfs_cache",
        "_process_node_features",
        "_connector_provenance",
        "_shortest_path_edge_union",
        "_induced_edge_ids",
        "_edge_key",
        "_append_sparse_row_entries",
        "get_structural_node_vectors",
        "get_graph_vector",
        "paired_graphs_vector_encoder",
        "copy",
        "hash_value",
    }
    for (filename, _line, name), (primitive, calls, total, cumulative, _callers) in stats.stats.items():
        if name not in wanted:
            continue
        normalized_path = filename.replace("\\", "/")
        if name in {
            "_hash", "hash_set", "hash_sequence", "node_hash", "rooted_graph_hashes",
            "edge_triplet_hash", "precompute_edge_triplet_hashes", "hash_value",
        } and not normalized_path.endswith("/hash.py"):
            continue
        if name == "copy" and "/networkx/classes/graph.py" not in normalized_path:
            continue
        if name not in {"copy", "_hash", "hash_set", "hash_sequence", "node_hash",
                        "rooted_graph_hashes", "edge_triplet_hash", "precompute_edge_triplet_hashes",
                        "hash_value"} and not normalized_path.endswith("/nsppk.py"):
            continue
        normalized = name
        values = selected[normalized]
        values[0] += calls
        values[1] += total
        values[2] += cumulative

    print("\nSelected cProfile functions (seconds; recursive time is cumulative)")
    print(f"{'function':38} {'calls':>9} {'self':>10} {'cumulative':>12}")
    for name, (calls, total, cumulative) in sorted(
        selected.items(), key=lambda item: item[1][2], reverse=True
    ):
        print(f"{name:38} {calls:9d} {total:10.4f} {cumulative:12.4f}")

    # Print other high-cost functions too, in case the selected list misses a hotspot.
    output = io.StringIO()
    with redirect_stdout(output):
        stats.strip_dirs().sort_stats("cumulative").print_stats(18)
    print("\nTop cumulative call stacks:")
    print(output.getvalue())


def instrumented_components(graphs):
    """Time selected operations in one unprofiled serial transform."""
    totals = defaultdict(lambda: [0, 0.0])
    restorers = []

    def wrap(owner, name, label=None):
        original = getattr(owner, name)
        metric = label or name

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                totals[metric][0] += 1
                totals[metric][1] += time.perf_counter() - start

        setattr(owner, name, timed)
        restorers.append(lambda: setattr(owner, name, original))

    for name in (
        "_build_bfs_cache",
        "_shortest_path_edge_union",
        "_connector_provenance",
        "_append_sparse_row_entries",
        "csr_matrix",
        "Counter",
    ):
        wrap(nsppk, name)
    wrap(nsppk.sp.sparse, "vstack", "sparse.vstack")
    wrap(nx.Graph, "copy", "Graph.copy")

    try:
        transform(graphs, parallel=False)
    finally:
        for restore in reversed(restorers):
            restore()

    print("\nInstrumented operation totals in one unprofiled serial transform")
    print(f"{'operation':34} {'calls':>9} {'elapsed (s)':>13}")
    for name, (calls, seconds) in sorted(totals.items(), key=lambda item: item[1][1], reverse=True):
        print(f"{name:34} {calls:9d} {seconds:13.4f}")


def median_elapsed(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


def parallel_sweep(graphs, sizes, repeats):
    worker_count = nsppk.mp.cpu_count()
    startup, startup_samples = median_elapsed(
        lambda: _open_close_pool(worker_count), repeats
    )
    print(f"\nPool startup/close ({worker_count} workers): median={startup:.4f}s; samples={startup_samples}")

    try:
        import dill

        serialization = dill.dumps
        serializer_name = "dill"
    except ImportError:
        serialization = pickle.dumps
        serializer_name = "pickle fallback"
    largest_batch = graphs[: min(max(sizes), len(graphs))]
    chunks = nsppk.split_into_chunks(largest_batch, worker_count)
    serialized_chunks = [serialization(chunk) for chunk in chunks]
    print(
        f"Input graph chunk payload ({len(largest_batch)} graphs, {serializer_name}): "
        f"{sum(map(len, serialized_chunks)) / 1024:.1f} KiB across {len(chunks)} chunks"
    )
    print("One-shot graph transform medians (seconds)")
    print(f"{'graphs':>8} {'serial':>10} {'parallel':>10} {'serial/parallel':>17}")
    for size in sizes:
        batch = graphs[: min(size, len(graphs))]
        timings = {False: [], True: []}
        for repeat in range(repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for parallel in order:
                start = time.perf_counter()
                transform(batch, parallel)
                timings[parallel].append(time.perf_counter() - start)
        serial = statistics.median(timings[False])
        parallel = statistics.median(timings[True])
        print(f"{len(batch):8d} {serial:10.4f} {parallel:10.4f} {serial / parallel:17.2f}x")


def _open_close_pool(worker_count):
    with nsppk.mp.Pool(worker_count):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-graphs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--parallel-sizes",
        type=int,
        nargs="+",
        default=[18, 19, 36, 72, 128],
    )
    args = parser.parse_args()

    graphs, dataset_count = load_graphs(max(args.profile_graphs, max(args.parallel_sizes)), args.seed)
    node_counts = np.asarray([graph.number_of_nodes() for graph in graphs])
    edge_counts = np.asarray([graph.number_of_edges() for graph in graphs])
    print(f"Dataset: {DATA_PATH.relative_to(ROOT)} ({dataset_count:,} cached graphs)")
    print(f"Python profile uses NetworkX {nx.__version__}, SciPy {scipy.__version__}, {nsppk.mp.cpu_count()} workers")
    print(
        f"Sample: {len(graphs)} graphs, node median/max={np.median(node_counts):.0f}/{node_counts.max()}, "
        f"edge median/max={np.median(edge_counts):.0f}/{edge_counts.max()}"
    )
    print("Parameters:", PARAMS, "hash=4-byte BLAKE2s")

    profile_serial(graphs[: args.profile_graphs])
    instrumented_components(graphs[: args.profile_graphs])
    parallel_sweep(graphs, args.parallel_sizes, args.repeats)


if __name__ == "__main__":
    main()

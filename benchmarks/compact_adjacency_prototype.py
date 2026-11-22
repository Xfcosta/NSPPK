"""Compare NetworkX BFS caching with a compact SciPy CSR prototype.

Run from the repository root with:

    PYTHONPATH=src python benchmarks/compact_adjacency_prototype.py

The prototype is deliberately isolated from the production encoder. It creates
one CSR adjacency matrix per graph, asks SciPy for bounded unweighted distances
from each node, and reconstructs *all* shortest-path predecessors from the CSR
edge arrays so connector features retain their current semantics.
"""

from __future__ import annotations

import argparse
import pickle
import platform
import statistics
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
from scipy.sparse.csgraph import dijkstra

import nsppk


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "HIV_networkx_graphs.pkl"
PARAMS = {
    "radius": 1,
    "distance": 4,
    "connector": 1,
    "nbits": 14,
}


def build_csr_bfs_cache(graph: nx.Graph, max_radius: int):
    """Build the existing cache shape using CSR distances and edge arrays."""
    nodes = tuple(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        weight=None,
        dtype=np.uint8,
        format="csr",
    )
    adjacency.sort_indices()
    edge_rows = np.repeat(
        np.arange(len(nodes), dtype=np.int32),
        np.diff(adjacency.indptr),
    )
    edge_cols = adjacency.indices
    bfs_cache = {}

    for source_index, source_node in enumerate(nodes):
        distances = dijkstra(
            adjacency,
            directed=False,
            indices=source_index,
            unweighted=True,
            limit=max_radius,
        )
        reachable = np.isfinite(distances)
        reachable_indices = np.flatnonzero(reachable)

        node_idxs_to_dist_dict = {
            nodes[index]: int(distances[index])
            for index in reachable_indices
        }
        dist_to_node_idxs_dict = {}
        for index in reachable_indices:
            dist_to_node_idxs_dict.setdefault(int(distances[index]), []).append(nodes[index])

        predecessor_mask = (
            reachable[edge_rows]
            & reachable[edge_cols]
            & (distances[edge_rows] + 1 == distances[edge_cols])
        )
        predecessors = {nodes[index]: [] for index in reachable_indices}
        for parent_index, child_index in zip(
            edge_rows[predecessor_mask],
            edge_cols[predecessor_mask],
        ):
            predecessors[nodes[child_index]].append(nodes[parent_index])

        balls_by_radius = []
        ball = set()
        for radius in range(max_radius + 1):
            ball.update(dist_to_node_idxs_dict.get(radius, []))
            balls_by_radius.append(frozenset(ball))

        bfs_cache[source_node] = {
            "node_idxs_to_dist_dict": node_idxs_to_dist_dict,
            "dist_to_node_idxs_dict": dist_to_node_idxs_dict,
            "predecessors": predecessors,
            "balls_by_radius": tuple(balls_by_radius),
            "shortest_path_union_cache": {},
        }

    return bfs_cache


def caches_match(left, right):
    """Compare cache contents without depending on predecessor list order."""
    if left.keys() != right.keys():
        return False
    for node in left:
        a, b = left[node], right[node]
        if a["node_idxs_to_dist_dict"] != b["node_idxs_to_dist_dict"]:
            return False
        if a["dist_to_node_idxs_dict"].keys() != b["dist_to_node_idxs_dict"].keys():
            return False
        if any(
            set(a["dist_to_node_idxs_dict"][distance])
            != set(b["dist_to_node_idxs_dict"][distance])
            for distance in a["dist_to_node_idxs_dict"]
        ):
            return False
        if a["predecessors"].keys() != b["predecessors"].keys():
            return False
        if any(
            set(a["predecessors"][child]) != set(b["predecessors"][child])
            for child in a["predecessors"]
        ):
            return False
        if a["balls_by_radius"] != b["balls_by_radius"]:
            return False
    return True


def load_graphs(count: int, seed: int):
    with DATA_PATH.open("rb") as stream:
        data = pickle.load(stream)
    all_graphs = data["graphs"]
    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(len(all_graphs), size=min(count, len(all_graphs)), replace=False)
    )
    return [all_graphs[index] for index in selected], len(all_graphs)


def elapsed(function, repeats: int):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


def end_to_end(graphs, bfs_builder):
    original_builder = nsppk._build_bfs_cache
    nsppk._build_bfs_cache = bfs_builder
    try:
        return [
            nsppk.get_structural_node_vectors(
                graph,
                **PARAMS,
            )
            for graph in graphs
        ]
    finally:
        nsppk._build_bfs_cache = original_builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=int, default=64, help="number of HIV graphs to sample")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per method")
    parser.add_argument("--seed", type=int, default=17, help="deterministic sample seed")
    parser.add_argument(
        "--synthetic-sizes",
        type=int,
        nargs="+",
        default=[250, 500, 1000],
        help="Barabasi-Albert node counts for traversal-only stress cases",
    )
    args = parser.parse_args()

    graphs, dataset_count = load_graphs(args.graphs, args.seed)
    sizes = np.asarray([graph.number_of_nodes() for graph in graphs])
    edge_counts = np.asarray([graph.number_of_edges() for graph in graphs])
    cutoff = max(PARAMS["radius"], PARAMS["distance"], PARAMS["connector"])

    print(f"Dataset: {DATA_PATH.relative_to(ROOT)} ({dataset_count:,} graphs)")
    print(f"Python {platform.python_version()}, NetworkX {nx.__version__}, SciPy {scipy.__version__}")
    print(
        f"Sample: {len(graphs)} graphs; nodes median/max={np.median(sizes):.0f}/{sizes.max()}, "
        f"edges median/max={np.median(edge_counts):.0f}/{edge_counts.max()}; cutoff={cutoff}"
    )

    # Verify cache semantics and final feature values on a small representative slice.
    for graph in graphs[:8]:
        nx_cache = nsppk._build_bfs_cache(graph, cutoff)
        csr_cache = build_csr_bfs_cache(graph, cutoff)
        if not caches_match(nx_cache, csr_cache):
            raise AssertionError("CSR and NetworkX BFS caches differ")

    nx_vectors = end_to_end(graphs[:8], nsppk._build_bfs_cache)
    csr_vectors = end_to_end(graphs[:8], build_csr_bfs_cache)
    if any((left != right).nnz for left, right in zip(nx_vectors, csr_vectors)):
        raise AssertionError("CSR and NetworkX end-to-end feature matrices differ")
    print("Correctness: BFS distances, predecessor DAGs, balls, and feature matrices match")

    traversal_nx, _ = elapsed(
        lambda: [nsppk._build_bfs_cache(graph, cutoff) for graph in graphs],
        args.repeats,
    )
    traversal_csr, _ = elapsed(
        lambda: [build_csr_bfs_cache(graph, cutoff) for graph in graphs],
        args.repeats,
    )
    e2e_nx, nx_samples = elapsed(
        lambda: end_to_end(graphs, nsppk._build_bfs_cache),
        args.repeats,
    )
    e2e_csr, csr_samples = elapsed(
        lambda: end_to_end(graphs, build_csr_bfs_cache),
        args.repeats,
    )

    print("\nMedian elapsed time (seconds)")
    print(
        f"BFS cache only: NetworkX={traversal_nx:.4f}, CSR={traversal_csr:.4f}, "
        f"speedup={traversal_nx / traversal_csr:.2f}x"
    )
    print(
        f"Full node-feature extraction: NetworkX={e2e_nx:.4f}, CSR={e2e_csr:.4f}, "
        f"speedup={e2e_nx / e2e_csr:.2f}x"
    )
    print(f"End-to-end samples NetworkX: {[round(value, 4) for value in nx_samples]}")
    print(f"End-to-end samples CSR:      {[round(value, 4) for value in csr_samples]}")

    print("\nTraversal-only Barabasi-Albert stress cases (m=2)")
    for node_count in args.synthetic_sizes:
        graph = nx.barabasi_albert_graph(node_count, 2, seed=args.seed)
        nx_time, _ = elapsed(lambda: nsppk._build_bfs_cache(graph, cutoff), args.repeats)
        csr_time, _ = elapsed(lambda: build_csr_bfs_cache(graph, cutoff), args.repeats)
        print(
            f"n={node_count}, edges={graph.number_of_edges()}: "
            f"NetworkX={nx_time:.4f}, CSR={csr_time:.4f}, "
            f"speedup={nx_time / csr_time:.2f}x"
        )


if __name__ == "__main__":
    main()

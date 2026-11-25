import math
from collections import defaultdict, Counter
import copy
from numbers import Integral, Real
import networkx as nx
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.utils.metaestimators import available_if
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.model_selection import train_test_split
import multiprocessing_on_dill as mp

import graph_io as _graph_io
from hash import (
    _hash,
    edge_triplet_hash,
    hash_sequence,
    hash_set,
    hash_value,
    invert_dict,
    masked_hash_value,
    node_hash,
    precompute_edge_triplet_hashes,
    rooted_graph_hashes,
)
from graph_io import ensure_graph_labels

__version__ = "2.1.0"

_make_rng = _graph_io._make_rng
_require_rdkit = _graph_io._require_rdkit
_require_torch_geometric = _graph_io._require_torch_geometric


def _sync_graph_io_hooks():
    _graph_io._make_rng = _make_rng
    _graph_io._require_rdkit = _require_rdkit
    _graph_io._require_torch_geometric = _require_torch_geometric


def _resolve_alias(primary_name, primary_value, alias_name, alias_value, default_value):
    if alias_value is None:
        return primary_value
    if primary_value == default_value:
        return alias_value
    if primary_value != alias_value:
        raise ValueError(
            f"Conflicting values for '{primary_name}' and '{alias_name}': "
            f"{primary_value!r} != {alias_value!r}."
        )
    return alias_value


def _validate_encoder_parameters(
    *, radius, distance, connector, nbits, degree_threshold, dense, parallel,
    node_attribute_key, edge_attribute_key, attribute_dim, attribute_alphabet_size,
    edge_attribute_dim, edge_attribute_alphabet_size, sigma, use_node_kernel,
    use_edges_as_features,
):
    integer_parameters = {"radius": radius, "distance": distance, "connector": connector}
    for name, value in integer_parameters.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    if isinstance(nbits, bool) or not isinstance(nbits, Integral) or not 2 <= nbits <= 30:
        raise ValueError("nbits must be an integer between 2 and 30.")

    if degree_threshold is not None and (
        isinstance(degree_threshold, bool)
        or not isinstance(degree_threshold, Integral)
        or degree_threshold < 0
    ):
        raise ValueError("degree_threshold must be None or a non-negative integer.")

    for name, value in (
        ("attribute_dim", attribute_dim),
        ("attribute_alphabet_size", attribute_alphabet_size),
        ("edge_attribute_dim", edge_attribute_dim),
        ("edge_attribute_alphabet_size", edge_attribute_alphabet_size),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{name} must be None or a positive integer.")

    uses_attribute_preprocessing = attribute_dim is not None or attribute_alphabet_size is not None
    if uses_attribute_preprocessing and not node_attribute_key:
        raise ValueError(
            "node_attribute_key is required when attribute_dim or "
            "attribute_alphabet_size is set."
        )

    uses_edge_attribute_preprocessing = (
        edge_attribute_dim is not None or edge_attribute_alphabet_size is not None
    )
    if uses_edge_attribute_preprocessing and not edge_attribute_key:
        raise ValueError(
            "edge_attribute_key is required when edge_attribute_dim or "
            "edge_attribute_alphabet_size is set."
        )

    if sigma is not None and (
        isinstance(sigma, bool)
        or not isinstance(sigma, Real)
        or not math.isfinite(float(sigma))
        or sigma <= 0
    ):
        raise ValueError("sigma must be None or a finite positive number.")

    for name, value in (
        ("dense", dense),
        ("parallel", parallel),
        ("use_node_kernel", use_node_kernel),
        ("use_edges_as_features", use_edges_as_features),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean.")


_PARAMETER_ALIASES = {"r": "radius", "d": "distance", "c": "connector"}


def _normalize_parameter_aliases(params):
    normalized = dict(params)
    for alias, canonical in _PARAMETER_ALIASES.items():
        if alias not in normalized:
            continue
        alias_value = normalized.pop(alias)
        if canonical in normalized and normalized[canonical] != alias_value:
            raise ValueError(
                f"Conflicting values for '{canonical}' and '{alias}': "
                f"{normalized[canonical]!r} != {alias_value!r}."
            )
        normalized[canonical] = alias_value
    return normalized


def _set_loaded_label_metadata(estimator, metadata):
    estimator.loaded_balance_candidate_size_ = None
    estimator.loaded_label_counts_ = None
    estimator.loaded_class_weight_ = None
    if metadata is None:
        return
    estimator.loaded_balance_candidate_size_ = metadata["candidate_size"]
    estimator.loaded_label_counts_ = dict(metadata["label_counts"])
    estimator.loaded_class_weight_ = dict(metadata["class_weight"])

def items_to_sparse_histogram(items, nbits):
    histogram_dict = Counter(items)
    # Create a LIL matrix first
    mat = sp.sparse.lil_matrix((1, 2**nbits), dtype=int)
    for col, value in histogram_dict.items():
        mat[0, col] = value
    return mat

def weighted_sparse_histogram(weighted_dict, nbits):
    # Create a LIL matrix first
    mat = sp.sparse.lil_matrix((1, 2**nbits), dtype=float)
    for col, value in weighted_dict.items():
        mat[0, col] = value
    return mat


def _append_sparse_row_entries(row_entries, row_idx, rows, cols, data):
    for col_idx, value in row_entries.items():
        rows.append(row_idx)
        cols.append(col_idx)
        data.append(value)

def gaussian_weight(dist, sigma):
    if sigma is None:
        return 1.0
    return np.exp(- (dist ** 2) / (sigma ** 2))


# ------------------------
# Accumulator Classes
# ------------------------

class ListAccumulator:
    def __init__(self):
        self.data = []
    def add(self, code, weight):
        # For unweighted case weight is always 1.0
        self.data.append(code)
    def get(self):
        return self.data

class DictAccumulator:
    def __init__(self):
        self.data = defaultdict(float)
    def add(self, code, weight):
        self.data[code] += weight
    def get(self):
        return self.data


def _edge_key(u, v):
    return tuple(sorted((u, v)))


def _edge_id(graph, u, v, key=None):
    if graph.is_directed():
        return (u, v, key) if graph.is_multigraph() else (u, v)
    try:
        endpoints = _edge_key(u, v)
    except TypeError:
        node_order = {node: idx for idx, node in enumerate(graph.nodes())}
        endpoints = (u, v) if node_order[u] <= node_order[v] else (v, u)
    return endpoints + (key,) if graph.is_multigraph() else endpoints


def _edge_records(graph):
    if graph.is_multigraph():
        for u, v, key, data in graph.edges(keys=True, data=True):
            yield u, v, key, data
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, None, data


def _undirected_pair_key(u, v):
    return frozenset((u, v))


def _incident_edge_records(graph, node):
    if graph.is_directed():
        if graph.is_multigraph():
            records = list(graph.out_edges(node, keys=True, data=True))
            incoming = graph.in_edges(node, keys=True, data=True)
        else:
            records = list(graph.out_edges(node, data=True))
            incoming = graph.in_edges(node, data=True)
        records.extend(edge for edge in incoming if edge[0] != edge[1])
    else:
        records = list(graph.edges(node, keys=True, data=True)) if graph.is_multigraph() else list(graph.edges(node, data=True))
    for edge in records:
        if graph.is_multigraph():
            u, v, key, data = edge
        else:
            u, v, data = edge
            key = None
        other = v if u == node else u
        yield other, key, data, u, v


def _node_ball(graph, center, radius):
    if radius < 0:
        return set()
    bfs_cache = graph.graph.get('_bfs_cache')
    if bfs_cache is not None:
        return set(bfs_cache[center]['balls_by_radius'][radius])
    traversal_graph = graph.graph.get('_traversal_graph', graph)
    return set(nx.single_source_shortest_path_length(traversal_graph, center, cutoff=radius).keys())


def _build_bfs_cache(graph, max_radius):
    bfs_cache = {}
    for node_idx in graph.nodes():
        if max_radius == 0:
            # NetworkX treats cutoff=0 as no cutoff in predecessor(), so this
            # radius-zero case is the source node alone.
            predecessors = {node_idx: []}
            node_idxs_to_dist_dict = {node_idx: 0}
        else:
            predecessors, node_idxs_to_dist_dict = nx.predecessor(
                graph,
                node_idx,
                cutoff=max_radius,
                return_seen=True,
            )
        dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
        balls_by_radius = []
        ball = set()
        for radius in range(max_radius + 1):
            ball.update(dist_to_node_idxs_dict.get(radius, []))
            balls_by_radius.append(frozenset(ball))
        bfs_cache[node_idx] = {
            'node_idxs_to_dist_dict': node_idxs_to_dist_dict,
            'dist_to_node_idxs_dict': dist_to_node_idxs_dict,
            'predecessors': predecessors,
            'balls_by_radius': tuple(balls_by_radius),
            'shortest_path_union_cache': {},
        }
    return bfs_cache


def _build_structural_cache(graph):
    cache = {
        'neighbors': {},
        'degrees': {},
        'node_order': {},
        'node_label_hashes': {},
        'node_hashes': {},
        'edge_triplet_hashes': {},
        'rooted_graph_hashes': {},
        'edge_ids_by_pair': defaultdict(list),
    }
    for node_order, node_idx in enumerate(graph.nodes()):
        neighbors = tuple(graph.graph['_traversal_graph'].neighbors(node_idx))
        cache['neighbors'][node_idx] = neighbors
        cache['degrees'][node_idx] = len(neighbors)
        cache['node_order'][node_idx] = node_order
    for u, v, key, _ in _edge_records(graph):
        cache['edge_ids_by_pair'][_undirected_pair_key(u, v)].append(_edge_id(graph, u, v, key))
    return cache


def _induced_edge_ids(source_node_ids, neighbors_by_node, node_order=None):
    source_edge_ids = set()
    for u in source_node_ids:
        for v in neighbors_by_node[u]:
            if v in source_node_ids and (node_order is None or node_order[u] < node_order[v]):
                source_edge_ids.add(_edge_key(u, v))
    return source_edge_ids


def _induced_edge_ids_from_graph(graph, source_node_ids):
    source_node_ids = set(source_node_ids)
    return {
        _edge_id(graph, u, v, key)
        for u, v, key, _ in _edge_records(graph)
        if u in source_node_ids and v in source_node_ids
    }


def _edge_ids_between(graph, u, v):
    edge_ids_by_pair = graph.graph.get('_structural_cache', {}).get('edge_ids_by_pair')
    if edge_ids_by_pair is not None:
        return edge_ids_by_pair.get(_undirected_pair_key(u, v), ())
    return tuple(
        _edge_id(graph, edge_u, edge_v, key)
        for edge_u, edge_v, key, _ in _edge_records(graph)
        if _undirected_pair_key(edge_u, edge_v) == _undirected_pair_key(u, v)
    )


def _shortest_path_edge_union(graph, source, target, include_edges=True):
    bfs_cache = graph.graph.get('_bfs_cache')
    if bfs_cache is not None:
        source_cache = bfs_cache[source]
        cached_union = source_cache['shortest_path_union_cache'].get(target)
        if cached_union is not None:
            nodes, edges = cached_union
            if not include_edges:
                return nodes, set()
            if edges is not None:
                return cached_union
        predecessors = source_cache['predecessors']
        if target not in predecessors:
            result = (set(), set())
            source_cache['shortest_path_union_cache'][target] = result
            return result

        nodes = set()
        edges = set() if include_edges else None
        stack = [target]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.add(node)
            for predecessor in predecessors.get(node, []):
                if include_edges:
                    edges.update(_edge_ids_between(graph, predecessor, node))
                stack.append(predecessor)

        result = (nodes, edges)
        source_cache['shortest_path_union_cache'][target] = result
        return result if include_edges else (nodes, set())

    traversal_graph = graph.graph.get('_traversal_graph', graph)
    shortest_paths = list(nx.all_shortest_paths(traversal_graph, source=source, target=target))
    nodes = set()
    edges = set()
    for path in shortest_paths:
        nodes.update(path)
        if include_edges:
            for first, second in zip(path, path[1:]):
                edges.update(_edge_ids_between(graph, first, second))
    return nodes, edges


def _induced_subgraph_from_occurrence(original_graph, occurrence):
    subgraph = original_graph.__class__()
    for key, value in original_graph.graph.items():
        subgraph.graph[key] = copy.deepcopy(value)
    source_nodes = set(occurrence['source_node_ids'])
    for node in (node for node in original_graph.nodes() if node in source_nodes):
        subgraph.add_node(node, **copy.deepcopy(original_graph.nodes[node]))
        subgraph.nodes[node]['original_node_id'] = node
    for edge_id in occurrence['source_edge_ids']:
        if original_graph.is_multigraph():
            u, v, key = edge_id
            if original_graph.has_edge(u, v, key):
                data = copy.deepcopy(original_graph.edges[u, v, key])
                subgraph.add_edge(u, v, key=key, **data)
                subgraph.edges[u, v, key]['original_edge'] = edge_id
        else:
            u, v = edge_id
            if original_graph.has_edge(u, v):
                data = copy.deepcopy(original_graph.edges[u, v])
                subgraph.add_edge(u, v, **data)
                subgraph.edges[u, v]['original_edge'] = edge_id
    subgraph.graph['source_graph_index'] = occurrence['source_graph_index']
    subgraph.graph['feature_id'] = occurrence['feature_id']
    subgraph.graph['feature_kind'] = occurrence['feature_kind']
    subgraph.graph['raw_signature'] = occurrence['raw_signature']
    subgraph.graph['root_node'] = occurrence['root_node']
    subgraph.graph['target_node'] = occurrence['target_node']
    return subgraph


class _FeatureArchiveCollector:
    def __init__(self, original_graph, graph_index):
        self.original_graph = original_graph
        self.graph_index = graph_index
        self.archive = defaultdict(list)

    def add(
        self,
        feature_id,
        raw_signature,
        feature_kind,
        root_node,
        target_node,
        source_node_ids,
        source_edge_ids,
    ):
        source_nodes = set(source_node_ids)
        source_edges = set(source_edge_ids)
        occurrence = {
            'feature_id': feature_id,
            'raw_signature': raw_signature,
            'feature_kind': feature_kind,
            'source_graph_index': self.graph_index,
            'root_node': root_node,
            'target_node': target_node,
            'source_node_ids': tuple(node for node in self.original_graph.nodes() if node in source_nodes),
            'source_edge_ids': tuple(
                edge_id for edge_id in (_edge_id(self.original_graph, u, v, key)
                    for u, v, key, _ in _edge_records(self.original_graph))
                if edge_id in source_edges
            ),
        }
        occurrence['subgraph'] = _induced_subgraph_from_occurrence(self.original_graph, occurrence)
        self.archive[feature_id].append(occurrence)


def _prepare_structural_graph(original_graph, radius, distance, connector, degree_threshold=None):
    graph = original_graph.copy()
    ensure_graph_labels(graph)
    traversal_graph = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    graph.graph['_traversal_graph'] = traversal_graph
    cutoff = max(radius, connector)
    bfs_cutoff = max(radius, connector, distance)
    graph.graph['_bfs_cache'] = _build_bfs_cache(traversal_graph, bfs_cutoff)
    graph.graph['_structural_cache'] = _build_structural_cache(graph)
    structural_cache = graph.graph['_structural_cache']

    for node_idx in graph.nodes():
        node_label_hash = _hash(graph.nodes[node_idx]['label'])
        structural_cache['node_label_hashes'][node_idx] = node_label_hash
        graph.nodes[node_idx]['node_label_hash'] = node_label_hash
    for node_idx in graph.nodes():
        node_hash_value = node_hash(node_idx, graph)
        structural_cache['node_hashes'][node_idx] = node_hash_value
        graph.nodes[node_idx]['node_hash'] = node_hash_value
    precompute_edge_triplet_hashes(graph)
    for u, v, key, data in _edge_records(graph):
        structural_cache['edge_triplet_hashes'][_edge_id(graph, u, v, key)] = data['triplet_hash']

    for node_idx in graph.nodes():
        rooted_hashes = np.zeros(cutoff + 1, dtype=int)
        graph.nodes[node_idx]['rooted_graph_hash'] = rooted_hashes
        degree = structural_cache['degrees'][node_idx]
        effective_cutoff = 0 if degree_threshold is not None and degree > degree_threshold else cutoff
        node_bfs_cache = graph.graph['_bfs_cache'][node_idx]
        for r, radius_r_rooted_graph_hash in enumerate(
            rooted_graph_hashes(
                node_idx,
                graph,
                radius=effective_cutoff,
                node_idxs_to_dist_dict=node_bfs_cache['node_idxs_to_dist_dict'],
                node_label_hashes=structural_cache['node_label_hashes'],
                node_hashes=structural_cache['node_hashes'],
            )
        ):
            rooted_hashes[r] = radius_r_rooted_graph_hash
        structural_cache['rooted_graph_hashes'][node_idx] = rooted_hashes
    return graph


def _connector_provenance(
    graph,
    node_idxs_to_dist_dict,
    union_of_shortest_paths,
    union_path_edges,
    connect,
    include_occurrence_provenance=True,
):
    if connect <= 0:
        return set(union_of_shortest_paths), set()

    structural_cache = graph.graph.get('_structural_cache', {})
    neighbors_by_node = structural_cache.get('neighbors', {})
    node_order = structural_cache.get('node_order')
    rooted_graph_hashes_by_node = structural_cache.get('rooted_graph_hashes', {})

    node_signatures = [
        hash_sequence((node_idxs_to_dist_dict[node], rooted_graph_hashes_by_node[node][connect - 1]))
        for node in union_of_shortest_paths
    ]
    if graph.is_directed() or graph.is_multigraph():
        edge_triplet_hashes = structural_cache.get('edge_triplet_hashes', {})
        edge_signatures = [
            hash_sequence(("path-edge", edge_triplet_hashes[edge_id]))
            for edge_id in union_path_edges
            if edge_id in edge_triplet_hashes
        ]
        union_signature = hash_set(node_signatures + edge_signatures)
    else:
        union_signature = hash_set(node_signatures)

    if not include_occurrence_provenance:
        return set(), set(), union_signature

    provenance_nodes = set(union_of_shortest_paths)
    if connect > 1:
        for path_node in union_of_shortest_paths:
            provenance_nodes.update(_node_ball(graph, path_node, connect - 1))
    provenance_edges = _induced_edge_ids_from_graph(graph, provenance_nodes)
    return provenance_nodes, provenance_edges, union_signature


def _process_node_features(node_idx, graph, distance, connector, nbits, sigma, accumulator, use_edges_as_features=True, archive_collector=None):
    # Weighted: inline computation of weight using the Gaussian function.
    structural_cache = graph.graph.get('_structural_cache', {})
    neighbors_by_node = structural_cache.get('neighbors', {})
    node_order = structural_cache.get('node_order')
    edge_triplet_hashes = structural_cache.get('edge_triplet_hashes', {})
    rooted_graph_hashes_by_node = structural_cache.get('rooted_graph_hashes', {})
    node_bfs_cache = graph.graph.get('_bfs_cache', {}).get(node_idx)
    if node_bfs_cache is not None:
        node_idxs_to_dist_dict = {
            curr_node_idx: dist
            for curr_node_idx, dist in node_bfs_cache['node_idxs_to_dist_dict'].items()
            if dist <= distance
        }
        dist_to_node_idxs_dict = {
            dist: node_idxs
            for dist, node_idxs in node_bfs_cache['dist_to_node_idxs_dict'].items()
            if dist <= distance
        }
    else:
        node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=distance)
        dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)

    if use_edges_as_features:
        for target_node, dist in node_idxs_to_dist_dict.items():
            w = gaussian_weight(dist, sigma)
            if archive_collector is not None:
                path_nodes, path_edges = _shortest_path_edge_union(
                    graph, node_idx, target_node, include_edges=True
                )
            for neighbor, edge_key, _, edge_u, edge_v in _incident_edge_records(graph, target_node):
                edge_id = _edge_id(graph, edge_u, edge_v, edge_key)
                triplet_hash = edge_triplet_hashes.get(edge_id)
                if triplet_hash is not None:
                    raw_signature = hash_sequence((dist, triplet_hash))
                    distance_triplet_hash = hash_value(raw_signature, nbits=nbits)
                    accumulator.add(distance_triplet_hash, w)
                    if archive_collector is not None:
                        source_node_ids = set(path_nodes)
                        source_node_ids.update({target_node, neighbor})
                        source_edge_ids = set(path_edges)
                        source_edge_ids.add(edge_id)
                        archive_collector.add(
                            feature_id=distance_triplet_hash,
                            raw_signature=raw_signature,
                            feature_kind='edge_triplet',
                            root_node=node_idx,
                            target_node=target_node,
                            source_node_ids=source_node_ids,
                            source_edge_ids=source_edge_ids,
                        )

    connector_data_by_target = {}
    for radius_i, code_i in enumerate(rooted_graph_hashes_by_node[node_idx]):
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items()):
            w = gaussian_weight(dist, sigma)
            for curr_node_idx in node_idxs:
                if connector > 0:
                    target_connector_data = connector_data_by_target.get(curr_node_idx)
                    if target_connector_data is None:
                        union_of_shortest_paths, union_path_edges = _shortest_path_edge_union(
                            graph,
                            node_idx,
                            curr_node_idx,
                            include_edges=archive_collector is not None,
                        )
                        provenance_by_connect = {
                            connect_depth: _connector_provenance(
                                graph,
                                node_idxs_to_dist_dict,
                                union_of_shortest_paths,
                                union_path_edges,
                                connect_depth,
                                include_occurrence_provenance=archive_collector is not None,
                            )
                            for connect_depth in range(1, connector + 1)
                        }
                        target_connector_data = (
                            union_of_shortest_paths,
                            union_path_edges,
                            provenance_by_connect,
                        )
                        connector_data_by_target[curr_node_idx] = target_connector_data
                    union_of_shortest_paths, union_path_edges, provenance_by_connect = (
                        target_connector_data
                    )
                for connect in range(connector + 1):
                    for radius_j, code_j in enumerate(rooted_graph_hashes_by_node[curr_node_idx]):
                        if connect == 0:
                            raw_signature = hash_sequence((code_i, dist, code_j))
                        else:
                            (
                                connector_nodes,
                                connector_edges,
                                union_of_shortest_paths_code,
                            ) = provenance_by_connect[connect]
                            raw_signature = hash_sequence((code_i, dist, code_j, union_of_shortest_paths_code))

                        paired_code = hash_value(raw_signature, nbits=nbits)
                        accumulator.add(paired_code, w)
                        if archive_collector is not None:
                            source_node_ids = _node_ball(graph, node_idx, radius_i)
                            source_node_ids.update(_node_ball(graph, curr_node_idx, radius_j))
                            source_edge_ids = _induced_edge_ids_from_graph(graph, source_node_ids)
                            if connect > 0:
                                source_node_ids.update(connector_nodes)
                                source_edge_ids.update(connector_edges)
                                source_edge_ids.update(union_path_edges)
                            archive_collector.add(
                                feature_id=paired_code,
                                raw_signature=raw_signature,
                                feature_kind='paired_rooted_subgraph',
                                root_node=node_idx,
                                target_node=curr_node_idx,
                                source_node_ids=source_node_ids,
                                source_edge_ids=source_edge_ids,
                            )

def get_structural_node_vectors(original_graph, radius, distance, connector, nbits, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Generates a feature vector for each node in the graph.
    Uses an unweighted (faster) version if sigma is None, in which the weight is inlined as 1.0.
    If sigma is provided, the Gaussian weighted variant is used.
    """
    graph = _prepare_structural_graph(original_graph, radius, distance, connector, degree_threshold)
    n_nodes = graph.number_of_nodes()
    n_features = 2 ** nbits
    rows = []
    cols = []
    data = []
    if sigma is None:
        for row_idx, node_idx in enumerate(graph.nodes()):
            accumulator = ListAccumulator()
            _process_node_features(node_idx, graph, distance, connector, nbits, sigma, accumulator, use_edges_as_features)
            row_entries = Counter(accumulator.get())
            row_entries[0] = 1
            row_entries[1] = graph.degree[node_idx]
            _append_sparse_row_entries(row_entries, row_idx, rows, cols, data)
    else:
        for row_idx, node_idx in enumerate(graph.nodes()):
            accumulator = DictAccumulator()
            _process_node_features(node_idx, graph, distance, connector, nbits, sigma, accumulator, use_edges_as_features)
            row_entries = dict(accumulator.get())
            row_entries[0] = 1.0
            row_entries[1] = float(graph.degree[node_idx])
            _append_sparse_row_entries(row_entries, row_idx, rows, cols, data)

    dtype = float if sigma is not None else int
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_features), dtype=dtype)


def get_feature_archive(original_graph, graph_index, radius, distance, connector, nbits, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Replays the structural hashing path and archives all occurrences per final hashed feature id.
    """
    graph = _prepare_structural_graph(original_graph, radius, distance, connector, degree_threshold)
    archive_collector = _FeatureArchiveCollector(original_graph, graph_index)

    for node_idx in graph.nodes():
        accumulator = DictAccumulator() if sigma is not None else ListAccumulator()
        _process_node_features(
            node_idx,
            graph,
            distance,
            connector,
            nbits,
            sigma,
            accumulator,
            use_edges_as_features,
            archive_collector=archive_collector,
        )

    return archive_collector.archive

def get_node_attribute_matrix(original_graph, node_attribute_key):
    """
    Extracts node attributes from a graph and stacks them into a matrix.

    Args:
        original_graph (networkx.Graph): The input graph.
        node_attribute_key (str): The key corresponding to the node attribute to extract.

    Returns:
        numpy.ndarray: The matrix of node attributes.
    """
    return np.vstack([
        original_graph.nodes[node_idx][node_attribute_key]
        for node_idx in original_graph.nodes()
    ])

def get_edge_attribute_mtx(original_graph, edge_attribute_key):
    """
    Extracts edge attributes from a graph and stacks them into a matrix.

    Args:
        original_graph (networkx.Graph): The input graph.
        edge_attribute_key (str): The key corresponding to the edge attribute to extract.

    Returns:
        numpy.ndarray: The matrix of edge attributes.
    """
    edge_attributes = [data[edge_attribute_key] for _, _, _, data in _edge_records(original_graph)]
    return np.vstack(edge_attributes) if edge_attributes else np.empty((0, 0), dtype=float)

def reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key):
    """
    Reweights node feature vectors based on node weights.

    Args:
        node_vectors_mtx (scipy.sparse.csr_matrix): The matrix of node feature vectors.
        original_graph (networkx.Graph): The input graph.
        weight_key (str): The key corresponding to the node weight to apply.

    Returns:
        scipy.sparse.csr_matrix: The reweighted node feature matrix.
    """
    weight_vector = np.array([
        original_graph.nodes[node_idx][weight_key]
        for node_idx in original_graph.nodes()
    ])
    node_vectors_mtx = (node_vectors_mtx.todense().A.T * weight_vector).T
    return csr_matrix(node_vectors_mtx)


def reweight_edge_vectors_mtx(edge_vectors_mtx, original_graph, weight_key):
    """
    Reweights edge feature vectors based on edge weights.

    Args:
        edge_vectors_mtx (scipy.sparse.csr_matrix): The matrix of edge feature vectors.
        original_graph (networkx.Graph): The input graph.
        weight_key (str): The key corresponding to the edge weight to apply.

    Returns:
        scipy.sparse.csr_matrix: The reweighted edge feature matrix.
    """
    weight_vector = np.array([
        data[weight_key] for _, _, _, data in _edge_records(original_graph)
    ])
    edge_vectors_mtx = (edge_vectors_mtx.todense().A.T * weight_vector).T
    return csr_matrix(edge_vectors_mtx)


def get_node_vectors(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, degree_threshold=None, add_structural_node_information=True, sigma=None, use_edges_as_features=True, structural_node_vectors_mtx=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature matrix representing the graph.
    """
    node_vectors_mtx = structural_node_vectors_mtx
    if node_vectors_mtx is None:
        node_vectors_mtx = get_structural_node_vectors(
            original_graph, radius, distance, connector, nbits, degree_threshold, sigma, use_edges_as_features
        )
    if weight_key is not None:
        node_vectors_mtx = reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key)
    if node_attribute_key is not None:
        attribute_mtx = get_node_attribute_matrix(original_graph, node_attribute_key)
        feature_node_vectors_mtx = node_vectors_mtx.todense().A
        feature_node_vectors_mtx = attribute_mtx.T.dot(feature_node_vectors_mtx).dot(feature_node_vectors_mtx.T).T
        feature_node_vectors_mtx = np.power(np.abs(feature_node_vectors_mtx),1/3) #reduce magnitude of entries to avoid kernel diagonal dominance issues
        feature_node_vectors_mtx = csr_matrix(feature_node_vectors_mtx)
        if add_structural_node_information: node_vectors_mtx = sp.sparse.hstack([csr_matrix(attribute_mtx), feature_node_vectors_mtx, node_vectors_mtx])
        else: node_vectors_mtx = sp.sparse.hstack([csr_matrix(attribute_mtx), feature_node_vectors_mtx])
    return node_vectors_mtx


def get_structural_edge_vectors(original_graph, radius, distance, connector, nbits,
                                degree_threshold=None, sigma=None, use_edges_as_features=True,
                                node_vectors_mtx=None):
    """Build one structural feature row per undirected edge by summing endpoint rows."""
    if node_vectors_mtx is None:
        node_vectors_mtx = get_structural_node_vectors(
            original_graph, radius, distance, connector, nbits, degree_threshold, sigma,
            use_edges_as_features,
        )
    node_idxs = list(original_graph.nodes())
    node_to_row = {node_idx: row_idx for row_idx, node_idx in enumerate(node_idxs)}
    edge_vectors = []
    for u, v, key, data in _edge_records(original_graph):
        row = node_vectors_mtx.getrow(node_to_row[u]) + node_vectors_mtx.getrow(node_to_row[v])
        if original_graph.is_directed():
            orientation_signature = hash_sequence((
                "directed-edge-row",
                _hash(original_graph.nodes[u].get('label', 0)),
                _hash(data.get('label', 0)),
                _hash(original_graph.nodes[v].get('label', 0)),
            ))
            orientation_code = hash_value(orientation_signature, nbits=nbits)
            row = row.tolil()
            row[0, orientation_code] += 1
            row = row.tocsr()
        edge_vectors.append(row)
    if not edge_vectors:
        return csr_matrix((0, node_vectors_mtx.shape[1]), dtype=node_vectors_mtx.dtype)
    return sp.sparse.vstack(edge_vectors, format="csr")


def get_edge_vectors(original_graph, radius, distance, connector, nbits,
                     edge_attribute_key=None, degree_threshold=None, sigma=None,
                     use_edges_as_features=True, edge_attribute_width=None):
    """Generate one feature row per edge, aligned with ``graph.edges()`` order."""
    edge_vectors_mtx = get_structural_edge_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold,
        sigma, use_edges_as_features,
    )
    if edge_attribute_key is None:
        return edge_vectors_mtx

    attribute_mtx = get_edge_attribute_mtx(original_graph, edge_attribute_key)
    if attribute_mtx.shape[1] == 0 and edge_attribute_width:
        attribute_mtx = np.empty((0, edge_attribute_width), dtype=float)
    if attribute_mtx.shape[0] != edge_vectors_mtx.shape[0]:
        raise ValueError("Each edge must have an attribute vector at edge_attribute_key.")

    if edge_vectors_mtx.shape[0] == 0:
        interactions = csr_matrix((0, attribute_mtx.shape[1]), dtype=float)
    else:
        dense_edge_vectors = edge_vectors_mtx.toarray()
        interaction_mtx = attribute_mtx.T.dot(dense_edge_vectors).dot(dense_edge_vectors.T).T
        interaction_mtx = np.power(np.abs(interaction_mtx), 1 / 3)
        interactions = csr_matrix(interaction_mtx)
    return sp.sparse.hstack(
        [csr_matrix(attribute_mtx), interactions, edge_vectors_mtx], format="csr"
    )


def get_edge_graph_vector(original_graph, radius, distance, connector, nbits,
                         edge_attribute_key, degree_threshold=None, sigma=None,
                         use_edges_as_features=True, edge_attribute_width=None,
                         node_vectors_mtx=None):
    """Aggregate structural edge rows through their continuous attributes."""
    edge_structural_mtx = get_structural_edge_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold,
        sigma, use_edges_as_features, node_vectors_mtx,
    )
    attribute_mtx = get_edge_attribute_mtx(original_graph, edge_attribute_key)
    if attribute_mtx.shape[1] == 0 and edge_attribute_width:
        attribute_mtx = np.empty((0, edge_attribute_width), dtype=float)
    if attribute_mtx.shape[0] != edge_structural_mtx.shape[0]:
        raise ValueError("Each edge must have an attribute vector at edge_attribute_key.")
    return csr_matrix(attribute_mtx.T.dot(edge_structural_mtx.toarray()).reshape(1, -1))


def get_node_graph_vectors(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True, edge_attribute_key=None, edge_attribute_width=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    node_vectors_mtx = get_structural_node_vectors(original_graph, radius, distance, connector, nbits, degree_threshold, sigma, use_edges_as_features)
    node_vectors_mtx = node_vectors_mtx.todense().A
    attribute_node_vectors_mtx = get_node_vectors(
        original_graph, radius, distance, connector, nbits, weight_key,
        node_attribute_key, degree_threshold, add_structural_node_information=False,
        sigma=sigma, use_edges_as_features=use_edges_as_features,
        structural_node_vectors_mtx=sp.sparse.csr_matrix(node_vectors_mtx),
    )
    vector_ = attribute_node_vectors_mtx.T.dot(node_vectors_mtx)
    if edge_attribute_key is not None:
        edge_graph_vector = get_edge_graph_vector(
            original_graph, radius, distance, connector, nbits,
            edge_attribute_key, degree_threshold, sigma, use_edges_as_features,
            edge_attribute_width,
            node_vectors_mtx=sp.sparse.csr_matrix(node_vectors_mtx),
        )
        vector_ = np.hstack([
            np.asarray(vector_).reshape(1, -1),
            edge_graph_vector.toarray(),
        ])
    vector = vector_.reshape(1, -1)
    vector = np.power(np.abs(vector), 1/2)
    vector = csr_matrix(vector)
    return csr_matrix(vector)


def get_graph_vector(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True, edge_attribute_width=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    structural_node_vectors_mtx = get_structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold, sigma, use_edges_as_features
    )
    node_vectors_mtx = structural_node_vectors_mtx
    if weight_key is not None:
        node_vectors_mtx = reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key)
    if node_attribute_key is None:
        node_graph_vector = csr_matrix(node_vectors_mtx.sum(axis=0))
    else:
        attribute_mtx = get_node_attribute_matrix(original_graph, node_attribute_key)
        node_vectors_mtx = node_vectors_mtx.todense().A
        vector_ = attribute_mtx.T.dot(node_vectors_mtx)
        node_graph_vector = csr_matrix(vector_.reshape(1, -1))

    if edge_attribute_key is None:
        return node_graph_vector

    edge_graph_vector = get_edge_graph_vector(
        original_graph, radius, distance, connector, nbits, edge_attribute_key,
        degree_threshold, sigma, use_edges_as_features, edge_attribute_width,
        structural_node_vectors_mtx,
    )
    return sp.sparse.hstack([node_graph_vector, edge_graph_vector], format="csr")


def split_into_chunks(lst, n):
    """
    Splits a list into 'n' nearly equal chunks.

    Args:
        lst (list): The list to split.
        n (int): The number of chunks.

    Returns:
        list of lists: A list containing 'n' sublists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def paired_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True, edge_attribute_width=None):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    n_cpus = mp.cpu_count()

    # A pool adds noticeable one-shot setup and graph-serialization cost.
    # Require roughly 1.5 graphs per worker before paying that cost.
    parallel_threshold = max(n_cpus + 1, (3 * n_cpus + 1) // 2)
    if parallel and len(graphs) >= parallel_threshold:
        def process_subset(subset):
            return [
                get_graph_vector(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, edge_attribute_key, degree_threshold, sigma, use_edges_as_features, edge_attribute_width)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            get_graph_vector(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, edge_attribute_key, degree_threshold, sigma, use_edges_as_features, edge_attribute_width)
            for graph in graphs
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_graphs_vector_encoder(graphs, radius, distance, connector, nbits,
                                     parallel=True, weight_key=None,
                                     node_attribute_key=None, edge_attribute_key=None,
                                     degree_threshold=None, sigma=None,
                                     use_edges_as_features=True,
                                     edge_attribute_width=None):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                get_node_graph_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features, edge_attribute_key=edge_attribute_key, edge_attribute_width=edge_attribute_width)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            get_node_graph_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features, edge_attribute_key=edge_attribute_key, edge_attribute_width=edge_attribute_width)
            for graph in graphs
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Encodes a list of graphs into node feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        list of scipy.sparse.csr_matrix: A list where each element is a matrix of node vectors for a graph.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                get_node_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_node_vectors_subsets = pool.map(process_subset, subsets)
        graph_node_vectors = [vec for subset in graph_node_vectors_subsets for vec in subset]
    else:
        graph_node_vectors = [
            get_node_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features)
            for graph in graphs
        ]

    return graph_node_vectors


def paired_edge_vector_encoder(graphs, radius, distance, connector, nbits,
                               parallel=True, edge_attribute_key=None,
                               degree_threshold=None, sigma=None,
                               use_edges_as_features=True, edge_attribute_width=None):
    """Encode each graph as a list of edge feature matrices."""
    def encode_subset(subset):
        return [
            get_edge_vectors(
                graph, radius, distance, connector, nbits,
                edge_attribute_key=edge_attribute_key,
                degree_threshold=degree_threshold,
                sigma=sigma,
                use_edges_as_features=use_edges_as_features,
                edge_attribute_width=edge_attribute_width,
            )
            for graph in subset
        ]

    n_cpus = mp.cpu_count()
    if parallel and len(graphs) > n_cpus:
        with mp.Pool(n_cpus) as pool:
            subsets = pool.map(encode_subset, split_into_chunks(graphs, n_cpus))
        edge_matrices = [matrix for subset in subsets for matrix in subset]
    else:
        edge_matrices = encode_subset(graphs)
    return edge_matrices


class BaseNSPPK(BaseEstimator, TransformerMixin):
    """
    BaseNSPPK is an abstract base class for encoding graphs into feature vectors.

    This class is initialized with generic components for embedding, clustering, and classification,
    allowing for flexibility and extensibility. Subclasses can specialize these components to 
    implement specific encoding strategies.

    Parameters:
        embedder (transformer, optional): A transformer for attribute embedding (e.g., TruncatedSVD).
        clustering_predictor (estimator, optional): A clustering model for attribute clustering (e.g., KMeans).
        classifier (estimator, optional): A classifier for predicting cluster labels (e.g., ExtraTreesClassifier).
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        use_node_kernel (bool, default=False): Whether to use node-level kernel encoding.
    """

    def __init__(self, embedder=None, clustering_predictor=None, classifier=None,
                 edge_embedder=None, edge_clustering_predictor=None, edge_classifier=None,
                 radius=1, distance=3, connector=0,
                 nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 node_attribute_key=None, edge_attribute_key=None, attribute_dim=None,
                 attribute_alphabet_size=None, edge_attribute_dim=None,
                 edge_attribute_alphabet_size=None, use_node_kernel=False, sigma=None,
                 use_edges_as_features=True):
        _validate_encoder_parameters(
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            node_attribute_key=node_attribute_key,
            edge_attribute_key=edge_attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size,
            edge_attribute_dim=edge_attribute_dim,
            edge_attribute_alphabet_size=edge_attribute_alphabet_size,
            sigma=sigma,
            use_node_kernel=use_node_kernel,
            use_edges_as_features=use_edges_as_features,
        )
        self.embedder = embedder
        self.clustering_predictor = clustering_predictor
        self.classifier = classifier
        self.edge_embedder = edge_embedder
        self.edge_clustering_predictor = edge_clustering_predictor
        self.edge_classifier = edge_classifier
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.edge_attribute_key = edge_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_attribute_alphabet_size = edge_attribute_alphabet_size
        self.use_node_kernel = use_node_kernel
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features
        self.is_fitted_ = not self.requires_fit()

    def __repr__(self):
        """
        Returns a string representation of the BaseNSPPK instance.

        Returns:
            str: The string representation.
        """
        infos = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({infos})"

    def requires_fit(self):
        """
        Return whether this encoder needs data-dependent fitting before ``transform``.

        Returns:
            bool: ``True`` when attribute embedding or attribute-label discretization must be learned.
        """
        return bool(
            (self.node_attribute_key and (self.attribute_dim or self.attribute_alphabet_size))
            or (self.edge_attribute_key and (self.edge_attribute_dim or self.edge_attribute_alphabet_size))
        )

    def fit(self, graphs, targets=None):
        """
        Fit the encoder on the given graphs.

        This method fits the embedder and clustering predictor if they are provided.
        If attribute clustering is specified, it also fits the classifier.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        if self.node_attribute_key and (self.attribute_dim or self.attribute_alphabet_size):
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.node_attribute_key]
                for graph in graphs
                for node_idx in graph.nodes()
            ])

            if self.attribute_dim and self.embedder:
                self.embedder.fit(attribute_mtx)
                attribute_mtx = self.embedder.transform(attribute_mtx)

            if self.attribute_alphabet_size and self.clustering_predictor:
                targets = self.clustering_predictor.fit_predict(attribute_mtx)
                self.classifier.fit(attribute_mtx, targets)

        if self.edge_attribute_key:
            edge_attribute_mtx = np.vstack([
                data[self.edge_attribute_key]
                for graph in graphs
                for _, _, _, data in _edge_records(graph)
            ]) if any(graph.number_of_edges() for graph in graphs) else np.empty((0, 0))
            if edge_attribute_mtx.shape[1]:
                self.edge_attribute_n_features_in_ = edge_attribute_mtx.shape[1]
            if self.edge_attribute_dim or self.edge_attribute_alphabet_size:
                if edge_attribute_mtx.shape[0] == 0:
                    raise ValueError("Edge attribute preprocessing requires at least one edge.")
                if self.edge_attribute_dim and self.edge_embedder:
                    self.edge_embedder.fit(edge_attribute_mtx)
                    edge_attribute_mtx = self.edge_embedder.transform(edge_attribute_mtx)
                if self.edge_attribute_alphabet_size and self.edge_clustering_predictor:
                    edge_targets = self.edge_clustering_predictor.fit_predict(edge_attribute_mtx)
                    self.edge_classifier.fit(edge_attribute_mtx, edge_targets)
                self.edge_attribute_n_features_in_ = edge_attribute_mtx.shape[1]

        self.is_fitted_ = True
        return self

    def embed_attributes(self, graphs):
        """
        Apply the embedder to reduce the dimensionality of node attributes.

        Args:
            graphs (list of networkx.Graph): List of graphs with attributes to embed.

        Returns:
            list of networkx.Graph: Graphs with embedded attributes.
        """
        if not self.embedder:
            return graphs

        out_graphs = []
        for graph in graphs:
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.node_attribute_key]
                for node_idx in graph.nodes()
            ])
            embeddings = self.embedder.transform(attribute_mtx)
            out_graph = graph.copy()
            for embedding, node_idx in zip(embeddings, out_graph.nodes()):
                out_graph.nodes[node_idx]['original_' + self.node_attribute_key] = graph.nodes[node_idx][self.node_attribute_key]
                out_graph.nodes[node_idx][self.node_attribute_key] = embedding
            out_graphs.append(out_graph)
        return out_graphs

    def set_discrete_labels(self, graphs):
        """
        Assign discrete labels to nodes in the graphs based on clustered attributes.

        Args:
            graphs (list of networkx.Graph): Input graphs.

        Returns:
            list of networkx.Graph: Graphs with updated node labels.
        """
        if not self.classifier:
            return graphs

        out_graphs = []
        for graph in graphs:
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.node_attribute_key]
                for node_idx in graph.nodes()
            ])
            labels = self.classifier.predict(attribute_mtx)
            out_graph = graph.copy()
            for label, node_idx in zip(labels, out_graph.nodes()):
                out_graph.nodes[node_idx]['label'] = label
            out_graphs.append(out_graph)
        return out_graphs

    def embed_edge_attributes(self, graphs):
        """Apply the fitted edge attribute reducer to copies of the input graphs."""
        if not self.edge_embedder:
            return graphs
        out_graphs = []
        for graph in graphs:
            edges = list(_edge_records(graph))
            if not edges:
                out_graphs.append(graph.copy())
                continue
            attribute_mtx = np.vstack([
                data[self.edge_attribute_key] for _, _, _, data in edges
            ])
            embeddings = self.edge_embedder.transform(attribute_mtx)
            out_graph = graph.copy()
            for (u, v, key, data), embedding in zip(edges, embeddings):
                out_data = out_graph.edges[u, v, key] if out_graph.is_multigraph() else out_graph.edges[u, v]
                out_data['original_' + self.edge_attribute_key] = data[self.edge_attribute_key]
                out_data[self.edge_attribute_key] = embedding
            out_graphs.append(out_graph)
        return out_graphs

    def set_discrete_edge_labels(self, graphs):
        """Replace edge labels with predicted edge attribute cluster labels."""
        if not self.edge_classifier:
            return graphs
        out_graphs = []
        for graph in graphs:
            edges = list(_edge_records(graph))
            if not edges:
                out_graphs.append(graph.copy())
                continue
            attribute_mtx = np.vstack([
                data[self.edge_attribute_key] for _, _, _, data in edges
            ])
            labels = self.edge_classifier.predict(attribute_mtx)
            out_graph = graph.copy()
            for (u, v, key, _), label in zip(edges, labels):
                out_data = out_graph.edges[u, v, key] if out_graph.is_multigraph() else out_graph.edges[u, v]
                out_data['label'] = label
            out_graphs.append(out_graph)
        return out_graphs

    def transform(self, graphs):
        """
        Transforms the input graphs into feature vectors.

        This method handles attribute embedding, label assignment, and encoding of graphs into feature vectors.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        check_is_fitted(self, 'is_fitted_')

        if self.node_attribute_key and self.attribute_dim:
            graphs = self.embed_attributes(graphs)
        if self.node_attribute_key and self.attribute_alphabet_size:
            graphs = self.set_discrete_labels(graphs)
        if self.edge_attribute_key and self.edge_attribute_dim:
            graphs = self.embed_edge_attributes(graphs)
        if self.edge_attribute_key and self.edge_attribute_alphabet_size:
            graphs = self.set_discrete_edge_labels(graphs)

        edge_attribute_width = getattr(self, 'edge_attribute_n_features_in_', None)
        if self.edge_attribute_key and self.edge_attribute_dim:
            edge_attribute_width = self.edge_attribute_dim
        if self.edge_attribute_key and edge_attribute_width is None:
            for graph in graphs:
                if graph.number_of_edges():
                    edge_attribute_width = len(next(_edge_records(graph))[3][self.edge_attribute_key])
                    break

        if self.use_node_kernel:
            data_mtx = paired_node_graphs_vector_encoder(
                graphs,
                self.radius,
                self.distance,
                self.connector,
                self.nbits,
                parallel=self.parallel,
                weight_key=self.weight_key,
                node_attribute_key=self.node_attribute_key,
                edge_attribute_key=self.edge_attribute_key,
                degree_threshold=self.degree_threshold,
                sigma=self.sigma,
                use_edges_as_features=self.use_edges_as_features,
                edge_attribute_width=edge_attribute_width,
            )
        else:
            data_mtx = paired_graphs_vector_encoder(
                graphs, 
                self.radius, 
                self.distance, 
                self.connector, 
                self.nbits,
                self.parallel, 
                self.weight_key,
                self.node_attribute_key,
                self.edge_attribute_key,
                self.degree_threshold,
                self.sigma,
                self.use_edges_as_features,
                edge_attribute_width,
            )

        if self.dense:
            data_mtx = data_mtx.todense().A

        return data_mtx

class NSPPK(BaseEstimator, TransformerMixin):
    """
    NSPPK (Neighborhood Subgraph Pairwise Propagation Kernel) class specialized from BaseNSPPK.

    This class encodes graphs into feature vectors suitable for machine learning models by capturing both
    local (node-level) and structural (subgraph-level) information. It specializes the BaseNSPPK by
    initializing it with specific components: TruncatedSVD for embedding, KMeans for clustering, and 
    ExtraTreesClassifier for classification.

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        use_node_kernel (bool, default=False): Whether to use node-level kernel encoding.
        sigma (float, optional): The sigma parameter for Gaussian weighting. Defaults to None.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None,
                 node_attribute_key=None, edge_attribute_key=None, attribute_dim=None, attribute_alphabet_size=None,
                 edge_attribute_dim=None, edge_attribute_alphabet_size=None, use_node_kernel=False, sigma=None,
                 use_edges_as_features=True, r=None, d=None, c=None):
        radius = _resolve_alias('radius', radius, 'r', r, 1)
        distance = _resolve_alias('distance', distance, 'd', d, 3)
        connector = _resolve_alias('connector', connector, 'c', c, 0)
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.edge_attribute_key = edge_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_attribute_alphabet_size = edge_attribute_alphabet_size
        self.use_node_kernel = use_node_kernel
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features
        self.r = radius
        self.d = distance
        self.c = connector
        self.loaded_balance_candidate_size_ = None
        self.loaded_label_counts_ = None
        self.loaded_class_weight_ = None

        self._initialize_components()

    def set_nbits(self, nbits):
        """
        Set the number of bits for hashing.

        Args:
            nbits (int): Number of bits for hashing.
        """
        self.nbits = nbits
        self._initialize_components()
        
    def _initialize_components(self):
        """
        Initializes the internal estimators based on the current parameters.
        """
        self.embedder = TruncatedSVD(n_components=self.attribute_dim) if self.attribute_dim else None
        self.clustering_predictor = KMeans(
            n_clusters=self.attribute_alphabet_size,
            random_state=0,
        ) if self.attribute_alphabet_size else None
        self.classifier = ExtraTreesClassifier(
            n_estimators=300, 
            n_jobs=-1 if self.parallel else None
        ) if self.attribute_alphabet_size else None
        self.edge_embedder = TruncatedSVD(n_components=self.edge_attribute_dim) if self.edge_attribute_dim else None
        self.edge_clustering_predictor = KMeans(
            n_clusters=self.edge_attribute_alphabet_size,
            random_state=0,
        ) if self.edge_attribute_alphabet_size else None
        self.edge_classifier = ExtraTreesClassifier(
            n_estimators=300,
            n_jobs=-1 if self.parallel else None,
        ) if self.edge_attribute_alphabet_size else None

        self.base_nsppk = BaseNSPPK(
            embedder=self.embedder,
            clustering_predictor=self.clustering_predictor,
            classifier=self.classifier,
            edge_embedder=self.edge_embedder,
            edge_clustering_predictor=self.edge_clustering_predictor,
            edge_classifier=self.edge_classifier,
            radius=self.radius,
            distance=self.distance,
            connector=self.connector,
            nbits=self.nbits,
            degree_threshold=self.degree_threshold,
            dense=self.dense,
            parallel=self.parallel,
            weight_key=self.weight_key,
            node_attribute_key=self.node_attribute_key,
            edge_attribute_key=self.edge_attribute_key,
            attribute_dim=self.attribute_dim,
            attribute_alphabet_size=self.attribute_alphabet_size,
            edge_attribute_dim=self.edge_attribute_dim,
            edge_attribute_alphabet_size=self.edge_attribute_alphabet_size,
            use_node_kernel=self.use_node_kernel,
            sigma=self.sigma,
            use_edges_as_features=self.use_edges_as_features
        )

    def __repr__(self):
        """
        Returns a string representation of the NSPPK instance.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}(radius={self.radius}, distance={self.distance}, connector={self.connector}, sigma={self.sigma}, use_edges_as_features={self.use_edges_as_features}" \
               f"nbits={self.nbits}, degree_threshold={self.degree_threshold}, dense={self.dense}, " \
               f"parallel={self.parallel}, weight_key={self.weight_key}, node_attribute_key={self.node_attribute_key}, " \
               f"attribute_dim={self.attribute_dim}, attribute_alphabet_size={self.attribute_alphabet_size}, " \
               f"edge_attribute_key={self.edge_attribute_key}, edge_attribute_dim={self.edge_attribute_dim}, " \
               f"edge_attribute_alphabet_size={self.edge_attribute_alphabet_size})"

    def fit(self, graphs, targets=None):
        """
        Fit the NSPPK encoder on the given graphs by delegating to BaseNSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.base_nsppk.fit(graphs, targets)
        return self

    def load_from(self, uri, type, reader=None, limit=None, random_state=None, verbose=False, balance=False, label_extractor=None, start_after_instance=0):
        """
        Load graphs from a local path or URL and materialize them as a list.

        Args:
            uri (str or pathlib.Path): Local path or remote URL.
            type (str): Built-in reader type, e.g. ``smiles``, ``sdf``, or ``pyg_pt``.
            reader (callable, optional): Custom reader override. It must accept ``uri`` and yield
                ``networkx.Graph`` objects.
            limit (int or float, optional): Selection control. ``None`` loads all graphs, a non-negative
                integer loads the first ``limit`` graphs, and a float in ``(0, 1)`` applies Bernoulli sampling.
            start_after_instance (int, default=0): Skip the first ``start_after_instance`` materialized graphs
                before applying ``limit`` and optional balancing.
            random_state (int or numpy.random.Generator, optional): Sampling seed used only for fractional ``limit``.
            verbose (bool, default=False): Whether to print cumulative loading statistics while streaming.
            balance (bool, default=False): Whether to rebalance the loaded graphs across class labels before returning them.
            label_extractor (callable, optional): Callable extracting a class label from each graph when ``balance=True``.
                Defaults to ``lambda graph: graph.graph['name']``.

        Returns:
            list of networkx.Graph: Materialized graphs. After the call, `loaded_balance_candidate_size_`,
            `loaded_label_counts_`, and `loaded_class_weight_` are populated when `balance=True` using the
            pre-balanced candidate pool and cleared otherwise.
        """
        _sync_graph_io_hooks()
        graphs, metadata = _graph_io._materialize_loaded_graphs(
            uri,
            type,
            reader=reader,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            balance=balance,
            label_extractor=label_extractor,
            start_after_instance=start_after_instance,
            return_metadata=True,
        )
        _set_loaded_label_metadata(self, metadata)
        return graphs

    def stream_from(self, uri, type, reader=None, limit=None, random_state=None, batch_size=128, verbose=False, label_extractor=None, start_after_instance=0):
        """
        Stream transformed graph batches from a local path or URL.

        Args:
            uri (str or pathlib.Path): Local path or remote URL.
            type (str): Built-in reader type, e.g. ``smiles``, ``sdf``, or ``pyg_pt``.
            reader (callable, optional): Custom reader override. It must accept ``uri`` and yield
                ``networkx.Graph`` objects.
            limit (int or float, optional): Selection control. ``None`` loads all graphs, a non-negative
                integer loads the first ``limit`` graphs, and a float in ``(0, 1)`` applies Bernoulli sampling.
            start_after_instance (int, default=0): Skip the first ``start_after_instance`` materialized graphs
                before applying ``limit``.
            random_state (int or numpy.random.Generator, optional): Sampling seed used only for fractional ``limit``.
            batch_size (int, default=128): Number of graphs per transformed batch.
            verbose (bool, default=False): Whether to print cumulative loading statistics while streaming.
            label_extractor (callable, optional): Callable extracting a target from each graph. When provided,
                the stream yields ``(X_batch, y_batch)`` pairs instead of feature batches alone. Defaults to
                ``lambda graph: graph.graph['name']``.

        Yields:
            numpy.ndarray or scipy.sparse.csr_matrix: Transformed batches produced by ``self.transform`` when
                ``label_extractor`` is omitted.
            tuple: ``(X_batch, y_batch)`` when ``label_extractor`` is provided.
        """
        check_is_fitted(self.base_nsppk, 'is_fitted_')
        _sync_graph_io_hooks()
        extractor = _graph_io._normalize_label_extractor(label_extractor) if label_extractor is not None else None
        graph_iterable = _graph_io._iter_loaded_graphs(
            uri,
            type,
            reader=reader,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            mode='stream',
            start_after_instance=start_after_instance,
        )
        for graph_batch in _graph_io._batched_graphs(graph_iterable, batch_size):
            X_batch = self.transform(graph_batch)
            if extractor is None:
                yield X_batch
                continue
            y_batch = np.asarray([extractor(graph) for graph in graph_batch])
            yield X_batch, y_batch

    def transform(self, graphs):
        """
        Transform the input graphs into feature vectors by delegating to BaseNSPPK.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        check_is_fitted(self.base_nsppk, 'is_fitted_')
        return self.base_nsppk.transform(graphs)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool, default=True): If True, will return the parameters for this estimator and
                                        contained sub-objects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = {
            'radius': self.radius,
            'distance': self.distance,
            'connector': self.connector,
            'r': self.radius,
            'd': self.distance,
            'c': self.connector,
            'nbits': self.nbits,
            'degree_threshold': self.degree_threshold,
            'dense': self.dense,
            'parallel': self.parallel,
            'weight_key': self.weight_key,
            'node_attribute_key': self.node_attribute_key,
            'edge_attribute_key': self.edge_attribute_key,
            'attribute_dim': self.attribute_dim,
            'attribute_alphabet_size': self.attribute_alphabet_size,
            'edge_attribute_dim': self.edge_attribute_dim,
            'edge_attribute_alphabet_size': self.edge_attribute_alphabet_size,
            'use_node_kernel': self.use_node_kernel,
            'sigma': self.sigma,
            'use_edges_as_features': self.use_edges_as_features
        }

        if deep:
            if self.embedder is not None:
                params.update({f'embedder__{k}': v for k, v in self.embedder.get_params().items()})
            if self.clustering_predictor is not None:
                params.update({f'clustering_predictor__{k}': v for k, v in self.clustering_predictor.get_params().items()})
            if self.classifier is not None:
                params.update({f'classifier__{k}': v for k, v in self.classifier.get_params().items()})
            if self.edge_embedder is not None:
                params.update({f'edge_embedder__{k}': v for k, v in self.edge_embedder.get_params().items()})
            if self.edge_clustering_predictor is not None:
                params.update({f'edge_clustering_predictor__{k}': v for k, v in self.edge_clustering_predictor.get_params().items()})
            if self.edge_classifier is not None:
                params.update({f'edge_classifier__{k}': v for k, v in self.edge_classifier.get_params().items()})

        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: The instance itself.
        """
        if not params:
            return self

        # Separate parameters for internal estimators and normalize aliases.
        embedder_params = {}
        clustering_predictor_params = {}
        classifier_params = {}
        edge_embedder_params = {}
        edge_clustering_predictor_params = {}
        edge_classifier_params = {}
        main_params = {}

        for key, value in params.items():
            if key.startswith('embedder__'):
                embedder_params[key.split('__', 1)[1]] = value
            elif key.startswith('clustering_predictor__'):
                clustering_predictor_params[key.split('__', 1)[1]] = value
            elif key.startswith('classifier__'):
                classifier_params[key.split('__', 1)[1]] = value
            elif key.startswith('edge_embedder__'):
                edge_embedder_params[key.split('__', 1)[1]] = value
            elif key.startswith('edge_clustering_predictor__'):
                edge_clustering_predictor_params[key.split('__', 1)[1]] = value
            elif key.startswith('edge_classifier__'):
                edge_classifier_params[key.split('__', 1)[1]] = value
            else:
                main_params[key] = value

        valid_main_params = set(self.get_params(deep=False))
        main_params = _normalize_parameter_aliases(main_params)
        unknown = set(main_params) - valid_main_params
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) for NSPPK: {sorted(unknown)!r}. "
                f"Valid parameters are: {sorted(valid_main_params)!r}."
            )

        for key, values, estimator in (
            ("embedder", embedder_params, self.embedder),
            ("clustering_predictor", clustering_predictor_params, self.clustering_predictor),
            ("classifier", classifier_params, self.classifier),
            ("edge_embedder", edge_embedder_params, self.edge_embedder),
            ("edge_clustering_predictor", edge_clustering_predictor_params, self.edge_clustering_predictor),
            ("edge_classifier", edge_classifier_params, self.edge_classifier),
        ):
            if values:
                valid_nested = estimator.get_params(deep=True) if estimator is not None else {}
                invalid = set(values) - set(valid_nested)
                if invalid:
                    raise ValueError(
                        f"Invalid parameter(s) for {key}: {sorted(invalid)!r}. "
                        f"Valid parameters are: {sorted(valid_nested)!r}."
                    )

        candidate_params = self.get_params(deep=False)
        candidate_params.update(main_params)
        _validate_encoder_parameters(
            radius=candidate_params['radius'],
            distance=candidate_params['distance'],
            connector=candidate_params['connector'],
            nbits=candidate_params['nbits'],
            degree_threshold=candidate_params['degree_threshold'],
            dense=candidate_params['dense'],
            parallel=candidate_params['parallel'],
            node_attribute_key=candidate_params['node_attribute_key'],
            edge_attribute_key=candidate_params['edge_attribute_key'],
            attribute_dim=candidate_params['attribute_dim'],
            attribute_alphabet_size=candidate_params['attribute_alphabet_size'],
            edge_attribute_dim=candidate_params['edge_attribute_dim'],
            edge_attribute_alphabet_size=candidate_params['edge_attribute_alphabet_size'],
            sigma=candidate_params['sigma'],
            use_node_kernel=candidate_params['use_node_kernel'],
            use_edges_as_features=candidate_params['use_edges_as_features'],
        )

        # Set validated main parameters and reinitialize components if needed.
        if main_params:
            for key, value in main_params.items():
                setattr(self, key, value)
            self.r, self.d, self.c = self.radius, self.distance, self.connector
            self._initialize_components()

        # Set parameters for internal estimators
        if embedder_params and self.embedder is not None:
            self.embedder.set_params(**embedder_params)
        if clustering_predictor_params and self.clustering_predictor is not None:
            self.clustering_predictor.set_params(**clustering_predictor_params)
        if classifier_params and self.classifier is not None:
            self.classifier.set_params(**classifier_params)
        if edge_embedder_params and self.edge_embedder is not None:
            self.edge_embedder.set_params(**edge_embedder_params)
        if edge_clustering_predictor_params and self.edge_clustering_predictor is not None:
            self.edge_clustering_predictor.set_params(**edge_clustering_predictor_params)
        if edge_classifier_params and self.edge_classifier is not None:
            self.edge_classifier.set_params(**edge_classifier_params)

        # Re-initialize BaseNSPPK with updated components
        if main_params:
            self.base_nsppk = BaseNSPPK(
                embedder=self.embedder,
                clustering_predictor=self.clustering_predictor,
                classifier=self.classifier,
                edge_embedder=self.edge_embedder,
                edge_clustering_predictor=self.edge_clustering_predictor,
                edge_classifier=self.edge_classifier,
                radius=self.radius,
                distance=self.distance,
                connector=self.connector,
                nbits=self.nbits,
                degree_threshold=self.degree_threshold,
                dense=self.dense,
                parallel=self.parallel,
                weight_key=self.weight_key,
                node_attribute_key=self.node_attribute_key,
                edge_attribute_key=self.edge_attribute_key,
                attribute_dim=self.attribute_dim,
                attribute_alphabet_size=self.attribute_alphabet_size,
                edge_attribute_dim=self.edge_attribute_dim,
                edge_attribute_alphabet_size=self.edge_attribute_alphabet_size,
                use_node_kernel=self.use_node_kernel,
                sigma=self.sigma,
                use_edges_as_features=self.use_edges_as_features
            )

        return self


class _NSPPKPredictor(BaseEstimator):
    """Shared graph-vectorization and prediction implementation."""

    _task = None

    def __init__(
        self,
        estimator="linear",
        estimator_params=None,
        radius=1,
        distance=3,
        connector=0,
        nbits=10,
        degree_threshold=None,
        dense=False,
        parallel=True,
        weight_key=None,
        node_attribute_key=None,
        edge_attribute_key=None,
        attribute_dim=None,
        attribute_alphabet_size=None,
        edge_attribute_dim=None,
        edge_attribute_alphabet_size=None,
        use_node_kernel=False,
        sigma=None,
        use_edges_as_features=True,
    ):
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.edge_attribute_key = edge_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_attribute_alphabet_size = edge_attribute_alphabet_size
        self.use_node_kernel = use_node_kernel
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features

    def _make_estimator(self):
        estimator = self.estimator
        if isinstance(estimator, str):
            shortcut = estimator.strip().lower()
            if shortcut == "linear":
                model = (
                    SGDClassifier(
                        loss="log_loss",
                        penalty=None,
                        max_iter=1,
                        tol=None,
                        shuffle=False,
                        learning_rate="constant",
                        eta0=1.0,
                        fit_intercept=False,
                        random_state=0,
                    )
                    if self._task == "classifier"
                    else Ridge()
                )
            elif shortcut == "rf":
                model = (
                    RandomForestClassifier(random_state=0)
                    if self._task == "classifier"
                    else RandomForestRegressor(random_state=0)
                )
            else:
                raise ValueError(
                    "estimator must be 'linear', 'rf', or a scikit-learn estimator; "
                    f"got {estimator!r}."
                )
        elif isinstance(estimator, type):
            try:
                model = estimator()
            except TypeError as exc:
                raise TypeError(
                    "Pass an estimator instance, or an estimator class that can be "
                    "constructed without arguments."
                ) from exc
        else:
            model = clone(estimator)

        if self.estimator_params is not None:
            if not isinstance(self.estimator_params, dict):
                raise TypeError("estimator_params must be a dictionary or None.")
            if not hasattr(model, "set_params"):
                raise TypeError("estimator must support scikit-learn set_params().")
            model.set_params(**self.estimator_params)

        if not callable(getattr(model, "fit", None)) or not callable(
            getattr(model, "predict", None)
        ):
            raise TypeError("estimator must implement scikit-learn fit() and predict().")

        correct_type = (
            is_classifier(model)
            if self._task == "classifier"
            else is_regressor(model)
        )
        if not correct_type:
            expected = "classifier" if self._task == "classifier" else "regressor"
            raise TypeError(f"estimator must be a scikit-learn {expected}.")
        return model

    def fit(self, X, y, **fit_params):
        """Fit the NSPPK encoder and the selected predictor on graph inputs."""
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        if y is None:
            raise ValueError("y is required.")
        check_consistent_length(graphs, y)

        encoder = NSPPK(
            radius=self.radius,
            distance=self.distance,
            connector=self.connector,
            nbits=self.nbits,
            degree_threshold=self.degree_threshold,
            dense=self.dense,
            parallel=self.parallel,
            weight_key=self.weight_key,
            node_attribute_key=self.node_attribute_key,
            edge_attribute_key=self.edge_attribute_key,
            attribute_dim=self.attribute_dim,
            attribute_alphabet_size=self.attribute_alphabet_size,
            edge_attribute_dim=self.edge_attribute_dim,
            edge_attribute_alphabet_size=self.edge_attribute_alphabet_size,
            use_node_kernel=self.use_node_kernel,
            sigma=self.sigma,
            use_edges_as_features=self.use_edges_as_features,
        )
        encoder.fit(graphs, y)
        features = encoder.transform(graphs)
        model = self._make_estimator()
        model.fit(features, y, **fit_params)

        self.encoder_ = encoder
        self.estimator_ = model
        if self._task == "classifier" and hasattr(model, "classes_"):
            self.classes_ = model.classes_
        return self

    def predict(self, X):
        """Predict targets for a list of NetworkX graphs."""
        check_is_fitted(self, attributes=["encoder_", "estimator_"])
        return self.estimator_.predict(self.encoder_.transform(list(X)))

    @available_if(lambda self: hasattr(getattr(self, "estimator_", None), "predict_proba"))
    def predict_proba(self, X):
        """Return class probabilities when the wrapped classifier provides them."""
        check_is_fitted(self, attributes=["encoder_", "estimator_"])
        return self.estimator_.predict_proba(self.encoder_.transform(list(X)))

    @available_if(lambda self: hasattr(getattr(self, "estimator_", None), "decision_function"))
    def decision_function(self, X):
        """Return decision scores when the wrapped classifier provides them."""
        check_is_fitted(self, attributes=["encoder_", "estimator_"])
        return self.estimator_.decision_function(self.encoder_.transform(list(X)))


class NSPPKClassifier(ClassifierMixin, _NSPPKPredictor):
    """Classify graph inputs using NSPPK features and a scikit-learn estimator.

    ``estimator`` defaults to ``"linear"``, a speed-first, one-pass
    ``SGDClassifier`` with logistic loss. Use ``"rf"`` for a
    ``RandomForestClassifier`` or pass a scikit-learn classifier instance.
    Pass estimator constructor overrides with ``estimator_params``.
    """

    _task = "classifier"


class NSPPKRegressor(RegressorMixin, _NSPPKPredictor):
    """Regress graph targets using NSPPK features and a scikit-learn estimator.

    ``estimator`` may be ``"linear"`` (Ridge), ``"rf"``
    (RandomForestRegressor), or a scikit-learn regressor instance. Pass
    estimator constructor overrides with ``estimator_params``.
    """

    _task = "regressor"


class NodeNSPPK(BaseEstimator, TransformerMixin):
    """
    NodeNSPPK (Node Neighborhood Subgraph Pairwise Propagation Kernel) class compatible with scikit-learn's Transformer interface.

    This class encodes graphs into node-level feature vectors suitable for
    node-level machine learning models. It captures both local (node-level) and structural (subgraph-level) information
    by hashing node labels and the structure of their neighborhoods. Additionally, it supports dimensionality reduction
    of node attributes using Singular Value Decomposition (SVD).

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): The target dimensionality for attribute vectors; if set, applies SVD for dimensionality reduction.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        sigma (float, optional): The sigma parameter for Gaussian weighting. Defaults to None.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None,
                 node_attribute_key=None, attribute_dim=None, attribute_alphabet_size=None, sigma=None, use_edges_as_features=True, r=None, d=None, c=None):
        radius = _resolve_alias('radius', radius, 'r', r, 1)
        distance = _resolve_alias('distance', distance, 'd', d, 3)
        connector = _resolve_alias('connector', connector, 'c', c, 0)
        self.nsppk = NSPPK(
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            weight_key=weight_key,
            node_attribute_key=node_attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size,
            sigma=sigma,
            use_edges_as_features=use_edges_as_features
        )
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features
        self.r = radius
        self.d = distance
        self.c = connector
        
    def get_params(self, deep=True):
        """Return the public encoder parameters used by scikit-learn tools."""
        return {
            'radius': self.radius,
            'distance': self.distance,
            'connector': self.connector,
            'r': self.radius,
            'd': self.distance,
            'c': self.connector,
            'nbits': self.nbits,
            'degree_threshold': self.degree_threshold,
            'dense': self.dense,
            'parallel': self.parallel,
            'weight_key': self.weight_key,
            'node_attribute_key': self.node_attribute_key,
            'attribute_dim': self.attribute_dim,
            'attribute_alphabet_size': self.attribute_alphabet_size,
            'sigma': self.sigma,
            'use_edges_as_features': self.use_edges_as_features,
        }

    def set_params(self, **params):
        """Set validated public parameters and keep the wrapped NSPPK in sync."""
        if not params:
            return self

        normalized = _normalize_parameter_aliases(dict(params))
        valid = set(self.get_params(deep=False))
        unknown = set(normalized) - valid
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) for NodeNSPPK: {sorted(unknown)!r}. "
                f"Valid parameters are: {sorted(valid)!r}."
            )

        candidate = self.get_params(deep=False)
        candidate.update(normalized)
        _validate_encoder_parameters(
            radius=candidate['radius'],
            distance=candidate['distance'],
            connector=candidate['connector'],
            nbits=candidate['nbits'],
            degree_threshold=candidate['degree_threshold'],
            dense=candidate['dense'],
            parallel=candidate['parallel'],
            node_attribute_key=candidate['node_attribute_key'],
            edge_attribute_key=None,
            attribute_dim=candidate['attribute_dim'],
            attribute_alphabet_size=candidate['attribute_alphabet_size'],
            edge_attribute_dim=None,
            edge_attribute_alphabet_size=None,
            sigma=candidate['sigma'],
            use_node_kernel=False,
            use_edges_as_features=candidate['use_edges_as_features'],
        )

        for key, value in normalized.items():
            setattr(self, key, value)
        self.r, self.d, self.c = self.radius, self.distance, self.connector
        self.nsppk.set_params(**normalized)
        return self

    def set_nbits(self, nbits):
        """
        Set the number of bits for hashing.

        Args:
            nbits (int): Number of bits for hashing.
        """
        self.set_params(nbits=nbits)

    def __repr__(self):
        """
        Generate a string representation of the NodeNSPPK instance.

        Returns:
            str: The string representation of the object.
        """
        return f"{self.__class__.__name__}({self.nsppk})"

    def fit(self, graphs, targets=None):
        """
        Fit the NodeNSPPK encoder on the given graphs by delegating to NSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.nsppk.fit(graphs, targets)
        return self

    def load_from(self, uri, type, reader=None, limit=None, random_state=None, verbose=False, balance=False, label_extractor=None, start_after_instance=0):
        """
        Load graphs from a local path or URL and materialize them as a list.

        Args:
            uri (str or pathlib.Path): Local path or remote URL.
            type (str): Built-in reader type, e.g. ``smiles``, ``sdf``, or ``pyg_pt``.
            reader (callable, optional): Custom reader override. It must accept ``uri`` and yield
                ``networkx.Graph`` objects.
            limit (int or float, optional): Selection control. ``None`` loads all graphs, a non-negative
                integer loads the first ``limit`` graphs, and a float in ``(0, 1)`` applies Bernoulli sampling.
            start_after_instance (int, default=0): Skip the first ``start_after_instance`` materialized graphs
                before applying ``limit`` and optional balancing.
            random_state (int or numpy.random.Generator, optional): Sampling seed used only for fractional ``limit``.
            verbose (bool, default=False): Whether to print cumulative loading statistics while streaming.
            balance (bool, default=False): Whether to rebalance the loaded graphs across class labels before returning them.
            label_extractor (callable, optional): Callable extracting a class label from each graph when ``balance=True``.
                Defaults to ``lambda graph: graph.graph['name']``.

        Returns:
            list of networkx.Graph: Materialized graphs. After the call, `loaded_balance_candidate_size_`,
            `loaded_label_counts_`, and `loaded_class_weight_` mirror the underlying `NSPPK` loader metadata
            when `balance=True` and are cleared otherwise.
        """
        graphs = self.nsppk.load_from(
            uri,
            type,
            reader=reader,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            balance=balance,
            label_extractor=label_extractor,
            start_after_instance=start_after_instance,
        )
        self.loaded_balance_candidate_size_ = self.nsppk.loaded_balance_candidate_size_
        self.loaded_label_counts_ = self.nsppk.loaded_label_counts_
        self.loaded_class_weight_ = self.nsppk.loaded_class_weight_
        return graphs

    def stream_from(self, uri, type, reader=None, limit=None, random_state=None, batch_size=128, verbose=False, label_extractor=None, start_after_instance=0):
        """
        Stream transformed node-feature batches from a local path or URL.

        Args:
            uri (str or pathlib.Path): Local path or remote URL.
            type (str): Built-in reader type, e.g. ``smiles``, ``sdf``, or ``pyg_pt``.
            reader (callable, optional): Custom reader override. It must accept ``uri`` and yield
                ``networkx.Graph`` objects.
            limit (int or float, optional): Selection control. ``None`` loads all graphs, a non-negative
                integer loads the first ``limit`` graphs, and a float in ``(0, 1)`` applies Bernoulli sampling.
            start_after_instance (int, default=0): Skip the first ``start_after_instance`` materialized graphs
                before applying ``limit``.
            random_state (int or numpy.random.Generator, optional): Sampling seed used only for fractional ``limit``.
            batch_size (int, default=128): Number of graphs per transformed batch.
            verbose (bool, default=False): Whether to print cumulative loading statistics while streaming.
            label_extractor (callable, optional): Callable extracting a target from each graph. When provided,
                the stream yields ``(X_batch, y_batch)`` pairs instead of feature batches alone. Defaults to
                ``lambda graph: graph.graph['name']``.

        Yields:
            list of numpy.ndarray or list of scipy.sparse.csr_matrix: Transformed batches produced by
                ``self.transform`` when ``label_extractor`` is omitted.
            tuple: ``(X_batch, y_batch)`` when ``label_extractor`` is provided.
        """
        check_is_fitted(self.nsppk.base_nsppk, 'is_fitted_')
        _sync_graph_io_hooks()
        extractor = _graph_io._normalize_label_extractor(label_extractor) if label_extractor is not None else None
        graph_iterable = _graph_io._iter_loaded_graphs(
            uri,
            type,
            reader=reader,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            mode='stream',
            start_after_instance=start_after_instance,
        )
        for graph_batch in _graph_io._batched_graphs(graph_iterable, batch_size):
            X_batch = self.transform(graph_batch)
            if extractor is None:
                yield X_batch
                continue
            y_batch = np.asarray([extractor(graph) for graph in graph_batch])
            yield X_batch, y_batch

    def transform(self, graphs):
        """
        Transform the input graphs into node-level feature vectors.

        This method encodes each graph into a list of node feature matrices, where each matrix corresponds to a graph
        and contains feature vectors for its nodes. If `attribute_dim` is specified, it applies SVD to reduce the
        dimensionality of node attributes before encoding. Additionally, if `attribute_alphabet_size` is specified,
        it assigns discrete labels to nodes based on clustered attributes.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            list of numpy.ndarray or list of scipy.sparse.csr_matrix: A list where each element is a node feature matrix for a graph.
        """
        check_is_fitted(self.nsppk.base_nsppk, 'is_fitted_')

        if self.nsppk.base_nsppk.node_attribute_key and self.nsppk.base_nsppk.attribute_dim:
            graphs = self.nsppk.base_nsppk.embed_attributes(graphs)
        if self.nsppk.base_nsppk.node_attribute_key and self.nsppk.base_nsppk.attribute_alphabet_size:
            graphs = self.nsppk.base_nsppk.set_discrete_labels(graphs)

        nodes_data_mtx_list = paired_node_vector_encoder(
            graphs,
            self.nsppk.base_nsppk.radius,
            self.nsppk.base_nsppk.distance,
            self.nsppk.base_nsppk.connector,
            self.nsppk.base_nsppk.nbits,
            self.nsppk.base_nsppk.parallel,
            self.nsppk.base_nsppk.weight_key,
            self.nsppk.base_nsppk.node_attribute_key,
            self.nsppk.base_nsppk.degree_threshold,
            self.nsppk.sigma,
            self.nsppk.use_edges_as_features
        )

        if self.nsppk.base_nsppk.dense:
            nodes_data_mtx_list = [mtx.todense().A for mtx in nodes_data_mtx_list]

        return nodes_data_mtx_list


class EdgeNSPPK(BaseEstimator, TransformerMixin):
    """Return one feature row per undirected edge in each input graph."""

    def __init__(self, radius=1, distance=3, connector=0, nbits=10,
                 degree_threshold=None, dense=True, parallel=True,
                 edge_attribute_key=None, edge_attribute_dim=None,
                 edge_attribute_alphabet_size=None, sigma=None,
                 use_edges_as_features=True, r=None, d=None, c=None):
        radius = _resolve_alias('radius', radius, 'r', r, 1)
        distance = _resolve_alias('distance', distance, 'd', d, 3)
        connector = _resolve_alias('connector', connector, 'c', c, 0)
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.edge_attribute_key = edge_attribute_key
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_attribute_alphabet_size = edge_attribute_alphabet_size
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features
        self.r, self.d, self.c = radius, distance, connector
        self.nsppk = NSPPK(
            radius=radius, distance=distance, connector=connector, nbits=nbits,
            degree_threshold=degree_threshold, dense=dense, parallel=parallel,
            edge_attribute_key=edge_attribute_key,
            edge_attribute_dim=edge_attribute_dim,
            edge_attribute_alphabet_size=edge_attribute_alphabet_size,
            sigma=sigma, use_edges_as_features=use_edges_as_features,
        )

    def get_params(self, deep=True):
        return {
            'radius': self.radius,
            'distance': self.distance,
            'connector': self.connector,
            'r': self.radius,
            'd': self.distance,
            'c': self.connector,
            'nbits': self.nbits,
            'degree_threshold': self.degree_threshold,
            'dense': self.dense,
            'parallel': self.parallel,
            'edge_attribute_key': self.edge_attribute_key,
            'edge_attribute_dim': self.edge_attribute_dim,
            'edge_attribute_alphabet_size': self.edge_attribute_alphabet_size,
            'sigma': self.sigma,
            'use_edges_as_features': self.use_edges_as_features,
        }

    def set_params(self, **params):
        if not params:
            return self
        normalized = _normalize_parameter_aliases(dict(params))
        valid = set(self.get_params(deep=False))
        unknown = set(normalized) - valid
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) for EdgeNSPPK: {sorted(unknown)!r}. "
                f"Valid parameters are: {sorted(valid)!r}."
            )
        self.nsppk.set_params(**normalized)
        for key, value in normalized.items():
            setattr(self, key, value)
        self.r, self.d, self.c = self.radius, self.distance, self.connector
        return self

    def fit(self, graphs, targets=None):
        self.nsppk.fit(graphs, targets)
        return self

    def transform(self, graphs):
        base = self.nsppk.base_nsppk
        check_is_fitted(base, 'is_fitted_')
        graphs = list(graphs)
        if base.edge_attribute_key and base.edge_attribute_dim:
            graphs = base.embed_edge_attributes(graphs)
        if base.edge_attribute_key and base.edge_attribute_alphabet_size:
            graphs = base.set_discrete_edge_labels(graphs)

        edge_attribute_width = getattr(base, 'edge_attribute_n_features_in_', None)
        if base.edge_attribute_key and base.edge_attribute_dim:
            edge_attribute_width = base.edge_attribute_dim
        if base.edge_attribute_key and edge_attribute_width is None:
            for graph in graphs:
                if graph.number_of_edges():
                    edge_attribute_width = len(next(_edge_records(graph))[3][base.edge_attribute_key])
                    break

        edge_matrices = paired_edge_vector_encoder(
            graphs, base.radius, base.distance, base.connector, base.nbits,
            parallel=base.parallel,
            edge_attribute_key=base.edge_attribute_key,
            degree_threshold=base.degree_threshold,
            sigma=base.sigma,
            use_edges_as_features=base.use_edges_as_features,
            edge_attribute_width=edge_attribute_width,
        )
        if base.dense:
            edge_matrices = [matrix.toarray() for matrix in edge_matrices]
        return edge_matrices

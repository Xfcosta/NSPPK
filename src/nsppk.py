import math
from collections import defaultdict, Counter
import copy
from dataclasses import dataclass
from itertools import islice
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
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
import multiprocessing_on_dill as mp

import graph_io as _graph_io
from hash import (
    _hash,
    _node_hash_directed,
    _node_hash_multi_directed,
    _node_hash_undirected_multi,
    _node_hash_undirected_simple,
    _edge_triplet_hash_directed,
    _edge_triplet_hash_undirected,
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

__version__ = "3.5.0"

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


def _inclusive_parameter_bounds(value, name):
    """Normalize a cutoff or inclusive (min, max) range to integer bounds."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return 0, int(value)
    if isinstance(value, tuple) and len(value) == 2:
        lower, upper = value
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
            raise ValueError(f"{name} range endpoints must be non-negative integers.")
        if lower < 0 or upper < 0 or lower > upper:
            raise ValueError(f"{name} range must satisfy 0 <= min <= max.")
        return int(lower), int(upper)
    raise ValueError(
        f"{name} must be a non-negative integer cutoff or an inclusive (min, max) tuple."
    )


def _inclusive_parameter_values(value, name):
    lower, upper = _inclusive_parameter_bounds(value, name)
    return range(lower, upper + 1)


def _resolve_simplex_feature_order(simplex_feature_order, simplex_feature_max_order):
    """Resolve the legacy max-order spelling to the canonical order range."""
    if simplex_feature_max_order is None:
        return simplex_feature_order
    if simplex_feature_order == 2 or simplex_feature_order == simplex_feature_max_order:
        return simplex_feature_max_order
    raise ValueError(
        "simplex_feature_order and its legacy alias "
        "simplex_feature_max_order cannot specify different values."
    )


def _validate_encoder_parameters(
    *, radius, distance, connector, nbits, degree_threshold, dense, parallel,
    node_attribute_key, edge_attribute_key, attribute_dim, attribute_alphabet_size,
    edge_attribute_dim, edge_attribute_alphabet_size, sigma, use_node_kernel,
    use_simplex_features, simplex_feature_order=2,
):
    for name, value in (("radius", radius), ("distance", distance), ("connector", connector)):
        _inclusive_parameter_bounds(value, name)
    _inclusive_parameter_bounds(simplex_feature_order, "simplex_feature_order")

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
        ("use_simplex_features", use_simplex_features),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean.")


def _normalize_feature_parameter_sets(
    parameters,
    *,
    radius,
    distance,
    connector,
    sigma,
    use_simplex_features,
    simplex_feature_order,
):
    """Resolve optional per-feature settings against the encoder defaults."""
    defaults = {
        "radius": radius,
        "distance": distance,
        "connector": connector,
        "sigma": sigma,
        "use_simplex_features": use_simplex_features,
        "simplex_feature_order": simplex_feature_order,
    }
    if parameters is None:
        parameter_sets = [defaults]
    else:
        if isinstance(parameters, (str, bytes)) or not isinstance(parameters, (list, tuple)):
            raise TypeError("parameters must be a list or tuple of parameter dictionaries, or None.")
        if not parameters:
            raise ValueError("parameters must contain at least one parameter dictionary.")
        parameter_sets = []
        allowed = {
            "radius", "distance", "connector", "sigma", "use_simplex_features",
            "simplex_feature_order", "simplex_feature_max_order", "r", "d", "c",
        }
        aliases = {"r": "radius", "d": "distance", "c": "connector"}
        for index, parameter_set in enumerate(parameters):
            if not isinstance(parameter_set, dict):
                raise TypeError(f"parameters[{index}] must be a dictionary.")
            unknown = set(parameter_set) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in parameters[{index}]: {sorted(unknown)!r}. "
                    f"Supported keys are: {sorted(allowed)!r}."
                )
            config = defaults.copy()
            for alias, canonical in aliases.items():
                if alias in parameter_set and canonical in parameter_set:
                    if parameter_set[alias] != parameter_set[canonical]:
                        raise ValueError(
                            f"Conflicting values for '{canonical}' and '{alias}' "
                            f"in parameters[{index}]."
                        )
                elif alias in parameter_set:
                    config[canonical] = parameter_set[alias]
            for key in ("radius", "distance", "connector", "sigma", "use_simplex_features"):
                if key in parameter_set:
                    config[key] = parameter_set[key]

            if (
                "simplex_feature_order" in parameter_set
                and "simplex_feature_max_order" in parameter_set
                and parameter_set["simplex_feature_order"] != parameter_set["simplex_feature_max_order"]
            ):
                raise ValueError(
                    f"Conflicting simplex feature order values in parameters[{index}]."
                )
            if "simplex_feature_order" in parameter_set:
                config["simplex_feature_order"] = parameter_set["simplex_feature_order"]
            elif "simplex_feature_max_order" in parameter_set:
                config["simplex_feature_order"] = parameter_set["simplex_feature_max_order"]
            parameter_sets.append(config)

    for index, config in enumerate(parameter_sets):
        for key in ("radius", "distance", "connector", "simplex_feature_order"):
            _inclusive_parameter_bounds(config[key], f"parameters[{index}].{key}")
        if not isinstance(config["use_simplex_features"], (bool, np.bool_)):
            raise ValueError(f"parameters[{index}].use_simplex_features must be a boolean.")
        value = config["sigma"]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"parameters[{index}].sigma must be None or a finite positive number.")
    return tuple(parameter_sets)


_PARAMETER_ALIASES = {
    "r": "radius",
    "d": "distance",
    "c": "connector",
    "simplex_feature_max_order": "simplex_feature_order",
}


class _SimplexFeatureOrderAliasMixin:
    @property
    def simplex_feature_max_order(self):
        """Legacy alias for ``simplex_feature_order``."""
        return self.simplex_feature_order

    @simplex_feature_max_order.setter
    def simplex_feature_max_order(self, value):
        self.simplex_feature_order = value


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


def _normalize_root_node_ids(graphs, root_node_ids):
    """Validate and normalize per-graph selections of outermost feature roots."""
    graphs = list(graphs)
    if root_node_ids is None:
        return [None] * len(graphs)
    try:
        selections = list(root_node_ids)
    except TypeError as exc:
        raise ValueError("root_node_ids must contain one node-ID collection per graph.") from exc
    if len(selections) != len(graphs):
        raise ValueError(
            "root_node_ids must contain one node-ID collection per graph; "
            f"got {len(selections)} selections for {len(graphs)} graphs."
        )

    normalized = []
    for graph_index, (graph, selection) in enumerate(zip(graphs, selections)):
        if selection is None:
            normalized.append(None)
            continue
        if isinstance(selection, (str, bytes)):
            raise ValueError(
                f"root_node_ids[{graph_index}] must be a collection of node IDs, not a string."
            )
        try:
            selected_nodes = set(selection)
        except TypeError as exc:
            raise ValueError(
                f"root_node_ids[{graph_index}] must be an iterable of node IDs."
            ) from exc
        unknown_nodes = selected_nodes - set(graph.nodes())
        if unknown_nodes:
            raise ValueError(
                f"root_node_ids[{graph_index}] contains node IDs absent from its graph: "
                f"{list(unknown_nodes)[:5]!r}."
            )
        normalized.append(selected_nodes)
    return normalized

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


def _undirected_simple_edge_id(graph, u, v, key=None):
    try:
        return _edge_key(u, v)
    except TypeError:
        node_order = {node: idx for idx, node in enumerate(graph.nodes())}
        return (u, v) if node_order[u] <= node_order[v] else (v, u)


def _undirected_multi_edge_id(graph, u, v, key=None):
    return _undirected_simple_edge_id(graph, u, v) + (key,)


def _directed_simple_edge_id(graph, u, v, key=None):
    return (u, v)


def _directed_multi_edge_id(graph, u, v, key=None):
    return (u, v, key)


def _simple_edge_records(graph):
    for u, v, data in graph.edges(data=True):
        yield u, v, None, data


def _multi_edge_records(graph):
    yield from graph.edges(keys=True, data=True)


def _simple_undirected_incident_edges(graph, node):
    for u, v, data in graph.edges(node, data=True):
        other = v if u == node else u
        yield other, None, data, u, v


def _multi_undirected_incident_edges(graph, node):
    for u, v, key, data in graph.edges(node, keys=True, data=True):
        other = v if u == node else u
        yield other, key, data, u, v


def _simple_directed_incident_edges(graph, node):
    for u, v, data in graph.out_edges(node, data=True):
        yield v, None, data, u, v
    for u, v, data in graph.in_edges(node, data=True):
        if u != v:
            yield u, None, data, u, v


def _multi_directed_incident_edges(graph, node):
    for u, v, key, data in graph.out_edges(node, keys=True, data=True):
        yield v, key, data, u, v
    for u, v, key, data in graph.in_edges(node, keys=True, data=True):
        if u != v:
            yield u, key, data, u, v


def _undirected_projection(graph):
    return graph


def _directed_undirected_projection(graph):
    return graph.to_undirected(as_view=True)


def _undirected_path_union_signature(graph, node_signatures, union_path_edges):
    return hash_set(node_signatures)


def _edge_aware_path_union_signature(graph, node_signatures, union_path_edges):
    edge_triplet_hashes = graph.graph.get('_structural_cache', {}).get('edge_triplet_hashes', {})
    edge_signatures = [
        hash_sequence(("path-edge", edge_triplet_hashes[edge_id]))
        for edge_id in union_path_edges
        if edge_id in edge_triplet_hashes
    ]
    return hash_set(node_signatures + edge_signatures)


def _undirected_edge_row(row, graph, u, v, data, nbits):
    return row


def _directed_edge_row(row, graph, u, v, data, nbits):
    orientation_signature = hash_sequence((
        "directed-edge-row",
        _hash(graph.nodes[u].get('label', 0)),
        _hash(data.get('label', 0)),
        _hash(graph.nodes[v].get('label', 0)),
    ))
    orientation_code = hash_value(orientation_signature, nbits=nbits)
    row = row.tolil()
    row[0, orientation_code] += 1
    return row.tocsr()


def _add_simple_subgraph_edge(subgraph, original_graph, edge_id):
    u, v = edge_id
    data = copy.deepcopy(original_graph.edges[u, v])
    subgraph.add_edge(u, v, **data)
    subgraph.edges[u, v]['original_edge'] = edge_id


def _add_multi_subgraph_edge(subgraph, original_graph, edge_id):
    u, v, key = edge_id
    data = copy.deepcopy(original_graph.edges[u, v, key])
    subgraph.add_edge(u, v, key=key, **data)
    subgraph.edges[u, v, key]['original_edge'] = edge_id


@dataclass(frozen=True)
class _GraphHashStrategy:
    name: str
    directed: bool
    multigraph: bool
    node_hasher: object
    edge_hasher: object
    edge_records: object
    incident_edges: object
    edge_id: object
    traversal_graph: object
    path_union_signature: object
    edge_row: object
    add_subgraph_edge: object


_GRAPH_HASH_STRATEGIES = {
    (False, False): _GraphHashStrategy(
        'Graph', False, False, _node_hash_undirected_simple,
        _edge_triplet_hash_undirected, _simple_edge_records,
        _simple_undirected_incident_edges, _undirected_simple_edge_id,
        _undirected_projection, _undirected_path_union_signature,
        _undirected_edge_row, _add_simple_subgraph_edge,
    ),
    (True, False): _GraphHashStrategy(
        'DiGraph', True, False, _node_hash_directed,
        _edge_triplet_hash_directed, _simple_edge_records,
        _simple_directed_incident_edges, _directed_simple_edge_id,
        _directed_undirected_projection, _edge_aware_path_union_signature,
        _directed_edge_row, _add_simple_subgraph_edge,
    ),
    (False, True): _GraphHashStrategy(
        'MultiGraph', False, True, _node_hash_undirected_multi,
        _edge_triplet_hash_undirected, _multi_edge_records,
        _multi_undirected_incident_edges, _undirected_multi_edge_id,
        _undirected_projection, _edge_aware_path_union_signature,
        _undirected_edge_row, _add_multi_subgraph_edge,
    ),
    (True, True): _GraphHashStrategy(
        'MultiDiGraph', True, True, _node_hash_multi_directed,
        _edge_triplet_hash_directed, _multi_edge_records,
        _multi_directed_incident_edges, _directed_multi_edge_id,
        _directed_undirected_projection, _edge_aware_path_union_signature,
        _directed_edge_row, _add_multi_subgraph_edge,
    ),
}


def _select_graph_hash_strategy(graphs):
    sample = list(islice(iter(graphs), 10))
    if not sample:
        return _GRAPH_HASH_STRATEGIES[(False, False)]
    first = sample[0]
    kind = (bool(first.is_directed()), bool(first.is_multigraph()))
    strategy = _GRAPH_HASH_STRATEGIES[kind]
    for graph in sample[1:]:
        graph_kind = (bool(graph.is_directed()), bool(graph.is_multigraph()))
        if graph_kind != kind:
            raise ValueError(
                "All graphs in a batch must have the same directed/multigraph type; "
                f"expected {strategy.name}, got {type(graph).__name__}."
            )
    return strategy


def _edge_id(graph, u, v, key=None, strategy=None):
    if strategy is None:
        strategy = _select_graph_hash_strategy((graph,))
    return strategy.edge_id(graph, u, v, key)


def _edge_records(graph, strategy=None):
    if strategy is None:
        strategy = _select_graph_hash_strategy((graph,))
    yield from strategy.edge_records(graph)


def _incident_edge_records(graph, node, strategy=None):
    if strategy is None:
        strategy = _select_graph_hash_strategy((graph,))
    yield from strategy.incident_edges(graph, node)


def _undirected_pair_key(u, v):
    return frozenset((u, v))


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


def _build_structural_cache(graph, strategy):
    cache = {
        'neighbors': {},
        'degrees': {},
        'node_order': {},
        'node_label_hashes': {},
        'node_hashes': {},
        'edge_triplet_hashes': {},
        'rooted_graph_hashes': {},
        'edge_ids_by_pair': defaultdict(list),
        'simplex_fragments': {},
    }
    for node_order, node_idx in enumerate(graph.nodes()):
        neighbors = tuple(graph.graph['_traversal_graph'].neighbors(node_idx))
        cache['neighbors'][node_idx] = neighbors
        cache['degrees'][node_idx] = len(neighbors)
        cache['node_order'][node_idx] = node_order
    for u, v, key, _ in strategy.edge_records(graph):
        cache['edge_ids_by_pair'][_undirected_pair_key(u, v)].append(
            strategy.edge_id(graph, u, v, key)
        )
    return cache


def _ensure_simplex_edge_cache(graph):
    structural_cache = graph.graph['_structural_cache']
    if 'simplex_edges_by_pair' in structural_cache:
        return structural_cache['simplex_edges_by_pair']

    strategy = graph.graph['_hash_strategy']
    edge_triplet_hashes = structural_cache['edge_triplet_hashes']
    edges_by_pair = defaultdict(list)
    for u, v, key, _ in strategy.edge_records(graph):
        if u == v:
            continue
        edge_id = strategy.edge_id(graph, u, v, key)
        edges_by_pair[frozenset((u, v))].append(edge_triplet_hashes[edge_id])
    structural_cache['simplex_edges_by_pair'] = edges_by_pair
    return edges_by_pair


def _simplex_fragment_hash(graph, simplex_nodes):
    """Hash a singleton by label, or a higher simplex by its edge hashes."""
    simplex_nodes = tuple(simplex_nodes)
    if len(simplex_nodes) == 1:
        return graph.graph['_structural_cache']['node_label_hashes'][simplex_nodes[0]]

    edges_by_pair = _ensure_simplex_edge_cache(graph)
    return hash_set(
        edge_hash
        for index, left in enumerate(simplex_nodes)
        for right in simplex_nodes[index + 1:]
        for edge_hash in edges_by_pair[frozenset((left, right))]
    )


def _simplex_fragments(graph, source_node_ids, simplex_feature_order):
    """Enumerate and cache all simplex fragments in one induced region."""
    source_node_ids = frozenset(source_node_ids)
    order_min, order_max = _inclusive_parameter_bounds(
        simplex_feature_order, "simplex_feature_order"
    )
    cache = graph.graph.get('_structural_cache', {}).get('simplex_fragments', {})
    cache_key = (source_node_ids, order_min, order_max)
    if cache_key in cache:
        return cache[cache_key]

    _ensure_simplex_edge_cache(graph)
    traversal_graph = graph.graph['_traversal_graph']
    simple_region = nx.Graph(traversal_graph.subgraph(source_node_ids))
    simple_region.remove_edges_from(nx.selfloop_edges(simple_region))
    fragments = []
    node_order = graph.graph['_structural_cache']['node_order']
    for clique in _bounded_cliques(simple_region, order_max, node_order):
        order = len(clique) - 1
        if order >= order_min:
            fragments.append((order, _simplex_fragment_hash(graph, clique), tuple(clique)))

    fragments = tuple(fragments)
    cache[cache_key] = fragments
    return fragments


def _simplex_fragment_edge_ids(graph, simplex_nodes):
    """Return non-loop graph edges that belong to one simplex vertex set."""
    edge_ids_by_pair = graph.graph['_structural_cache']['edge_ids_by_pair']
    simplex_nodes = tuple(simplex_nodes)
    return {
        edge_id
        for index, left in enumerate(simplex_nodes)
        for right in simplex_nodes[index + 1:]
        for edge_id in edge_ids_by_pair.get(frozenset((left, right)), ())
    }


_FAST_CLIQUE_MAX_SIMPLEX_DIMENSION = 3


def _bounded_cliques(graph, max_simplex_dimension, node_order):
    """Enumerate bounded cliques with bitset intersections for dimensions <= 3."""
    max_size = max_simplex_dimension + 1
    if max_simplex_dimension > _FAST_CLIQUE_MAX_SIMPLEX_DIMENSION:
        for clique in nx.enumerate_all_cliques(graph):
            if len(clique) > max_size:
                break
            yield tuple(sorted(clique, key=node_order.__getitem__))
        return

    ordered_nodes = sorted(graph.nodes(), key=node_order.__getitem__)
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    higher_neighbor_masks = []
    for index, node in enumerate(ordered_nodes):
        mask = 0
        for neighbor in graph.neighbors(node):
            neighbor_index = node_index[neighbor]
            if neighbor_index > index:
                mask |= 1 << neighbor_index
        higher_neighbor_masks.append(mask)

    def extend(prefix, candidates):
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            index = bit.bit_length() - 1
            clique = prefix + (ordered_nodes[index],)
            yield clique
            if len(clique) < max_size:
                yield from extend(
                    clique,
                    candidates & higher_neighbor_masks[index],
                )

    for index, node in enumerate(ordered_nodes):
        yield (node,)
        if max_size > 1:
            yield from extend((node,), higher_neighbor_masks[index])


def _expanded_path_union_nodes(graph, union_of_shortest_paths, connect):
    expanded_nodes = set(union_of_shortest_paths)
    if connect > 1:
        for path_node in union_of_shortest_paths:
            expanded_nodes.update(_node_ball(graph, path_node, connect - 1))
    return expanded_nodes


def _induced_edge_ids(source_node_ids, neighbors_by_node, node_order=None):
    source_edge_ids = set()
    for u in source_node_ids:
        for v in neighbors_by_node[u]:
            if v in source_node_ids and (node_order is None or node_order[u] < node_order[v]):
                source_edge_ids.add(_edge_key(u, v))
    return source_edge_ids


def _induced_edge_ids_from_graph(graph, source_node_ids, strategy=None):
    if strategy is None:
        strategy = graph.graph.get('_hash_strategy') or _select_graph_hash_strategy((graph,))
    source_node_ids = set(source_node_ids)
    return {
        strategy.edge_id(graph, u, v, key)
        for u, v, key, _ in strategy.edge_records(graph)
        if u in source_node_ids and v in source_node_ids
    }


def _edge_ids_between(graph, u, v):
    edge_ids_by_pair = graph.graph.get('_structural_cache', {}).get('edge_ids_by_pair')
    if edge_ids_by_pair is not None:
        return edge_ids_by_pair.get(_undirected_pair_key(u, v), ())
    strategy = graph.graph.get('_hash_strategy') or _select_graph_hash_strategy((graph,))
    return tuple(
        strategy.edge_id(graph, edge_u, edge_v, key)
        for edge_u, edge_v, key, _ in strategy.edge_records(graph)
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


def _induced_subgraph_from_occurrence(original_graph, occurrence, strategy):
    subgraph = original_graph.__class__()
    for key, value in original_graph.graph.items():
        subgraph.graph[key] = copy.deepcopy(value)
    source_nodes = set(occurrence['source_node_ids'])
    for node in (node for node in original_graph.nodes() if node in source_nodes):
        subgraph.add_node(node, **copy.deepcopy(original_graph.nodes[node]))
        subgraph.nodes[node]['original_node_id'] = node
    for edge_id in occurrence['source_edge_ids']:
        strategy.add_subgraph_edge(subgraph, original_graph, edge_id)
    subgraph.graph['source_graph_index'] = occurrence['source_graph_index']
    subgraph.graph['feature_id'] = occurrence['feature_id']
    subgraph.graph['feature_kind'] = occurrence['feature_kind']
    subgraph.graph['raw_signature'] = occurrence['raw_signature']
    subgraph.graph['root_node'] = occurrence['root_node']
    subgraph.graph['target_node'] = occurrence['target_node']
    return subgraph


class _FeatureArchiveCollector:
    def __init__(self, original_graph, graph_index, strategy):
        self.original_graph = original_graph
        self.graph_index = graph_index
        self.strategy = strategy
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
                edge_id for edge_id in (
                    self.strategy.edge_id(self.original_graph, u, v, key)
                    for u, v, key, _ in self.strategy.edge_records(self.original_graph)
                ) if edge_id in source_edges
            ),
        }
        occurrence['subgraph'] = _induced_subgraph_from_occurrence(
            self.original_graph, occurrence, self.strategy
        )
        self.archive[feature_id].append(occurrence)


def _prepare_structural_graph(
    original_graph, radius, distance, connector, degree_threshold=None,
    graph_strategy=None,
):
    graph = original_graph.copy()
    ensure_graph_labels(graph)
    strategy = graph_strategy or _select_graph_hash_strategy((graph,))
    graph.graph['_hash_strategy'] = strategy
    traversal_graph = strategy.traversal_graph(graph)
    graph.graph['_traversal_graph'] = traversal_graph
    _, radius_max = _inclusive_parameter_bounds(radius, "radius")
    _, distance_max = _inclusive_parameter_bounds(distance, "distance")
    _, connector_max = _inclusive_parameter_bounds(connector, "connector")
    cutoff = max(radius_max, connector_max)
    bfs_cutoff = max(radius_max, connector_max, distance_max)
    graph.graph['_bfs_cache'] = _build_bfs_cache(traversal_graph, bfs_cutoff)
    graph.graph['_structural_cache'] = _build_structural_cache(graph, strategy)
    structural_cache = graph.graph['_structural_cache']

    for node_idx in graph.nodes():
        node_label_hash = _hash(graph.nodes[node_idx]['label'])
        structural_cache['node_label_hashes'][node_idx] = node_label_hash
        graph.nodes[node_idx]['node_label_hash'] = node_label_hash
    for node_idx in graph.nodes():
        node_hash_value = strategy.node_hasher(node_idx, graph)
        structural_cache['node_hashes'][node_idx] = node_hash_value
        graph.nodes[node_idx]['node_hash'] = node_hash_value
    for u, v, key, data in strategy.edge_records(graph):
        data['triplet_hash'] = strategy.edge_hasher(u, v, data, graph)
        structural_cache['edge_triplet_hashes'][
            strategy.edge_id(graph, u, v, key)
        ] = data['triplet_hash']

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
    strategy = graph.graph['_hash_strategy']
    union_signature = strategy.path_union_signature(
        graph, node_signatures, union_path_edges
    )

    if not include_occurrence_provenance:
        return set(), set(), union_signature

    provenance_nodes = _expanded_path_union_nodes(
        graph, union_of_shortest_paths, connect
    )
    provenance_edges = _induced_edge_ids_from_graph(
        graph, provenance_nodes, strategy
    )
    return provenance_nodes, provenance_edges, union_signature


def _process_node_features(
    node_idx, graph, radius, distance, connector, nbits, sigma, accumulator,
    use_simplex_features=False, simplex_feature_order=2,
    archive_collector=None,
):
    # Weighted: inline computation of weight using the Gaussian function.
    structural_cache = graph.graph.get('_structural_cache', {})
    strategy = graph.graph['_hash_strategy']
    neighbors_by_node = structural_cache.get('neighbors', {})
    node_order = structural_cache.get('node_order')
    rooted_graph_hashes_by_node = structural_cache.get('rooted_graph_hashes', {})
    radius_values = _inclusive_parameter_values(radius, "radius")
    distance_min, distance_max = _inclusive_parameter_bounds(distance, "distance")
    connector_values = _inclusive_parameter_values(connector, "connector")
    positive_connector_values = tuple(value for value in connector_values if value > 0)
    node_bfs_cache = graph.graph.get('_bfs_cache', {}).get(node_idx)
    if node_bfs_cache is not None:
        node_idxs_to_dist_dict = {
            curr_node_idx: dist
            for curr_node_idx, dist in node_bfs_cache['node_idxs_to_dist_dict'].items()
            if dist <= distance_max
        }
        dist_to_node_idxs_dict = {
            dist: node_idxs
            for dist, node_idxs in node_bfs_cache['dist_to_node_idxs_dict'].items()
            if distance_min <= dist <= distance_max
        }
    else:
        node_idxs_to_dist_dict = nx.single_source_shortest_path_length(
            graph, node_idx, cutoff=distance_max
        )
        selected_distances = {
            target: dist for target, dist in node_idxs_to_dist_dict.items()
            if distance_min <= dist <= distance_max
        }
        dist_to_node_idxs_dict = invert_dict(selected_distances)

    connector_data_by_target = {}
    for radius_i in radius_values:
        code_i = rooted_graph_hashes_by_node[node_idx][radius_i]
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items()):
            w = gaussian_weight(dist, sigma)
            for curr_node_idx in node_idxs:
                if positive_connector_values:
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
                            for connect_depth in positive_connector_values
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
                root_label_pair_hash = hash_set((
                    structural_cache['node_label_hashes'][node_idx],
                    structural_cache['node_label_hashes'][curr_node_idx],
                )) if use_simplex_features and dist > 0 else None
                for connect in connector_values:
                    if (
                        use_simplex_features
                        and connect > 0
                        and dist > 0
                        and radius_i == 0
                    ):
                        union_nodes = _expanded_path_union_nodes(
                            graph, union_of_shortest_paths, connect
                        )
                        for _, simplex_hash, simplex_nodes in _simplex_fragments(
                            graph, union_nodes, simplex_feature_order
                        ):
                            raw_signature = hash_sequence((
                                root_label_pair_hash, dist, connect, simplex_hash,
                            ))
                            feature_code = hash_value(raw_signature, nbits=nbits)
                            accumulator.add(feature_code, w)
                            if archive_collector is not None:
                                source_node_ids = set(simplex_nodes)
                                source_edge_ids = _simplex_fragment_edge_ids(
                                    graph, source_node_ids
                                )
                                archive_collector.add(
                                    feature_id=feature_code,
                                    raw_signature=raw_signature,
                                    feature_kind='simplex_path_union',
                                    root_node=node_idx,
                                    target_node=curr_node_idx,
                                    source_node_ids=source_node_ids,
                                    source_edge_ids=source_edge_ids,
                                )

                    if use_simplex_features and connect == 0 and dist == 0:
                        single_region = _node_ball(graph, node_idx, radius_i)
                        root_label_hash = structural_cache['node_label_hashes'][node_idx]
                        for _, simplex_hash, simplex_nodes in _simplex_fragments(
                            graph, single_region, simplex_feature_order
                        ):
                            raw_simplex_signature = hash_sequence((
                                root_label_hash, radius_i, simplex_hash,
                            ))
                            feature_code = hash_value(raw_simplex_signature, nbits=nbits)
                            accumulator.add(feature_code, w)
                            if archive_collector is not None:
                                source_node_ids = set(simplex_nodes)
                                archive_collector.add(
                                    feature_id=feature_code,
                                    raw_signature=raw_simplex_signature,
                                    feature_kind='simplex_neighborhood_single',
                                    root_node=node_idx,
                                    target_node=curr_node_idx,
                                    source_node_ids=source_node_ids,
                                    source_edge_ids=_simplex_fragment_edge_ids(graph, source_node_ids),
                                )

                    for radius_j in radius_values:
                        code_j = rooted_graph_hashes_by_node[curr_node_idx][radius_j]
                        if use_simplex_features and connect == 0 and dist > 0:
                            first_region = _node_ball(graph, node_idx, radius_i)
                            second_region = _node_ball(graph, curr_node_idx, radius_j)
                            first_fragments = _simplex_fragments(
                                graph, first_region, simplex_feature_order
                            )
                            second_fragments = _simplex_fragments(
                                graph, second_region, simplex_feature_order
                            )
                            fragments_by_nodes = {
                                frozenset(fragment_nodes): (fragment_hash, fragment_nodes)
                                for _, fragment_hash, fragment_nodes in first_fragments
                            }
                            for _, fragment_hash, fragment_nodes in second_fragments:
                                fragments_by_nodes.setdefault(
                                    frozenset(fragment_nodes),
                                    (fragment_hash, fragment_nodes),
                                )

                            for simplex_hash, simplex_nodes in fragments_by_nodes.values():
                                raw_simplex_signature = hash_sequence((
                                    root_label_pair_hash, dist, simplex_hash,
                                ))
                                feature_code = hash_value(
                                    raw_simplex_signature, nbits=nbits
                                )
                                accumulator.add(feature_code, w)
                                if archive_collector is not None:
                                    source_node_ids = set(simplex_nodes)
                                    source_edge_ids = _simplex_fragment_edge_ids(
                                        graph, simplex_nodes
                                    )
                                    archive_collector.add(
                                        feature_id=feature_code,
                                        raw_signature=raw_simplex_signature,
                                        feature_kind='simplex_neighborhood_pair',
                                        root_node=node_idx,
                                        target_node=curr_node_idx,
                                        source_node_ids=source_node_ids,
                                        source_edge_ids=source_edge_ids,
                                    )
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
                            source_edge_ids = _induced_edge_ids_from_graph(
                                graph, source_node_ids, strategy
                            )
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

def get_structural_node_vectors(
    original_graph, radius, distance, connector, nbits, degree_threshold=None,
    sigma=None, use_simplex_features=False, graph_strategy=None,
    simplex_feature_order=2, root_node_ids=None, simplex_feature_max_order=None,
    parameters=None,
):
    """
    Generates a feature vector for each node in the graph.
    Uses an unweighted (faster) version if sigma is None, in which the weight is inlined as 1.0.
    If sigma is provided, the Gaussian weighted variant is used.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    parameter_sets = _normalize_feature_parameter_sets(
        parameters,
        radius=radius,
        distance=distance,
        connector=connector,
        sigma=sigma,
        use_simplex_features=use_simplex_features,
        simplex_feature_order=simplex_feature_order,
    )
    prepare_radius = max(_inclusive_parameter_bounds(config['radius'], 'radius')[1] for config in parameter_sets)
    prepare_distance = max(_inclusive_parameter_bounds(config['distance'], 'distance')[1] for config in parameter_sets)
    prepare_connector = max(_inclusive_parameter_bounds(config['connector'], 'connector')[1] for config in parameter_sets)
    graph = _prepare_structural_graph(
        original_graph, prepare_radius, prepare_distance, prepare_connector, degree_threshold,
        graph_strategy=graph_strategy,
    )
    n_nodes = graph.number_of_nodes()
    n_features = 2 ** nbits
    rows = []
    cols = []
    data = []
    selected_roots = _normalize_root_node_ids([graph], [root_node_ids])[0]
    use_weighted_accumulator = any(config['sigma'] is not None for config in parameter_sets)
    if not use_weighted_accumulator:
        for row_idx, node_idx in enumerate(graph.nodes()):
            accumulator = ListAccumulator()
            if selected_roots is None or node_idx in selected_roots:
                for config in parameter_sets:
                    _process_node_features(
                        node_idx, graph, config['radius'], config['distance'],
                        config['connector'], nbits, config['sigma'], accumulator,
                        config['use_simplex_features'], config['simplex_feature_order'],
                    )
            row_entries = Counter(accumulator.get())
            row_entries[0] = 1
            row_entries[1] = graph.degree[node_idx]
            _append_sparse_row_entries(row_entries, row_idx, rows, cols, data)
    else:
        for row_idx, node_idx in enumerate(graph.nodes()):
            accumulator = DictAccumulator()
            if selected_roots is None or node_idx in selected_roots:
                for config in parameter_sets:
                    _process_node_features(
                        node_idx, graph, config['radius'], config['distance'],
                        config['connector'], nbits, config['sigma'], accumulator,
                        config['use_simplex_features'], config['simplex_feature_order'],
                    )
            row_entries = dict(accumulator.get())
            row_entries[0] = 1.0
            row_entries[1] = float(graph.degree[node_idx])
            _append_sparse_row_entries(row_entries, row_idx, rows, cols, data)

    dtype = float if use_weighted_accumulator else int
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_features), dtype=dtype)


def get_feature_archive(
    original_graph, graph_index, radius, distance, connector, nbits,
    degree_threshold=None, sigma=None, use_simplex_features=False,
    graph_strategy=None, simplex_feature_order=2, simplex_feature_max_order=None,
    parameters=None,
):
    """
    Replays the structural hashing path and archives all occurrences per final hashed feature id.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    parameter_sets = _normalize_feature_parameter_sets(
        parameters,
        radius=radius,
        distance=distance,
        connector=connector,
        sigma=sigma,
        use_simplex_features=use_simplex_features,
        simplex_feature_order=simplex_feature_order,
    )
    prepare_radius = max(_inclusive_parameter_bounds(config['radius'], 'radius')[1] for config in parameter_sets)
    prepare_distance = max(_inclusive_parameter_bounds(config['distance'], 'distance')[1] for config in parameter_sets)
    prepare_connector = max(_inclusive_parameter_bounds(config['connector'], 'connector')[1] for config in parameter_sets)
    graph = _prepare_structural_graph(
        original_graph, prepare_radius, prepare_distance, prepare_connector, degree_threshold,
        graph_strategy=graph_strategy,
    )
    archive_collector = _FeatureArchiveCollector(
        original_graph, graph_index, graph.graph['_hash_strategy']
    )

    for node_idx in graph.nodes():
        accumulator = (
            DictAccumulator()
            if any(config['sigma'] is not None for config in parameter_sets)
            else ListAccumulator()
        )
        for config in parameter_sets:
            _process_node_features(
                node_idx,
                graph,
                config['radius'],
                config['distance'],
                config['connector'],
                nbits,
                config['sigma'],
                accumulator,
                config['use_simplex_features'],
                config['simplex_feature_order'],
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


def get_node_vectors(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, degree_threshold=None, add_structural_node_information=True, sigma=None, use_simplex_features=False, structural_node_vectors_mtx=None, graph_strategy=None, simplex_feature_order=2, root_node_ids=None, simplex_feature_max_order=None, parameters=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int or tuple[int, int]): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int]): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int]): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature matrix representing the graph.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    node_vectors_mtx = structural_node_vectors_mtx
    if node_vectors_mtx is None:
        node_vectors_mtx = get_structural_node_vectors(
            original_graph, radius, distance, connector, nbits, degree_threshold,
            sigma, use_simplex_features, graph_strategy,
            simplex_feature_order, root_node_ids, parameters=parameters,
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
                                degree_threshold=None, sigma=None, use_simplex_features=False,
                                node_vectors_mtx=None, graph_strategy=None,
                                simplex_feature_order=2,
                                simplex_feature_max_order=None, parameters=None):
    """Build one structural feature row per undirected edge by summing endpoint rows."""
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    strategy = graph_strategy or _select_graph_hash_strategy((original_graph,))
    if node_vectors_mtx is None:
        node_vectors_mtx = get_structural_node_vectors(
            original_graph, radius, distance, connector, nbits, degree_threshold, sigma,
            use_simplex_features, strategy,
            simplex_feature_order, parameters=parameters,
        )
    node_idxs = list(original_graph.nodes())
    node_to_row = {node_idx: row_idx for row_idx, node_idx in enumerate(node_idxs)}
    edge_vectors = []
    for u, v, key, data in strategy.edge_records(original_graph):
        row = node_vectors_mtx.getrow(node_to_row[u]) + node_vectors_mtx.getrow(node_to_row[v])
        row = strategy.edge_row(row, original_graph, u, v, data, nbits)
        edge_vectors.append(row)
    if not edge_vectors:
        return csr_matrix((0, node_vectors_mtx.shape[1]), dtype=node_vectors_mtx.dtype)
    return sp.sparse.vstack(edge_vectors, format="csr")


def get_edge_vectors(original_graph, radius, distance, connector, nbits,
                     edge_attribute_key=None, degree_threshold=None, sigma=None,
                     use_simplex_features=False, edge_attribute_width=None,
                     graph_strategy=None, simplex_feature_order=2,
                     simplex_feature_max_order=None, parameters=None):
    """Generate one feature row per edge, aligned with ``graph.edges()`` order."""
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graph_strategy = graph_strategy or _select_graph_hash_strategy((original_graph,))
    edge_vectors_mtx = get_structural_edge_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold,
        sigma, use_simplex_features, graph_strategy=graph_strategy,
        simplex_feature_order=simplex_feature_order, parameters=parameters,
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
                         use_simplex_features=False, edge_attribute_width=None,
                         node_vectors_mtx=None, graph_strategy=None,
                         simplex_feature_order=2, simplex_feature_max_order=None,
                         parameters=None):
    """Aggregate structural edge rows through their continuous attributes."""
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graph_strategy = graph_strategy or _select_graph_hash_strategy((original_graph,))
    edge_structural_mtx = get_structural_edge_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold,
        sigma, use_simplex_features, node_vectors_mtx, graph_strategy,
        simplex_feature_order, parameters=parameters,
    )
    attribute_mtx = get_edge_attribute_mtx(original_graph, edge_attribute_key)
    if attribute_mtx.shape[1] == 0 and edge_attribute_width:
        attribute_mtx = np.empty((0, edge_attribute_width), dtype=float)
    if attribute_mtx.shape[0] != edge_structural_mtx.shape[0]:
        raise ValueError("Each edge must have an attribute vector at edge_attribute_key.")
    return csr_matrix(attribute_mtx.T.dot(edge_structural_mtx.toarray()).reshape(1, -1))


def get_node_graph_vectors(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, degree_threshold=None, sigma=None, use_simplex_features=False, edge_attribute_key=None, edge_attribute_width=None, graph_strategy=None, simplex_feature_order=2, root_node_ids=None, simplex_feature_max_order=None, parameters=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int or tuple[int, int]): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int]): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int]): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graph_strategy = graph_strategy or _select_graph_hash_strategy((original_graph,))
    node_vectors_mtx = get_structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold,
        sigma, use_simplex_features, graph_strategy,
        simplex_feature_order, root_node_ids, parameters=parameters,
    )
    node_vectors_mtx = node_vectors_mtx.todense().A
    attribute_node_vectors_mtx = get_node_vectors(
        original_graph, radius, distance, connector, nbits, weight_key,
        node_attribute_key, degree_threshold, add_structural_node_information=False,
        sigma=sigma, use_simplex_features=use_simplex_features,
        structural_node_vectors_mtx=sp.sparse.csr_matrix(node_vectors_mtx),
        graph_strategy=graph_strategy,
        simplex_feature_order=simplex_feature_order,
        root_node_ids=root_node_ids,
        parameters=parameters,
    )
    vector_ = attribute_node_vectors_mtx.T.dot(node_vectors_mtx)
    if edge_attribute_key is not None:
        edge_graph_vector = get_edge_graph_vector(
            original_graph, radius, distance, connector, nbits,
            edge_attribute_key, degree_threshold, sigma, use_simplex_features,
            edge_attribute_width,
            node_vectors_mtx=sp.sparse.csr_matrix(node_vectors_mtx),
            graph_strategy=graph_strategy,
            simplex_feature_order=simplex_feature_order,
            parameters=parameters,
        )
        vector_ = np.hstack([
            np.asarray(vector_).reshape(1, -1),
            edge_graph_vector.toarray(),
        ])
    vector = vector_.reshape(1, -1)
    vector = np.power(np.abs(vector), 1/2)
    vector = csr_matrix(vector)
    return csr_matrix(vector)


def get_graph_vector(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_simplex_features=False, edge_attribute_width=None, graph_strategy=None, simplex_feature_order=2, root_node_ids=None, simplex_feature_max_order=None, parameters=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int or tuple[int, int]): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int]): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int]): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graph_strategy = graph_strategy or _select_graph_hash_strategy((original_graph,))
    structural_node_vectors_mtx = get_structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold,
        sigma, use_simplex_features, graph_strategy,
        simplex_feature_order, root_node_ids, parameters=parameters,
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
        degree_threshold, sigma, use_simplex_features, edge_attribute_width,
        structural_node_vectors_mtx, graph_strategy,
        simplex_feature_order, parameters=parameters,
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


def paired_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_simplex_features=False, edge_attribute_width=None, simplex_feature_order=2, root_node_ids=None, simplex_feature_max_order=None, parameters=None):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int or tuple[int, int]): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int]): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int]): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graphs = list(graphs)
    root_node_ids = _normalize_root_node_ids(graphs, root_node_ids)
    strategy = _select_graph_hash_strategy(graphs)
    n_cpus = mp.cpu_count()

    # A pool adds noticeable one-shot setup and graph-serialization cost.
    # Require roughly 1.5 graphs per worker before paying that cost.
    parallel_threshold = max(n_cpus + 1, (3 * n_cpus + 1) // 2)
    if parallel and len(graphs) >= parallel_threshold:
        def process_subset(indexed_subset):
            return [
                get_graph_vector(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, edge_attribute_key, degree_threshold, sigma, use_simplex_features, edge_attribute_width, strategy, simplex_feature_order, selected_roots, parameters=parameters)
                for graph, selected_roots in indexed_subset
            ]

        subsets = split_into_chunks(list(zip(graphs, root_node_ids)), n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            get_graph_vector(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, edge_attribute_key, degree_threshold, sigma, use_simplex_features, edge_attribute_width, strategy, simplex_feature_order, selected_roots, parameters=parameters)
            for graph, selected_roots in zip(graphs, root_node_ids)
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_graphs_vector_encoder(graphs, radius, distance, connector, nbits,
                                     parallel=True, weight_key=None,
                                     node_attribute_key=None, edge_attribute_key=None,
                                     degree_threshold=None, sigma=None,
                                     use_simplex_features=False,
                                     edge_attribute_width=None,
                                     simplex_feature_order=2,
                                     root_node_ids=None,
                                     simplex_feature_max_order=None,
                                     parameters=None):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int or tuple[int, int]): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int]): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int]): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graphs = list(graphs)
    root_node_ids = _normalize_root_node_ids(graphs, root_node_ids)
    strategy = _select_graph_hash_strategy(graphs)
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(indexed_subset):
            return [
                get_node_graph_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_simplex_features=use_simplex_features, edge_attribute_key=edge_attribute_key, edge_attribute_width=edge_attribute_width, graph_strategy=strategy, simplex_feature_order=simplex_feature_order, root_node_ids=selected_roots, parameters=parameters)
                for graph, selected_roots in indexed_subset
            ]

        subsets = split_into_chunks(list(zip(graphs, root_node_ids)), n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            get_node_graph_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_simplex_features=use_simplex_features, edge_attribute_key=edge_attribute_key, edge_attribute_width=edge_attribute_width, graph_strategy=strategy, simplex_feature_order=simplex_feature_order, root_node_ids=selected_roots, parameters=parameters)
            for graph, selected_roots in zip(graphs, root_node_ids)
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_simplex_features=False, simplex_feature_order=2, root_node_ids=None, simplex_feature_max_order=None, parameters=None):
    """
    Encodes a list of graphs into node feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int or tuple[int, int]): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int]): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int]): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        list of scipy.sparse.csr_matrix: A list where each element is a matrix of node vectors for a graph.
    """
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    graphs = list(graphs)
    root_node_ids = _normalize_root_node_ids(graphs, root_node_ids)
    strategy = _select_graph_hash_strategy(graphs)
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(indexed_subset):
            return [
                get_node_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_simplex_features=use_simplex_features, graph_strategy=strategy, simplex_feature_order=simplex_feature_order, root_node_ids=selected_roots, parameters=parameters)
                for graph, selected_roots in indexed_subset
            ]

        subsets = split_into_chunks(list(zip(graphs, root_node_ids)), n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_node_vectors_subsets = pool.map(process_subset, subsets)
        graph_node_vectors = [vec for subset in graph_node_vectors_subsets for vec in subset]
    else:
        graph_node_vectors = [
            get_node_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_simplex_features=use_simplex_features, graph_strategy=strategy, simplex_feature_order=simplex_feature_order, root_node_ids=selected_roots, parameters=parameters)
            for graph, selected_roots in zip(graphs, root_node_ids)
        ]

    return graph_node_vectors


def paired_edge_vector_encoder(graphs, radius, distance, connector, nbits,
                               parallel=True, edge_attribute_key=None,
                               degree_threshold=None, sigma=None,
                               use_simplex_features=False, edge_attribute_width=None,
                               simplex_feature_order=2,
                               simplex_feature_max_order=None,
                               parameters=None):
    simplex_feature_order = _resolve_simplex_feature_order(
        simplex_feature_order, simplex_feature_max_order
    )
    """Encode each graph as a list of edge feature matrices."""
    graphs = list(graphs)
    strategy = _select_graph_hash_strategy(graphs)
    def encode_subset(subset):
        return [
            get_edge_vectors(
                graph, radius, distance, connector, nbits,
                edge_attribute_key=edge_attribute_key,
                degree_threshold=degree_threshold,
                sigma=sigma,
                use_simplex_features=use_simplex_features,
                edge_attribute_width=edge_attribute_width,
                graph_strategy=strategy,
                simplex_feature_order=simplex_feature_order,
                parameters=parameters,
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


class BaseNSPPK(_SimplexFeatureOrderAliasMixin, BaseEstimator, TransformerMixin):
    """
    BaseNSPPK is an abstract base class for encoding graphs into feature vectors.

    This class is initialized with generic components for embedding, clustering, and classification,
    allowing for flexibility and extensibility. Subclasses can specialize these components to 
    implement specific encoding strategies.

    Parameters:
        embedder (transformer, optional): A transformer for attribute embedding (e.g., TruncatedSVD).
        clustering_predictor (estimator, optional): A clustering model for attribute clustering (e.g., KMeans).
        classifier (estimator, optional): A classifier for predicting cluster labels (e.g., ExtraTreesClassifier).
        radius (int or tuple[int, int], default=1): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int], default=3): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int], default=0): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
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
                 use_simplex_features=False, simplex_feature_order=2,
                 simplex_feature_max_order=None, parameters=None):
        simplex_feature_order = _resolve_simplex_feature_order(
            simplex_feature_order, simplex_feature_max_order
        )
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
            use_simplex_features=use_simplex_features,
            simplex_feature_order=simplex_feature_order,
        )
        _normalize_feature_parameter_sets(
            parameters,
            radius=radius,
            distance=distance,
            connector=connector,
            sigma=sigma,
            use_simplex_features=use_simplex_features,
            simplex_feature_order=simplex_feature_order,
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
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = simplex_feature_order
        self.parameters = parameters
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
            output_edge_data = [data for _, _, _, data in _edge_records(out_graph)]
            for (_, _, _, data), out_data, embedding in zip(edges, output_edge_data, embeddings):
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
            output_edge_data = [data for _, _, _, data in _edge_records(out_graph)]
            for out_data, label in zip(output_edge_data, labels):
                out_data['label'] = label
            out_graphs.append(out_graph)
        return out_graphs

    def transform(self, graphs, root_node_ids=None):
        """
        Transforms the input graphs into feature vectors.

        This method handles attribute embedding, label assignment, and encoding of graphs into feature vectors.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        check_is_fitted(self, 'is_fitted_')

        graphs = list(graphs)
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

        root_node_ids = _normalize_root_node_ids(graphs, root_node_ids)
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
                use_simplex_features=self.use_simplex_features,
                simplex_feature_order=self.simplex_feature_order,
                edge_attribute_width=edge_attribute_width,
                root_node_ids=root_node_ids,
                parameters=self.parameters,
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
                self.use_simplex_features,
                edge_attribute_width,
                self.simplex_feature_order,
                root_node_ids,
                parameters=self.parameters,
            )

        if self.dense:
            data_mtx = data_mtx.todense().A

        return data_mtx

class NSPPK(_SimplexFeatureOrderAliasMixin, BaseEstimator, TransformerMixin):
    """
    NSPPK (Neighborhood Subgraph Pairwise Propagation Kernel) class specialized from BaseNSPPK.

    This class encodes graphs into feature vectors suitable for machine learning models by capturing both
    local (node-level) and structural (subgraph-level) information. It specializes the BaseNSPPK by
    initializing it with specific components: TruncatedSVD for embedding, KMeans for clustering, and 
    ExtraTreesClassifier for classification.

    Parameters:
        radius (int or tuple[int, int], default=1): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int], default=3): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int], default=0): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
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
                 use_simplex_features=False, simplex_feature_order=2,
                 r=None, d=None, c=None, simplex_feature_max_order=None,
                 parameters=None):
        radius = _resolve_alias('radius', radius, 'r', r, 1)
        distance = _resolve_alias('distance', distance, 'd', d, 3)
        connector = _resolve_alias('connector', connector, 'c', c, 0)
        simplex_feature_order = _resolve_simplex_feature_order(
            simplex_feature_order, simplex_feature_max_order
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
        self.edge_attribute_key = edge_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_attribute_alphabet_size = edge_attribute_alphabet_size
        self.use_node_kernel = use_node_kernel
        self.sigma = sigma
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = simplex_feature_order
        self.parameters = parameters
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
            use_simplex_features=self.use_simplex_features,
            simplex_feature_order=self.simplex_feature_order,
            parameters=self.parameters,
        )

    def __repr__(self):
        """
        Returns a string representation of the NSPPK instance.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}(radius={self.radius}, distance={self.distance}, connector={self.connector}, sigma={self.sigma}, use_simplex_features={self.use_simplex_features}, simplex_feature_order={self.simplex_feature_order}, " \
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

    def transform(self, graphs, root_node_ids=None):
        """
        Transform the input graphs into feature vectors by delegating to BaseNSPPK.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        check_is_fitted(self.base_nsppk, 'is_fitted_')
        return self.base_nsppk.transform(graphs, root_node_ids=root_node_ids)

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
            'use_simplex_features': self.use_simplex_features,
            'simplex_feature_order': self.simplex_feature_order,
            'simplex_feature_max_order': self.simplex_feature_order,
            'parameters': self.parameters,
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
            use_simplex_features=candidate_params['use_simplex_features'],
            simplex_feature_order=candidate_params['simplex_feature_order'],
        )
        _normalize_feature_parameter_sets(
            candidate_params['parameters'],
            radius=candidate_params['radius'],
            distance=candidate_params['distance'],
            connector=candidate_params['connector'],
            sigma=candidate_params['sigma'],
            use_simplex_features=candidate_params['use_simplex_features'],
            simplex_feature_order=candidate_params['simplex_feature_order'],
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
                use_simplex_features=self.use_simplex_features,
                simplex_feature_order=self.simplex_feature_order,
            )

        return self


class _NSPPKPredictor(_SimplexFeatureOrderAliasMixin, BaseEstimator):
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
        use_simplex_features=False,
        simplex_feature_order=2,
        simplex_feature_max_order=None,
        parameters=None,
    ):
        simplex_feature_order = _resolve_simplex_feature_order(
            simplex_feature_order, simplex_feature_max_order
        )
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
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = simplex_feature_order
        self.parameters = parameters

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
            elif shortcut == "sgd" and self._task == "regressor":
                model = SGDRegressor(random_state=0)
            else:
                raise ValueError(
                    "estimator must be 'linear', 'rf', 'sgd' for regression, or a "
                    "scikit-learn estimator; "
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

    def _supports_partial_fit(self):
        """Whether the configured predictor has a scikit-learn partial_fit method."""
        try:
            model = self._make_estimator()
        except (TypeError, ValueError):
            return False
        return callable(getattr(model, "partial_fit", None))

    def _make_encoder(self):
        """Create a fresh NSPPK encoder with this predictor's configuration."""
        return NSPPK(
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
            use_simplex_features=self.use_simplex_features,
            simplex_feature_order=self.simplex_feature_order,
            parameters=self.parameters,
        )

    def fit(self, X, y, **fit_params):
        """Fit the NSPPK encoder and the selected predictor on graph inputs."""
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        if y is None:
            raise ValueError("y is required.")
        check_consistent_length(graphs, y)

        encoder = self._make_encoder()
        encoder.fit(graphs, y)
        features = encoder.transform(graphs)
        model = self._make_estimator()
        model.fit(features, y, **fit_params)

        self.encoder_ = encoder
        self.estimator_ = model
        if self._task == "classifier" and hasattr(model, "classes_"):
            self.classes_ = model.classes_
        return self

    def tune(
        self,
        X,
        y,
        *,
        search_space=None,
        scoring=None,
        cv=5,
        n_samples=40,
        n_initial_points=20,
        random_state=0,
        trial_callback=None,
        verbose=False,
        validation_data=None,
    ):
        """Tune NSPPK encoder parameters by Bayesian optimization.

        Numeric two-tuples in ``search_space`` are inclusive bounds; lists
        specify explicit choices. The configured scikit-learn estimator stays
        fixed during the search. The selected configuration is fitted on all
        supplied graphs and targets, and the search result is returned.

        If ``search_space`` is omitted, the search varies ``radius`` over
        ``(0, 2)``, ``distance`` over ``(0, 4)``, ``connector`` over ``(0, 2)``,
        and ``use_simplex_features`` over ``[False, True]``. Classification
        defaults to average precision; regression defaults to R².

        Bayesian tuning requires the optional ``scikit-optimize`` dependency:
        install it with ``python -m pip install 'nsppk[tuning]'``.

        Parameters
        ----------
        X : iterable of graph-like objects
            Input NetworkX graphs.
        y : array-like
            One class label or numeric regression target per graph.
        search_space : mapping, optional
            NSPPK encoder parameter names mapped to numeric ``(low, high)``
            bounds or non-empty lists of explicit choices. Classifier and
            regressor settings cannot be searched.
        scoring : str or callable, optional
            Scikit-learn scorer name or scorer callable. Defaults to
            ``"average_precision"`` for classification and ``"r2"`` for
            regression.
        cv : int, default=5
            Number of stratified classification or shuffled regression folds.
        n_samples : int, default=40
            Number of parameter configurations to evaluate.
        n_initial_points : int, default=20
            Maximum number of Latin-hypercube initial trials.
        random_state : int or None, default=0
            Seed for the sampler and cross-validation folds.
        trial_callback : callable, optional
            Called after each trial with its parameters, fold scores, score,
            best-so-far information, and whether it is the best trial.
        verbose : bool, default=False
            Whether to print trial progress.
        validation_data : pair of (graphs, targets), optional
            When provided, fit each trial on ``(X, y)`` and score it on this
            explicit validation pair instead of cross-validation. After the
            search, refit the wrapper on ``(X, y)`` only.

        Returns
        -------
        skopt.OptimizeResult
            Search history and best configuration. The wrapper is refitted
            using that configuration and exposes ``best_params_``,
            ``best_score_``, and ``tune_result_``.
        """
        from nsppk_tuning import tune_predictor

        return tune_predictor(
            self,
            X,
            y,
            search_space=search_space,
            scoring=scoring,
            cv=cv,
            n_samples=n_samples,
            n_initial_points=n_initial_points,
            random_state=random_state,
            trial_callback=trial_callback,
            verbose=verbose,
            validation_data=validation_data,
        )

    def _partial_fit_batch(self, X, y, classes=None, **fit_params):
        if not self._supports_partial_fit():
            model = self._make_estimator()
            raise TypeError(
                f"{type(model).__name__} does not implement partial_fit(); "
                "choose an incremental estimator."
            )

        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        if y is None:
            raise ValueError("y is required.")
        check_consistent_length(graphs, y)

        if self._task == "regressor" and classes is not None:
            raise ValueError("classes is only supported by NSPPKClassifier.")

        first_update = not hasattr(self, "estimator_")
        if self._task == "classifier" and first_update and classes is None:
            raise ValueError("classes must be provided on the first classifier partial_fit call.")
        if self._task == "classifier" and classes is not None:
            classes = np.asarray(classes)
            if classes.ndim != 1 or classes.size == 0 or np.unique(classes).size != classes.size:
                raise ValueError("classes must be a non-empty one-dimensional array of unique labels.")

        if first_update:
            encoder = self._make_encoder()
            # Keep a single feature mapping across every update. In particular,
            # attribute projections and clusters must not be refit per batch.
            encoder.fit(graphs, y)
            features = encoder.transform(graphs)
            model = self._make_estimator()
        else:
            encoder = self.encoder_
            model = self.estimator_
            features = encoder.transform(graphs)

        if self._task == "classifier" and classes is not None:
            model.partial_fit(features, y, classes=classes, **fit_params)
        else:
            model.partial_fit(features, y, **fit_params)

        if first_update:
            self.encoder_ = encoder
            self.estimator_ = model
        if self._task == "classifier" and hasattr(model, "classes_"):
            self.classes_ = model.classes_
        return features

    @available_if(lambda self: self._supports_partial_fit())
    def partial_fit(self, X, y, classes=None, **fit_params):
        """Incrementally train the predictor on one batch of input graphs.

        On the first call, the NSPPK encoder is fitted on this batch so any
        learned attribute preprocessing is established. The same batch is then
        transformed and used to update the predictor. Later calls reuse the
        fitted encoder and predictor. For classification, ``classes`` is
        required on the first call and may be omitted afterward.
        """
        self._partial_fit_batch(X, y, classes=classes, **fit_params)
        return self

    @available_if(lambda self: self._supports_partial_fit())
    def partial_fit_from(
        self,
        uri,
        type,
        reader=None,
        limit=None,
        random_state=None,
        batch_size=128,
        verbose=False,
        label_extractor=None,
        start_after_instance=0,
        classes=None,
        batch_callback=None,
        **fit_params,
    ):
        """Incrementally train from a graph source using NSPPK's built-in readers.

        The first batch initializes learned encoder preprocessing and is also
        used to train the predictor. Classifiers require ``classes`` to include
        the complete label set. If supplied, ``batch_callback`` is called after
        each update as ``batch_callback(self, X_batch, y_batch, n_samples_seen)``.
        """
        if batch_callback is not None and not callable(batch_callback):
            raise TypeError("batch_callback must be callable when provided.")
        _sync_graph_io_hooks()
        extractor = _graph_io._normalize_label_extractor(label_extractor)
        graph_iterable = _graph_io._iter_loaded_graphs(
            uri,
            type,
            reader=reader,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            mode="stream",
            start_after_instance=start_after_instance,
        )
        first_batch = True
        n_samples_seen = 0
        for graphs in _graph_io._batched_graphs(graph_iterable, batch_size):
            targets = np.asarray([extractor(graph) for graph in graphs])
            features = self._partial_fit_batch(
                graphs,
                targets,
                classes=classes if first_batch else None,
                **fit_params,
            )
            n_samples_seen += len(targets)
            if batch_callback is not None:
                batch_callback(self, features, targets, n_samples_seen)
            first_batch = False
        if first_batch:
            raise ValueError("The graph source produced no training batches.")
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
    (RandomForestRegressor), ``"sgd"`` (SGDRegressor), or a scikit-learn
    regressor instance. Ridge and random forests do not support
    ``partial_fit``; SGDRegressor and compatible custom estimators do. Pass
    estimator constructor overrides with ``estimator_params``.
    """

    _task = "regressor"


class NodeNSPPK(_SimplexFeatureOrderAliasMixin, BaseEstimator, TransformerMixin):
    """
    NodeNSPPK (Node Neighborhood Subgraph Pairwise Propagation Kernel) class compatible with scikit-learn's Transformer interface.

    This class encodes graphs into node-level feature vectors suitable for
    node-level machine learning models. It captures both local (node-level) and structural (subgraph-level) information
    by hashing node labels and the structure of their neighborhoods. Additionally, it supports dimensionality reduction
    of node attributes using Singular Value Decomposition (SVD).

    Parameters:
        radius (int or tuple[int, int], default=1): Scalar n selects radii 0..n; a tuple selects inclusive min..max.
        distance (int or tuple[int, int], default=3): Scalar n selects distances 0..n; a tuple selects inclusive min..max.
        connector (int or tuple[int, int], default=0): Scalar n selects connectors 0..n; a tuple selects inclusive min..max.
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
                 node_attribute_key=None, attribute_dim=None, attribute_alphabet_size=None, sigma=None, use_simplex_features=False, simplex_feature_order=2, r=None, d=None, c=None, simplex_feature_max_order=None, parameters=None):
        radius = _resolve_alias('radius', radius, 'r', r, 1)
        distance = _resolve_alias('distance', distance, 'd', d, 3)
        connector = _resolve_alias('connector', connector, 'c', c, 0)
        simplex_feature_order = _resolve_simplex_feature_order(
            simplex_feature_order, simplex_feature_max_order
        )
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
            use_simplex_features=use_simplex_features,
            simplex_feature_order=simplex_feature_order,
            parameters=parameters,
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
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = simplex_feature_order
        self.parameters = parameters
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
            'use_simplex_features': self.use_simplex_features,
            'simplex_feature_order': self.simplex_feature_order,
            'simplex_feature_max_order': self.simplex_feature_order,
            'parameters': self.parameters,
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
            use_simplex_features=candidate['use_simplex_features'],
            simplex_feature_order=candidate['simplex_feature_order'],
        )

        self.nsppk.set_params(**normalized)
        for key, value in normalized.items():
            setattr(self, key, value)
        self.r, self.d, self.c = self.radius, self.distance, self.connector
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

    def transform(self, graphs, root_node_ids=None):
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

        graphs = list(graphs)
        if self.nsppk.base_nsppk.node_attribute_key and self.nsppk.base_nsppk.attribute_dim:
            graphs = self.nsppk.base_nsppk.embed_attributes(graphs)
        if self.nsppk.base_nsppk.node_attribute_key and self.nsppk.base_nsppk.attribute_alphabet_size:
            graphs = self.nsppk.base_nsppk.set_discrete_labels(graphs)

        nodes_data_mtx_list = paired_node_vector_encoder(
            graphs=graphs,
            radius=self.nsppk.base_nsppk.radius,
            distance=self.nsppk.base_nsppk.distance,
            connector=self.nsppk.base_nsppk.connector,
            nbits=self.nsppk.base_nsppk.nbits,
            parallel=self.nsppk.base_nsppk.parallel,
            weight_key=self.nsppk.base_nsppk.weight_key,
            node_attribute_key=self.nsppk.base_nsppk.node_attribute_key,
            degree_threshold=self.nsppk.base_nsppk.degree_threshold,
            sigma=self.nsppk.sigma,
            use_simplex_features=self.nsppk.use_simplex_features,
            simplex_feature_order=self.nsppk.simplex_feature_order,
            root_node_ids=root_node_ids,
            parameters=self.nsppk.parameters,
        )

        if self.nsppk.base_nsppk.dense:
            nodes_data_mtx_list = [mtx.todense().A for mtx in nodes_data_mtx_list]

        return nodes_data_mtx_list


class EdgeNSPPK(_SimplexFeatureOrderAliasMixin, BaseEstimator, TransformerMixin):
    """Return one feature row per undirected edge in each input graph."""

    def __init__(self, radius=1, distance=3, connector=0, nbits=10,
                 degree_threshold=None, dense=True, parallel=True,
                 edge_attribute_key=None, edge_attribute_dim=None,
                 edge_attribute_alphabet_size=None, sigma=None,
                 use_simplex_features=False, simplex_feature_order=2,
                 r=None, d=None, c=None, simplex_feature_max_order=None,
                 parameters=None):
        radius = _resolve_alias('radius', radius, 'r', r, 1)
        distance = _resolve_alias('distance', distance, 'd', d, 3)
        connector = _resolve_alias('connector', connector, 'c', c, 0)
        simplex_feature_order = _resolve_simplex_feature_order(
            simplex_feature_order, simplex_feature_max_order
        )
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
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = simplex_feature_order
        self.parameters = parameters
        self.r, self.d, self.c = radius, distance, connector
        self.nsppk = NSPPK(
            radius=radius, distance=distance, connector=connector, nbits=nbits,
            degree_threshold=degree_threshold, dense=dense, parallel=parallel,
            edge_attribute_key=edge_attribute_key,
            edge_attribute_dim=edge_attribute_dim,
            edge_attribute_alphabet_size=edge_attribute_alphabet_size,
            sigma=sigma, use_simplex_features=use_simplex_features,
            simplex_feature_order=simplex_feature_order,
            parameters=parameters,
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
            'use_simplex_features': self.use_simplex_features,
            'simplex_feature_order': self.simplex_feature_order,
            'simplex_feature_max_order': self.simplex_feature_order,
            'parameters': self.parameters,
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
            use_simplex_features=base.use_simplex_features,
            edge_attribute_width=edge_attribute_width,
            simplex_feature_order=base.simplex_feature_order,
            parameters=base.parameters,
        )
        if base.dense:
            edge_matrices = [matrix.toarray() for matrix in edge_matrices]
        return edge_matrices


class _EntityNSPPKPredictor(BaseEstimator):
    """Shared supervised interface for node- and edge-level NSPPK features."""

    _entity_level = True
    _task = None

    def _make_estimator(self):
        return _NSPPKPredictor._make_estimator(self)

    def _supports_partial_fit(self):
        return _NSPPKPredictor._supports_partial_fit(self)

    def _make_encoder(self):
        raise NotImplementedError

    def _features_per_graph(self, encoder, graphs, selectors=None):
        matrices = encoder.transform(graphs)
        if selectors is None:
            return matrices
        return [matrix[selector] for matrix, selector in zip(matrices, selectors)]

    def _normalize_entity_indices(self, graphs, entity_indices):
        if entity_indices is None:
            return None
        if len(graphs) == 1 and isinstance(entity_indices, np.ndarray) and entity_indices.ndim == 1:
            selectors = [entity_indices]
        elif len(graphs) == 1 and isinstance(entity_indices, (list, tuple)) and (
            not entity_indices or np.asarray(entity_indices).ndim == 1
        ):
            selectors = [entity_indices]
        else:
            try:
                selectors = list(entity_indices)
            except TypeError as exc:
                raise TypeError("entity_indices must provide one selector per graph.") from exc
        if len(selectors) != len(graphs):
            raise ValueError("entity_indices must contain exactly one selector per graph.")

        normalized = []
        for graph_index, (graph, selector) in enumerate(zip(graphs, selectors)):
            values = np.asarray(selector)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(
                    f"entity_indices[{graph_index}] must be a one-dimensional integer array."
                )
            count = int(
                graph.number_of_nodes()
                if self._entity_name == "node"
                else graph.number_of_edges()
            )
            values = values.astype(np.int64, copy=False)
            if np.any(values < 0) or np.any(values >= count):
                raise ValueError(
                    f"entity_indices[{graph_index}] contains an index outside [0, {count})."
                )
            if np.unique(values).size != values.size:
                raise ValueError(f"entity_indices[{graph_index}] cannot contain duplicates.")
            normalized.append(values)
        return normalized

    @staticmethod
    def _stack_entity_features(matrices):
        if not matrices:
            raise ValueError("X must contain at least one graph.")
        if any(sp.sparse.issparse(matrix) for matrix in matrices):
            sparse_matrices = [
                matrix.tocsr() if sp.sparse.issparse(matrix) else csr_matrix(matrix)
                for matrix in matrices
            ]
            return sp.sparse.vstack(sparse_matrices, format="csr")
        return np.concatenate([np.asarray(matrix) for matrix in matrices], axis=0)

    def _fit_transform_graphs(self, graphs, entity_indices=None):
        encoder = self._make_encoder()
        encoder.fit(graphs)
        selectors = self._normalize_entity_indices(graphs, entity_indices)
        matrices = self._features_per_graph(encoder, graphs, selectors)
        counts = [int(matrix.shape[0]) for matrix in matrices]
        return encoder, self._stack_entity_features(matrices), counts

    @staticmethod
    def _normalize_targets(y, counts):
        if y is None:
            raise ValueError("y is required.")
        nested = False
        if isinstance(y, (list, tuple)) and len(y) == len(counts):
            candidate = [np.asarray(target) for target in y]
            nested = bool(candidate) and all(target.ndim == 1 for target in candidate) and any(
                target.ndim > 0 for target in candidate
            )
        elif isinstance(y, np.ndarray) and y.ndim == 2 and len(y) == len(counts):
            nested = all(y.shape[1] == count for count in counts)

        if nested:
            targets_by_graph = [np.asarray(target) for target in y]
            for graph_index, (target, count) in enumerate(zip(targets_by_graph, counts)):
                if target.ndim != 1:
                    raise ValueError(
                        f"Targets for graph {graph_index} must be one-dimensional."
                    )
                if len(target) != count:
                    raise ValueError(
                        f"Graph {graph_index} has {count} entities "
                        f"but received {len(target)} targets."
                    )
            flat_targets = (
                np.concatenate(targets_by_graph)
                if targets_by_graph
                else np.asarray([], dtype=float)
            )
            return flat_targets, "per_graph"

        flat_targets = np.asarray(y)
        if flat_targets.ndim != 1:
            raise ValueError("Flat y must be a one-dimensional target array.")
        expected = int(sum(counts))
        if len(flat_targets) != expected:
            raise ValueError(
                f"Received {len(flat_targets)} targets for {expected} encoded entities."
            )
        return flat_targets, "flat"

    def _validate_targets(self, targets, *, require_multiple_classes=True):
        if self._task == "classifier":
            from sklearn.utils.multiclass import check_classification_targets

            check_classification_targets(targets)
            if require_multiple_classes and np.unique(targets).size < 2:
                raise ValueError("Classification requires at least two target classes.")
        else:
            if targets.dtype.kind not in "iuf" or not np.isfinite(targets).all():
                raise ValueError("Regression targets must be finite numeric values.")

    def _check_layout(self, layout):
        if hasattr(self, "target_layout_") and self.target_layout_ != layout:
            raise ValueError(
                "partial_fit target layout must match the layout used by the first batch."
            )

    def fit(self, X, y, *, entity_indices=None, **fit_params):
        """Fit on node or edge rows flattened in graph order."""
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        encoder, features, counts = self._fit_transform_graphs(graphs, entity_indices)
        targets, layout = self._normalize_targets(y, counts)
        if features.shape[0] == 0:
            raise ValueError(f"X contains no {self._entity_name} feature rows.")
        self._validate_targets(targets)
        model = self._make_estimator()
        model.fit(features, targets, **fit_params)
        self.encoder_ = encoder
        self.estimator_ = model
        self.target_layout_ = layout
        self.n_features_in_ = features.shape[1]
        if self._task == "classifier":
            self.classes_ = model.classes_
        return self

    def _transform_fitted(self, X, entity_indices=None):
        check_is_fitted(self, attributes=["encoder_", "estimator_"])
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        selectors = self._normalize_entity_indices(graphs, entity_indices)
        matrices = self._features_per_graph(self.encoder_, graphs, selectors)
        counts = [int(matrix.shape[0]) for matrix in matrices]
        return self._stack_entity_features(matrices), counts

    def _format_predictions(self, values, counts):
        if getattr(self, "target_layout_", "flat") == "flat":
            return values
        boundaries = np.cumsum(counts)[:-1]
        return list(np.split(values, boundaries, axis=0))

    def predict(self, X, *, entity_indices=None):
        features, counts = self._transform_fitted(X, entity_indices)
        if features.shape[0] == 0:
            values = np.asarray([], dtype=getattr(self, "classes_", np.asarray([])).dtype)
        else:
            values = self.estimator_.predict(features)
        return self._format_predictions(values, counts)

    @available_if(lambda self: hasattr(getattr(self, "estimator_", None), "predict_proba"))
    def predict_proba(self, X, *, entity_indices=None):
        features, counts = self._transform_fitted(X, entity_indices)
        values = (
            self.estimator_.predict_proba(features)
            if features.shape[0]
            else np.empty((0, len(self.classes_)), dtype=float)
        )
        return self._format_predictions(values, counts)

    @available_if(lambda self: hasattr(getattr(self, "estimator_", None), "decision_function"))
    def decision_function(self, X, *, entity_indices=None):
        features, counts = self._transform_fitted(X, entity_indices)
        values = (
            self.estimator_.decision_function(features)
            if features.shape[0]
            else np.empty((0,), dtype=float)
        )
        return self._format_predictions(values, counts)

    def score(self, X, y, *, entity_indices=None):
        from sklearn.metrics import accuracy_score, r2_score

        features, counts = self._transform_fitted(X, entity_indices)
        targets, _ = self._normalize_targets(y, counts)
        predictions = self.estimator_.predict(features)
        if self._task == "classifier":
            return accuracy_score(targets, predictions)
        return r2_score(targets, predictions)

    def tune(
        self,
        X,
        y,
        *,
        search_space=None,
        scoring=None,
        cv=5,
        n_samples=40,
        n_initial_points=20,
        random_state=0,
        trial_callback=None,
        verbose=False,
        entity_indices=None,
        validation_data=None,
        validation_entity_indices=None,
    ):
        """Tune encoder settings by entity-row CV or explicit validation data."""
        from nsppk_tuning import tune_entity_predictor

        return tune_entity_predictor(
            self,
            X,
            y,
            search_space=search_space,
            scoring=scoring,
            cv=cv,
            n_samples=n_samples,
            n_initial_points=n_initial_points,
            random_state=random_state,
            trial_callback=trial_callback,
            verbose=verbose,
            entity_indices=entity_indices,
            validation_data=validation_data,
            validation_entity_indices=validation_entity_indices,
        )

    def _partial_fit_batch(self, X, y, classes=None, **fit_params):
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        first_update = not hasattr(self, "estimator_")
        if first_update:
            encoder = self._make_encoder()
            encoder.fit(graphs)
        else:
            encoder = self.encoder_
        matrices = self._features_per_graph(encoder, graphs)
        counts = [int(matrix.shape[0]) for matrix in matrices]
        features = self._stack_entity_features(matrices)
        targets, layout = self._normalize_targets(y, counts)
        self._check_layout(layout)
        if features.shape[0] == 0:
            raise ValueError(f"X contains no {self._entity_name} feature rows.")
        self._validate_targets(targets, require_multiple_classes=False)
        if self._task == "regressor" and classes is not None:
            raise ValueError("classes is only supported by classifiers.")

        model = self._make_estimator() if first_update else self.estimator_
        if not callable(getattr(model, "partial_fit", None)):
            raise TypeError(
                f"{type(model).__name__} does not implement partial_fit(); "
                "choose an incremental estimator."
            )
        if self._task == "classifier":
            if first_update and classes is None:
                raise ValueError("classes is required on the first classifier partial_fit call.")
            if classes is not None:
                classes = np.asarray(classes)
                if classes.ndim != 1 or classes.size == 0 or np.unique(classes).size != classes.size:
                    raise ValueError("classes must be a non-empty one-dimensional array of unique labels.")
                model.partial_fit(features, targets, classes=classes, **fit_params)
            else:
                model.partial_fit(features, targets, **fit_params)
        else:
            model.partial_fit(features, targets, **fit_params)

        if first_update:
            self.encoder_ = encoder
            self.estimator_ = model
            self.target_layout_ = layout
            self.n_features_in_ = features.shape[1]
        if self._task == "classifier" and hasattr(model, "classes_"):
            self.classes_ = model.classes_
        return features, targets

    @available_if(lambda self: self._supports_partial_fit())
    def partial_fit(self, X, y, classes=None, **fit_params):
        """Incrementally fit a batch of graphs and their entity targets."""
        self._partial_fit_batch(X, y, classes=classes, **fit_params)
        return self

    @available_if(lambda self: self._supports_partial_fit())
    def partial_fit_from(
        self,
        uri,
        type,
        reader=None,
        limit=None,
        random_state=None,
        batch_size=128,
        verbose=False,
        label_extractor=None,
        start_after_instance=0,
        classes=None,
        batch_callback=None,
        **fit_params,
    ):
        """Stream graph batches; ``label_extractor`` returns targets per entity."""
        if label_extractor is None or not callable(label_extractor):
            raise ValueError("label_extractor must return one target array per graph.")
        if batch_callback is not None and not callable(batch_callback):
            raise ValueError("batch_callback must be callable when provided.")
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        _sync_graph_io_hooks()
        graph_iterable = _graph_io._iter_loaded_graphs(
            uri,
            type,
            reader=reader,
            limit=limit,
            random_state=random_state,
            verbose=verbose,
            mode="stream",
            start_after_instance=start_after_instance,
        )
        n_entities_seen = 0
        first_batch = True
        for graphs in _graph_io._batched_graphs(graph_iterable, int(batch_size)):
            targets = [label_extractor(graph) for graph in graphs]
            features, flat_targets = self._partial_fit_batch(
                graphs,
                targets,
                classes=classes if first_batch else None,
                **fit_params,
            )
            n_entities_seen += len(flat_targets)
            if batch_callback is not None:
                batch_callback(self, features, flat_targets, n_entities_seen)
            first_batch = False
        if first_batch:
            raise ValueError("The graph source produced no training batches.")
        return self


class _NodeNSPPKPredictor(_EntityNSPPKPredictor):
    _entity_name = "node"

    def __init__(
        self,
        estimator="rf",
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
        attribute_dim=None,
        attribute_alphabet_size=None,
        sigma=None,
        use_simplex_features=False,
        simplex_feature_order=2,
        parameters=None,
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
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.sigma = sigma
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = _resolve_simplex_feature_order(simplex_feature_order, None)
        self.parameters = parameters

    def _make_encoder(self):
        return NodeNSPPK(
            radius=self.radius,
            distance=self.distance,
            connector=self.connector,
            nbits=self.nbits,
            degree_threshold=self.degree_threshold,
            dense=self.dense,
            parallel=self.parallel,
            weight_key=self.weight_key,
            node_attribute_key=self.node_attribute_key,
            attribute_dim=self.attribute_dim,
            attribute_alphabet_size=self.attribute_alphabet_size,
            sigma=self.sigma,
            use_simplex_features=self.use_simplex_features,
            simplex_feature_order=self.simplex_feature_order,
            parameters=self.parameters,
        )

    def _features_per_graph(self, encoder, graphs, selectors=None):
        if selectors is None:
            return encoder.transform(graphs)
        graph_node_ids = [list(graph.nodes()) for graph in graphs]
        root_node_ids = [
            [graph_node_ids[index][int(row)] for row in selector]
            for index, selector in enumerate(selectors)
        ]
        matrices = encoder.transform(graphs, root_node_ids=root_node_ids)
        return [matrix[selector] for matrix, selector in zip(matrices, selectors)]


class _EdgeNSPPKPredictor(_EntityNSPPKPredictor):
    _entity_name = "edge"

    def __init__(
        self,
        estimator="rf",
        estimator_params=None,
        radius=1,
        distance=3,
        connector=0,
        nbits=10,
        degree_threshold=None,
        dense=False,
        parallel=True,
        edge_attribute_key=None,
        edge_attribute_dim=None,
        edge_attribute_alphabet_size=None,
        sigma=None,
        use_simplex_features=False,
        simplex_feature_order=2,
        parameters=None,
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
        self.edge_attribute_key = edge_attribute_key
        self.edge_attribute_dim = edge_attribute_dim
        self.edge_attribute_alphabet_size = edge_attribute_alphabet_size
        self.sigma = sigma
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = _resolve_simplex_feature_order(simplex_feature_order, None)
        self.parameters = parameters

    def _make_encoder(self):
        return EdgeNSPPK(
            radius=self.radius,
            distance=self.distance,
            connector=self.connector,
            nbits=self.nbits,
            degree_threshold=self.degree_threshold,
            dense=self.dense,
            parallel=self.parallel,
            edge_attribute_key=self.edge_attribute_key,
            edge_attribute_dim=self.edge_attribute_dim,
            edge_attribute_alphabet_size=self.edge_attribute_alphabet_size,
            sigma=self.sigma,
            use_simplex_features=self.use_simplex_features,
            simplex_feature_order=self.simplex_feature_order,
            parameters=self.parameters,
        )


class NodeNSPPKClassifier(ClassifierMixin, _NodeNSPPKPredictor):
    """Classify node rows encoded by :class:`NodeNSPPK`."""

    _task = "classifier"

    def score(self, X, y, *, entity_indices=None):
        return _EntityNSPPKPredictor.score(self, X, y, entity_indices=entity_indices)


class NodeNSPPKRegressor(RegressorMixin, _NodeNSPPKPredictor):
    """Regress node targets from :class:`NodeNSPPK` feature rows."""

    _task = "regressor"

    def score(self, X, y, *, entity_indices=None):
        return _EntityNSPPKPredictor.score(self, X, y, entity_indices=entity_indices)


class EdgeNSPPKClassifier(ClassifierMixin, _EdgeNSPPKPredictor):
    """Classify edge rows encoded by :class:`EdgeNSPPK`."""

    _task = "classifier"

    def score(self, X, y, *, entity_indices=None):
        return _EntityNSPPKPredictor.score(self, X, y, entity_indices=entity_indices)


class EdgeNSPPKRegressor(RegressorMixin, _EdgeNSPPKPredictor):
    """Regress edge targets from :class:`EdgeNSPPK` feature rows."""

    _task = "regressor"

    def score(self, X, y, *, entity_indices=None):
        return _EntityNSPPKPredictor.score(self, X, y, entity_indices=entity_indices)

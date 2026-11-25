import hashlib
from collections import defaultdict


def _hash(value):
    digest = hashlib.blake2s(str(value).encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def masked_hash_value(value, bitmask=(2**20 - 1)):
    return _hash(value) & bitmask


def hash_value(value, nbits=10):
    max_index = 2 ** nbits
    return (_hash(value) & (max_index - 3)) + 2


def hash_sequence(iterable):
    return _hash(tuple(iterable))


def hash_set(iterable):
    sorted_iterable = sorted(iterable)
    tuple_representation = tuple(sorted_iterable)
    return _hash(tuple_representation)


def _hash_unordered_triplet(first, second, third):
    """Hash three unordered values without allocating a sorted list."""
    if first > second:
        first, second = second, first
    if second > third:
        second, third = third, second
    if first > second:
        first, second = second, first
    return _hash((first, second, third))


def _node_hash_undirected_simple(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    incident_patterns = (
        _hash((_hash(graph.nodes[neighbor]['label']), _hash(graph[node_idx][neighbor]['label'])))
        for neighbor in graph.neighbors(node_idx)
    )
    nh = hash_set(incident_patterns)
    return _hash((uh, nh))


def _node_hash_undirected_multi(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    incident_patterns = (
        _hash((_hash(graph.nodes[neighbor]['label']), _hash(data['label'])))
        for neighbor in graph.neighbors(node_idx)
        for data in graph[node_idx][neighbor].values()
    )
    nh = hash_set(incident_patterns)
    return _hash((uh, nh))


def _node_hash_directed(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    incident_patterns = [
        _hash(("out", _hash(graph.nodes[v]['label']), _hash(data['label'])))
        for _, v, data in graph.out_edges(node_idx, data=True)
    ]
    incident_patterns.extend(
        _hash(("in", _hash(graph.nodes[u]['label']), _hash(data['label'])))
        for u, v, data in graph.in_edges(node_idx, data=True)
        if u != v
    )
    nh = hash_set(incident_patterns)
    return _hash((uh, nh))


def _node_hash_multi_directed(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    incident_patterns = [
        _hash(("out", _hash(graph.nodes[v]['label']), _hash(data['label'])))
        for _, v, _, data in graph.out_edges(node_idx, keys=True, data=True)
    ]
    incident_patterns.extend(
        _hash(("in", _hash(graph.nodes[u]['label']), _hash(data['label'])))
        for u, v, _, data in graph.in_edges(node_idx, keys=True, data=True)
        if u != v
    )
    nh = hash_set(incident_patterns)
    return _hash((uh, nh))


def node_hash(node_idx, graph):
    """Hash one node, dispatching once for this standalone call."""
    if graph.is_directed():
        hasher = _node_hash_multi_directed if graph.is_multigraph() else _node_hash_directed
    else:
        hasher = _node_hash_undirected_multi if graph.is_multigraph() else _node_hash_undirected_simple
    return hasher(node_idx, graph)


def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items():
        reversed_dict[value].append(key)
    return reversed_dict


def rooted_graph_hashes(
    node_idx,
    graph,
    radius=1,
    node_idxs_to_dist_dict=None,
    node_label_hashes=None,
    node_hashes=None,
):
    if node_idxs_to_dist_dict is None:
        raise ValueError("node_idxs_to_dist_dict must be provided for rooted_graph_hashes().")
    if node_label_hashes is None:
        node_label_hashes = {
            curr_node_idx: graph.nodes[curr_node_idx]['node_label_hash']
            for curr_node_idx in graph.nodes()
        }
    if node_hashes is None:
        node_hashes = {
            curr_node_idx: graph.nodes[curr_node_idx]['node_hash']
            for curr_node_idx in graph.nodes()
        }
    dist_to_node_idxs_dict = invert_dict({
        curr_node_idx: dist
        for curr_node_idx, dist in node_idxs_to_dist_dict.items()
        if dist <= radius
    })
    iso_distance_codes = tuple(
        hash_set(
            node_label_hashes[curr_node_idx]
            if dist == 0 else node_hashes[curr_node_idx]
            for curr_node_idx in node_idxs
        )
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items())
    )
    h_list = [
        hash_sequence(iso_distance_codes[:i])
        for i in range(1, len(iso_distance_codes) + 1)
    ]
    return h_list


def _edge_triplet_hash_undirected(u, v, data, graph):
    return _hash_unordered_triplet(
        _hash(graph.nodes[u]['label']),
        _hash(data['label']),
        _hash(graph.nodes[v]['label']),
    )


def _edge_triplet_hash_directed(u, v, data, graph):
    return _hash((
        "directed-edge",
        _hash(graph.nodes[u]['label']),
        _hash(data['label']),
        _hash(graph.nodes[v]['label']),
    ))


def _simple_edge_records(graph):
    for u, v, data in graph.edges(data=True):
        yield u, v, None, data


def _multi_edge_records(graph):
    yield from graph.edges(keys=True, data=True)


def _precompute_edge_triplet_hashes(graph, edge_records, edge_hasher):
    for u, v, key, data in edge_records(graph):
        data['triplet_hash'] = edge_hasher(u, v, data, graph)


def edge_triplet_hash(u, v, graph, edge_key=None):
    """Hash one edge, dispatching once for this standalone call."""
    data = graph.edges[u, v, edge_key] if graph.is_multigraph() else graph.edges[u, v]
    hasher = _edge_triplet_hash_directed if graph.is_directed() else _edge_triplet_hash_undirected
    return hasher(u, v, data, graph)


def precompute_edge_triplet_hashes(graph):
    """Precompute edge hashes using a strategy selected once per graph."""
    if graph.is_directed():
        edge_hasher = _edge_triplet_hash_directed
    else:
        edge_hasher = _edge_triplet_hash_undirected
    edge_records = _multi_edge_records if graph.is_multigraph() else _simple_edge_records
    _precompute_edge_triplet_hashes(graph, edge_records, edge_hasher)

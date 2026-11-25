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


def node_hash(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    incident_patterns = []
    if graph.is_directed():
        incident = []
        for u, v, data in graph.out_edges(node_idx, data=True):
            incident.append(("out", v, data))
        for u, v, data in graph.in_edges(node_idx, data=True):
            if u == v:  # A loop is represented once.
                continue
            incident.append(("in", u, data))
        for direction, neighbor, data in incident:
            incident_patterns.append(_hash((
                direction,
                _hash(graph.nodes[neighbor]['label']),
                _hash(data['label']),
            )))
    else:
        for neighbor in graph.neighbors(node_idx):
            data_items = graph[node_idx][neighbor].values() if graph.is_multigraph() else (graph[node_idx][neighbor],)
            for data in data_items:
                incident_patterns.append(_hash((
                    _hash(graph.nodes[neighbor]['label']),
                    _hash(data['label']),
                )))
    nh = hash_set(incident_patterns)
    ext_node_h = _hash((uh, nh))
    return ext_node_h


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


def edge_triplet_hash(u, v, graph, edge_key=None):
    data = graph.edges[u, v, edge_key] if graph.is_multigraph() else graph.edges[u, v]
    if graph.is_directed():
        return _hash((
            "directed-edge",
            _hash(graph.nodes[u]['label']),
            _hash(data['label']),
            _hash(graph.nodes[v]['label']),
        ))
    return _hash_unordered_triplet(
        _hash(graph.nodes[u]['label']),
        _hash(data['label']),
        _hash(graph.nodes[v]['label']),
    )


def precompute_edge_triplet_hashes(graph):
    records = graph.edges(keys=True) if graph.is_multigraph() else graph.edges()
    for edge in records:
        u, v = edge[:2]
        key = edge[2] if graph.is_multigraph() else None
        data = graph.edges[u, v, key] if graph.is_multigraph() else graph.edges[u, v]
        data['triplet_hash'] = edge_triplet_hash(u, v, graph, key)

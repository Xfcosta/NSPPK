import hashlib
from collections import defaultdict

import networkx as nx


def _hash(value):
    value_str = str(value)
    value_bytes = value_str.encode()
    sha256_hash = hashlib.sha256(value_bytes).hexdigest()
    hash_int = int(sha256_hash, 16) & (2**30 - 1)
    return hash_int


def masked_hash_value(value, bitmask=(2**20 - 1)):
    return _hash(value) & bitmask


def hash_value(value, nbits=10):
    max_index = 2 ** nbits
    h = masked_hash_value(value, max_index - 3)
    h += 2
    return h


def hash_sequence(iterable):
    return _hash(tuple(iterable))


def hash_set(iterable):
    sorted_iterable = sorted(iterable)
    tuple_representation = tuple(sorted_iterable)
    return _hash(tuple_representation)


def node_hash(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    edges_h = [
        _hash((_hash(graph.nodes[v]['label']), _hash(graph.edges[node_idx, v]['label'])))
        for v in graph.neighbors(node_idx)
    ]
    nh = hash_set(edges_h)
    ext_node_h = _hash((uh, nh))
    return ext_node_h


def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items():
        reversed_dict[value].append(key)
    return reversed_dict


def rooted_graph_hashes(node_idx, graph, radius=1):
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=radius)
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
    iso_distance_codes_list = [
        hash_set([
            graph.nodes[curr_node_idx]['node_label_hash']
            if dist == 0 else graph.nodes[curr_node_idx]['node_hash']
            for curr_node_idx in node_idxs
        ])
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items())
    ]
    h_list = [
        hash_sequence(iso_distance_codes_list[:i])
        for i in range(1, len(iso_distance_codes_list) + 1)
    ]
    return h_list


def edge_triplet_hash(u, v, graph):
    return hash_set([
        _hash(graph.nodes[u]['label']),
        _hash(graph.edges[u, v]['label']),
        _hash(graph.nodes[v]['label']),
    ])


def precompute_edge_triplet_hashes(graph):
    for u, v in graph.edges():
        graph.edges[u, v]['triplet_hash'] = edge_triplet_hash(u, v, graph)

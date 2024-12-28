"""
Module for encoding graphs into feature vectors using Neighborhood Subgraph Pairwise Propagation Kernel (NSPPK).
"""

from collections import Counter, defaultdict
from copy import copy
from scipy.sparse import csr_matrix
import multiprocessing_on_dill as mp
import networkx as nx
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
# Import necessary modules for data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

def hash_list(seq):
    """
    Hashes a list by converting it to a tuple and then hashing.

    Args:
        seq (list): The list to hash.

    Returns:
        int: The hash value of the tuple.
    """
    return hash(tuple(seq))


def masked_hash_value(value, bitmask=4294967295):
    """
    Applies a bitmask to the hash of a value to limit its size.

    Args:
        value: The value to hash and mask.
        bitmask (int, optional): The bitmask to apply. Defaults to 4294967295.

    Returns:
        int: The masked hash value.
    """
    return hash(value) & bitmask


def hash_value(value, nbits=10):
    """
    Hashes a value and limits it to a specified number of bits, ensuring it is at least 2.

    Args:
        value: The value to hash.
        nbits (int, optional): Number of bits to limit the hash. Defaults to 10.

    Returns:
        int: The hashed value limited to nbits, and in the range [2, 2**nbits - 1]
    """
    max_index = 2 ** nbits
    h = masked_hash_value(value, max_index - 3)
    h += 2
    return h


def node_hash(node_idx, graph):
    """
    Computes a hash for a node based on its label and the labels of its neighbors.

    Args:
        node_idx (int): The index of the node in the graph.
        graph (networkx.Graph): The graph containing the node.

    Returns:
        int: The computed hash for the node.
    """
    uh = hash(graph.nodes[node_idx]['label'])
    edges_h = [
        hash((hash(graph.nodes[v]['label']), hash(graph.edges[node_idx, v]['label'])))
        for v in graph.neighbors(node_idx)
    ]
    nh = hash_list(sorted(edges_h))
    ext_node_h = hash((uh, nh))
    return ext_node_h


def invert_dict(mydict):
    """
    Inverts a dictionary mapping keys to single values into a dictionary mapping values to lists of keys.

    Args:
        mydict (dict): The dictionary to invert.

    Returns:
        defaultdict(list): The inverted dictionary.
    """
    reversed_dict = defaultdict(list)
    for key, value in mydict.items():
        reversed_dict[value].append(key)
    return reversed_dict


def rooted_graph_hash(node_idx, graph, radius=1):
    """
    Computes a hash for the subgraph rooted at a given node up to a certain radius.

    Args:
        node_idx (int): The index of the root node.
        graph (networkx.Graph): The graph containing the node.
        radius (int, optional): The radius up to which to consider the subgraph. Defaults to 1.

    Returns:
        int: The hash representing the rooted subgraph.
    """
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=radius)
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
    iso_distance_codes_list = [
        hash_list(sorted([graph.nodes[curr_node_idx]['node_hash'] for curr_node_idx in node_idxs]))
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items())
    ]
    return hash_list(iso_distance_codes_list)


def items_to_sparse_histogram(items, nbits):
    """
    Converts a list of items into a sparse histogram vector.

    Args:
        items (list of int): The list of items to count.
        nbits (int): The number of bits used to define the histogram size (2^nbits).

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse histogram vector.
    """
    histogram_dict = Counter(items)
    num_items = len(histogram_dict)
    rows = np.zeros(num_items, dtype=int)
    cols = np.array(list(histogram_dict.keys()), dtype=int)
    data = np.array(list(histogram_dict.values()), dtype=int)
    return sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, 2**nbits))


def structural_node_vectors(original_graph, radius, distance, connector, nbits, degree_threshold=None):
    """
    Generates a feature vector for each node in the graph.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix of node feature vectors.
    """
    graph = original_graph.copy()
    for node_idx in graph.nodes():
        graph.nodes[node_idx]['node_hash'] = node_hash(node_idx, graph)

    cutoff = max(radius, connector)
    for node_idx in graph.nodes():
        graph.nodes[node_idx]['rooted_graph_hash'] = np.zeros(cutoff, dtype=int)
        degree = graph.degree[node_idx]
        effective_cutoff = 0 if degree_threshold is not None and degree > degree_threshold else cutoff
        for r in range(effective_cutoff):
            graph.nodes[node_idx]['rooted_graph_hash'][r] = rooted_graph_hash(node_idx, graph, radius=r)

    node_vectors = []
    for node_idx in graph.nodes():
        node_codes_list = []
        for code_i in graph.nodes[node_idx]['rooted_graph_hash']:
            node_idxs_to_dist_dict = nx.single_source_shortest_path_length(
                graph, node_idx, cutoff=distance
            )
            dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
            for connect in range(connector + 1):
                for dist, node_idxs in sorted(dist_to_node_idxs_dict.items()):
                    for curr_node_idx in node_idxs:
                        if connect > 0:
                            union_of_shortest_paths = set()
                            shortest_paths = list(nx.all_shortest_paths(graph, source=node_idx, target=curr_node_idx))
                            for path in shortest_paths:
                                union_of_shortest_paths.update(path)
                        for code_j in graph.nodes[curr_node_idx]['rooted_graph_hash']:
                            if connect == 0:
                                paired_code = hash_list([code_i, dist, code_j])
                            else:
                                union_of_shortest_paths_code = hash_list(
                                    sorted([
                                        hash_list([node_idxs_to_dist_dict[node], graph.nodes[node]['rooted_graph_hash'][connect - 1]])
                                        for node in union_of_shortest_paths
                                    ])
                                )
                                paired_code = hash_list([code_i, dist, code_j, union_of_shortest_paths_code])
                            paired_code = hash_value(paired_code, nbits=nbits)
                            node_codes_list.append(paired_code)
        node_codes_list.append(0)
        node_codes_list.extend([1] * graph.degree[node_idx])
        node_vector = items_to_sparse_histogram(node_codes_list, nbits)
        node_vectors.append(node_vector)

    return sp.sparse.vstack(node_vectors)


def get_attribute_mtx(original_graph, attribute_key):
    """
    Extracts node attributes from a graph and stacks them into a matrix.

    Args:
        original_graph (networkx.Graph): The input graph.
        attribute_key (str): The key corresponding to the node attribute to extract.

    Returns:
        numpy.ndarray: The matrix of node attributes.
    """
    return np.vstack([
        original_graph.nodes[node_idx][attribute_key]
        for node_idx in original_graph.nodes()
    ])


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


def node_vector(original_graph, radius, distance, connector, nbits, weight_key=None, attribute_key=None, degree_threshold=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature matrix representing the graph.
    """
    node_vectors_mtx = structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold
    )
    if weight_key is not None:
        node_vectors_mtx = reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key)
    if attribute_key is not None:
        attribute_mtx = get_attribute_mtx(original_graph, attribute_key)
        feature_node_vectors_mtx = node_vectors_mtx.todense().A
        feature_node_vectors_mtx = attribute_mtx.T.dot(feature_node_vectors_mtx).dot(feature_node_vectors_mtx.T).T
        feature_node_vectors_mtx = csr_matrix(feature_node_vectors_mtx)
        node_vectors_mtx = sp.sparse.hstack([csr_matrix(attribute_mtx), feature_node_vectors_mtx, node_vectors_mtx])
    return node_vectors_mtx


def node_graph_vector(original_graph, radius, distance, connector, nbits, weight_key=None, attribute_key=None, degree_threshold=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    node_vectors_mtx = structural_node_vectors(original_graph, radius, distance, connector, nbits, degree_threshold)
    node_vectors_mtx = node_vectors_mtx.todense().A
    attribute_node_vectors_mtx = node_vector(original_graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
    vector_ = attribute_node_vectors_mtx.T.dot(node_vectors_mtx)
    vector = vector_.reshape(1, -1)
    vector = csr_matrix(vector)
    return csr_matrix(vector)


def graph_vector(original_graph, radius, distance, connector, nbits, weight_key=None, attribute_key=None, degree_threshold=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    node_vectors_mtx = structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold
    )
    if weight_key is not None:
        node_vectors_mtx = reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key)
    if attribute_key is None:
        vector = node_vectors_mtx.sum(axis=0)
        vector = csr_matrix(vector)
    else:
        attribute_mtx = get_attribute_mtx(original_graph, attribute_key)
        node_vectors_mtx = node_vectors_mtx.todense().A
        vector_ = attribute_mtx.T.dot(node_vectors_mtx)
        vector = vector_.reshape(1, -1)
        vector = csr_matrix(vector)
    return vector


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


def paired_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, attribute_key=None, degree_threshold=None):
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
        attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                graph_vector(graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            graph_vector(graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
            for graph in graphs
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, attribute_key=None, degree_threshold=None):
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
        attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                node_graph_vector(graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            node_graph_vector(graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
            for graph in graphs
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, attribute_key=None, degree_threshold=None):
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
        attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        list of scipy.sparse.csr_matrix: A list where each element is a matrix of node vectors for a graph.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                node_vector(graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_node_vectors_subsets = pool.map(process_subset, subsets)
        graph_node_vectors = [vec for subset in graph_node_vectors_subsets for vec in subset]
    else:
        graph_node_vectors = [
            node_vector(graph, radius, distance, connector, nbits, weight_key, attribute_key, degree_threshold)
            for graph in graphs
        ]

    return graph_node_vectors


class AbstractNSPPK(BaseEstimator, TransformerMixin):
    """
    AbstractNSPPK is an abstract base class for encoding graphs into feature vectors.

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
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        use_node_kernel (bool, default=False): Whether to use node-level kernel encoding.
    """

    def __init__(self, embedder=None, clustering_predictor=None, classifier=None, radius=1, distance=3, connector=0,
                 nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None, use_node_kernel=False):
        self.embedder = embedder
        self.clustering_predictor = clustering_predictor
        self.classifier = classifier
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.attribute_key = attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.use_node_kernel = use_node_kernel

    def __repr__(self):
        """
        Returns a string representation of the AbstractNSPPK instance.

        Returns:
            str: The string representation.
        """
        infos = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({infos})"

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
        if self.attribute_key and (self.attribute_dim or self.attribute_alphabet_size):
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.attribute_key]
                for graph in graphs
                for node_idx in graph.nodes()
            ])

            if self.attribute_dim and self.embedder:
                self.embedder.fit(attribute_mtx)
                attribute_mtx = self.embedder.transform(attribute_mtx)

            if self.attribute_alphabet_size and self.clustering_predictor:
                targets = self.clustering_predictor.fit_predict(attribute_mtx)
                self.classifier.fit(attribute_mtx, targets)

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
                graph.nodes[node_idx][self.attribute_key]
                for node_idx in graph.nodes()
            ])
            embeddings = self.embedder.transform(attribute_mtx)
            out_graph = graph.copy()
            for embedding, node_idx in zip(embeddings, out_graph.nodes()):
                out_graph.nodes[node_idx]['original_' + self.attribute_key] = graph.nodes[node_idx][self.attribute_key]
                out_graph.nodes[node_idx][self.attribute_key] = embedding
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
                graph.nodes[node_idx][self.attribute_key]
                for node_idx in graph.nodes()
            ])
            labels = self.classifier.predict(attribute_mtx)
            out_graph = graph.copy()
            for label, node_idx in zip(labels, out_graph.nodes()):
                out_graph.nodes[node_idx]['label'] = label
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
        if self.attribute_key and self.attribute_dim:
            graphs = self.embed_attributes(graphs)
        if self.attribute_key and self.attribute_alphabet_size:
            graphs = self.set_discrete_labels(graphs)

        if self.use_node_kernel:
            data_mtx = paired_node_graphs_vector_encoder(
                graphs, 
                self.radius, 
                self.distance, 
                self.connector, 
                self.nbits,
                self.parallel, 
                self.weight_key,
                self.attribute_key,
                self.degree_threshold
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
                self.attribute_key,
                self.degree_threshold
            )

        if self.dense:
            data_mtx = data_mtx.todense().A

        return data_mtx


class NSPPK(BaseEstimator, TransformerMixin):
    """
    NSPPK (Neighborhood Subgraph Pairwise Propagation Kernel) class specialized from AbstractNSPPK.

    This class encodes graphs into feature vectors suitable for machine learning models by capturing both
    local (node-level) and structural (subgraph-level) information. It specializes the AbstractNSPPK by
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
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None):
        embedder = TruncatedSVD(n_components=attribute_dim) if attribute_dim else None
        clustering_predictor = KMeans(n_clusters=attribute_alphabet_size) if attribute_alphabet_size else None
        classifier = ExtraTreesClassifier(
            n_estimators=300, 
            n_jobs=-1 if parallel else None
        ) if attribute_alphabet_size else None

        self.abstract_nsppk = AbstractNSPPK(
            embedder=embedder,
            clustering_predictor=clustering_predictor,
            classifier=classifier,
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            weight_key=weight_key,
            attribute_key=attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size
        )

    def __repr__(self):
        """
        Returns a string representation of the NSPPK instance.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}({self.abstract_nsppk})"

    def fit(self, graphs, targets=None):
        """
        Fit the NSPPK encoder on the given graphs by delegating to AbstractNSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.abstract_nsppk.fit(graphs, targets)
        return self

    def transform(self, graphs):
        """
        Transform the input graphs into feature vectors by delegating to AbstractNSPPK.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        return self.abstract_nsppk.transform(graphs)


class NodeGraphNSPPK(BaseEstimator, TransformerMixin):
    """
    NSPPK (Neighborhood Subgraph Pairwise Propagation Kernel) class specialized from AbstractNSPPK.

    This class encodes graphs into feature vectors suitable for machine learning models by capturing both
    local (node-level) and structural (subgraph-level) information. It specializes the AbstractNSPPK by
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
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None):
        embedder = TruncatedSVD(n_components=attribute_dim) if attribute_dim else None
        clustering_predictor = KMeans(n_clusters=attribute_alphabet_size) if attribute_alphabet_size else None
        classifier = ExtraTreesClassifier(
            n_estimators=300, 
            n_jobs=-1 if parallel else None
        ) if attribute_alphabet_size else None

        self.abstract_nsppk = AbstractNSPPK(
            embedder=embedder,
            clustering_predictor=clustering_predictor,
            classifier=classifier,
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            weight_key=weight_key,
            attribute_key=attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size,
            use_node_kernel=True
        )

    def __repr__(self):
        """
        Returns a string representation of the NSPPK instance.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}({self.abstract_nsppk})"

    def fit(self, graphs, targets=None):
        """
        Fit the NSPPK encoder on the given graphs by delegating to AbstractNSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.abstract_nsppk.fit(graphs, targets)
        return self

    def transform(self, graphs):
        """
        Transform the input graphs into feature vectors by delegating to AbstractNSPPK.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        return self.abstract_nsppk.transform(graphs)


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
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): The target dimensionality for attribute vectors; if set, applies SVD for dimensionality reduction.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None):
        self.nsp_pk = NSPPK(
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            weight_key=weight_key,
            attribute_key=attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size
        )

    def __repr__(self):
        """
        Generate a string representation of the NodeNSPPK instance.

        Returns:
            str: The string representation of the object.
        """
        return f"{self.__class__.__name__}({self.nsp_pk})"

    def fit(self, graphs, targets=None):
        """
        Fit the NodeNSPPK encoder on the given graphs by delegating to NSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.nsp_pk.fit(graphs, targets)
        return self

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
        if self.nsp_pk.abstract_nsppk.attribute_key and self.nsp_pk.abstract_nsppk.attribute_dim:
            graphs = self.nsp_pk.abstract_nsppk.embed_attributes(graphs)
        if self.nsp_pk.abstract_nsppk.attribute_key and self.nsp_pk.abstract_nsppk.attribute_alphabet_size:
            graphs = self.nsp_pk.abstract_nsppk.set_discrete_labels(graphs)

        nodes_data_mtx_list = paired_node_vector_encoder(
            graphs,
            self.nsp_pk.abstract_nsppk.radius,
            self.nsp_pk.abstract_nsppk.distance,
            self.nsp_pk.abstract_nsppk.connector,
            self.nsp_pk.abstract_nsppk.nbits,
            self.nsp_pk.abstract_nsppk.parallel,
            self.nsp_pk.abstract_nsppk.weight_key,
            self.nsp_pk.abstract_nsppk.attribute_key,
            self.nsp_pk.abstract_nsppk.degree_threshold
        )

        if self.nsp_pk.abstract_nsppk.dense:
            nodes_data_mtx_list = [mtx.todense().A for mtx in nodes_data_mtx_list]

        return nodes_data_mtx_list


class ImportanceNSPPK(object):
    """
    A class to visualize node importance in graphs using feature importance derived from an ensemble classifier.

    This class leverages the NSPPK (Neighborhood Subgraph Pairwise Path Kernel) method
    to decompose graphs into substructures and vectorize them. It then uses an ExtraTreesClassifier
    to compute feature importances, which are used to assign importance weights to nodes in the graphs.

    Attributes:
        node_nsppk (object): An instance of NSPPK or a similar class responsible for graph decomposition and vectorization.
        importance_key (str): The key under which node importance weights will be stored in the graph's node attributes.
        n_iter (int): Number of iterations for training the classifier to ensure stable feature importance estimates.
        n_estimators (int): Number of trees in the ExtraTreesClassifier.
        quantile (float): Quantile threshold for filtering out low-importance features.
        parallel (bool): Whether to utilize parallel processing during classifier training.
        normalize (bool): Whether to normalize node and edge weights to [0,1] range.
        feature_importance_vector (np.ndarray): Normalized feature importance scores after processing.
    """
    
    def __init__(self, node_nsppk, importance_key='att', n_iter=10, n_estimators=100, quantile=0.5, parallel=True, normalize=True):
        """
        Initializes the ImportanceNSPPK instance with specified parameters.

        Args:
            node_nsppk (object): An instance responsible for graph decomposition and vectorization (e.g., NSPPK).
            importance_key (str, optional): The key for storing node importance in graph attributes. Defaults to 'att'.
            n_iter (int, optional): Number of iterations for classifier training. Defaults to 10.
            n_estimators (int, optional): Number of trees in the ExtraTreesClassifier. Defaults to 100.
            quantile (float, optional): Quantile threshold for filtering feature importances. Defaults to 0.5.
            parallel (bool, optional): Whether to run classifier training in parallel. Defaults to True.
            normalize (bool, optional): Whether to normalize weights to [0,1]. Defaults to True.
        """
        self.node_nsppk = node_nsppk
        self.importance_key = importance_key
        self.n_iter = n_iter
        self.n_estimators = n_estimators
        self.quantile = quantile
        self.parallel = parallel
        self.normalize = normalize
        self.feature_importance_vector = None  # To store the computed normalized feature importance scores

    def fit(self, graphs, targets=None):
        """
        Fits the model by computing feature importances from the provided graphs and corresponding target labels.

        This method vectorizes the input graphs, aggregates node features, and trains an ExtraTreesClassifier
        multiple times to obtain stable estimates of feature importances. It then processes these importances
        by averaging, subtracting the standard deviation, thresholding, and normalizing.

        Args:
            graphs (list): A list of graph objects to be vectorized and used for training.
            targets (array-like, optional): Target labels corresponding to each graph. Required for supervised training.

        Returns:
            self: Returns the instance itself to allow method chaining.
        """
        # Vectorize the graphs using the provided NSPPK instance
        node_feature_mtx_list = self.node_nsppk.fit_transform(graphs)
        
        # Aggregate node features for each graph by summing across nodes
        X = np.vstack([node_feature_mtx.sum(axis=0) for node_feature_mtx in node_feature_mtx_list])

        feature_importances = []  # List to store feature importances from each iteration

        # Perform multiple iterations to compute stable feature importances
        for it in range(self.n_iter):
            # Split the data into training and testing sets with a fixed random state for reproducibility
            train_X, test_X, train_targets, test_targets = train_test_split(
                X, targets, train_size=0.7, random_state=it+1
            )
            
            # Initialize and train the ExtraTreesClassifier
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators, 
                n_jobs=-1 if self.parallel else None,
                random_state=it+1  # Ensure reproducibility across iterations
            ).fit(train_X, train_targets)
            
            # Append the feature importances from the trained classifier
            feature_importances.append(clf.feature_importances_)
        
        # Convert the list of feature importances into a NumPy matrix for statistical processing
        feature_importances_mtx = np.vstack(feature_importances)
        
        # Compute the mean and standard deviation of feature importances across all iterations
        mean_importances = np.mean(feature_importances_mtx, axis=0)
        std_importances = np.std(feature_importances_mtx, axis=0)
        
        # Calculate the adjusted feature importance by subtracting the standard deviation from the mean
        adjusted_importances = mean_importances - std_importances
        
        # Set any negative importance values to zero to ensure non-negativity
        adjusted_importances[adjusted_importances < 0] = 0
        
        # Apply a quantile threshold to filter out features with low importance
        threshold = np.quantile(adjusted_importances, self.quantile)
        adjusted_importances[adjusted_importances < threshold] = 0
        
        # Normalize the feature importance vector by dividing by its maximum value to scale between 0 and 1
        max_importance = np.max(adjusted_importances)
        if max_importance > 0:
            self.feature_importance_vector = adjusted_importances / max_importance
        else:
            self.feature_importance_vector = adjusted_importances  # Remain zero if max is zero
        
        return self  # Allow method chaining

    def transform(self, graphs):
        """
        Transforms the input graphs by assigning normalized importance weights to each node based on feature importances.

        This method vectorizes the nodes of each graph, computes weights by combining node features with the
        precomputed feature importance vector, and normalizes these weights to assign importance scores to nodes.
        Additionally, edge weights are assigned based on the product of the connected nodes' importance scores.

        Args:
            graphs (list): A list of graph objects to be transformed with node and edge importance weights.

        Returns:
            out_graphs (list): A list of transformed graph objects with updated node and edge importance attributes.
        """
        # Vectorize the nodes of the graphs using the provided NSPPK instance
        node_feature_mtx_list = self.node_nsppk.transform(graphs)
        
        out_graphs = []  # List to store the transformed graphs

        # Iterate over each graph and its corresponding node feature matrix
        for graph, node_feature_mtx in zip(graphs, node_feature_mtx_list):
            out_graph = graph.copy()  # Create a copy of the graph to avoid modifying the original
            
            # Compute the weights by multiplying node features with the feature importance vector
            weights = node_feature_mtx * self.feature_importance_vector
            
            # Sum the weights across all feature dimensions for each node to obtain a single weight per node
            weights = np.sum(weights, axis=1)
            
            if self.normalize:
                # Normalize the weights by the maximum weight to scale between 0 and 1
                max_weight = np.max(weights) if np.max(weights) > 0 else 1
                normalized_weights = weights / max_weight
            else:
                normalized_weights = weights
            
            # Assign the normalized weight to each node in the graph
            for i, weight in enumerate(normalized_weights):
                out_graph.nodes[i][self.importance_key] = weight
            
            # Assign edge weights based on the product of the connected nodes' importance scores
            for u, v in out_graph.edges():
                out_graph.edges[u, v][self.importance_key] = out_graph.nodes[u][self.importance_key] * out_graph.nodes[v][self.importance_key]
        
            out_graphs.append(out_graph)  # Add the transformed graph to the output list
        
        return out_graphs  # Return the list of transformed graphs

# Import necessary modules and libraries
from collections import Counter, defaultdict  # For counting elements and creating dictionaries with default types
from copy import copy                         # For copying objects
from scipy.sparse import csr_matrix           # For handling sparse matrices and vertical stacking
import multiprocessing_on_dill as mp          # For parallel processing with support for more object types
import networkx as nx                         # For graph operations
import numpy as np                            # For numerical operations
import scipy as sp                            # For scientific computing
from sklearn.base import BaseEstimator, TransformerMixin  # Import scikit-learn base classes
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.decomposition import TruncatedSVD

def hash_list(seq):
    """
    Hashes a list by converting it to a tuple and then hashing.

    Args:
        seq (list): The list to hash.

    Returns:
        int: The hash value of the tuple.
    """
    # Convert the list to a tuple (since lists are not hashable) and then compute the hash
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
    # Compute the hash of the value and apply the bitmask using bitwise AND
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
    # Compute the masked hash value, adjusting to exclude indices 0 and 1
    h = masked_hash_value(value, max_index - 1 - 2)
    # Shift the hash to start from 2 to ensure it's within the desired range
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
    # Hash the label of the current node
    uh = hash(graph.nodes[node_idx]['label'])
    
    edges_h = []
    # Iterate over the neighbors of the node
    for v in graph.neighbors(node_idx):
        # Hash the label of the neighbor node
        neighbor_label_hash = hash(graph.nodes[v]['label'])
        # Hash the label of the edge between node_idx and v
        edge_label_hash = hash(graph.edges[node_idx, v]['label'])
        # Combine the neighbor label hash and edge label hash into a tuple and hash it
        neighbor_edge_hash = hash((neighbor_label_hash, edge_label_hash))
        edges_h.append(neighbor_edge_hash)
    
    # Hash the sorted list of neighbor-edge hashes to ensure consistency
    nh = hash_list(sorted(edges_h))
    
    # Combine the node's own label hash and the neighbors' hash
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
    # Create a defaultdict of lists to store the inverted mapping
    reversed_dict = defaultdict(list)
    # Iterate over the original dictionary and reverse the mapping
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
    # Compute shortest path lengths from the root node up to the given radius
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=radius)
    # node_idxs_to_dist_dict maps node indices to their distance from node_idx

    # Invert the dictionary to map distances to node indices
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
    # dist_to_node_idxs_dict maps distances to lists of node indices at that distance

    iso_distance_codes_list = []
    
    # Iterate over each distance in sorted order
    for dist in sorted(dist_to_node_idxs_dict):
        node_idxs = dist_to_node_idxs_dict[dist]
        # Collect hashes of all nodes at the current distance
        codes_list = [graph.nodes[curr_node_idx]['node_hash'] for curr_node_idx in node_idxs]
        # Hash the sorted list of node hashes to maintain consistency
        code = hash_list(sorted(codes_list))
        # Append the code for this distance
        iso_distance_codes_list.append(code)
    
    # Hash the list of distance-based codes to get the final rooted graph hash
    code = hash_list(iso_distance_codes_list)
    return code

def items_to_sparse_histogram(items, nbits):
    """
    Converts a list of items into a sparse histogram vector.

    Args:
        items (list of int): The list of items to count.
        nbits (int): The number of bits used to define the histogram size (2^nbits).

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse histogram vector.
    """
    # Count occurrences of each item using Counter
    histogram_dict = Counter(items)
    # Number of unique items
    num_items = len(histogram_dict)
    # Create arrays for constructing the sparse matrix
    rows = np.zeros(num_items, dtype=int)  # All entries belong to row 0
    cols = np.array(list(histogram_dict.keys()), dtype=int)  # Column indices correspond to item values
    data = np.array(list(histogram_dict.values()), dtype=int)  # Data are counts of each item

    # Create a CSR (Compressed Sparse Row) matrix with the counts
    vector = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, 2**nbits))
    return vector 

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
    # Create a copy of the original graph to avoid modifying it
    graph = original_graph.copy()
    
    # Step 1: Compute and store node hashes for all nodes in the graph
    for node_idx in graph.nodes():
        # Compute a unique hash for the node based on its label and the labels of its immediate neighbors
        graph.nodes[node_idx]['node_hash'] = node_hash(node_idx, graph)
    
    # Step 2: Compute and store rooted graph hashes for each node up to the specified radius
    # The 'cutoff' is the maximum of the 'radius' and 'connector' parameters
    cutoff = max(radius, connector)
    for node_idx in graph.nodes():
        # Initialize an array to store rooted graph hashes up to the cutoff radius
        graph.nodes[node_idx]['rooted_graph_hash'] = np.zeros(cutoff, dtype=int)
        # Determine effective cutoff based on degree threshold
        degree = graph.degree[node_idx]
        if degree_threshold is not None and degree > degree_threshold:
            effective_cutoff = 0
        else:
            effective_cutoff = cutoff
        # For each radius from 0 up to (but not including) cutoff
        for r in range(effective_cutoff):
            # Compute the rooted graph hash for the node at radius 'r'
            label = rooted_graph_hash(node_idx, graph, radius=r)
            # Store the computed hash in the array at position 'r'
            graph.nodes[node_idx]['rooted_graph_hash'][r] = label

    node_vectors = []  # List to store feature vectors for each node
    
    # Step 3: Iterate over all nodes to generate paired codes based on distances and rooted graph hashes
    for node_idx in graph.nodes():
        node_codes_list = []  # List to store code identifiers for the current node
        # For each rooted graph hash 'code_i' of the node (up to the cutoff radius)
        for code_i in graph.nodes[node_idx]['rooted_graph_hash']:
            # Step 3a: Find all nodes within the specified 'distance' from the current node
            node_idxs_to_dist_dict = nx.single_source_shortest_path_length(
                graph, node_idx, cutoff=distance
            )
            # node_idxs_to_dist_dict is a dictionary mapping node indices to their shortest path distance from 'node_idx'

            # Invert the dictionary to map distances to lists of node indices at that distance
            dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)

            # Step 3b: Iterate over each level of connector thickness from 0 up to 'connector'
            for connect in range(connector + 1):
                # Step 3c: Iterate over each distance and corresponding list of nodes at that distance
                for dist in sorted(dist_to_node_idxs_dict):
                    node_idxs = dist_to_node_idxs_dict[dist]  # List of node indices at distance 'dist'
                    for curr_node_idx in node_idxs:
                        if connect > 0:
                            # If connector thickness is greater than 0, include the union of shortest paths
                            # Initialize a set to collect nodes on the union of all shortest paths between 'node_idx' and 'curr_node_idx'
                            union_of_shortest_paths = set()
                            # Compute all shortest paths between 'node_idx' and 'curr_node_idx'
                            shortest_paths = list(nx.all_shortest_paths(graph, source=node_idx, target=curr_node_idx))
                            for path in shortest_paths:
                                # Add all nodes in each path to the union set
                                union_of_shortest_paths.update(path)

                        # For each rooted graph hash 'code_j' of the current neighbor node 'curr_node_idx'
                        for code_j in graph.nodes[curr_node_idx]['rooted_graph_hash']:
                            
                            if connect == 0:
                                # When connector thickness is 0, compute the paired code without the union of shortest paths
                                # Paired code includes the rooted graph hash of 'node_idx', the distance, and the rooted graph hash of 'curr_node_idx'
                                paired_code = hash_list([code_i, dist, code_j])
                            else:
                                # When connector thickness is greater than 0, include the union of shortest paths in the code
                                # For each node in the union of shortest paths, extract its rooted graph hash at level (connect - 1)
                                union_of_shortest_paths_node_hashes = [
                                    # Create a hash of (distance to the node, rooted graph hash at level 'connect - 1')
                                    hash_list([node_idxs_to_dist_dict[node], graph.nodes[node]['rooted_graph_hash'][connect - 1]])
                                    for node in union_of_shortest_paths
                                ]

                                # Sort the list of hashes to ensure consistent ordering
                                sorted_list = sorted(union_of_shortest_paths_node_hashes)
                                # Hash the sorted list to obtain a code representing the union of shortest paths
                                union_of_shortest_paths_code = hash_list(sorted_list)
                                # Compute the paired code including the union of shortest paths code
                                paired_code = hash_list([code_i, dist, code_j, union_of_shortest_paths_code])

                            # Limit the hash to the specified number of bits to control the feature space size
                            paired_code = hash_value(paired_code, nbits=nbits)
                            # Append the computed paired code to the list for the current node
                            node_codes_list.append(paired_code)

        # **Add fixed features to the node codes list by appending**
        # Append '0' once to set node_vector[0,0] = 1 (bias term)
        node_codes_list.append(0)
        # Append '1' 'degree' times to set node_vector[0,1] = degree of the node
        node_codes_list.extend([1] * graph.degree[node_idx])

        # Step 4: Convert the list of paired codes into a sparse histogram vector
        node_vector = items_to_sparse_histogram(node_codes_list, nbits)
        
        # Append the node vector to the list of node vectors
        node_vectors.append(node_vector)
    
    # Stack all node vectors vertically to create a sparse matrix
    node_vectors_mtx = sp.sparse.vstack(node_vectors)

    # Return the matrix of node feature vectors
    return node_vectors_mtx


def graph_vector(original_graph, radius, distance, connector, nbits, attribute_key=None, degree_threshold=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        attribute_key (str, optional): Node attribute key to use for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    # Generate node vectors (feature vectors for each node)
    node_vectors_mtx = structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold
    )
    if attribute_key is None:
        # If no attribute key is specified, sum the node vectors to get the graph vector
        vector = node_vectors_mtx.sum(axis=0)
        vector = csr_matrix(vector)  # Ensure the result is a CSR matrix
    else:
        # If an attribute key is specified, use the node attributes to weight the node vectors
        attribute_mtx = np.vstack([
            original_graph.nodes[node_idx][attribute_key]
            for node_idx in original_graph.nodes()
        ])
        # Multiply attributes with node vectors to get the graph vector
        node_vectors_mtx = node_vectors_mtx.todense().A
        vector_ = attribute_mtx.T.dot(node_vectors_mtx)
        vector = vector_.reshape(1, -1)
        vector = csr_matrix(vector)  # Ensure the result is a CSR matrix
    return vector

def node_vector(original_graph, radius, distance, connector, nbits, attribute_key=None, degree_threshold=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        attribute_key (str, optional): Node attribute key to use for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    # Generate node vectors (feature vectors for each node)
    node_vectors_mtx = structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold
    )
    if attribute_key is not None:
        # If an attribute key is specified, use the node attributes to create additional features
        attribute_mtx = np.vstack([
            original_graph.nodes[node_idx][attribute_key]
            for node_idx in original_graph.nodes()
        ])
        # Multiply attributes with node vectors to get the graph vector
        feature_node_vectors_mtx = node_vectors_mtx.todense().A
        feature_node_vectors_mtx = attribute_mtx.T.dot(node_vectors_mtx).dot(feature_node_vectors_mtx.T).T
        feature_node_vectors_mtx = csr_matrix(feature_node_vectors_mtx)  # Ensure the result is a CSR matrix
        # Horizontally stack the attribute matrix, feature node vectors, and original node vectors
        node_vectors_mtx = sp.sparse.hstack([csr_matrix(attribute_mtx), feature_node_vectors_mtx, node_vectors_mtx])
    return node_vectors_mtx

def paired_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, attribute_key=None, degree_threshold=None):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        attribute_key (str, optional): Node attribute key to use for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    if parallel:
        # Define a helper function for parallel processing
        def func(graph): 
            return graph_vector(graph, radius, distance, connector, nbits, attribute_key, degree_threshold)
        
        # Get the number of available CPUs
        n_cpus = mp.cpu_count()
        # Create a pool of worker processes
        pool = mp.Pool(n_cpus)
        # Encode graphs in parallel
        graph_vectors = pool.map(func, graphs)
        # Close the pool after processing
        pool.close()
    else:
        # Encode graphs sequentially
        graph_vectors = [
            graph_vector(graph, radius, distance, connector, nbits, attribute_key, degree_threshold)
            for graph in graphs
        ]
    # Stack all graph vectors vertically into a single sparse matrix
    graphs_vector_mtx = sp.sparse.vstack(graph_vectors)
    return graphs_vector_mtx

def paired_node_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, attribute_key=None, degree_threshold=None):
    """
    Encodes a list of graphs into node feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        attribute_key (str, optional): Node attribute key to use for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        list of scipy.sparse.csr_matrix: A list where each element is a matrix of node vectors for a graph.
    """
    if parallel:
        # Define a helper function for parallel processing
        def func(graph): 
            return node_vector(graph, radius, distance, connector, nbits, attribute_key, degree_threshold)
        
        # Get the number of available CPUs
        n_cpus = mp.cpu_count()
        # Create a pool of worker processes
        pool = mp.Pool(n_cpus)
        # Encode graphs in parallel
        graph_node_vectors = pool.map(func, graphs)
        # Close the pool after processing
        pool.close()
    else:
        # Encode graphs sequentially
        graph_node_vectors = [
            node_vector(graph, radius, distance, connector, nbits, attribute_key, degree_threshold)
            for graph in graphs
        ]
    return graph_node_vectors


# Import necessary modules and libraries
from sklearn.base import BaseEstimator, TransformerMixin  # Import scikit-learn base classes
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import multiprocessing_on_dill as mp  # For parallel processing with support for more object types
import networkx as nx  # For graph operations
from collections import Counter, defaultdict  # For counting elements and creating dictionaries with default types
from copy import copy  # For copying objects

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
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
    """
    def __init__(self, embedder=None, clustering_predictor=None, classifier=None, radius=1, distance=3, connector=0,
                 nbits=10, degree_threshold=None, dense=True, parallel=True,
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None):
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
        self.attribute_key = attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size

    def __repr__(self):
        """
        Returns a string representation of the AbstractNSPPK instance.
        
        Returns:
            str: The string representation.
        """
        infos = ['%s=%s' % (key, value) for key, value in self.__dict__.items()]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

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
        # Check if attribute embedding or clustering is required
        if self.attribute_key is not None and (self.attribute_dim is not None or self.attribute_alphabet_size is not None):
            # Extract node attributes from all graphs and stack them into a matrix
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.attribute_key]
                for graph in graphs
                for node_idx in graph.nodes()
            ])

        # If attribute dimensionality reduction is specified, fit the embedder
        if self.attribute_key is not None and self.attribute_dim is not None and self.embedder is not None:
            self.embedder.fit(attribute_mtx)
            attribute_mtx = self.embedder.transform(attribute_mtx)

        # If attribute clustering is specified, fit the clustering predictor and classifier
        if self.attribute_key is not None and self.attribute_alphabet_size is not None:
            # Perform clustering on the attribute matrix to generate target labels
            targets = self.clustering_predictor.fit_predict(attribute_mtx)
            # Fit the classifier to predict cluster labels from attributes
            if self.classifier is not None:
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
        if self.embedder is None:
            return graphs  # No embedding to perform

        out_graphs = []
        for graph in graphs:
            # Extract and stack node attributes into a matrix
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.attribute_key]
                for node_idx in graph.nodes()
            ])
            # Transform the attributes using the fitted embedder
            embeddings = self.embedder.transform(attribute_mtx)
            # Create a copy of the graph to avoid modifying the original
            out_graph = graph.copy()
            # Assign the original and embedded attributes to each node
            for embedding, node_idx in zip(embeddings, out_graph.nodes()):
                out_graph.nodes[node_idx]['original_' + self.attribute_key] = graph.nodes[node_idx][self.attribute_key]
                out_graph.nodes[node_idx][self.attribute_key] = embedding
            # Append the modified graph to the output list
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
        if self.classifier is None:
            return graphs  # No classifier to assign labels

        out_graphs = []
        for graph in graphs:
            # Extract and stack node attributes into a matrix
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.attribute_key]
                for node_idx in graph.nodes()
            ])
            # Predict discrete labels using the trained classifier
            labels = self.classifier.predict(attribute_mtx)
            # Create a copy of the graph to avoid modifying the original
            out_graph = graph.copy()
            # Assign the predicted label to each node
            for label, node_idx in zip(labels, out_graph.nodes()):
                out_graph.nodes[node_idx]['label'] = label
            # Append the modified graph to the output list
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
        # If attribute dimensionality reduction is specified, apply embedding
        if self.attribute_key is not None and self.attribute_dim is not None:
            graphs = self.embed_attributes(graphs)
        # If attribute clustering is specified, assign discrete labels to nodes
        if self.attribute_key is not None and self.attribute_alphabet_size is not None:
            graphs = self.set_discrete_labels(graphs)

        # Encode the graphs into a sparse feature matrix using the encoder function
        data_mtx = paired_graphs_vector_encoder(
            graphs, 
            self.radius, 
            self.distance, 
            self.connector, 
            self.nbits,
            self.parallel, 
            self.attribute_key,
            self.degree_threshold
        )

        # Convert the feature matrix to dense format if specified
        if self.dense:
            data_mtx = data_mtx.todense().A

        return data_mtx


class NSPPK(AbstractNSPPK):
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
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
    """
    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, 
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None):
        """
        Initializes the NSPPK encoder with specific components: TruncatedSVD, KMeans, and ExtraTreesClassifier.
        
        Args:
            radius (int, optional): The radius for rooted graph hashing. Defaults to 1.
            distance (int, optional): The distance parameter for paired hashing. Defaults to 3.
            connector (int, optional): Connector thickness. Defaults to 0.
            nbits (int, optional): Number of bits for hashing. Defaults to 10.
            degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
            dense (bool, optional): Whether to convert the feature matrix to a dense format. Defaults to True.
            parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
            attribute_key (str, optional): Node attribute key to use for additional features.
            attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension. Defaults to None.
            attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes. Defaults to None.
        """
        # Initialize the embedder if attribute_dim is specified
        embedder = TruncatedSVD(n_components=attribute_dim) if attribute_dim else None

        # Initialize the clustering predictor if attribute_alphabet_size is specified
        clustering_predictor = KMeans(n_clusters=attribute_alphabet_size) if attribute_alphabet_size else None

        # Initialize the classifier if attribute_alphabet_size is specified
        classifier = ExtraTreesClassifier(
            n_estimators=300, 
            n_jobs=-1 if parallel else None
        ) if attribute_alphabet_size else None

        # Initialize the superclass (AbstractNSPPK) with the specific components and parameters
        super().__init__(
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
            attribute_key=attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size
        )


    # The fit_transform method is inherited from TransformerMixin,
    # which uses the fit and transform methods defined above.


class NodeNSPPK(NSPPK):
    """
    NodeNSPPK (Node Neighborhood Subgraph Pairwise Propagation Kernel) class compatible with scikit-learn's Transformer interface.

    Inherits from NSPPK and specializes the transformation process to generate node-level feature vectors suitable for
    node-level machine learning models. It captures both local (node-level) and structural (subgraph-level) information
    by hashing node labels and the structure of their neighborhoods. Additionally, it supports dimensionality reduction
    of node attributes using Singular Value Decomposition (SVD).

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): The target dimensionality for attribute vectors; if set, applies SVD for dimensionality reduction.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
    """
    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, 
                 attribute_key=None, attribute_dim=None, attribute_alphabet_size=None):
        """
        Initialize the NodeNSPPK encoder with the specified parameters.

        Args:
            radius (int, optional): The radius for rooted graph hashing. Defaults to 1.
            distance (int, optional): The distance parameter for paired hashing. Defaults to 3.
            connector (int, optional): Connector thickness. Defaults to 0.
            nbits (int, optional): Number of bits for hashing. Defaults to 10.
            dense (bool, optional): Whether to convert the feature matrix to a dense format. Defaults to True.
            parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
            attribute_key (str, optional): Node attribute key to use for additional features.
            attribute_dim (int, optional): The target dimensionality for attribute vectors; if set, applies SVD for dimensionality reduction. Defaults to None.
            attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes. Defaults to None.
        """
        # Initialize the superclass (NSPPK) with the provided parameters
        super().__init__(
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            attribute_key=attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size
        )

    def transform(self, graphs):
        """
        Transform input graphs into node-level feature vectors.

        This method encodes each graph into a list of node feature matrices, where each matrix corresponds to a graph
        and contains feature vectors for its nodes. If `attribute_dim` is specified, it applies SVD to reduce the
        dimensionality of node attributes before encoding. Additionally, if `attribute_alphabet_size` is specified,
        it assigns discrete labels to nodes based on clustered attributes.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            list of numpy.ndarray or scipy.sparse.csr_matrix: A list where each element is a node feature matrix for a graph.
        """
        # If attribute dimensionality reduction is specified, apply embedding
        if self.attribute_key and self.attribute_dim:
            graphs = self.embed_attributes(graphs)
        # If attribute clustering is specified, assign discrete labels to nodes
        if self.attribute_key is not None and self.attribute_alphabet_size is not None:
            graphs = self.set_discrete_labels(graphs)

        # Encode the graphs into node feature vectors using the node-specific encoder function
        nodes_data_mtx_list = paired_node_vector_encoder(
            graphs,
            self.radius,
            self.distance,
            self.connector,
            self.nbits,
            self.parallel,
            self.attribute_key,
            self.degree_threshold
        )

        # Convert each node feature matrix to dense format if specified
        if self.dense:
            nodes_data_mtx_list = [mtx.todense().A for mtx in nodes_data_mtx_list]

        return nodes_data_mtx_list

    def __repr__(self):
        """
        Generate a string representation of the NodeNSPPK instance.

        Returns:
            str: The string representation of the object.
        """
        # Reuse the NSPPK __repr__ method for consistency
        return super().__repr__()

    # The fit_transform method is inherited from TransformerMixin,
    # which uses the fit and transform methods defined above.

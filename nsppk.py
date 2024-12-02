# Import necessary modules and libraries
from collections import Counter, defaultdict  # For counting elements and creating dictionaries with default types
from copy import copy                         # For copying objects
from scipy.sparse import csr_matrix           # For handling sparse matrices and vertical stacking
import multiprocessing_on_dill as mp          # For parallel processing with support for more object types
import networkx as nx                         # For graph operations
import numpy as np                            # For numerical operations
import scipy as sp                            # For scientific computing
from sklearn.base import BaseEstimator, TransformerMixin  # Import scikit-learn base classes

def circular_convolution(signal, kernel):
    """
    Performs circular convolution between two signals using FFT.

    If either of the inputs is a csr_matrix (sparse matrix),
    it converts them to dense arrays before performing the convolution,
    and converts the result back to csr_matrix if needed.

    Args:
        signal (array-like or csr_matrix): The first input signal.
        kernel (array-like or csr_matrix): The second input signal.

    Returns:
        array-like or csr_matrix: The convolution result, in csr_matrix if any input was sparse.
    """
    # Check if either signal or kernel is a csr_matrix (sparse matrix)
    signal_is_sparse = isinstance(signal, csr_matrix)
    kernel_is_sparse = isinstance(kernel, csr_matrix)
    
    # Convert signals to dense arrays and flatten them
    if signal_is_sparse:
        signal = signal.toarray().flatten()
    else:
        signal = np.asarray(signal).flatten()
        
    if kernel_is_sparse:
        kernel = kernel.toarray().flatten()
    else:
        kernel = np.asarray(kernel).flatten()
        
    # Determine the dimension for zero-padding (length of the longest signal)
    dim = max(signal.shape[0], kernel.shape[0])
    
    # Zero-pad the signal and kernel to make their lengths equal
    padded_signal = np.zeros(dim)
    padded_signal[:signal.shape[0]] = signal
    padded_kernel = np.zeros(dim)
    padded_kernel[:kernel.shape[0]] = kernel
    
    # Perform the circular convolution using FFT (Fast Fourier Transform)
    conv = np.real(np.fft.ifft(np.fft.fft(padded_signal) * np.fft.fft(padded_kernel)))
    
    # Convert the result back to csr_matrix if any input was sparse
    if signal_is_sparse or kernel_is_sparse:
        conv = csr_matrix(conv)
    
    return conv

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
    Hashes a value and limits it to a specified number of bits.

    Args:
        value: The value to hash.
        nbits (int, optional): Number of bits to limit the hash. Defaults to 10.

    Returns:
        int: The hashed value limited to nbits.
    """
    # Create a bitmask with nbits set to 1 (e.g., if nbits=10, bitmask=1023)
    bitmask = pow(2, nbits) - 1
    # Apply the bitmask to the hashed value
    return masked_hash_value(value, bitmask)

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
    
    # Hash the sorted list of neighbor-edge hashes
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
        # Hash the sorted list of node hashes
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
    # Count occurrences of each item
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

def node_vectors(original_graph, radius, distance, connector, nbits):
    """
    Generates a feature vector for each node in the graph.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix of node feature vectors.
    """
    # Create a copy to avoid modifying the original graph
    graph = original_graph.copy()
    
    # Step 1: Compute and store node hashes for all nodes in the graph
    for node_idx in graph.nodes():
        # Compute node hash based on its label and the labels of its neighbors
        graph.nodes[node_idx]['node_hash'] = node_hash(node_idx, graph)
    
    # Step 2: Compute and store rooted graph hashes for each node up to the specified radius
    # The 'cutoff' is the maximum between the specified 'radius' and 'connector' values
    cutoff = max(radius, connector)
    for node_idx in graph.nodes():
        # Initialize the array to store hashes up to the maximum radius = cutoff
        graph.nodes[node_idx]['rooted_graph_hash'] = np.zeros(cutoff, dtype=int)
        # For each radius from 0 up to cutoff
        for r in range(cutoff):
            # Compute rooted graph hash at radius r
            label = rooted_graph_hash(node_idx, graph, radius=r)
            # Store the hash at position r
            graph.nodes[node_idx]['rooted_graph_hash'][r] = label

    node_vectors = []  # List to store feature vectors for each node
    
    # Step 3: Iterate over all nodes to generate paired codes based on distances and hashes
    for node_idx in graph.nodes():
        node_codes_list = []  # List to store codes for the current node
        # For each rooted graph hash code_i of the node
        for code_i in graph.nodes[node_idx]['rooted_graph_hash']:
            # Step 3a: Find all nodes within the specified distance from the current node
            node_idxs_to_dist_dict = nx.single_source_shortest_path_length(
                graph, node_idx, cutoff=distance
            )
            # Invert the dictionary to map distances to node indices
            dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)

            # Step 3b: Iterate over each connector thickness level
            for connect in range(connector + 1):
                # Step 3c: Iterate over each distance and corresponding nodes
                for dist in sorted(dist_to_node_idxs_dict):
                    node_idxs = dist_to_node_idxs_dict[dist]
                    for curr_node_idx in node_idxs:
                        if connect > 0:
                            # If connector thickness is greater than 0
                            # Initialize the set to collect nodes on the union of shortest paths
                            union_of_shortest_paths = set()
                            # Compute all shortest paths between node_idx and curr_node_idx
                            shortest_paths = list(nx.all_shortest_paths(graph, source=node_idx, target=curr_node_idx))
                            for path in shortest_paths:
                                # Add all nodes in the path to the set
                                union_of_shortest_paths.update(path)

                        # For each rooted graph hash code_j of the current neighbor node
                        for code_j in graph.nodes[curr_node_idx]['rooted_graph_hash']:
                            
                            if connect == 0:
                                # When connector thickness is 0, compute paired_code without union_of_shortest_paths_code
                                paired_code = hash_list([code_i, dist, code_j])
                            else:
                                # When connector thickness is greater than 0, compute union_of_shortest_paths_code
                                # For each node in the union of shortest paths, extract its rooted_graph_hash at level (connect - 1)
                                union_of_shortest_paths_node_hashes = [
                                    # Create a hash of (distance, rooted_graph_hash)
                                    hash_list([node_idxs_to_dist_dict[node], graph.nodes[node]['rooted_graph_hash'][connect - 1]])
                                    for node in union_of_shortest_paths
                                ]

                                # Sort the list to ensure consistent ordering
                                sorted_list = sorted(union_of_shortest_paths_node_hashes)
                                # Hash the sorted list to obtain union_of_shortest_paths_code
                                union_of_shortest_paths_code = hash_list(sorted_list)
                                # Compute paired_code with union_of_shortest_paths_code
                                paired_code = hash_list([code_i, dist, code_j, union_of_shortest_paths_code])

                            # Limit the hash to the specified number of bits
                            paired_code = hash_value(paired_code, nbits=nbits)
                            node_codes_list.append(paired_code)

        # Step 4: Convert the list of paired codes into a sparse histogram vector
        node_vector = items_to_sparse_histogram(node_codes_list, nbits)
        node_vectors.append(node_vector)
    
    # Stack node vectors vertically to create a matrix
    node_vectors_mtx = sp.sparse.vstack(node_vectors)

    return node_vectors_mtx

def graph_vector(original_graph, radius, distance, connector, nbits, attribute_key=None):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        attribute_key (str, optional): Node attribute key to use for additional features.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    # Generate node vectors (feature vectors for each node)
    node_vectors_mtx = node_vectors(
        original_graph, radius, distance, connector, nbits
    )
    if attribute_key is None:
        # If no attribute key is specified or convolution is used, sum the node vectors to get the graph vector
        vector = node_vectors_mtx.sum(axis=0)
        vector = csr_matrix(vector)  # Ensure the result is a CSR matrix
    else:
        # If an attribute key is specified and convolution is not used, use the node attributes to weight the node vectors
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

def paired_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, attribute_key=None):
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

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    if parallel:
        # Define a helper function for parallel processing
        def func(graph): 
            return graph_vector(graph, radius, distance, connector, nbits, attribute_key)
        
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
            graph_vector(graph, radius, distance, connector, nbits, attribute_key)
            for graph in graphs
        ]
    # Stack all graph vectors vertically into a single sparse matrix
    graphs_vector_mtx = sp.sparse.vstack(graph_vectors)
    return graphs_vector_mtx

def paired_node_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, attribute_key=None):
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

    Returns:
        list of scipy.sparse.csr_matrix: A list where each element is a matrix of node vectors for a graph.
    """
    if parallel:
        # Define a helper function for parallel processing
        def func(graph): 
            return node_vectors(graph, radius, distance, connector, nbits)
        
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
            node_vectors(graph, radius, distance, connector, nbits)
            for graph in graphs
        ]
    return graph_node_vectors

class NSPPK(BaseEstimator, TransformerMixin):
    """
    NSPPK (Neighborhood Subgraph Pairwise Propagation Kernel) class compatible with scikit-learn's Transformer interface.

    This class encodes graphs into feature vectors suitable for machine learning models. It captures both
    local (node-level) and structural (subgraph-level) information by hashing node labels and the structure
    of their neighborhoods.

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        attribute_key (str, optional): Node attribute key to use for additional features.
    """
    def __init__(self, radius=1, distance=3, connector=0, nbits=10, dense=True, parallel=True, attribute_key=None):
        """
        Initializes the NSPPK encoder with the given parameters.

        Args:
            radius (int, optional): The radius for rooted graph hashing. Defaults to 1.
            distance (int, optional): The distance parameter for paired hashing. Defaults to 3.
            connector (int, optional): Connector thickness. Defaults to 0.
            nbits (int, optional): Number of bits for hashing. Defaults to 10.
            dense (bool, optional): Whether to convert the feature matrix to a dense format. Defaults to True.
            parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
            attribute_key (str, optional): Node attribute key to use for additional features.
        """
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.dense = dense
        self.parallel = parallel
        self.attribute_key = attribute_key

    def __repr__(self):
        """
        Returns a string representation of the NSPPK instance.

        Returns:
            str: The string representation.
        """
        infos = ['%s=%s' % (key, value) for key, value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)' % (self.__class__.__name__, infos)
    
    def fit(self, X, y=None):
        """
        Fit method for compatibility with scikit-learn's Transformer interface.
        This transformer does not learn any parameters from the data, so it simply returns itself.

        Args:
            X (list of networkx.Graph): The input graphs to fit on.
            y (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        """
        Transforms the input graphs into feature vectors.

        Args:
            X (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        # Encode the graphs into a sparse feature matrix using the encoder function
        data_mtx = paired_graphs_vector_encoder(
            X, self.radius, self.distance, self.connector, self.nbits,
            self.parallel, self.attribute_key
        )

        if self.dense:
            # Convert to dense format if specified
            data_mtx = data_mtx.todense().A
        return data_mtx
    
    # The fit_transform method is inherited from TransformerMixin,
    # which uses the fit and transform methods defined above.

class NodeNSPPK(BaseEstimator, TransformerMixin):
    """
    NodeNSPPK (Node Neighborhood Subgraph Pairwise Propagation Kernel) class compatible with scikit-learn's Transformer interface.

    This class encodes graphs into node feature vectors suitable for node-level machine learning models.
    It captures both local (node-level) and structural (subgraph-level) information by hashing node labels
    and the structure of their neighborhoods.

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        attribute_key (str, optional): Node attribute key to use for additional features.
    """
    def __init__(self, radius=1, distance=3, connector=0, nbits=10, dense=True, parallel=True, attribute_key=None):
        """
        Initializes the NodeNSPPK encoder with the given parameters.

        Args:
            radius (int, optional): The radius for rooted graph hashing. Defaults to 1.
            distance (int, optional): The distance parameter for paired hashing. Defaults to 3.
            connector (int, optional): Connector thickness. Defaults to 0.
            nbits (int, optional): Number of bits for hashing. Defaults to 10.
            dense (bool, optional): Whether to convert the feature matrix to a dense format. Defaults to True.
            parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
            attribute_key (str, optional): Node attribute key to use for additional features.
        """
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.dense = dense
        self.parallel = parallel
        self.attribute_key = attribute_key

    def __repr__(self):
        """
        Returns a string representation of the NodeNSPPK instance.

        Returns:
            str: The string representation.
        """
        infos = ['%s=%s' % (key, value) for key, value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)' % (self.__class__.__name__, infos)
    
    def fit(self, X, y=None):
        """
        Fit method for compatibility with scikit-learn's Transformer interface.
        This transformer does not learn any parameters from the data, so it simply returns itself.

        Args:
            X (list of networkx.Graph): The input graphs to fit on.
            y (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        """
        Transforms the input graphs into node feature vectors.

        Args:
            X (list of networkx.Graph): The list of graphs to transform.

        Returns:
            list of numpy.ndarray or scipy.sparse.csr_matrix: A list of node feature matrices for each graph.
        """
        # Encode the graphs into node feature vectors using the encoder function
        nodes_data_mtx_list = paired_node_vector_encoder(
            X, self.radius, self.distance, self.connector, self.nbits,
            self.parallel, self.attribute_key
        )

        if self.dense:
            # Convert to dense format if specified
            nodes_data_mtx_list = [mtx.todense().A for mtx in nodes_data_mtx_list]
        return nodes_data_mtx_list

    # The fit_transform method is inherited from TransformerMixin,
    # which uses the fit and transform methods defined above.

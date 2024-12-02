# NSPPK
Neighbourhood Subgraph Pairwise Pathway Kernel 


The **Neighborhood Subgraph Pairwise Pathway Kernel (NSPPK)** is a graph kernel designed to encode graphs into feature vectors suitable for machine learning tasks, such as classification or regression. NSPPK captures both local (node-level) and global (subgraph-level) structural information of graphs by combining node labels and the structure of their neighborhoods through hashing techniques.

**Key Concepts of NSPPK:**

1. **Node Label Hashing**: Each node is assigned a hash value based on its own label and the labels of its immediate neighbors. This initial hashing step captures the immediate neighborhood information around each node.

2. **Rooted Graph Hashing**: For each node, NSPPK computes a hash that represents the subgraph rooted at that node up to a certain radius. This involves considering the node's neighborhood structure within the specified radius, effectively capturing more extensive local structural information.

3. **Paired Hashing of Nodes**: NSPPK considers pairs of nodes within a certain distance and computes combined hashes that include:
   - The rooted graph hashes of both nodes.
   - The shortest path distance between the nodes.
   - Optionally, the union of shortest paths (if the connector thickness is greater than 0).
   
   This step captures the relational information between pairs of nodes, integrating both their individual structures and their connectivity in the graph.

4. **Feature Vector Construction**: The hashes generated from the above steps are used to create a sparse histogram (feature vector) for the graph. Each unique hash corresponds to a feature, and its frequency (count) represents how often that particular substructure occurs in the graph.

5. **Graph Encoding**: The feature vectors of all nodes are aggregated (typically summed) to form a single feature vector representing the entire graph. This vector can then be used as input to machine learning models.

**Detailed Explanation of NSPPK Components:**

1. **Node Hashing (`node_hash` function)**:
   - **Purpose**: Compute a unique hash for each node based on its label and the labels of its immediate neighbors.
   - **Process**:
     - Hash the label of the node itself.
     - Iterate over all neighboring nodes:
       - Hash the neighbor's label.
       - Hash the label of the edge connecting the node and the neighbor.
       - Combine these hashes into a tuple and hash it.
     - Collect all neighbor-edge hashes into a list, sort it, and hash the list.
     - Combine the node's own label hash and the hashed list of neighbor-edge hashes to get the final node hash.

2. **Rooted Graph Hashing (`rooted_graph_hash` function)**:
   - **Purpose**: Compute a hash that represents the subgraph rooted at a node up to a certain radius.
   - **Process**:
     - Use breadth-first search to find all nodes within the specified radius.
     - For each distance from the root node (0 to radius):
       - Collect the node hashes of all nodes at that distance.
       - Sort and hash this list to get a code representing that level.
     - Collect all level codes into a list, and hash the list to get the rooted graph hash for the root node.

3. **Paired Hashing (`node_vectors` function)**:
   - **Purpose**: Generate feature vectors for nodes by considering pairs of nodes within a certain distance.
   - **Parameters**:
     - **Distance**: Maximum distance between node pairs to consider.
     - **Connector Thickness**: If greater than 0, includes the union of shortest paths between node pairs in the hash.
   - **Process**:
     - For each node, iterate over its rooted graph hashes at different radii.
     - For each node within the specified distance:
       - If connector thickness is greater than 0:
         - Compute the union of shortest paths between the two nodes.
         - Hash the rooted graph hashes of nodes on these paths.
         - Include this in the final paired hash.
       - Compute a paired hash that includes:
         - The rooted graph hash of the source node.
         - The distance between the nodes.
         - The rooted graph hash of the target node.
         - (Optional) The hash of the union of shortest paths.
     - Limit the hash to a specified number of bits to control the feature space size.
     - Collect all paired hashes for a node to form its feature vector.

4. **Feature Vector Construction (`items_to_sparse_histogram` function)**:
   - **Purpose**: Convert the list of paired hashes for each node into a sparse histogram.
   - **Process**:
     - Count the occurrences of each unique hash (feature).
     - Create a sparse vector where indices correspond to hashed feature values, and values correspond to feature counts.

5. **Graph Encoding (`graph_vector` function)**:
   - **Purpose**: Aggregate node feature vectors to represent the entire graph.
   - **Process**:
     - Sum the node feature vectors to obtain a single feature vector for the graph.
     - Optionally, use node attributes to weight the node vectors before summing.

**Implementation in NSPPK Class:**

- **NSPPK Class**:
  - Inherits from scikit-learn's `BaseEstimator` and `TransformerMixin` classes to integrate seamlessly with scikit-learn pipelines.
  - **Parameters**:
    - `radius`: Radius for rooted graph hashing.
    - `distance`: Maximum distance for paired hashing.
    - `connector`: Connector thickness for including union of shortest paths.
    - `nbits`: Number of bits for hashing (controls feature space size).
    - `dense`: Whether to output dense or sparse feature vectors.
    - `parallel`: Whether to use parallel processing for encoding graphs.
    - `attribute_key`: Node attribute key for additional features (optional).
  - **Methods**:
    - `fit`: Does nothing as NSPPK doesn't learn from data; included for compatibility.
    - `transform`: Transforms input graphs into feature vectors using the NSPPK method.

**Applications of NSPPK:**

- **Graph Classification**: NSPPK can be used to generate feature vectors for graphs, enabling classification models to distinguish between different types of graphs based on their structural properties.
- **Node Classification**: With the `NodeNSPPK` class, NSPPK can generate feature vectors for individual nodes, facilitating tasks like node classification within a graph.
- **Graph Similarity**: By comparing the feature vectors generated by NSPPK, one can measure the similarity between graphs or subgraphs.

**Advantages of NSPPK:**

- **Captures Rich Structural Information**: By considering both node labels and their neighborhoods, NSPPK effectively captures the local and global structure of graphs.
- **Flexible Parameters**: The radius, distance, connector thickness, and number of bits can be tuned to capture different levels of detail and control the feature space size.
- **Scalable with Parallel Processing**: NSPPK can process multiple graphs in parallel, making it suitable for large datasets.

**Example Usage:**

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Instantiate the NSPPK transformer
nsppk_transformer = NSPPK(radius=2, distance=3, connector=1, nbits=12, dense=True, parallel=True)

# Create a pipeline with NSPPK and a classifier
pipeline = Pipeline([
    ('nsppk', nsppk_transformer),
    ('classifier', SVC())
])

# Fit the pipeline on training data (graphs and labels)
pipeline.fit(training_graphs, training_labels)

# Predict on new data
predictions = pipeline.predict(test_graphs)
```

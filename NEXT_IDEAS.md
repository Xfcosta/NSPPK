# Next Ideas

For directed graph support, use the underlying undirected graph to compute distances, neighbourhoods, and unions of shortest paths, while preserving each edge’s original direction when constructing hashes. Encode direction explicitly in the relevant local and edge patterns, so graphs with the same undirected topology but different arc orientations can produce different features.

For each neighbourhood or union of shortest paths, treat the induced subgraph as a source of simplex fragments of every order: individual nodes (0), edges (1), triangles (2), tetrahedra (3), and higher-order simplices. A feature would combine one fragment from the neighbourhood of each endpoint with the distance between the roots, and include the unordered pair of root-node labels to contextualise the fragments; for directed graphs, retain the orientation of the fragment’s edges in its hash while determining simplex membership on the underlying undirected graph.

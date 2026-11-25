# Next Ideas

For directed graph support, use the underlying undirected graph to compute distances, neighbourhoods, and unions of shortest paths, while preserving each edge’s original direction when constructing hashes. Encode direction explicitly in the relevant local and edge patterns, so graphs with the same undirected topology but different arc orientations can produce different features.

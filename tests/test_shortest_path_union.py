import unittest

import networkx as nx

import nsppk


class ShortestPathUnionTests(unittest.TestCase):
    def test_shortest_path_union_matches_all_shortest_paths(self):
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (0, 1),
                (1, 4),
                (0, 2),
                (2, 4),
                (0, 3),
                (3, 4),
                (4, 5),
            ]
        )

        graph.graph["_bfs_cache"] = nsppk._build_bfs_cache(graph, max_radius=4)

        nodes, edges = nsppk._shortest_path_edge_union(graph, 0, 4)

        self.assertEqual(nodes, {0, 1, 2, 3, 4})
        self.assertEqual(edges, {(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)})

    def test_shortest_path_union_cache_is_reused(self):
        graph = nx.path_graph(4)
        graph.graph["_bfs_cache"] = nsppk._build_bfs_cache(graph, max_radius=3)

        first_nodes, first_edges = nsppk._shortest_path_edge_union(graph, 0, 3)
        second_nodes, second_edges = nsppk._shortest_path_edge_union(graph, 0, 3)

        self.assertIs(first_nodes, second_nodes)
        self.assertIs(first_edges, second_edges)

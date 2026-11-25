import unittest

import networkx as nx
import numpy as np

import nsppk
from nsppk import EdgeNSPPK, NSPPK, get_feature_archive, get_structural_edge_vectors, get_structural_node_vectors


def _directed_pair(reverse=False):
    graph = nx.DiGraph()
    graph.add_node("source", label="A")
    graph.add_node("target", label="B")
    edge = ("target", "source") if reverse else ("source", "target")
    graph.add_edge(*edge, label="x")
    return graph


class DirectedMultigraphTests(unittest.TestCase):
    def test_directed_reversal_changes_node_and_edge_features(self):
        forward = _directed_pair()
        reverse = _directed_pair(reverse=True)
        forward_nodes = get_structural_node_vectors(forward, 1, 1, 0, 8).toarray()
        reverse_nodes = get_structural_node_vectors(reverse, 1, 1, 0, 8).toarray()
        self.assertFalse(np.array_equal(forward_nodes, reverse_nodes))

        forward_edges = get_structural_edge_vectors(forward, 1, 1, 0, 8).toarray()
        reverse_edges = get_structural_edge_vectors(reverse, 1, 1, 0, 8).toarray()
        self.assertFalse(np.array_equal(forward_edges, reverse_edges))

        forward_graph = NSPPK(nbits=8, dense=False, parallel=False).transform([forward])
        reverse_graph = NSPPK(nbits=8, dense=False, parallel=False).transform([reverse])
        self.assertFalse((forward_graph != reverse_graph).nnz == 0)

    def test_directed_distances_and_shortest_path_edges_use_undirected_projection(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (2, 1)])
        prepared = nsppk._prepare_structural_graph(graph, 1, 2, 1)
        cache = prepared.graph["_bfs_cache"]
        self.assertEqual(cache[0]["node_idxs_to_dist_dict"][2], 2)

        nodes, edges = nsppk._shortest_path_edge_union(prepared, 0, 2)
        self.assertEqual(nodes, {0, 1, 2})
        self.assertEqual(edges, {(0, 1), (2, 1)})

    def test_directed_connector_signature_retains_path_arc_direction(self):
        signatures = []
        for graph in (_directed_pair(), _directed_pair(reverse=True)):
            prepared = nsppk._prepare_structural_graph(graph, 0, 1, 1)
            distances = {"source": 0, "target": 1}
            path_nodes, path_edges = nsppk._shortest_path_edge_union(
                prepared, "source", "target"
            )
            signatures.append(nsppk._connector_provenance(
                prepared,
                distances,
                path_nodes,
                path_edges,
                1,
                include_occurrence_provenance=False,
            )[2])
        self.assertNotEqual(signatures[0], signatures[1])

    def test_multigraph_parallel_edges_and_loops_are_preserved(self):
        graph = nx.MultiGraph()
        graph.add_node(0, label="a")
        graph.add_node(1, label="b")
        graph.add_edge(0, 1, key="first", label="x")
        graph.add_edge(0, 1, key="second", label="y")
        graph.add_edge(1, 1, key="loop", label="loop")

        edge_rows = get_structural_edge_vectors(graph, 1, 1, 0, 8)
        self.assertEqual(edge_rows.shape[0], 3)
        archive = get_feature_archive(graph, 0, 1, 1, 0, 8)
        occurrences = [occ for values in archive.values() for occ in values]
        archived_edges = {
            edge_id
            for occurrence in occurrences
            for edge_id in occurrence["source_edge_ids"]
        }
        self.assertIn((0, 1, "first"), archived_edges)
        self.assertIn((0, 1, "second"), archived_edges)
        self.assertIn((1, 1, "loop"), archived_edges)
        archived_graphs = [occ["subgraph"] for occ in occurrences]
        self.assertTrue(any(g.is_multigraph() and g.number_of_edges() for g in archived_graphs))

    def test_directed_multigraph_edge_attribute_rows_follow_keyed_edges(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, key="out", label="a", feature=np.array([1.0]))
        graph.add_edge(1, 0, key="back", label="b", feature=np.array([2.0]))
        graph.add_edge(0, 1, key="parallel", label="c", feature=np.array([3.0]))

        encoder = EdgeNSPPK(
            nbits=5,
            dense=False,
            parallel=False,
            edge_attribute_key="feature",
        )
        rows = encoder.transform([graph])[0]
        self.assertEqual(rows.shape[0], 3)
        np.testing.assert_array_equal(rows[:, 0].toarray().ravel(), [1.0, 3.0, 2.0])


if __name__ == "__main__":
    unittest.main()

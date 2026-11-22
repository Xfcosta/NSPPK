import unittest

import networkx as nx
import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone

from nsppk import (
    EdgeNSPPK,
    NSPPK,
    get_edge_vectors,
    get_structural_edge_vectors,
    get_structural_node_vectors,
)


def _graph_with_edge_attributes():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "atom", "label")
    nx.set_edge_attributes(graph, "bond", "label")
    for edge_index, edge in enumerate(graph.edges()):
        graph.edges[edge]["feature"] = np.asarray([edge_index + 1.0, 2.0])
    for node in graph:
        graph.nodes[node]["feature"] = np.asarray([node + 1.0])
    return graph


class EdgeAttributeTests(unittest.TestCase):
    def test_edge_structural_rows_sum_endpoint_node_rows_in_edge_order(self):
        graph = _graph_with_edge_attributes()
        node_rows = get_structural_node_vectors(graph, 1, 2, 0, 5).toarray()
        edge_rows = get_structural_edge_vectors(graph, 1, 2, 0, 5).toarray()
        node_order = {node: idx for idx, node in enumerate(graph.nodes())}

        self.assertEqual(edge_rows.shape, (graph.number_of_edges(), 2**5))
        for edge_idx, (u, v) in enumerate(graph.edges()):
            np.testing.assert_array_equal(
                edge_rows[edge_idx], node_rows[node_order[u]] + node_rows[node_order[v]]
            )

    def test_edge_encoder_adds_attribute_and_interaction_columns(self):
        graph = _graph_with_edge_attributes()
        structural = get_structural_edge_vectors(graph, 1, 2, 0, 5).toarray()
        vectors = get_edge_vectors(
            graph, 1, 2, 0, 5, edge_attribute_key="feature"
        ).toarray()
        attributes = np.vstack([graph.edges[e]["feature"] for e in graph.edges()])

        self.assertEqual(vectors.shape, (2, 2**5 + 2 * attributes.shape[1]))
        np.testing.assert_array_equal(vectors[:, :2], attributes)
        np.testing.assert_array_equal(vectors[:, -2**5:], structural)
        self.assertTrue(np.isfinite(vectors).all())

    def test_graph_encoder_concatenates_node_and_edge_blocks(self):
        graph = _graph_with_edge_attributes()
        encoder = NSPPK(
            nbits=5,
            dense=False,
            parallel=False,
            node_attribute_key="feature",
            edge_attribute_key="feature",
        )
        vector = encoder.transform([graph])
        self.assertTrue(issparse(vector))
        self.assertEqual(vector.shape, (1, (1 + 2) * 2**5))

        edge_only = NSPPK(
            nbits=5, dense=False, parallel=False, edge_attribute_key="feature"
        ).transform([graph])
        self.assertEqual(edge_only.shape, (1, 3 * 2**5))

        node_kernel = NSPPK(
            nbits=5,
            dense=False,
            parallel=False,
            edge_attribute_key="feature",
            use_node_kernel=True,
        ).transform([graph])
        self.assertEqual(node_kernel.shape, (1, 2**10 + 2 * 2**5))

    def test_empty_edge_graph_has_fitted_zero_edge_block(self):
        graph = _graph_with_edge_attributes()
        empty_graph = nx.empty_graph(2)
        nx.set_node_attributes(empty_graph, "atom", "label")
        encoder = NSPPK(
            nbits=4, dense=False, parallel=False, edge_attribute_key="feature"
        ).fit([graph])

        vectors = encoder.transform([empty_graph, graph])
        self.assertEqual(vectors.shape, (2, 2**4 + 2 * 2**4))
        np.testing.assert_array_equal(vectors[0, 2**4:].toarray(), np.zeros((1, 2 * 2**4)))

    def test_edge_encoder_empty_graph_and_attribute_preprocessing(self):
        empty_graph = nx.empty_graph(2)
        encoder = EdgeNSPPK(nbits=4, dense=False, parallel=False)
        empty_rows = encoder.transform([empty_graph])[0]
        self.assertEqual(empty_rows.shape, (0, 2**4))

        graphs = [_graph_with_edge_attributes(), _graph_with_edge_attributes()]
        for index, graph in enumerate(graphs):
            for edge_idx, edge in enumerate(graph.edges()):
                graph.edges[edge]["feature"] = np.asarray([index + edge_idx, edge_idx + 2.0])
        clustered = EdgeNSPPK(
            nbits=4,
            dense=False,
            parallel=False,
            edge_attribute_key="feature",
            edge_attribute_alphabet_size=2,
        )
        clustered.fit(graphs)
        rows = clustered.transform(graphs)
        self.assertEqual(rows[0].shape, (2, 2**4 + 2 * 2))
        self.assertTrue(all(data["label"] == "bond" for _, _, data in graphs[0].edges(data=True)))
        prepared = clustered.nsppk.base_nsppk.set_discrete_edge_labels(graphs)
        self.assertTrue(all(data["label"] != "bond" for _, _, data in prepared[0].edges(data=True)))
        self.assertEqual(clone(EdgeNSPPK(edge_attribute_key="feature")).edge_attribute_key, "feature")
        cloned_nsppk = clone(NSPPK(
            edge_attribute_key="feature", edge_attribute_dim=1, parallel=False
        ))
        self.assertEqual(cloned_nsppk.edge_attribute_dim, 1)

        embedded = EdgeNSPPK(
            nbits=4,
            dense=False,
            parallel=False,
            edge_attribute_key="feature",
            edge_attribute_dim=1,
        ).fit(graphs)
        self.assertEqual(embedded.transform(graphs)[0].shape, (2, 2**4 + 2))


if __name__ == "__main__":
    unittest.main()

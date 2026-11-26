import unittest

import networkx as nx
import numpy as np
from sklearn.base import clone

from importance_nsppk import ImportanceNSPPK
from nsppk import EdgeNSPPK, NSPPK, NodeNSPPK, get_node_vectors, get_structural_node_vectors


def _labeled_path():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "atom", "label")
    nx.set_edge_attributes(graph, "bond", "label")
    for node in graph:
        graph.nodes[node]["feature"] = np.asarray([float(node), float(node + 1)])
    return graph


class EncoderApiTests(unittest.TestCase):
    def test_parameter_sets_union_graph_node_and_edge_features(self):
        graph = _labeled_path()
        parameter_sets = [
            {"radius": 0, "distance": 4, "connector": 0, "use_simplex_features": False},
            {"radius": 3, "distance": 0, "connector": 0, "use_simplex_features": True},
        ]

        union_graph = NSPPK(nbits=6, dense=False, parallel=False, parameters=parameter_sets)
        union_graph.fit([graph])
        actual_graph = union_graph.transform([graph]).toarray()[0]
        expected_graph = np.zeros_like(actual_graph)
        for config in parameter_sets:
            single = NSPPK(nbits=6, dense=False, parallel=False, **config)
            single.fit([graph])
            expected_graph += single.transform([graph]).toarray()[0]
        np.testing.assert_array_equal(actual_graph[2:], expected_graph[2:])

        union_node = NodeNSPPK(nbits=6, dense=False, parallel=False, parameters=parameter_sets)
        union_node.fit([graph])
        actual_nodes = union_node.transform([graph])[0].toarray()
        expected_nodes = np.zeros_like(actual_nodes)
        for config in parameter_sets:
            single = NodeNSPPK(nbits=6, dense=False, parallel=False, **config)
            single.fit([graph])
            expected_nodes += single.transform([graph])[0].toarray()
        np.testing.assert_array_equal(actual_nodes[:, 2:], expected_nodes[:, 2:])

        union_edge = EdgeNSPPK(nbits=6, dense=False, parallel=False, parameters=parameter_sets)
        union_edge.fit([graph])
        actual_edges = union_edge.transform([graph])[0].toarray()
        expected_edges = np.zeros_like(actual_edges)
        for config in parameter_sets:
            single = EdgeNSPPK(nbits=6, dense=False, parallel=False, **config)
            single.fit([graph])
            expected_edges += single.transform([graph])[0].toarray()
        np.testing.assert_array_equal(actual_edges[:, 2:], expected_edges[:, 2:])
        np.testing.assert_array_equal(actual_graph[:2], expected_graph[:2] / len(parameter_sets))

    def test_parameter_sets_are_validated_and_cloneable_on_all_encoders(self):
        parameter_sets = [{"radius": 0}, {"distance": 2, "use_simplex_features": True}]
        for encoder in (
            NSPPK(parameters=parameter_sets, parallel=False),
            NodeNSPPK(parameters=parameter_sets, parallel=False),
            EdgeNSPPK(parameters=parameter_sets, parallel=False),
        ):
            with self.subTest(encoder=type(encoder).__name__):
                self.assertEqual(clone(encoder).get_params(deep=False)["parameters"], parameter_sets)
                with self.assertRaisesRegex(ValueError, "at least one"):
                    encoder.set_params(parameters=[])
                with self.assertRaisesRegex(ValueError, "Unknown key"):
                    encoder.set_params(parameters=[{"not_a_parameter": 1}])

    def test_alias_updates_sync_estimator_and_wrapped_encoder(self):
        encoder = NSPPK(parallel=False).set_params(r=2, d=4, c=1)
        self.assertEqual((encoder.radius, encoder.distance, encoder.connector), (2, 4, 1))
        self.assertEqual(
            (encoder.base_nsppk.radius, encoder.base_nsppk.distance, encoder.base_nsppk.connector),
            (2, 4, 1),
        )

        node_encoder = NodeNSPPK(parallel=False).set_params(r=2, nbits=6)
        self.assertEqual((node_encoder.radius, node_encoder.nbits), (2, 6))
        self.assertEqual((node_encoder.nsppk.radius, node_encoder.nsppk.nbits), (2, 6))

    def test_parameter_updates_reject_unknown_invalid_and_conflicting_values(self):
        encoder = NSPPK(parallel=False)
        with self.assertRaisesRegex(ValueError, "Invalid parameter"):
            encoder.set_params(unknown_option=True)
        with self.assertRaisesRegex(ValueError, "nbits"):
            encoder.set_params(nbits=1)
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
            encoder.set_params(radius=2, r=3)

        node_encoder = NodeNSPPK(parallel=False)
        with self.assertRaisesRegex(ValueError, "Invalid parameter"):
            node_encoder.set_params(unknown_option=True)

    def test_encoders_can_be_cloned_after_alias_support(self):
        encoder = NSPPK(r=2, d=4, c=1, parallel=False)
        cloned_encoder = clone(encoder)
        self.assertEqual((cloned_encoder.radius, cloned_encoder.distance, cloned_encoder.connector), (2, 4, 1))

        node_encoder = NodeNSPPK(r=2, d=4, c=1, parallel=False)
        cloned_node_encoder = clone(node_encoder)
        self.assertEqual(
            (cloned_node_encoder.radius, cloned_node_encoder.distance, cloned_node_encoder.connector),
            (2, 4, 1),
        )

    def test_encoder_parameters_are_validated_at_construction(self):
        for params, name in (
            ({"radius": -1}, "radius"),
            ({"distance": -1}, "distance"),
            ({"connector": -1}, "connector"),
            ({"nbits": 1}, "nbits"),
            ({"nbits": 31}, "nbits"),
            ({"degree_threshold": -1}, "degree_threshold"),
            ({"sigma": 0}, "sigma"),
            ({"attribute_dim": 0}, "attribute_dim"),
            ({"attribute_alphabet_size": 0}, "attribute_alphabet_size"),
        ):
            with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, name):
                NSPPK(**params)

        with self.assertRaisesRegex(ValueError, "parallel"):
            NSPPK(parallel=1)
        with self.assertRaisesRegex(ValueError, "node_attribute_key"):
            NSPPK(attribute_dim=2)

    def test_default_attribute_clustering_has_fixed_random_state(self):
        encoder = NSPPK(
            node_attribute_key="feature",
            attribute_alphabet_size=2,
            parallel=False,
        )
        self.assertEqual(encoder.clustering_predictor.random_state, 0)

    def test_structural_features_have_reserved_columns_and_are_repeatable(self):
        graph = _labeled_path()
        first = get_structural_node_vectors(graph, radius=1, distance=2, connector=1, nbits=6)
        second = get_structural_node_vectors(graph, radius=1, distance=2, connector=1, nbits=6)

        self.assertEqual(first.shape, (3, 64))
        np.testing.assert_array_equal(first.toarray(), second.toarray())
        np.testing.assert_array_equal(first[:, 0].toarray().ravel(), np.ones(3))
        np.testing.assert_array_equal(first[:, 1].toarray().ravel(), np.asarray([1, 2, 1]))
        self.assertNotIn("node_hash", graph.nodes[0])

    def test_root_node_filter_preserves_rows_and_limits_outer_roots(self):
        graph = _labeled_path()
        encoder = NodeNSPPK(radius=1, distance=2, nbits=6, dense=False, parallel=False)
        encoder.fit([graph])
        full = encoder.transform([graph])[0].toarray()
        filtered = encoder.transform([graph], root_node_ids=[[1]])[0].toarray()

        self.assertEqual(filtered.shape, full.shape)
        np.testing.assert_array_equal(filtered[1], full[1])
        np.testing.assert_array_equal(filtered[0, 2:], np.zeros(full.shape[1] - 2))
        np.testing.assert_array_equal(filtered[2, 2:], np.zeros(full.shape[1] - 2))
        np.testing.assert_array_equal(filtered[:, :2], full[:, :2])

    def test_graph_encoder_root_filter_matches_selected_node_rows(self):
        graph = _labeled_path()
        node_encoder = NodeNSPPK(radius=1, distance=2, nbits=6, dense=False, parallel=False)
        node_encoder.fit([graph])
        node_mtx = node_encoder.transform([graph])[0]
        graph_encoder = NSPPK(radius=1, distance=2, nbits=6, dense=False, parallel=False)
        graph_encoder.fit([graph])
        filtered_graph = graph_encoder.transform([graph], root_node_ids=[[1]])

        expected = np.asarray(node_mtx.sum(axis=0)).ravel()
        expected[2:] = node_mtx[1, 2:].toarray().ravel()
        np.testing.assert_array_equal(filtered_graph.toarray().ravel(), expected)
        with self.assertRaisesRegex(ValueError, "one node-ID collection per graph"):
            graph_encoder.transform([graph], root_node_ids=[])
        with self.assertRaisesRegex(ValueError, "absent from its graph"):
            graph_encoder.transform([graph], root_node_ids=[[99]])

    def test_node_attribute_features_extend_sparse_node_vectors(self):
        graph = _labeled_path()
        vectors = get_node_vectors(
            graph,
            radius=1,
            distance=2,
            connector=0,
            nbits=5,
            node_attribute_key="feature",
        )

        self.assertEqual(vectors.shape, (3, 2 * 2 + 2**5))
        self.assertTrue(np.isfinite(vectors.data).all())

    def test_importance_fit_validates_targets_and_configuration_before_encoding(self):
        model = ImportanceNSPPK(NodeNSPPK(parallel=False), parallel=False)
        with self.assertRaisesRegex(ValueError, "targets are required"):
            model.fit([_labeled_path()], None)
        with self.assertRaisesRegex(ValueError, "n_iter"):
            ImportanceNSPPK(NodeNSPPK(parallel=False), n_iter=0).fit([], [0, 1])
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            ImportanceNSPPK(NodeNSPPK(parallel=False), n_estimators=0).fit([], [0, 1])
        with self.assertRaisesRegex(ValueError, "quantile"):
            ImportanceNSPPK(NodeNSPPK(parallel=False), quantile=1.1).fit([], [0, 1])

        graphs = [_labeled_path(), _labeled_path()]
        with self.assertRaisesRegex(ValueError, "one label per graph"):
            model.fit(graphs, [0])
        with self.assertRaisesRegex(ValueError, "two distinct classes"):
            model.fit(graphs, [0, 0])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            model.fit(graphs, [[0], [1]])
        with self.assertRaisesRegex(ValueError, "at least two graphs"):
            model.fit(graphs * 2, [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()

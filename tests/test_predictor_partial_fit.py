import unittest

import networkx as nx
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor

from nsppk import NSPPKClassifier, NSPPKRegressor


def _graph(target, offset=0.0):
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "atom", "label")
    nx.set_edge_attributes(graph, "bond", "label")
    for node in graph:
        graph.nodes[node]["feature"] = np.asarray([offset + node, offset + node + 1.0])
    graph.graph["target"] = target
    return graph


class PredictorPartialFitTests(unittest.TestCase):
    def test_classifier_updates_batches_and_keeps_first_batch_preprocessing(self):
        model = NSPPKClassifier(
            estimator="linear",
            nbits=5,
            dense=False,
            parallel=False,
            node_attribute_key="feature",
            attribute_alphabet_size=2,
        )
        model.partial_fit(
            [_graph(0), _graph(1, 1.0)],
            [0, 1],
            classes=np.asarray([0, 1]),
        )
        centers_after_warmup = (
            model.encoder_.base_nsppk.clustering_predictor.cluster_centers_.copy()
        )

        model.partial_fit([_graph(1, 20.0), _graph(0, 21.0)], [1, 0])

        np.testing.assert_array_equal(
            model.encoder_.base_nsppk.clustering_predictor.cluster_centers_,
            centers_after_warmup,
        )
        np.testing.assert_array_equal(model.classes_, np.asarray([0, 1]))
        self.assertEqual(model.predict([_graph(0)]).shape, (1,))

    def test_classifier_requires_classes_on_first_partial_fit_call(self):
        model = NSPPKClassifier(estimator="linear", nbits=5, parallel=False)

        with self.assertRaisesRegex(ValueError, "classes must be provided"):
            model.partial_fit([_graph(0)], [0])

    def test_partial_fit_from_streams_labeled_graph_batches(self):
        graphs = [_graph(0), _graph(1, 1.0), _graph(0, 2.0), _graph(1, 3.0)]
        model = NSPPKClassifier(estimator="linear", nbits=5, dense=False, parallel=False)
        batches = []

        def record_batch(estimator, X_batch, y_batch, n_samples_seen):
            batches.append((estimator, X_batch.shape[0], y_batch.shape[0], n_samples_seen))

        returned = model.partial_fit_from(
            "unused-source",
            "custom",
            reader=lambda _uri: iter(graphs),
            batch_size=2,
            label_extractor=lambda graph: graph.graph["target"],
            classes=[0, 1],
            batch_callback=record_batch,
        )

        self.assertIs(returned, model)
        np.testing.assert_array_equal(model.classes_, np.asarray([0, 1]))
        self.assertEqual(model.predict(graphs).shape, (4,))
        self.assertEqual([(x, y, n) for _, x, y, n in batches], [(2, 2, 2), (2, 2, 4)])

    def test_regressor_sgd_shortcut_supports_partial_fit(self):
        model = NSPPKRegressor(estimator="sgd", nbits=5, dense=False, parallel=False)

        model.partial_fit([_graph(0)], [0.25])
        model.partial_fit([_graph(1, 1.0)], [0.75])

        self.assertEqual(model.predict([_graph(0)]).shape, (1,))
        self.assertIsInstance(model.estimator_, SGDRegressor)

    def test_non_incremental_estimators_do_not_expose_partial_fit(self):
        self.assertFalse(hasattr(NSPPKRegressor(), "partial_fit"))
        self.assertFalse(
            hasattr(NSPPKClassifier(estimator="rf"), "partial_fit")
        )
        self.assertTrue(
            hasattr(NSPPKClassifier(estimator=SGDClassifier()), "partial_fit")
        )


if __name__ == "__main__":
    unittest.main()

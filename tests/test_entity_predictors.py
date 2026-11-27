import unittest

import networkx as nx
import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor

from nsppk import (
    EdgeNSPPKClassifier,
    EdgeNSPPKRegressor,
    NodeNSPPKClassifier,
    NodeNSPPKRegressor,
)


def _graph(size=5, cycle=False):
    graph = nx.cycle_graph(size) if cycle else nx.path_graph(size)
    nx.set_node_attributes(graph, "node", "label")
    nx.set_edge_attributes(graph, "edge", "label")
    return graph


class EntityPredictorTests(unittest.TestCase):
    def setUp(self):
        self.graphs = [_graph(5), _graph(6, cycle=True)]
        self.node_counts = [graph.number_of_nodes() for graph in self.graphs]
        self.edge_counts = [graph.number_of_edges() for graph in self.graphs]

    def test_classifier_regressor_types_clone_and_random_forest_defaults(self):
        wrappers = (
            (NodeNSPPKClassifier, True, False, RandomForestClassifier),
            (EdgeNSPPKClassifier, True, False, RandomForestClassifier),
            (NodeNSPPKRegressor, False, True, RandomForestRegressor),
            (EdgeNSPPKRegressor, False, True, RandomForestRegressor),
        )
        for wrapper, classifier, regressor, estimator_type in wrappers:
            model = wrapper(nbits=4, parallel=False)
            self.assertIsInstance(model._make_estimator(), estimator_type)
            self.assertEqual(is_classifier(model), classifier)
            self.assertEqual(is_regressor(model), regressor)
            self.assertEqual(type(clone(model)), wrapper)

    def test_node_classifier_accepts_flat_and_per_graph_targets(self):
        flat_y = np.arange(sum(self.node_counts)) % 2
        flat_model = NodeNSPPKClassifier(nbits=4, parallel=False).fit(self.graphs, flat_y)
        self.assertEqual(flat_model.predict(self.graphs).shape, flat_y.shape)
        self.assertEqual(flat_model.predict_proba(self.graphs).shape, (len(flat_y), 2))

        nested_y = [np.arange(count) % 2 for count in self.node_counts]
        nested_model = NodeNSPPKClassifier(nbits=4, parallel=False).fit(self.graphs, nested_y)
        predictions = nested_model.predict(self.graphs)
        self.assertEqual([len(values) for values in predictions], self.node_counts)
        self.assertGreaterEqual(nested_model.score(self.graphs, nested_y), 0.0)

    def test_edge_classifier_and_regressor_preserve_per_graph_layout(self):
        labels = [np.arange(count) % 2 for count in self.edge_counts]
        classifier = EdgeNSPPKClassifier(nbits=4, parallel=False).fit(self.graphs, labels)
        self.assertEqual([len(values) for values in classifier.predict(self.graphs)], self.edge_counts)
        self.assertEqual(
            [values.shape for values in classifier.predict_proba(self.graphs)],
            [(count, 2) for count in self.edge_counts],
        )

        targets = [np.linspace(0.0, 1.0, count) for count in self.edge_counts]
        regressor = EdgeNSPPKRegressor(nbits=4, parallel=False).fit(self.graphs, targets)
        self.assertEqual([len(values) for values in regressor.predict(self.graphs)], self.edge_counts)
        self.assertTrue(np.isfinite(regressor.score(self.graphs, targets)))

    def test_node_regressor_flat_targets(self):
        targets = np.linspace(0.0, 1.0, sum(self.node_counts))
        regressor = NodeNSPPKRegressor(nbits=4, parallel=False).fit(self.graphs, targets)
        self.assertEqual(regressor.predict(self.graphs).shape, targets.shape)
        self.assertTrue(np.isfinite(regressor.score(self.graphs, targets)))

    def test_entity_selectors_and_validation_tuning(self):
        try:
            from nsppk_tuning import _load_skopt

            _load_skopt()
        except ImportError:
            self.skipTest("scikit-optimize is not installed")

        graph = _graph(12, cycle=True)
        labels = np.arange(12) % 2
        train_indices = np.arange(8)
        valid_indices = np.arange(8, 12)
        model = NodeNSPPKClassifier(nbits=4, parallel=False)
        result = model.tune(
            [graph],
            labels[train_indices],
            entity_indices=[train_indices],
            validation_data=([graph], labels[valid_indices]),
            validation_entity_indices=[valid_indices],
            search_space={"radius": [0, 1]},
            n_samples=2,
            n_initial_points=2,
            random_state=3,
        )
        self.assertTrue(result.validation_data_used)
        self.assertEqual(result.scoring, "average_precision")
        self.assertEqual(
            model.predict([graph], entity_indices=[valid_indices]).shape,
            valid_indices.shape,
        )

        edge_graphs = [_graph(5), _graph(6, cycle=True)]
        edge_selectors = [np.asarray([0]), np.asarray([0])]
        edge_model = EdgeNSPPKClassifier(nbits=4, parallel=False)
        edge_model.fit(edge_graphs, [0, 1], entity_indices=edge_selectors)
        self.assertEqual(
            edge_model.predict(edge_graphs, entity_indices=edge_selectors).shape,
            (2,),
        )

    def test_partial_fit_accepts_single_class_batches_and_streaming(self):
        classifier = NodeNSPPKClassifier(
            estimator=SGDClassifier(loss="log_loss", random_state=0),
            nbits=4,
            parallel=False,
        )
        classifier.partial_fit(
            [self.graphs[0]], [np.zeros(self.node_counts[0], dtype=int)], classes=[0, 1]
        )
        classifier.partial_fit(
            [self.graphs[1]], [np.ones(self.node_counts[1], dtype=int)]
        )
        self.assertIsInstance(classifier.estimator_, SGDClassifier)
        self.assertEqual([len(values) for values in classifier.predict(self.graphs)], self.node_counts)

        regressor = EdgeNSPPKRegressor(
            estimator=SGDRegressor(random_state=0), nbits=4, parallel=False
        )
        batches = []
        returned = regressor.partial_fit_from(
            "unused",
            "custom",
            reader=lambda _uri: iter(self.graphs),
            batch_size=1,
            label_extractor=lambda graph: np.arange(graph.number_of_edges(), dtype=float),
            batch_callback=lambda model, X, y, seen: batches.append((X.shape[0], len(y), seen)),
        )
        self.assertIs(returned, regressor)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[-1][2], sum(self.edge_counts))
        self.assertEqual([len(values) for values in regressor.predict(self.graphs)], self.edge_counts)

    def test_tune_uses_entity_level_cv_and_refits(self):
        try:
            from nsppk_tuning import _load_skopt

            _load_skopt()
        except ImportError:
            self.skipTest("scikit-optimize is not installed")
        graph = _graph(12, cycle=True)
        labels = np.arange(12) % 2
        model = NodeNSPPKClassifier(nbits=4, parallel=False)
        result = model.tune(
            [graph],
            labels,
            search_space={"radius": [0, 1]},
            cv=2,
            n_samples=2,
            n_initial_points=2,
            random_state=3,
        )
        self.assertEqual(result.task, "classifier")
        self.assertEqual(result.scoring, "average_precision")
        self.assertEqual(model.predict([graph]).shape, labels.shape)


if __name__ == "__main__":
    unittest.main()

import unittest

import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

from recursive_nsppk import RecursiveNSPPKClassifier


def _graphs_and_targets():
    graphs = []
    targets = []
    for idx in range(12):
        graph = nx.path_graph(4 + (idx % 2)) if idx % 2 == 0 else nx.cycle_graph(4 + (idx % 2))
        label = idx % 2
        for node in graph:
            graph.nodes[node]["label"] = str((node + label) % 3)
        for u, v in graph.edges():
            graph.edges[u, v]["label"] = str((u + v + label) % 2)
        graphs.append(graph)
        targets.append(label)
    return graphs, np.asarray(targets)


class RecursiveNSPPKClassifierTests(unittest.TestCase):
    def _classifier(self):
        return RecursiveNSPPKClassifier(
            estimator=LogisticRegression(solver="liblinear", max_iter=100),
            stages=(
                {"radius": 0, "distance": 1, "connector": 0},
                {"radius": 1, "distance": 2, "connector": 1},
            ),
            root_retention=0.5,
            nbits=5,
            dense=False,
            parallel=False,
            importance_n_iter=1,
            importance_n_estimators=5,
        )

    def test_classifier_clones_and_exposes_stage_and_cumulative_features(self):
        graphs, targets = _graphs_and_targets()
        classifier = self._classifier()
        cloned = clone(classifier)
        self.assertEqual(cloned.root_retention, 0.5)
        self.assertEqual(len(cloned.stages), 2)

        classifier.fit(graphs, targets)
        stage_matrices = classifier.transform_stages(graphs[:3])
        cumulative = classifier.transform(graphs[:3])
        self.assertEqual(len(stage_matrices), 2)
        self.assertEqual(stage_matrices[0].shape[0], 3)
        self.assertEqual(cumulative.shape[1], sum(matrix.shape[1] for matrix in stage_matrices))
        self.assertTrue(issparse(cumulative))
        self.assertEqual(len(classifier.stage_timings_), 2)
        for previous, current in zip(
            classifier.stage_selected_roots_[:-1], classifier.stage_input_roots_[1:]
        ):
            self.assertEqual(previous, current)

        predictions = classifier.predict(graphs[:3])
        probabilities = classifier.predict_proba(graphs[:3])
        self.assertEqual(predictions.shape, (3,))
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertGreaterEqual(classifier.score(graphs, targets), 0.0)

    def test_classifier_supports_cross_validation(self):
        graphs, targets = _graphs_and_targets()
        pipeline = Pipeline([("classifier", self._classifier())])
        scores = cross_val_score(pipeline, graphs, targets, cv=2)
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())

    def test_invalid_stage_and_retention_settings_fail_clearly(self):
        for stages, retention in (([], 0.25), ([1], 0.25), (None, 0.0)):
            with self.subTest(stages=stages, retention=retention):
                classifier = RecursiveNSPPKClassifier(stages=stages, root_retention=retention)
                graphs, targets = _graphs_and_targets()
                with self.assertRaises((ValueError, TypeError)):
                    classifier.fit(graphs, targets)


if __name__ == "__main__":
    unittest.main()

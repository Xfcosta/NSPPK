import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

from nsppk import NSPPKClassifier, NSPPKRegressor
from nsppk_tuning import _DEFAULT_SEARCH_SPACE, _load_skopt


def _graphs_and_targets(n=12):
    graphs = []
    labels = []
    regression_targets = []
    for index in range(n):
        label = index % 2
        size = 4 + index % 3
        graph = nx.path_graph(size) if label == 0 else nx.cycle_graph(size)
        for node in graph:
            graph.nodes[node]["label"] = str((node + label) % 3)
        for u, v in graph.edges():
            graph.edges[u, v]["label"] = str((u + v + label) % 2)
        graphs.append(graph)
        labels.append(label)
        regression_targets.append(float(graph.number_of_nodes() + graph.number_of_edges() * 0.1))
    return graphs, np.asarray(labels), np.asarray(regression_targets)


class NSPPKTuningTests(unittest.TestCase):
    def test_classifier_tune_runs_search_and_refits_best_configuration(self):
        graphs, labels, _ = _graphs_and_targets()
        model = NSPPKClassifier(
            estimator=LogisticRegression(solver="liblinear", max_iter=100),
            nbits=5,
            dense=False,
            parallel=False,
        )
        callbacks = []
        result = model.tune(
            graphs,
            labels,
            search_space={"radius": [0, 1], "use_simplex_features": [False, True]},
            cv=2,
            n_samples=2,
            n_initial_points=2,
            random_state=7,
            trial_callback=callbacks.append,
        )

        self.assertEqual(len(result.x_iters), 2)
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(result.best_params, model.best_params_)
        self.assertAlmostEqual(result.best_score, float(np.max(result.cv_scores)))
        self.assertEqual(result.cv_scores.shape, (2,))
        np.testing.assert_equal(model.predict(graphs[:3]).shape, (3,))

    def test_regressor_tune_defaults_to_r2_and_refits(self):
        graphs, _, targets = _graphs_and_targets()
        model = NSPPKRegressor(
            estimator="linear",
            nbits=5,
            dense=False,
            parallel=False,
        )
        result = model.tune(
            graphs,
            targets,
            search_space={"connector": (0, 1)},
            cv=2,
            n_samples=2,
            n_initial_points=2,
            random_state=5,
        )

        self.assertEqual(result.scoring, "r2")
        self.assertEqual(result.task, "regressor")
        self.assertEqual(model.best_params_, result.best_params)
        self.assertEqual(model.predict(graphs[:2]).shape, (2,))

    def test_classifier_tune_accepts_an_explicit_validation_set(self):
        graphs, labels, _ = _graphs_and_targets()
        train_graphs, validation_graphs = graphs[:8], graphs[8:]
        train_labels, validation_labels = labels[:8], labels[8:]
        model = NSPPKClassifier(
            estimator=LogisticRegression(solver="liblinear", max_iter=100),
            nbits=5,
            dense=False,
            parallel=False,
        )
        result = model.tune(
            train_graphs,
            train_labels,
            validation_data=(validation_graphs, validation_labels),
            search_space={"radius": [0, 1]},
            scoring="accuracy",
            n_samples=2,
            n_initial_points=2,
            random_state=11,
        )
        self.assertTrue(result.validation_data_used)
        self.assertEqual(result.scoring, "accuracy")
        self.assertEqual(model.predict(validation_graphs).shape, validation_labels.shape)

    def test_classifier_default_space_matches_documented_encoder_ranges(self):
        graphs, labels, _ = _graphs_and_targets()
        model = NSPPKClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            nbits=5,
            dense=False,
            parallel=False,
        )
        result = model.tune(
            graphs,
            labels,
            cv=2,
            n_samples=1,
            n_initial_points=1,
            random_state=3,
        )
        self.assertEqual(result.search_space, _DEFAULT_SEARCH_SPACE)
        self.assertEqual(result.scoring, "average_precision")

    def test_invalid_search_space_and_fold_settings_fail_clearly(self):
        graphs, labels, _ = _graphs_and_targets()
        model = NSPPKClassifier(estimator=LogisticRegression(solver="liblinear"))
        for search_space, message in (
            ({"estimator_params": [{"C": 1.0}]}, "Only NSPPK encoder settings"),
            ({"use_simplex_features": (False, True)}, "numeric values"),
            ({"radius": []}, "non-empty list"),
            ({"radius": (2, 1)}, "finite and increasing"),
        ):
            with self.subTest(search_space=search_space):
                with self.assertRaisesRegex(ValueError, message):
                    model.tune(graphs, labels, search_space=search_space, cv=2, n_samples=1)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            model.tune(graphs, labels, cv=1, n_samples=1)

    def test_missing_scikit_optimize_has_install_instructions(self):
        graphs, labels, _ = _graphs_and_targets()
        model = NSPPKClassifier(estimator=LogisticRegression(solver="liblinear"))
        with patch(
            "nsppk_tuning._load_skopt",
            side_effect=ImportError(
                "NSPPK .tune() requires scikit-optimize. Install it with "
                "`python -m pip install 'nsppk[tuning]'`."
            ),
        ):
            with self.assertRaisesRegex(ImportError, r"nsppk\[tuning\]"):
                model.tune(graphs, labels, cv=2, n_samples=1)


if __name__ == "__main__":
    unittest.main()

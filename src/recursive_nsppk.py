"""Scikit-learn classifier for recursive, importance-guided NSPPK features."""

from numbers import Integral, Real
import math
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.base import ClassifierMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from importance_nsppk import ImportanceNSPPK
from nsppk import NodeNSPPK, _NSPPKPredictor, _resolve_simplex_feature_order


_DEFAULT_STAGES = (
    {"radius": 1, "distance": 4, "connector": 0, "use_simplex_features": False},
    {"radius": 2, "distance": 4, "connector": 1, "use_simplex_features": False},
    {"radius": 2, "distance": 4, "connector": 1, "use_simplex_features": True},
)


class RecursiveNSPPKClassifier(ClassifierMixin, _NSPPKPredictor):
    """Classify graphs using importance-guided, recursively selected NSPPK roots.

    ``stages`` is an ordered sequence of dictionaries. Each stage dictionary
    overrides the shared ``NodeNSPPK`` settings for that pass. At each stage,
    an ``ImportanceNSPPK`` model is fitted on the training graphs and the
    current candidate roots. The next stage keeps the top ``root_retention``
    nodes per graph. Integer values select up to that many nodes; a fraction
    in ``(0, 1)`` selects that fraction of the graph's nodes (rounded up).

    The final classifier receives the horizontal concatenation of all stage
    feature matrices. Use :meth:`transform_stages` to inspect the individual
    stage matrices.

    Parameters follow ``NSPPKClassifier`` for the final predictor. The
    ``parallel`` option controls both NSPPK encoding and the internal Extra
    Trees importance estimators.
    """

    _task = "classifier"

    def __init__(
        self,
        estimator="linear",
        estimator_params=None,
        stages=None,
        root_retention=0.25,
        nbits=10,
        degree_threshold=None,
        dense=False,
        parallel=True,
        weight_key=None,
        node_attribute_key=None,
        attribute_dim=None,
        attribute_alphabet_size=None,
        sigma=None,
        use_simplex_features=False,
        simplex_feature_order=2,
        simplex_feature_max_order=None,
        importance_n_iter=3,
        importance_n_estimators=100,
        importance_quantile=0.5,
        importance_normalize=True,
    ):
        simplex_feature_order = _resolve_simplex_feature_order(
            simplex_feature_order, simplex_feature_max_order
        )
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.stages = stages
        self.root_retention = root_retention
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.sigma = sigma
        self.use_simplex_features = use_simplex_features
        self.simplex_feature_order = simplex_feature_order
        self.importance_n_iter = importance_n_iter
        self.importance_n_estimators = importance_n_estimators
        self.importance_quantile = importance_quantile
        self.importance_normalize = importance_normalize

    def _resolved_stages(self):
        stages = _DEFAULT_STAGES if self.stages is None else self.stages
        if isinstance(stages, (str, bytes)):
            raise ValueError("stages must be a non-empty sequence of parameter dictionaries.")
        try:
            stages = list(stages)
        except TypeError as exc:
            raise ValueError(
                "stages must be a non-empty sequence of parameter dictionaries."
            ) from exc
        if not stages:
            raise ValueError("stages must contain at least one parameter dictionary.")
        if any(not isinstance(stage, dict) for stage in stages):
            raise TypeError("Each stages entry must be a dictionary of NodeNSPPK parameters.")
        return stages

    def _shared_encoder_params(self):
        return {
            "nbits": self.nbits,
            "degree_threshold": self.degree_threshold,
            "dense": self.dense,
            "parallel": self.parallel,
            "weight_key": self.weight_key,
            "node_attribute_key": self.node_attribute_key,
            "attribute_dim": self.attribute_dim,
            "attribute_alphabet_size": self.attribute_alphabet_size,
            "sigma": self.sigma,
            "use_simplex_features": self.use_simplex_features,
            "simplex_feature_order": self.simplex_feature_order,
        }

    def _make_stage_encoder(self, stage_params):
        encoder_params = self._shared_encoder_params()
        encoder_params.update(stage_params)
        return NodeNSPPK(**encoder_params)

    def _validate_root_retention(self):
        value = self.root_retention
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("root_retention must be a positive integer or a fraction in (0, 1).")
        if isinstance(value, Integral):
            if value < 1:
                raise ValueError("Integer root_retention must be at least 1.")
            return
        if isinstance(value, Real) and math.isfinite(float(value)) and 0 < value < 1:
            return
        raise ValueError("root_retention must be a positive integer or a fraction in (0, 1).")

    def _select_roots(self, graphs, scored_graphs, candidates):
        selected_by_graph = []
        for graph, scored_graph, candidate_nodes in zip(graphs, scored_graphs, candidates):
            candidate_set = set(candidate_nodes)
            nodes = [node for node in graph.nodes() if node in candidate_set]
            scores = [
                scored_graph.nodes[node].get("importance", 0.0) for node in nodes
            ]
            requested = (
                int(self.root_retention)
                if isinstance(self.root_retention, Integral)
                else int(math.ceil(float(self.root_retention) * graph.number_of_nodes()))
            )
            ranked = sorted(
                range(len(nodes)),
                key=lambda idx: (-scores[idx], idx),
            )
            selected_indices = set(ranked[: min(requested, len(nodes))])
            selected_by_graph.append(
                [node for idx, node in enumerate(nodes) if idx in selected_indices]
            )
        return selected_by_graph

    @staticmethod
    def _graph_vectors(node_matrices):
        rows = [
            np.asarray(matrix.sum(axis=0)).reshape(-1)
            for matrix in node_matrices
        ]
        return csr_matrix(np.vstack(rows))

    def _stack_stage_features(self, stage_features):
        if self.dense:
            dense_stages = [
                matrix.toarray() if issparse(matrix) else np.asarray(matrix)
                for matrix in stage_features
            ]
            if len(dense_stages) == 1:
                return dense_stages[0]
            return np.hstack(dense_stages)
        sparse_stages = [
            matrix if issparse(matrix) else csr_matrix(matrix)
            for matrix in stage_features
        ]
        if len(sparse_stages) == 1:
            return sparse_stages[0]
        return hstack(sparse_stages, format="csr")

    def _make_importance_model(self, node_encoder):
        return ImportanceNSPPK(
            node_nsppk=node_encoder,
            importance_key="importance",
            n_iter=self.importance_n_iter,
            n_estimators=self.importance_n_estimators,
            quantile=self.importance_quantile,
            parallel=self.parallel,
            normalize=self.importance_normalize,
        )

    def fit(self, X, y, **fit_params):
        """Fit stage-wise node selectors and the final classifier on graph inputs."""
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        if y is None:
            raise ValueError("y is required.")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array of class labels.")
        check_consistent_length(graphs, y)
        check_classification_targets(y)
        classes, class_counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("y must contain at least two distinct classes.")
        if np.min(class_counts) < 2:
            raise ValueError("each class must contain at least two graphs for importance fitting.")
        self._validate_root_retention()
        stage_params_list = self._resolved_stages()

        current_roots = [list(graph.nodes()) for graph in graphs]
        stage_models = []
        stage_input_roots = []
        stage_output_roots = []
        stage_feature_matrices = []
        stage_timings = []

        for stage_params in stage_params_list:
            stage_input_roots.append([list(roots) for roots in current_roots])
            node_encoder = self._make_stage_encoder(stage_params)
            importance_model = self._make_importance_model(node_encoder)

            started = time.perf_counter()
            importance_model.fit(graphs, y, root_node_ids=current_roots)
            importance_fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            scored_graphs = importance_model.transform(
                graphs, root_node_ids=current_roots
            )
            next_roots = self._select_roots(graphs, scored_graphs, current_roots)
            root_selection_seconds = time.perf_counter() - started

            started = time.perf_counter()
            node_matrices = node_encoder.transform(
                graphs, root_node_ids=current_roots
            )
            stage_features = self._graph_vectors(node_matrices)
            if self.dense:
                stage_features = stage_features.toarray()
            feature_generation_seconds = time.perf_counter() - started

            stage_models.append(importance_model)
            stage_feature_matrices.append(stage_features)
            stage_output_roots.append([list(roots) for roots in next_roots])
            stage_timings.append({
                "importance_fit_seconds": importance_fit_seconds,
                "root_selection_seconds": root_selection_seconds,
                "feature_generation_seconds": feature_generation_seconds,
            })
            current_roots = next_roots

        training_features = self._stack_stage_features(stage_feature_matrices)
        final_estimator = self._make_estimator()
        started = time.perf_counter()
        final_estimator.fit(training_features, y, **fit_params)
        classifier_fit_seconds = time.perf_counter() - started

        self.stage_models_ = stage_models
        self.stage_params_ = stage_params_list
        self.stage_input_roots_ = stage_input_roots
        self.stage_selected_roots_ = stage_output_roots
        self.stage_feature_matrices_ = stage_feature_matrices
        self.stage_timings_ = stage_timings
        self.training_features_ = training_features
        self.estimator_ = final_estimator
        self.classes_ = getattr(final_estimator, "classes_", classes)
        self.classifier_fit_seconds_ = classifier_fit_seconds
        self.n_features_in_ = training_features.shape[1]
        return self

    def transform_stages(self, X):
        """Return one graph-feature matrix for each fitted stage, in stage order."""
        check_is_fitted(self, attributes=["stage_models_", "stage_params_"])
        graphs = list(X)
        if not graphs:
            raise ValueError("X must contain at least one graph.")
        current_roots = [list(graph.nodes()) for graph in graphs]
        stage_features = []
        for importance_model in self.stage_models_:
            scored_graphs = importance_model.transform(
                graphs, root_node_ids=current_roots
            )
            next_roots = self._select_roots(graphs, scored_graphs, current_roots)
            node_matrices = importance_model.node_nsppk.transform(
                graphs, root_node_ids=current_roots
            )
            features = self._graph_vectors(node_matrices)
            if self.dense:
                features = features.toarray()
            stage_features.append(features)
            current_roots = next_roots
        return stage_features

    def transform(self, X):
        """Return the cumulative graph-feature matrix used by the final classifier."""
        return self._stack_stage_features(self.transform_stages(X))

    def predict(self, X):
        """Predict graph labels."""
        check_is_fitted(self, attributes=["stage_models_", "estimator_"])
        return self.estimator_.predict(self.transform(X))

    @available_if(lambda self: hasattr(getattr(self, "estimator_", None), "predict_proba"))
    def predict_proba(self, X):
        """Return class probabilities if supported by the wrapped estimator."""
        check_is_fitted(self, attributes=["stage_models_", "estimator_"])
        return self.estimator_.predict_proba(self.transform(X))

    @available_if(lambda self: hasattr(getattr(self, "estimator_", None), "decision_function"))
    def decision_function(self, X):
        """Return decision scores if supported by the wrapped estimator."""
        check_is_fitted(self, attributes=["stage_models_", "estimator_"])
        return self.estimator_.decision_function(self.transform(X))

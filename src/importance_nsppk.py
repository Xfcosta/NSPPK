from collections import defaultdict
from numbers import Integral, Real
import math

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from nsppk import _select_graph_hash_strategy, get_feature_archive


class ImportanceNSPPK(object):
    """
    A class to visualize node importance in graphs using feature importance derived from an ensemble classifier.

    This class leverages the NSPPK (Neighborhood Subgraph Pairwise Path Kernel) method
    to decompose graphs into substructures and vectorize them. It then uses an ExtraTreesClassifier
    to compute feature importances, which are used to assign importance weights to nodes in the graphs.
    """

    def __init__(self, node_nsppk, importance_key='att', n_iter=10, n_estimators=100, quantile=0.5, parallel=True, normalize=True):
        self.node_nsppk = node_nsppk
        self.importance_key = importance_key
        self.n_iter = n_iter
        self.n_estimators = n_estimators
        self.quantile = quantile
        self.parallel = parallel
        self.normalize = normalize
        self.feature_importance_vector = None
        self.feature_archive_ = None
        self.feature_archive_summary_ = None

    def _prepare_graphs_for_node_nsppk(self, graphs):
        prepared_graphs = graphs
        if self.node_nsppk.nsppk.base_nsppk.node_attribute_key and self.node_nsppk.nsppk.base_nsppk.attribute_dim:
            prepared_graphs = self.node_nsppk.nsppk.base_nsppk.embed_attributes(prepared_graphs)
        if self.node_nsppk.nsppk.base_nsppk.node_attribute_key and self.node_nsppk.nsppk.base_nsppk.attribute_alphabet_size:
            prepared_graphs = self.node_nsppk.nsppk.base_nsppk.set_discrete_labels(prepared_graphs)
        return prepared_graphs

    def fit(self, graphs, targets=None, root_node_ids=None):
        if isinstance(self.n_iter, bool) or not isinstance(self.n_iter, Integral) or self.n_iter < 1:
            raise ValueError("n_iter must be a positive integer.")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if (
            isinstance(self.quantile, bool)
            or not isinstance(self.quantile, Real)
            or not math.isfinite(float(self.quantile))
            or not 0 <= self.quantile <= 1
        ):
            raise ValueError("quantile must be a finite number between 0 and 1.")

        if graphs is None:
            raise ValueError("graphs must be an iterable containing at least one graph.")
        try:
            graphs = list(graphs)
        except TypeError as exc:
            raise ValueError("graphs must be an iterable containing at least one graph.") from exc
        if not graphs:
            raise ValueError("graphs must contain at least one graph.")
        if targets is None:
            raise ValueError("targets are required and must contain one class label per graph.")
        targets = np.asarray(targets)
        if targets.ndim != 1:
            raise ValueError("targets must be a one-dimensional array of class labels.")
        if len(targets) != len(graphs):
            raise ValueError(
                f"targets must contain one label per graph; got {len(targets)} labels "
                f"for {len(graphs)} graphs."
            )
        if targets.dtype.kind in "fc" and np.isnan(targets).any():
            raise ValueError("targets cannot contain missing or NaN class labels.")
        if targets.dtype.kind == "O" and any(target is None for target in targets):
            raise ValueError("targets cannot contain missing class labels.")
        try:
            classes, class_counts = np.unique(targets, return_counts=True)
        except TypeError as exc:
            raise ValueError("targets must contain mutually comparable class labels.") from exc
        if len(classes) < 2:
            raise ValueError("targets must contain at least two distinct classes.")
        if np.min(class_counts) < 2:
            raise ValueError("each class must contain at least two graphs for repeated train/test splits.")
        n_train = int(np.floor(0.7 * len(graphs)))
        n_test = len(graphs) - n_train
        if n_train < len(classes) or n_test < len(classes):
            raise ValueError(
                "the 70/30 train/test split must leave at least one graph from each class "
                "in both subsets; provide more graphs or fewer classes."
            )

        self.node_nsppk.fit(graphs, targets)
        node_feature_mtx_list = self.node_nsppk.transform(
            graphs, root_node_ids=root_node_ids
        )
        X = np.vstack([
            np.asarray(node_feature_mtx.sum(axis=0)).reshape(-1)
            for node_feature_mtx in node_feature_mtx_list
        ])
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("node_nsppk produced no features to fit importance weights.")

        feature_importances = []
        for it in range(self.n_iter):
            train_X, _, train_targets, _ = train_test_split(
                X,
                targets,
                train_size=0.7,
                random_state=it + 1,
                stratify=targets,
            )
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                n_jobs=-1 if self.parallel else None,
                random_state=it + 1,
            ).fit(train_X, train_targets)
            feature_importances.append(clf.feature_importances_)

        feature_importances_mtx = np.vstack(feature_importances)
        mean_importances = np.mean(feature_importances_mtx, axis=0)
        std_importances = np.std(feature_importances_mtx, axis=0)

        adjusted_importances = mean_importances - std_importances
        adjusted_importances[adjusted_importances < 0] = 0

        threshold = np.quantile(adjusted_importances, self.quantile)
        adjusted_importances[adjusted_importances < threshold] = 0

        max_importance = np.max(adjusted_importances)
        if max_importance > 0:
            self.feature_importance_vector = adjusted_importances / max_importance
        else:
            self.feature_importance_vector = adjusted_importances

        self.feature_archive_ = None
        self.feature_archive_summary_ = None
        return self

    def build_feature_archive(self, graphs):
        check_is_fitted(self.node_nsppk.nsppk.base_nsppk, 'is_fitted_')
        prepared_graphs = list(self._prepare_graphs_for_node_nsppk(graphs))
        graph_strategy = _select_graph_hash_strategy(prepared_graphs)

        feature_archive = defaultdict(list)
        feature_archive_summary = {}

        for graph_index, graph in enumerate(prepared_graphs):
            graph_archive = get_feature_archive(
                graph,
                graph_index=graph_index,
                radius=self.node_nsppk.nsppk.base_nsppk.radius,
                distance=self.node_nsppk.nsppk.base_nsppk.distance,
                connector=self.node_nsppk.nsppk.base_nsppk.connector,
                nbits=self.node_nsppk.nsppk.base_nsppk.nbits,
                degree_threshold=self.node_nsppk.nsppk.base_nsppk.degree_threshold,
                sigma=self.node_nsppk.nsppk.sigma,
                use_simplex_features=self.node_nsppk.nsppk.use_simplex_features,
                simplex_feature_max_order=(
                    self.node_nsppk.nsppk.simplex_feature_max_order
                ),
                graph_strategy=graph_strategy,
            )
            for feature_id, occurrences in graph_archive.items():
                feature_archive[feature_id].extend(occurrences)

        for feature_id, occurrences in feature_archive.items():
            feature_archive_summary[feature_id] = {
                'occurrence_count': len(occurrences),
                'distinct_signature_count': len({occurrence['raw_signature'] for occurrence in occurrences}),
            }

        self.feature_archive_ = dict(feature_archive)
        self.feature_archive_summary_ = feature_archive_summary
        return self

    def get_top_features(self, top_k=10, min_importance=0.0):
        if self.feature_importance_vector is None:
            raise ValueError("fit must be called before get_top_features.")
        if self.feature_archive_ is None:
            raise ValueError("build_feature_archive must be called before get_top_features.")

        ranked_features = []
        for feature_id, occurrences in self.feature_archive_.items():
            if feature_id >= len(self.feature_importance_vector):
                continue
            importance = float(self.feature_importance_vector[feature_id])
            if importance <= 0 or importance < min_importance:
                continue
            summary = self.feature_archive_summary_.get(feature_id, {})
            representative = occurrences[0]
            ranked_features.append(
                {
                    'feature_id': feature_id,
                    'importance': importance,
                    'occurrence_count': summary.get('occurrence_count', len(occurrences)),
                    'distinct_signature_count': summary.get(
                        'distinct_signature_count',
                        len({occurrence['raw_signature'] for occurrence in occurrences}),
                    ),
                    'representative_graph': representative['subgraph'],
                    'representative_occurrence': representative,
                    'source_graph_index': representative['source_graph_index'],
                    'occurrences': occurrences,
                }
            )

        ranked_features.sort(key=lambda item: (-item['importance'], item['feature_id']))
        return ranked_features[:top_k]

    def transform(self, graphs, root_node_ids=None):
        graphs = list(graphs)
        graph_strategy = _select_graph_hash_strategy(graphs)
        if root_node_ids is not None:
            root_node_ids = [
                None if roots is None else list(roots) for roots in root_node_ids
            ]
        node_feature_mtx_list = self.node_nsppk.transform(
            graphs, root_node_ids=root_node_ids
        )
        if root_node_ids is None:
            selected_roots_by_graph = [None] * len(graphs)
        else:
            selected_roots_by_graph = [
                None if roots is None else set(roots) for roots in root_node_ids
            ]
        out_graphs = []

        for graph, node_feature_mtx, selected_roots in zip(
            graphs, node_feature_mtx_list, selected_roots_by_graph
        ):
            out_graph = graph.copy()
            if hasattr(node_feature_mtx, 'multiply'):
                weighted_features = node_feature_mtx.multiply(
                    self.feature_importance_vector
                )
            else:
                weighted_features = node_feature_mtx * self.feature_importance_vector
            weights = np.asarray(weighted_features.sum(axis=1)).reshape(-1)

            if self.normalize:
                max_weight = np.max(weights) if np.max(weights) > 0 else 1
                normalized_weights = weights / max_weight
            else:
                normalized_weights = weights

            for node_idx, weight in zip(out_graph.nodes(), normalized_weights):
                out_graph.nodes[node_idx][self.importance_key] = (
                    weight if selected_roots is None or node_idx in selected_roots else 0.0
                )

            edge_records = list(graph_strategy.edge_records(out_graph))
            for u, v, _, edge_data in edge_records:
                edge_data[self.importance_key] = out_graph.nodes[u][self.importance_key] * out_graph.nodes[v][self.importance_key]

            for u, node_feature in zip(out_graph.nodes(), node_feature_mtx):
                out_graph.nodes[u]['node_feature'] = node_feature

            for u, v, _, edge_data in edge_records:
                edge_data['edge_feature'] = out_graph.nodes[u]['node_feature'] + out_graph.nodes[v]['node_feature']

            out_graphs.append(out_graph)

        return out_graphs

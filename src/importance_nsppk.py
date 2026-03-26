from collections import defaultdict

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from nsppk import get_feature_archive


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

    def fit(self, graphs, targets=None):
        node_feature_mtx_list = self.node_nsppk.fit_transform(graphs)
        X = np.vstack([node_feature_mtx.sum(axis=0) for node_feature_mtx in node_feature_mtx_list])

        feature_importances = []
        for it in range(self.n_iter):
            train_X, test_X, train_targets, test_targets = train_test_split(
                X, targets, train_size=0.7, random_state=it + 1
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
        prepared_graphs = self._prepare_graphs_for_node_nsppk(graphs)

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
                use_edges_as_features=self.node_nsppk.nsppk.use_edges_as_features,
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

    def transform(self, graphs):
        node_feature_mtx_list = self.node_nsppk.transform(graphs)
        out_graphs = []

        for graph, node_feature_mtx in zip(graphs, node_feature_mtx_list):
            out_graph = graph.copy()
            weights = node_feature_mtx * self.feature_importance_vector
            weights = np.sum(weights, axis=1)

            if self.normalize:
                max_weight = np.max(weights) if np.max(weights) > 0 else 1
                normalized_weights = weights / max_weight
            else:
                normalized_weights = weights

            for node_idx, weight in zip(out_graph.nodes(), normalized_weights):
                out_graph.nodes[node_idx][self.importance_key] = weight

            for u, v in out_graph.edges():
                out_graph.edges[u, v][self.importance_key] = out_graph.nodes[u][self.importance_key] * out_graph.nodes[v][self.importance_key]

            for u, node_feature in zip(out_graph.nodes(), node_feature_mtx):
                out_graph.nodes[u]['node_feature'] = node_feature

            for u, v in out_graph.edges():
                out_graph.edges[u, v]['edge_feature'] = out_graph.nodes[u]['node_feature'] + out_graph.nodes[v]['node_feature']

            out_graphs.append(out_graph)

        return out_graphs

import unittest

import matplotlib
import networkx as nx

from importance_nsppk import ImportanceNSPPK
from nsppk import NodeNSPPK
from utils import plot_graph_importance


matplotlib.use("Agg")


def _labeled_graph(edges, node_offset=0, edge_label=0):
    graph = nx.Graph()
    nodes = sorted({node for edge in edges for node in edge})
    for node in nodes:
        graph.add_node(node + node_offset, label=(node + node_offset) % 3)
    for u, v in edges:
        graph.add_edge(u + node_offset, v + node_offset, label=edge_label)
    return graph


def _dataset():
    graphs = [
        _labeled_graph([(0, 1), (1, 2), (2, 3)], node_offset=0, edge_label=0),
        _labeled_graph([(0, 1), (1, 2), (2, 3), (3, 4)], node_offset=0, edge_label=1),
        _labeled_graph([(0, 1), (1, 2), (2, 3), (3, 0)], node_offset=0, edge_label=2),
        _labeled_graph([(0, 1), (0, 2), (0, 3)], node_offset=10, edge_label=0),
        _labeled_graph([(0, 1), (0, 2), (0, 3), (0, 4)], node_offset=10, edge_label=1),
        _labeled_graph([(0, 1), (1, 2), (2, 0), (2, 3)], node_offset=10, edge_label=2),
    ]
    labels = [0, 0, 0, 1, 1, 1]
    return graphs, labels


def _importance_model(nbits=6):
    node_encoder = NodeNSPPK(
        radius=1,
        distance=2,
        connector=1,
        nbits=nbits,
        dense=True,
        parallel=False,
    )
    return ImportanceNSPPK(
        node_nsppk=node_encoder,
        importance_key="importance",
        n_iter=2,
        n_estimators=16,
        quantile=0.0,
        parallel=False,
        normalize=True,
    )


class ImportanceArchiveTests(unittest.TestCase):
    def test_build_feature_archive_populates_occurrences_and_summary(self):
        graphs, labels = _dataset()
        model = _importance_model(nbits=6)

        model.fit(graphs, labels)
        model.build_feature_archive(graphs)

        self.assertTrue(model.feature_archive_)
        feature_id, occurrences = next(iter(model.feature_archive_.items()))
        occurrence = occurrences[0]

        self.assertIn(feature_id, model.feature_archive_summary_)
        self.assertEqual(
            model.feature_archive_summary_[feature_id]["occurrence_count"],
            len(occurrences),
        )
        self.assertIn("subgraph", occurrence)
        self.assertIn("source_node_ids", occurrence)
        self.assertIn("source_edge_ids", occurrence)
        self.assertIn("raw_signature", occurrence)
        self.assertIn("feature_kind", occurrence)
        self.assertTrue(all("label" in occurrence["subgraph"].nodes[node] for node in occurrence["subgraph"].nodes()))
        self.assertTrue(all("original_node_id" in occurrence["subgraph"].nodes[node] for node in occurrence["subgraph"].nodes()))

    def test_collision_bucket_keeps_distinct_raw_signatures(self):
        graphs, labels = _dataset()
        model = _importance_model(nbits=2)

        model.fit(graphs, labels)
        model.build_feature_archive(graphs)

        collision_feature_ids = [
            feature_id
            for feature_id, summary in model.feature_archive_summary_.items()
            if summary["distinct_signature_count"] > 1
        ]

        self.assertTrue(collision_feature_ids)
        collision_feature_id = collision_feature_ids[0]
        raw_signatures = {
            occurrence["raw_signature"]
            for occurrence in model.feature_archive_[collision_feature_id]
        }
        self.assertGreater(len(raw_signatures), 1)
        self.assertGreaterEqual(
            len(model.feature_archive_[collision_feature_id]),
            len(raw_signatures),
        )

    def test_get_top_features_returns_sorted_entries_with_occurrences(self):
        graphs, labels = _dataset()
        model = _importance_model(nbits=6)

        model.fit(graphs, labels)
        model.build_feature_archive(graphs)
        top_features = model.get_top_features(top_k=5)

        self.assertTrue(top_features)
        self.assertLessEqual(len(top_features), 5)
        importances = [item["importance"] for item in top_features]
        self.assertEqual(importances, sorted(importances, reverse=True))

        top_entry = top_features[0]
        self.assertIn("representative_graph", top_entry)
        self.assertIn("occurrences", top_entry)
        self.assertEqual(top_entry["occurrence_count"], len(top_entry["occurrences"]))
        self.assertGreaterEqual(
            top_entry["occurrence_count"],
            top_entry["distinct_signature_count"],
        )
        plot_graph_importance([top_entry["representative_graph"]], titles=["top"], use_pos=False)

    def test_transform_assigns_importance_to_actual_node_ids(self):
        graphs, labels = _dataset()
        model = _importance_model(nbits=6)

        model.fit(graphs, labels)
        out_graph = model.transform([graphs[3]])[0]

        for node in graphs[3].nodes():
            self.assertIn("importance", out_graph.nodes[node])


if __name__ == "__main__":
    unittest.main()

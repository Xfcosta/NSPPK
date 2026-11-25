import unittest

import networkx as nx

import nsppk
from nsppk import EdgeNSPPK, NSPPK, NodeNSPPK, get_feature_archive


def _feature_occurrences(archive, feature_kind):
    return [
        occurrence
        for occurrences in archive.values()
        for occurrence in occurrences
        if occurrence['feature_kind'] == feature_kind
    ]


class SimplexFeatureTests(unittest.TestCase):
    def test_fragment_enumeration_includes_each_order_once(self):
        graph = nx.Graph()
        graph.add_nodes_from((node, {'label': str(node)}) for node in range(4))
        graph.add_edges_from([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
        prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)

        fragments = nsppk._simplex_fragments(prepared, graph.nodes(), 3)
        order_counts = {}
        for order, _, _ in fragments:
            order_counts[order] = order_counts.get(order, 0) + 1

        self.assertEqual(order_counts, {0: 4, 1: 6, 2: 4, 3: 1})

    def test_neighborhood_fragments_are_cached_by_region_and_order(self):
        graph = nx.path_graph(3)
        prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)
        region = nsppk._node_ball(prepared, 1, 1)

        first = nsppk._simplex_fragments(prepared, region, 2)
        second = nsppk._simplex_fragments(prepared, region, 2)

        self.assertIs(first, second)
        self.assertEqual(len(prepared.graph['_structural_cache']['simplex_fragments']), 1)

    def test_feature_flag_defaults_off_and_max_order_is_validated(self):
        graph = nx.path_graph(2)
        graph_archive = get_feature_archive(graph, 0, 1, 1, 0, 6)
        self.assertEqual(_feature_occurrences(graph_archive, 'simplex_neighborhood_pair'), [])

        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    NSPPK(simplex_feature_max_order=invalid)

    def test_simplex_parameters_are_exposed_by_encoder_wrappers(self):
        for encoder in (
            NSPPK(use_simplex_features=True, simplex_feature_max_order=3),
            NodeNSPPK(use_simplex_features=True, simplex_feature_max_order=3),
            EdgeNSPPK(use_simplex_features=True, simplex_feature_max_order=3),
        ):
            with self.subTest(encoder=type(encoder).__name__):
                self.assertTrue(encoder.get_params()['use_simplex_features'])
                self.assertEqual(encoder.get_params()['simplex_feature_max_order'], 3)
                encoder.set_params(simplex_feature_max_order=1)
                self.assertEqual(encoder.get_params()['simplex_feature_max_order'], 1)

    def test_neighborhood_pair_hash_is_unordered_and_label_contextualized(self):
        graph = nx.Graph()
        graph.add_node('a', label='A')
        graph.add_node('b', label='B')
        graph.add_edge('a', 'b', label='bond')
        archive = get_feature_archive(
            graph, 0, 1, 1, 0, 8, use_simplex_features=True,
            simplex_feature_max_order=1,
        )
        occurrences = _feature_occurrences(archive, 'simplex_neighborhood_pair')

        self.assertTrue(occurrences)
        forward = [
            occurrence['raw_signature'] for occurrence in occurrences
            if occurrence['root_node'] == 'a' and occurrence['target_node'] == 'b'
        ]
        reverse = [
            occurrence['raw_signature'] for occurrence in occurrences
            if occurrence['root_node'] == 'b' and occurrence['target_node'] == 'a'
        ]
        self.assertEqual(set(forward), set(reverse))

    def test_directed_fragment_hash_retains_arc_orientation(self):
        signatures = []
        for source, target in (('a', 'b'), ('b', 'a')):
            graph = nx.DiGraph()
            graph.add_node('a', label='A')
            graph.add_node('b', label='B')
            graph.add_edge(source, target, label='bond')
            prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)
            fragments = nsppk._simplex_fragments(prepared, {'a', 'b'}, 1)
            signatures.append(next(fragment_hash for order, fragment_hash, _ in fragments if order == 1))

        self.assertNotEqual(signatures[0], signatures[1])

    def test_multigraph_parallel_edge_patterns_are_hashed_once_per_vertex_set(self):
        signatures = []
        for labels in (('x',), ('x', 'y')):
            graph = nx.MultiGraph()
            graph.add_node(0, label='A')
            graph.add_node(1, label='B')
            for index, label in enumerate(labels):
                graph.add_edge(0, 1, key=index, label=label)
            prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)
            edge_fragments = [
                fragment_hash
                for order, fragment_hash, _ in nsppk._simplex_fragments(prepared, {0, 1}, 1)
                if order == 1
            ]
            self.assertEqual(len(edge_fragments), 1)
            signatures.append(edge_fragments[0])

        self.assertNotEqual(signatures[0], signatures[1])

    def test_directed_multigraph_fragment_hash_includes_arc_pattern(self):
        signatures = []
        for arcs in (
            (('a', 'b', 'x'), ('b', 'a', 'y')),
            (('a', 'b', 'x'), ('a', 'b', 'y')),
        ):
            graph = nx.MultiDiGraph()
            graph.add_node('a', label='A')
            graph.add_node('b', label='B')
            for source, target, label in arcs:
                graph.add_edge(source, target, label=label)
            prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)
            edge_fragments = [
                fragment_hash
                for order, fragment_hash, _ in nsppk._simplex_fragments(prepared, {'a', 'b'}, 1)
                if order == 1
            ]
            self.assertEqual(len(edge_fragments), 1)
            signatures.append(edge_fragments[0])

        self.assertNotEqual(signatures[0], signatures[1])

    def test_connector_path_fragments_use_expanded_union(self):
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
        archive = get_feature_archive(
            graph, 0, 1, 2, 2, 8, use_simplex_features=True,
            simplex_feature_max_order=0,
        )
        occurrences = [
            occurrence for occurrence in _feature_occurrences(archive, 'simplex_path_union')
            if occurrence['root_node'] == 0 and occurrence['target_node'] == 2
        ]

        self.assertTrue(occurrences)
        self.assertTrue(any(3 in occurrence['source_node_ids'] for occurrence in occurrences))


if __name__ == '__main__':
    unittest.main()

import unittest

import networkx as nx

import nsppk
from hash import edge_triplet_hash, hash_sequence, hash_set
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

    def test_undirected_edge_triplet_keeps_roles_and_ignores_endpoint_order(self):
        graph = nx.Graph()
        graph.add_node(0, label='A')
        graph.add_node(1, label='B')
        graph.add_edge(0, 1, label='C')

        role_swapped_graph = nx.Graph()
        role_swapped_graph.add_node(0, label='A')
        role_swapped_graph.add_node(1, label='C')
        role_swapped_graph.add_edge(0, 1, label='B')

        self.assertEqual(
            edge_triplet_hash(0, 1, graph),
            edge_triplet_hash(1, 0, graph),
        )
        self.assertNotEqual(
            edge_triplet_hash(0, 1, graph),
            edge_triplet_hash(0, 1, role_swapped_graph),
        )

    def test_singleton_hash_is_its_node_label_hash(self):
        graph = nx.Graph()
        graph.add_node('root', label='A')
        graph.add_edge('root', 'leaf', label='x')
        prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)

        self.assertEqual(
            nsppk._simplex_fragment_hash(prepared, ('root',)),
            nsppk._hash('A'),
        )

    def test_clique_hash_is_hash_set_of_cached_edge_triplets(self):
        graph = nx.Graph()
        graph.add_nodes_from((node, {'label': label}) for node, label in zip('abc', 'ABC'))
        graph.add_edges_from((left, right, {'label': 'bond'}) for left, right in [('a', 'b'), ('a', 'c'), ('b', 'c')])
        prepared = nsppk._prepare_structural_graph(graph, 1, 1, 0)
        structural_cache = prepared.graph['_structural_cache']
        expected = hash_set(structural_cache['edge_triplet_hashes'].values())

        self.assertEqual(nsppk._simplex_fragment_hash(prepared, ('a', 'b', 'c')), expected)

        relabeled = nx.relabel_nodes(graph, {'a': 3, 'b': 1, 'c': 2})
        relabeled_prepared = nsppk._prepare_structural_graph(relabeled, 1, 1, 0)
        self.assertEqual(
            nsppk._simplex_fragment_hash(relabeled_prepared, (3, 1, 2)),
            expected,
        )

    def test_neighborhood_pair_hash_uses_union_and_root_pair_context(self):
        graph = nx.Graph()
        graph.add_node('a', label='A')
        graph.add_node('b', label='B')
        graph.add_edge('a', 'b', label='bond')
        archive = get_feature_archive(
            graph, 0, 0, 1, 0, 8, use_simplex_features=True,
            simplex_feature_order=(0, 0),
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
        root_pair_hash = hash_set((nsppk._hash('A'), nsppk._hash('B')))
        expected = {
            hash_sequence((root_pair_hash, 1, nsppk._hash(label)))
            for label in ('A', 'B')
        }
        self.assertEqual(set(forward), expected)
        self.assertEqual(set(reverse), expected)

    def test_unary_context_uses_root_label_hash_and_radius(self):
        graph = nx.Graph()
        graph.add_node('root', label='A')
        graph.add_nodes_from([('leaf1', {'label': 'B'}), ('leaf2', {'label': 'C'})])
        graph.add_edges_from([('root', 'leaf1'), ('root', 'leaf2')])
        archive = get_feature_archive(
            graph, 0, 1, 0, 0, 8, use_simplex_features=True,
            simplex_feature_order=(0, 0),
        )
        occurrences = [
            occurrence for occurrence in _feature_occurrences(archive, 'simplex_neighborhood_single')
            if occurrence['root_node'] == 'root'
            and occurrence['source_node_ids'] == ('root',)
        ]

        self.assertEqual(
            {occurrence['raw_signature'] for occurrence in occurrences},
            {
                hash_sequence((nsppk._hash('A'), radius, nsppk._hash('A')))
                for radius in (0, 1)
            },
        )

    def test_neighborhood_pair_union_deduplicates_shared_clique_occurrences(self):
        graph = nx.path_graph(3)
        archive = get_feature_archive(
            graph, 0, (1, 1), 2, 0, 8, use_simplex_features=True,
            simplex_feature_order=(0, 0),
        )
        occurrences = [
            occurrence for occurrence in _feature_occurrences(archive, 'simplex_neighborhood_pair')
            if occurrence['root_node'] == 0 and occurrence['target_node'] == 2
        ]

        self.assertEqual(
            sorted(occurrence['source_node_ids'] for occurrence in occurrences),
            [(0,), (1,), (2,)],
        )

    def test_path_union_hash_context_includes_connector_thickness(self):
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
        archive = get_feature_archive(
            graph, 0, 0, 2, 2, 8, use_simplex_features=True,
            simplex_feature_order=(0, 0),
        )
        occurrences = [
            occurrence for occurrence in _feature_occurrences(archive, 'simplex_path_union')
            if occurrence['root_node'] == 0
            and occurrence['target_node'] == 2
            and occurrence['source_node_ids'] == (1,)
        ]

        expected = {
            hash_sequence((hash_set((nsppk._hash(0), nsppk._hash(0))), 2, connect, nsppk._hash(0)))
            for connect in (1, 2)
        }
        self.assertEqual({occurrence['raw_signature'] for occurrence in occurrences}, expected)

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
        for labels in (('x',), ('x', 'y'), ('x', 'x')):
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
        self.assertNotEqual(signatures[0], signatures[2])

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

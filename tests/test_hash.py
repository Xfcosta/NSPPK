import unittest

import hash as graph_hash


class StableHashTests(unittest.TestCase):
    def test_hash_is_deterministic_for_nested_values(self):
        value = ("node", 7, ("edge", 3), [1, 2, 3], None, True)

        hash_a = graph_hash._hash(value)
        hash_b = graph_hash._hash(value)

        self.assertEqual(hash_a, hash_b)

    def test_hash_sequence_depends_on_order(self):
        self.assertNotEqual(
            graph_hash.hash_sequence([1, 2, 3]),
            graph_hash.hash_sequence([3, 2, 1]),
        )

    def test_hash_set_is_order_invariant(self):
        self.assertEqual(
            graph_hash.hash_set([3, 1, 2]),
            graph_hash.hash_set([2, 3, 1]),
        )

    def test_hash_matches_string_coercion_semantics(self):
        self.assertEqual(graph_hash._hash(1), graph_hash._hash("1"))
        self.assertNotEqual(graph_hash._hash((1, 2)), graph_hash._hash([1, 2]))

    def test_hash_value_reserves_special_columns(self):
        for value in [0, 1, "abc", ("x", 2)]:
            hashed = graph_hash.hash_value(value, nbits=8)
            self.assertGreaterEqual(hashed, 2)
            self.assertLess(hashed, 2**8)

import functools
import importlib.util
import io
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
from sklearn.exceptions import NotFittedError

import nsppk


HAS_RDKIT = importlib.util.find_spec("rdkit") is not None


class _SequenceRng:
    def __init__(self, values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


class _HttpServer:
    def __init__(self, directory):
        self.directory = directory
        self.server = None
        self.thread = None
        self.url = None

    def __enter__(self):
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(self.directory))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@unittest.skipUnless(HAS_RDKIT, "RDKit is required for streaming tests.")
class StreamingIOTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.smiles_path = self.data_dir / "mols.smi"
        self.smiles_path.write_text(
            "\n".join(
                [
                    "CCO ethanol",
                    "CC propane",
                    "O water",
                    "N ammonia",
                    "C methane",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_from_returns_graph_list_with_integer_limit(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)

        self.assertEqual(len(graphs), 2)
        self.assertTrue(all(isinstance(graph, nx.Graph) for graph in graphs))
        self.assertEqual([graph.graph["smiles"] for graph in graphs], ["CCO", "CC"])

    def test_load_from_skips_smiles_header_and_handles_csv_first_column(self):
        csv_smiles_path = self.data_dir / "mols.csv"
        csv_smiles_path.write_text(
            "\n".join(
                [
                    "smiles,name",
                    "CCO,ethanol",
                    "CC,propane",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(csv_smiles_path, "smiles")

        self.assertEqual(len(graphs), 2)
        self.assertEqual([graph.graph["smiles"] for graph in graphs], ["CCO", "CC"])
        self.assertEqual([graph.graph.get("name") for graph in graphs], ["ethanol", "propane"])

    def test_load_from_preserves_named_csv_columns(self):
        csv_smiles_path = self.data_dir / "mols_targets.csv"
        csv_smiles_path.write_text(
            "\n".join(
                [
                    "smiles,activity,HIV_active",
                    "CCO,CI,0",
                    "CC,CA,1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(csv_smiles_path, "smiles")

        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].graph["activity"], "CI")
        self.assertEqual(graphs[0].graph["HIV_active"], "0")
        self.assertEqual(graphs[1].graph["activity"], "CA")
        self.assertEqual(graphs[1].graph["HIV_active"], "1")

    def test_load_from_fractional_limit_is_reproducible(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs_a = vectorizer.load_from(self.smiles_path, "smiles", limit=0.5, random_state=7)
        graphs_b = vectorizer.load_from(self.smiles_path, "smiles", limit=0.5, random_state=7)

        self.assertEqual(
            [graph.graph["smiles"] for graph in graphs_a],
            [graph.graph["smiles"] for graph in graphs_b],
        )

    def test_load_from_fractional_limit_without_seed_can_vary(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        with mock.patch(
            "nsppk._make_rng",
            side_effect=[
                _SequenceRng([0.1, 0.9, 0.1, 0.9, 0.1]),
                _SequenceRng([0.9, 0.1, 0.9, 0.1, 0.9]),
            ],
        ):
            graphs_a = vectorizer.load_from(self.smiles_path, "smiles", limit=0.5)
            graphs_b = vectorizer.load_from(self.smiles_path, "smiles", limit=0.5)

        self.assertNotEqual(
            [graph.graph["smiles"] for graph in graphs_a],
            [graph.graph["smiles"] for graph in graphs_b],
        )

    def test_load_from_can_skip_initial_instances(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(self.smiles_path, "smiles", start_after_instance=2, limit=2)

        self.assertEqual(len(graphs), 2)
        self.assertEqual([graph.graph["smiles"] for graph in graphs], ["O", "N"])

    def test_stream_from_batches_transformed_output(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        vectorizer.fit(fit_graphs)

        batches = list(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2))

        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape[0], 2)
        self.assertEqual(batches[1].shape[0], 2)
        self.assertEqual(batches[2].shape[0], 1)

    def test_stream_from_requires_fit_when_attribute_learning_is_enabled(self):
        vectorizer = nsppk.NSPPK(
            parallel=False,
            dense=False,
            node_attribute_key="feat",
            attribute_dim=2,
        )

        def reader(_):
            for value in [0.0, 1.0, 2.0]:
                graph = nx.Graph()
                graph.add_node(0, label="x", feat=np.array([value, value + 1.0, value + 2.0]))
                yield graph

        with self.assertRaises(NotFittedError):
            next(vectorizer.stream_from("ignored", "smiles", reader=reader, batch_size=2))

    def test_node_nsppk_has_same_loading_and_streaming_interface(self):
        vectorizer = nsppk.NodeNSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        self.assertEqual(len(fit_graphs), 2)

        vectorizer.fit(fit_graphs)
        batches = list(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2, limit=3))

        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 1)

    def test_stream_from_can_skip_initial_instances(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        vectorizer.fit(fit_graphs)

        batches = list(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2, start_after_instance=2))

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape[0], 2)
        self.assertEqual(batches[1].shape[0], 1)

    def test_stream_from_does_not_require_explicit_fit_for_default_discrete_labels(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)

        batches = list(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2, limit=3))

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape[0], 2)
        self.assertEqual(batches[1].shape[0], 1)

    def test_load_from_supports_url(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        with _HttpServer(self.data_dir) as server:
            graphs = vectorizer.load_from(f"{server.url}/mols.smi", "smiles", limit=3)

        self.assertEqual(len(graphs), 3)
        self.assertEqual([graph.graph["smiles"] for graph in graphs], ["CCO", "CC", "O"])

    def test_stream_from_supports_url(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        vectorizer.fit(fit_graphs)

        with _HttpServer(self.data_dir) as server:
            batches = list(vectorizer.stream_from(f"{server.url}/mols.smi", "smiles", batch_size=2, limit=4))

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape[0], 2)
        self.assertEqual(batches[1].shape[0], 2)

    def test_stream_from_can_yield_targets(self):
        csv_smiles_path = self.data_dir / "mols_targets.csv"
        csv_smiles_path.write_text(
            "\n".join(
                [
                    "smiles,activity,HIV_active",
                    "CCO,CI,0",
                    "CC,CA,1",
                    "O,CI,0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)
        batches = list(
            vectorizer.stream_from(
                csv_smiles_path,
                "smiles",
                batch_size=2,
                limit=3,
                label_extractor=lambda graph: int(graph.graph["HIV_active"]),
            )
        )

        self.assertEqual(len(batches), 2)
        X0, y0 = batches[0]
        X1, y1 = batches[1]
        self.assertEqual(X0.shape[0], 2)
        self.assertEqual(X1.shape[0], 1)
        np.testing.assert_array_equal(y0, np.array([0, 1]))
        np.testing.assert_array_equal(y1, np.array([0]))

    def test_stream_from_uses_default_name_as_target_when_label_extractor_omitted(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        vectorizer.fit(fit_graphs)

        def reader(_):
            for label in ["a", "b", "c"]:
                graph = nx.Graph()
                graph.add_node(0, label=label)
                graph.graph["name"] = label
                yield graph

        batches = list(
            vectorizer.stream_from(
                "ignored",
                "smiles",
                reader=reader,
                batch_size=2,
                limit=3,
                label_extractor=lambda graph: graph.graph["name"],
            )
        )

        self.assertEqual(len(batches), 2)
        _, y0 = batches[0]
        _, y1 = batches[1]
        np.testing.assert_array_equal(y0, np.array(["a", "b"]))
        np.testing.assert_array_equal(y1, np.array(["c"]))

    def test_verbose_load_from_emits_progress_logs(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        buffer = io.StringIO()

        with redirect_stderr(buffer):
            graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2, verbose=True)

        self.assertEqual(len(graphs), 2)
        log_output = buffer.getvalue()
        self.assertIn("[load_from]", log_output)
        self.assertRegex(log_output, r"loaded=\s*2")

    def test_invalid_limit_raises_value_error(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)

        with self.assertRaises(ValueError):
            vectorizer.load_from(self.smiles_path, "smiles", limit=-1)

        with self.assertRaises(ValueError):
            vectorizer.load_from(self.smiles_path, "smiles", limit=1.5)

    def test_invalid_start_after_instance_raises_value_error(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)

        with self.assertRaises(ValueError):
            vectorizer.load_from(self.smiles_path, "smiles", start_after_instance=-1)

        with self.assertRaises(ValueError):
            vectorizer.load_from(self.smiles_path, "smiles", start_after_instance=1.5)

    def test_missing_rdkit_raises_targeted_import_error(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        with mock.patch("nsppk._require_rdkit", side_effect=ImportError("rdkit missing")):
            with self.assertRaises(ImportError):
                vectorizer.load_from(self.smiles_path, "smiles")

    def test_missing_torch_geometric_raises_targeted_import_error(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        with mock.patch("nsppk._require_torch_geometric", side_effect=ImportError("torch_geometric missing")):
            with self.assertRaises(ImportError):
                vectorizer.load_from("ignored.pt", "pyg_pt")


class BalancedLoadTests(unittest.TestCase):
    @staticmethod
    def _reader(_):
        labels = [0, 0, 0, 0, 1, 1]
        for idx, label in enumerate(labels):
            graph = nx.Graph()
            graph.add_node(0, label=idx)
            graph.graph["name"] = label
            graph.graph["idx"] = idx
            yield graph

    def test_load_from_can_balance_binary_classes(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(
            "ignored",
            "smiles",
            reader=self._reader,
            balance=True,
            random_state=7,
        )

        labels = [graph.graph["name"] for graph in graphs]
        self.assertEqual(len(graphs), 4)
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)
        self.assertEqual(vectorizer.loaded_balance_candidate_size_, 6)
        self.assertEqual(vectorizer.loaded_label_counts_, {0: 4, 1: 2})
        self.assertEqual(vectorizer.loaded_class_weight_, {0: 0.75, 1: 1.5})

    def test_load_from_balance_respects_integer_limit(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(
            "ignored",
            "smiles",
            reader=self._reader,
            balance=True,
            limit=3,
            random_state=7,
        )

        labels = [graph.graph["name"] for graph in graphs]
        self.assertEqual(len(graphs), 3)
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 1)

    def test_load_from_balance_applies_after_start_offset(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        graphs = vectorizer.load_from(
            "ignored",
            "smiles",
            reader=self._reader,
            balance=True,
            start_after_instance=1,
            random_state=7,
        )

        labels = [graph.graph["name"] for graph in graphs]
        idxs = [graph.graph["idx"] for graph in graphs]
        self.assertEqual(len(graphs), 4)
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)
        self.assertNotIn(0, idxs)

    def test_load_from_balance_applies_fractional_limit_before_rebalancing(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        with mock.patch(
            "nsppk._make_rng",
            side_effect=[
                _SequenceRng([0.1, 0.9, 0.1, 0.9, 0.1, 0.9]),
                np.random.default_rng(7),
            ],
        ):
            graphs = vectorizer.load_from(
                "ignored",
                "smiles",
                reader=self._reader,
                balance=True,
                limit=0.5,
            )

        labels = [graph.graph["name"] for graph in graphs]
        self.assertEqual(len(graphs), 2)
        self.assertEqual(labels.count(0), 1)
        self.assertEqual(labels.count(1), 1)
        self.assertEqual(vectorizer.loaded_balance_candidate_size_, 3)
        self.assertEqual(vectorizer.loaded_label_counts_, {0: 2, 1: 1})
        self.assertEqual(vectorizer.loaded_class_weight_, {0: 0.75, 1: 1.5})

    def test_load_from_without_balance_clears_loaded_label_metadata(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)
        vectorizer.load_from(
            "ignored",
            "smiles",
            reader=self._reader,
            balance=True,
            random_state=7,
        )

        vectorizer.load_from(
            "ignored",
            "smiles",
            reader=self._reader,
            balance=False,
        )

        self.assertIsNone(vectorizer.loaded_balance_candidate_size_)
        self.assertIsNone(vectorizer.loaded_label_counts_)
        self.assertIsNone(vectorizer.loaded_class_weight_)

    def test_node_nsppk_exposes_loaded_label_metadata(self):
        vectorizer = nsppk.NodeNSPPK(parallel=False, dense=False)
        vectorizer.load_from(
            "ignored",
            "smiles",
            reader=self._reader,
            balance=True,
            random_state=7,
        )

        self.assertEqual(vectorizer.loaded_balance_candidate_size_, 6)
        self.assertEqual(vectorizer.loaded_label_counts_, {0: 4, 1: 2})
        self.assertEqual(vectorizer.loaded_class_weight_, {0: 0.75, 1: 1.5})


class ConstructorAliasTests(unittest.TestCase):
    def test_nsppk_accepts_short_form_aliases(self):
        vectorizer = nsppk.NSPPK(r=1, d=4, c=1, nbits=8, parallel=False)

        self.assertEqual(vectorizer.radius, 1)
        self.assertEqual(vectorizer.distance, 4)
        self.assertEqual(vectorizer.connector, 1)
        self.assertEqual(vectorizer.r, 1)
        self.assertEqual(vectorizer.d, 4)
        self.assertEqual(vectorizer.c, 1)

    def test_node_nsppk_accepts_short_form_aliases(self):
        vectorizer = nsppk.NodeNSPPK(r=1, d=4, c=1, nbits=8, parallel=False)

        self.assertEqual(vectorizer.radius, 1)
        self.assertEqual(vectorizer.distance, 4)
        self.assertEqual(vectorizer.connector, 1)
        self.assertEqual(vectorizer.nsppk.radius, 1)
        self.assertEqual(vectorizer.nsppk.distance, 4)
        self.assertEqual(vectorizer.nsppk.connector, 1)

    def test_conflicting_aliases_raise_value_error(self):
        with self.assertRaises(ValueError):
            nsppk.NSPPK(radius=2, r=3)

        with self.assertRaises(ValueError):
            nsppk.NodeNSPPK(distance=4, d=5)


if __name__ == "__main__":
    unittest.main()

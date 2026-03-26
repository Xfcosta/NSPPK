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

    def test_stream_from_batches_transformed_output(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        vectorizer.fit(fit_graphs)

        batches = list(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2))

        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape[0], 2)
        self.assertEqual(batches[1].shape[0], 2)
        self.assertEqual(batches[2].shape[0], 1)

    def test_stream_from_requires_fit(self):
        vectorizer = nsppk.NSPPK(parallel=False, dense=False)

        with self.assertRaises(NotFittedError):
            next(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2))

    def test_node_nsppk_has_same_loading_and_streaming_interface(self):
        vectorizer = nsppk.NodeNSPPK(parallel=False, dense=False, nbits=6)
        fit_graphs = vectorizer.load_from(self.smiles_path, "smiles", limit=2)
        self.assertEqual(len(fit_graphs), 2)

        vectorizer.fit(fit_graphs)
        batches = list(vectorizer.stream_from(self.smiles_path, "smiles", batch_size=2, limit=3))

        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 1)

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


if __name__ == "__main__":
    unittest.main()

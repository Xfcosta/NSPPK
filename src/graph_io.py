import io
import sys
import time
import urllib.request
from contextlib import contextmanager
from numbers import Integral, Real
from urllib.parse import urlparse

import networkx as nx
import numpy as np


def ensure_graph_labels(graph, default_label=0):
    """
    Ensure every node and edge has a discrete label.

    NSPPK fundamentally hashes node and edge labels. For plain NetworkX graphs
    with unlabeled nodes or edges, fall back to a shared default label so the
    estimator remains usable without extra preprocessing.
    """
    for node_idx in graph.nodes():
        graph.nodes[node_idx].setdefault("label", default_label)
    for u, v in graph.edges():
        graph.edges[u, v].setdefault("label", default_label)
    return graph


def _is_url(uri):
    scheme = urlparse(str(uri)).scheme
    return scheme in {"http", "https"}


@contextmanager
def _open_binary_uri(uri):
    if _is_url(uri):
        stream = urllib.request.urlopen(str(uri))
        try:
            yield stream
        finally:
            stream.close()
    else:
        with open(uri, "rb") as stream:
            yield stream


@contextmanager
def _open_text_uri(uri, encoding="utf-8"):
    with _open_binary_uri(uri) as stream:
        text_stream = io.TextIOWrapper(stream, encoding=encoding)
        try:
            yield text_stream
        finally:
            text_stream.close()


def _require_rdkit(reader_name):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            f"RDKit is required for '{reader_name}' input. Install rdkit to use this reader."
        ) from exc
    return Chem


def _require_torch_geometric():
    try:
        import torch
        from torch_geometric.data import Batch, Data
        from torch_geometric.utils import to_networkx
    except ImportError as exc:
        raise ImportError(
            "torch and torch_geometric are required for 'pyg_pt' input. "
            "Install both packages to use this reader."
        ) from exc
    return torch, Data, Batch, to_networkx


def _atom_to_discrete_label(atom):
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
    )


def _bond_to_discrete_label(bond):
    return (
        str(bond.GetBondType()),
        int(bond.GetStereo()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    )


def _rdkit_mol_to_nx_graph(mol):
    graph = nx.Graph()

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), label=_atom_to_discrete_label(atom))

    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=_bond_to_discrete_label(bond),
        )

    return graph


def _hashable_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable_value(val)) for key, val in value.items()))
    return value


def _extract_smiles_and_name(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, None

    lowered = stripped.lower()
    if lowered == "smiles" or lowered.startswith("smiles,") or lowered.startswith("smiles\t") or lowered.startswith("smiles "):
        return None, None

    if "," in stripped:
        parts = [part.strip() for part in stripped.split(",")]
        smiles = parts[0].strip('"')
        name = parts[1].strip('"') if len(parts) > 1 and parts[1] else None
        return smiles, name

    parts = stripped.split()
    smiles = parts[0]
    name = parts[1] if len(parts) > 1 else None
    return smiles, name


def _iter_smiles_graphs(uri):
    return _iter_smiles_graphs_with_stats(uri, stats=None)


def _iter_smiles_graphs_with_stats(uri, stats=None):
    Chem = _require_rdkit("smiles")

    with _open_text_uri(uri) as stream:
        for line_number, line in enumerate(stream, start=1):
            smiles, name = _extract_smiles_and_name(line)
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if stats is not None:
                    stats["errors"] += 1
                continue

            graph = _rdkit_mol_to_nx_graph(mol)
            graph.graph["smiles"] = smiles
            graph.graph["source_uri"] = str(uri)
            graph.graph["source_line"] = line_number
            if name is not None:
                graph.graph["name"] = name
            yield graph


def _iter_sdf_graphs(uri):
    return _iter_sdf_graphs_with_stats(uri, stats=None)


def _iter_sdf_graphs_with_stats(uri, stats=None):
    Chem = _require_rdkit("sdf")

    with _open_binary_uri(uri) as stream:
        supplier = Chem.ForwardSDMolSupplier(stream)
        for mol in supplier:
            if mol is None:
                if stats is not None:
                    stats["errors"] += 1
                continue

            graph = _rdkit_mol_to_nx_graph(mol)
            if mol.HasProp("_Name"):
                graph.graph["name"] = mol.GetProp("_Name")
            graph.graph["source_uri"] = str(uri)
            yield graph


def _iter_pyg_pt_graphs(uri):
    return _iter_pyg_pt_graphs_with_stats(uri, stats=None)


def _iter_pyg_pt_graphs_with_stats(uri, stats=None):
    torch, Data, Batch, to_networkx = _require_torch_geometric()

    with _open_binary_uri(uri) as stream:
        payload = io.BytesIO(stream.read())

    obj = torch.load(payload, map_location="cpu")
    if isinstance(obj, Batch):
        data_objects = obj.to_data_list()
    elif isinstance(obj, Data):
        data_objects = [obj]
    else:
        try:
            data_objects = list(obj)
        except TypeError as exc:
            raise TypeError(
                "The 'pyg_pt' reader expects a torch_geometric Data, Batch, or iterable of Data objects."
            ) from exc

    for data in data_objects:
        if not isinstance(data, Data):
            raise TypeError("The 'pyg_pt' reader expects each item to be a torch_geometric.data.Data instance.")

        keys = list(data.keys()) if callable(getattr(data, "keys", None)) else list(data.keys)
        node_attrs = [key for key in keys if key not in {"batch", "ptr"} and data.is_node_attr(key)]
        edge_attrs = [key for key in keys if data.is_edge_attr(key)]
        graph = to_networkx(
            data,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            to_undirected=True,
        )

        for node_idx in graph.nodes():
            if "label" not in graph.nodes[node_idx]:
                if "x" in graph.nodes[node_idx]:
                    graph.nodes[node_idx]["label"] = _hashable_value(graph.nodes[node_idx]["x"])
                else:
                    graph.nodes[node_idx]["label"] = 0
            else:
                graph.nodes[node_idx]["label"] = _hashable_value(graph.nodes[node_idx]["label"])

        for u, v in graph.edges():
            if "label" not in graph.edges[u, v]:
                if "edge_attr" in graph.edges[u, v]:
                    graph.edges[u, v]["label"] = _hashable_value(graph.edges[u, v]["edge_attr"])
                else:
                    graph.edges[u, v]["label"] = 0
            else:
                graph.edges[u, v]["label"] = _hashable_value(graph.edges[u, v]["label"])

        graph.graph["source_uri"] = str(uri)
        yield graph


_BUILTIN_READERS = {
    "smiles": _iter_smiles_graphs_with_stats,
    "sdf": _iter_sdf_graphs_with_stats,
    "pyg_pt": _iter_pyg_pt_graphs_with_stats,
}


def _validate_graph(graph):
    if not isinstance(graph, nx.Graph):
        raise TypeError(
            "Readers must yield networkx.Graph instances. "
            f"Received {type(graph).__name__}."
        )
    return graph


def _validate_limit(limit):
    if limit is None:
        return

    if isinstance(limit, bool):
        raise ValueError("limit must be None, a non-negative integer, or a float strictly between 0 and 1.")

    if isinstance(limit, Integral):
        if limit < 0:
            raise ValueError("Integer limit must be non-negative.")
        return

    if isinstance(limit, Real):
        if not 0 < float(limit) < 1:
            raise ValueError("Float limit must be strictly between 0 and 1.")
        return

    raise ValueError("limit must be None, a non-negative integer, or a float strictly between 0 and 1.")


def _make_rng(random_state):
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _apply_limit(graph_iterable, limit=None, random_state=None):
    _validate_limit(limit)

    if limit is None:
        yield from graph_iterable
        return

    if isinstance(limit, Integral):
        remaining = int(limit)
        for graph in graph_iterable:
            if remaining <= 0:
                break
            yield graph
            remaining -= 1
        return

    probability = float(limit)
    rng = _make_rng(random_state)
    for graph in graph_iterable:
        if rng.random() < probability:
            yield graph


def _normalize_label_extractor(label_extractor):
    if label_extractor is None:
        return lambda graph: graph.graph["name"]
    if not callable(label_extractor):
        raise TypeError("label_extractor must be callable when provided.")
    return label_extractor


def _select_balanced_indices(labels, target_size, rng):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("balance=True requires at least two classes.")

    class_indices = {label: np.flatnonzero(labels == label) for label in unique_labels}
    minority_count = min(len(indices) for indices in class_indices.values())
    max_balanced_size = minority_count * unique_labels.size

    if target_size is None:
        target_size = max_balanced_size
    else:
        target_size = min(int(target_size), len(labels))

    if target_size <= 0:
        return np.empty(0, dtype=int)

    per_class = min(minority_count, target_size // unique_labels.size)
    remainder = target_size - per_class * unique_labels.size

    selected_parts = []
    leftover_parts = []
    for label in unique_labels:
        indices = np.array(class_indices[label], copy=True)
        chosen = rng.choice(indices, size=per_class, replace=False) if per_class else np.empty(0, dtype=int)
        selected_parts.append(chosen)
        if remainder > 0:
            remaining = np.setdiff1d(indices, chosen, assume_unique=False)
            leftover_parts.append(remaining)

    selected_idx = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=int)

    if remainder > 0:
        leftovers = np.concatenate([part for part in leftover_parts if len(part)]) if leftover_parts else np.empty(0, dtype=int)
        if leftovers.size:
            extra_size = min(remainder, leftovers.size)
            extra_idx = rng.choice(leftovers, size=extra_size, replace=False)
            selected_idx = np.concatenate([selected_idx, extra_idx])

    return np.sort(selected_idx)


def _materialize_loaded_graphs(uri, source_type, reader=None, limit=None, random_state=None, verbose=False, balance=False, label_extractor=None):
    if balance and isinstance(limit, Real) and not isinstance(limit, Integral):
        raise ValueError("balance=True requires limit to be None or a non-negative integer.")

    graph_iterable = _iter_loaded_graphs(
        uri,
        source_type,
        reader=reader,
        limit=None if balance else limit,
        random_state=random_state,
        verbose=verbose,
        mode="load",
    )
    graphs = list(graph_iterable)

    if not balance:
        return graphs

    rng = _make_rng(random_state)
    extractor = _normalize_label_extractor(label_extractor)
    labels = np.asarray([extractor(graph) for graph in graphs])
    selected_idx = _select_balanced_indices(labels, target_size=limit, rng=rng)
    return [graphs[idx] for idx in selected_idx]


def _iter_loaded_graphs(uri, source_type, reader=None, limit=None, random_state=None, verbose=False, mode="load", log_every=100, log_stream=None):
    stats = {
        "seen": 0,
        "loaded": 0,
        "errors": 0,
        "verbose": verbose,
        "mode": mode,
        "stream": sys.stderr if log_stream is None else log_stream,
    }
    start_time = time.perf_counter()
    stats["timer"] = lambda: time.perf_counter() - start_time

    if reader is None:
        normalized_type = str(source_type).strip().lower()
        if normalized_type not in _BUILTIN_READERS:
            available = ", ".join(sorted(_BUILTIN_READERS))
            raise ValueError(f"Unsupported source type '{source_type}'. Available built-in types: {available}.")
        graph_iterable = _BUILTIN_READERS[normalized_type](uri, stats=stats)
    else:
        graph_iterable = reader(uri)

    validated_graphs = (_validate_graph(graph) for graph in graph_iterable)
    limited_graphs = _apply_limit(validated_graphs, limit=limit, random_state=random_state)

    try:
        for graph in limited_graphs:
            stats["seen"] += 1
            stats["loaded"] += 1
            if verbose and (stats["loaded"] == 1 or stats["loaded"] % log_every == 0):
                _emit_stream_stats(stats, final=False)
            yield graph
    finally:
        if verbose:
            _emit_stream_stats(stats, final=True)


def _batched_graphs(graph_iterable, batch_size):
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    batch = []
    for graph in graph_iterable:
        batch.append(graph)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _format_stream_stats(stats):
    elapsed = max(stats["timer"](), 1e-12)
    loaded = stats["loaded"]
    seen = stats["seen"]
    errors = stats["errors"]
    sec_per_graph = elapsed / loaded if loaded else float("nan")
    graphs_per_sec = loaded / elapsed if loaded else 0.0
    return (
        f"seen={seen:>7d}  "
        f"loaded={loaded:>7d}  "
        f"errors={errors:>5d}  "
        f"elapsed={elapsed:>8.2f}s  "
        f"graphs/sec={graphs_per_sec:>8.2f}  "
        f"sec/graph={sec_per_graph:>10.6f}"
    )


def _emit_stream_stats(stats, final=False):
    if not stats["verbose"]:
        return

    prefix = "[load_from]" if stats["mode"] == "load" else "[stream_from]"
    stage = "final" if final else "progress"
    print(f"{prefix} {stage}: {_format_stream_stats(stats)}", file=stats["stream"])
    stats["stream"].flush()

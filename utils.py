import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


PASTEL_PALETTE = [
    "#FFB3BA",
    "#FFDFBA",
    "#FFFFBA",
    "#BAFFC9",
    "#BAE1FF",
    "#E4C1F9",
    "#FFD6E0",
    "#CDE7BE",
    "#B5EAD7",
    "#C7CEEA",
]


def _node_colors(graph, node_color_key):
    if node_color_key is None:
        return [PASTEL_PALETTE[0] for _ in graph.nodes()]

    labels = [graph.nodes[node].get(node_color_key, 0) for node in graph.nodes()]
    unique_labels = list(dict.fromkeys(labels))
    color_map = {
        label: PASTEL_PALETTE[idx % len(PASTEL_PALETTE)]
        for idx, label in enumerate(unique_labels)
    }
    return [color_map[label] for label in labels]


def plot_graphs(graphs, titles=None, num_graphs_per_row=7, node_color_key='label', size=3.5):
    """
    Plot a list of graphs using a Kamada-Kawai layout and bright pastel node colors.

    Args:
        graphs (list[nx.Graph]): Graphs to plot.
        titles (list[str] | None): Optional title for each graph.
        num_graphs_per_row (int, default=7): Number of graphs per subplot row.
        node_color_key (str | None, default='label'): Node attribute used to group pastel colors.
        size (float, default=4.0): Base subplot size used in ``figsize=(size * cols, size * rows)``.
    """
    if not graphs:
        raise ValueError("graphs must contain at least one graph.")
    if num_graphs_per_row < 1:
        raise ValueError("num_graphs_per_row must be at least 1.")
    if size <= 0:
        raise ValueError("size must be positive.")

    if titles is None:
        titles = [None] * len(graphs)
    if len(titles) != len(graphs):
        raise ValueError("titles must be None or have the same length as graphs.")

    cols = min(num_graphs_per_row, len(graphs))
    rows = math.ceil(len(graphs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, graph, title in zip(axes, graphs, titles):
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(
            graph,
            pos,
            ax=ax,
            node_color=_node_colors(graph, node_color_key),
            node_size=150,
            edge_color="#4F5D75",
            edgecolors="#2F4858",
            linewidths=1.5,
            with_labels=True,
            font_size=8,
        )
        if title is not None:
            ax.set_title(title)
        ax.set_axis_off()

    for ax in axes[len(graphs):]:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

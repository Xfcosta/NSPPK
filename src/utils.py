import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


DEFAULT_PALETTE = [
    "#FF6B6B",
    "#FFD166",
    "#F4F06E",
    "#7AE582",
    "#6ED3FF",
    "#A78BFA",
    "#FF8FAB",
    "#B8E986",
    "#5EEAD4",
    "#C4B5FD",
]


def quadratic_loess(x, y, frac=0.8, grid=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of the same length.")
    if len(x) < 3:
        raise ValueError("quadratic_loess requires at least three data points.")
    if not 0 < frac <= 1:
        raise ValueError("frac must lie in (0, 1].")

    if grid is None:
        grid = x
    grid = np.asarray(grid, dtype=float)
    span = max(3, int(np.ceil(frac * len(x))))
    fitted = []

    for x0 in grid:
        distances = np.abs(x - x0)
        bandwidth = np.partition(distances, span - 1)[span - 1]
        if bandwidth == 0:
            weights = (distances == 0).astype(float)
        else:
            scaled = np.clip(distances / bandwidth, 0, 1)
            weights = (1 - scaled ** 3) ** 3
        design = np.column_stack([np.ones_like(x), x - x0, (x - x0) ** 2])
        weighted_design = design * weights[:, None]
        beta, *_ = np.linalg.lstsq(weighted_design, y * weights, rcond=None)
        fitted.append(beta[0])

    return np.column_stack([grid, np.asarray(fitted)])


def plot_series_with_band_loess(
    ax,
    x,
    y,
    *,
    y_std=None,
    frac=0.8,
    color=None,
    marker="o",
    alpha=0.6,
    linewidth=3,
    label=None,
    band_alpha=0.15,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of the same length.")

    line, = ax.plot(x, y, marker=marker, alpha=alpha, color=color, label=label)

    if y_std is not None:
        y_std = np.asarray(y_std, dtype=float)
        if y_std.shape != y.shape:
            raise ValueError("y_std must have the same shape as y.")
        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=line.get_color(),
            alpha=band_alpha,
            label="_nolegend_",
        )

    smoothed = quadratic_loess(x, y, frac=frac)
    ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=line.get_color(),
        linewidth=linewidth,
        label="_nolegend_",
    )
    return line


def _node_colors(graph, node_color_key, cmap=None):
    if node_color_key is None:
        return [DEFAULT_PALETTE[0] for _ in graph.nodes()]

    labels = [graph.nodes[node].get(node_color_key, 0) for node in graph.nodes()]
    unique_labels = list(dict.fromkeys(labels))
    if cmap is None:
        color_map = {
            label: DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
            for idx, label in enumerate(unique_labels)
        }
    else:
        cmap_obj = plt.get_cmap(cmap)
        denom = max(len(unique_labels) - 1, 1)
        color_map = {
            label: cmap_obj(idx / denom)
            for idx, label in enumerate(unique_labels)
        }
    return [color_map[label] for label in labels]


def _graph_pos(graph, use_pos):
    if use_pos:
        node_pos = {
            node: graph.nodes[node].get("pos")
            for node in graph.nodes()
        }
        if all(pos is not None for pos in node_pos.values()):
            return node_pos
    return nx.kamada_kawai_layout(graph)


def plot_graphs(
    graphs,
    titles=None,
    num_graphs_per_row=7,
    node_color_key='label',
    cmap=None,
    size=3.75,
    use_pos=False,
):
    """
    Plot a list of graphs using stored node positions or a Kamada-Kawai layout.

    Args:
        graphs (list[nx.Graph]): Graphs to plot.
        titles (list[str] | None): Optional title for each graph.
        num_graphs_per_row (int, default=7): Number of graphs per subplot row.
        node_color_key (str | None, default='label'): Node attribute used to group colors.
        cmap (str | matplotlib.colors.Colormap | None, default=None): Optional colormap used for node groups.
        size (float, default=4.0): Base subplot size used in ``figsize=(size * cols, size * rows)``.
        use_pos (bool, default=False): Use each node's ``pos`` attribute when present for all nodes.
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
        pos = _graph_pos(graph, use_pos=use_pos)
        nx.draw(
            graph,
            pos,
            ax=ax,
            node_color=_node_colors(graph, node_color_key, cmap=cmap),
            node_size=150,
            edge_color="#000000",
            edgecolors="#000000",
            width=1.5,
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


def plot_graph_importance(
    graphs,
    titles=None,
    importance_key="importance",
    cols=4,
    size=4.5,
    use_pos=False,
):
    """
    Plot graphs with node and edge colors driven by an importance attribute.

    Args:
        graphs (list[nx.Graph]): Graphs to plot.
        titles (list[str] | None): Optional title for each graph.
        importance_key (str, default='importance'): Node and edge attribute used for coloring.
        cols (int, default=4): Number of subplot columns.
        size (float, default=4.5): Base subplot size used in ``figsize=(size * cols, size * rows)``.
        use_pos (bool, default=False): Use each node's ``pos`` attribute when present for all nodes.
    """
    if not graphs:
        raise ValueError("graphs must contain at least one graph.")
    if cols < 1:
        raise ValueError("cols must be at least 1.")
    if size <= 0:
        raise ValueError("size must be positive.")

    if titles is None:
        titles = [None] * len(graphs)
    if len(titles) != len(graphs):
        raise ValueError("titles must be None or have the same length as graphs.")

    rows = math.ceil(len(graphs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.cm.viridis

    for ax, graph, title in zip(axes, graphs, titles):
        pos = _graph_pos(graph, use_pos=use_pos)
        node_values = np.asarray(
            [graph.nodes[node].get(importance_key, 0.0) for node in graph.nodes()],
            dtype=float,
        )
        edge_values = [
            graph.edges[u, v].get(importance_key, 0.0)
            for u, v in graph.edges()
        ]
        edge_widths = 1.0 + 2.5 * np.asarray(edge_values, dtype=float)

        nx.draw(
            graph,
            pos,
            ax=ax,
            node_color=node_values,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            node_size=220,
            edge_color=edge_values,
            edge_cmap=cmap,
            edge_vmin=0.0,
            edge_vmax=1.0,
            width=edge_widths,
            edgecolors="#111111",
            linewidths=1.0,
            with_labels=True,
            font_size=8,
        )
        if title is not None:
            ax.set_title(title)
        ax.set_axis_off()

    for ax in axes[len(graphs):]:
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, label="Importance")
    plt.show()

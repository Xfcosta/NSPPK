"""Fetch and compare results from HTML leaderboards."""

from __future__ import annotations

from html.parser import HTMLParser
import math
import re
from numbers import Real
from urllib.parse import urldefrag
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_UNCERTAINTY = re.compile(
    r"(?:±|\+/-)\s*(\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


class _HTMLTableParser(HTMLParser):
    """Collect HTML tables and the nearest heading id before each table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.section = None
        self._table = None
        self._row = None
        self._cell = None
        self._cell_tag = None
        self._heading_tag = None
        self._heading_parts = []
        self._heading_id = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("name"):
            self.section = attrs["name"]
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_tag = tag
            self._heading_parts = []
            self._heading_id = attrs.get("id")
        elif tag == "table":
            self._table = {"section": self.section, "rows": []}
        elif self._table is not None and tag == "tr":
            self._row = []
        elif self._row is not None and tag in {"th", "td"}:
            self._cell_tag = tag
            self._cell = []

    def handle_endtag(self, tag):
        if self._cell is not None and tag == self._cell_tag:
            value = " ".join("".join(self._cell).split())
            self._row.append((self._cell_tag, value))
            self._cell = None
            self._cell_tag = None
        elif self._row is not None and tag == "tr":
            if self._row:
                self._table["rows"].append(self._row)
            self._row = None
        elif self._table is not None and tag == "table":
            self.tables.append(self._table)
            self._table = None
        elif self._heading_tag == tag:
            if self._heading_id and self.section is None:
                self.section = self._heading_id
            self._heading_tag = None
            self._heading_id = None
            self._heading_parts = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        if self._heading_tag is not None:
            self._heading_parts.append(data)


def _parse_score(value):
    """Parse a score and optional ``±``/``+/-`` standard deviation."""
    text = str(value).replace(",", "")
    match = _NUMBER.search(text)
    if match is None:
        return None, None
    score = float(match.group(0))
    spread = _UNCERTAINTY.search(text)
    return score, (float(spread.group(1)) if spread else None)


def _table_records(table):
    rows = table["rows"]
    header_index = next(
        (i for i, row in enumerate(rows) if any(kind == "th" for kind, _ in row)),
        None,
    )
    if header_index is None:
        return [], None
    headers = [value for _, value in rows[header_index]]
    records = []
    for row in rows[header_index + 1 :]:
        values = [value for _, value in row]
        if len(values) != len(headers):
            continue
        records.append(dict(zip(headers, values)))
    return records, headers


def fetch_leaderboard(
    url,
    *,
    section=None,
    metric_column=None,
    timeout=20,
    user_agent="NSPPK leaderboard utility/1.0",
):
    """Fetch one HTML leaderboard table and return all rows as dictionaries.

    ``section`` selects a table by its preceding heading id (for example,
    ``"ogbg-molhiv"``). If omitted, a URL fragment is used when present. If
    ``metric_column`` is omitted, the first column with ``test`` in its name is
    used. Every returned row keeps the source text and adds normalized
    ``_score`` and ``_reported_std`` fields for that metric.
    """
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("url must be an HTTP or HTTPS URL.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds.")

    clean_url, fragment = urldefrag(url)
    section = section or fragment or None
    request = Request(clean_url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = _HTMLTableParser()
    parser.feed(html)
    candidates = []
    for table in parser.tables:
        records, headers = _table_records(table)
        if records:
            candidates.append((table["section"], records, headers))
    if section is not None:
        section_key = section.casefold().lstrip("#")
        candidates = [
            candidate
            for candidate in candidates
            if str(candidate[0] or "").casefold() == section_key
        ]
    if not candidates:
        raise ValueError(f"No leaderboard table found for section {section!r} at {url!r}.")
    if len(candidates) > 1 and metric_column is None:
        raise ValueError(
            "The page has multiple leaderboard tables; pass section or metric_column."
        )

    if metric_column is None:
        table = candidates[0]
        metric_column = next(
            (header for header in table[2] if "test" in header.casefold()), None
        )
    else:
        desired = metric_column.casefold()
        matching = [
            candidate
            for candidate in candidates
            if any(header.casefold() == desired for header in candidate[2])
        ]
        if not matching:
            matching = [
                candidate
                for candidate in candidates
                if any(desired in header.casefold() for header in candidate[2])
            ]
        candidates = matching
        if not candidates:
            raise ValueError(
                f"Metric column {metric_column!r} was not found in the selected table."
            )
        if len(candidates) > 1:
            raise ValueError("Metric column matches multiple tables; pass section.")
        metric_column = next(
            header
            for header in candidates[0][2]
            if header.casefold() == desired or desired in header.casefold()
        )

    _, records, headers = candidates[0]
    if metric_column is None:
        raise ValueError("Could not identify a test metric column; pass metric_column.")
    normalized = []
    for record in records:
        score, spread = _parse_score(record.get(metric_column, ""))
        normalized.append(
            {
                **record,
                "_section": candidates[0][0],
                "_metric_column": metric_column,
                "_score": score,
                "_reported_std": spread,
            }
        )
    return normalized


def compare_leaderboard_result(
    results,
    score,
    *,
    result_name="Our result",
    result_std=None,
    higher_is_better=True,
    comparison_note=None,
):
    """Summarize a score's rank and standardized position among leaderboard rows."""
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
        raise ValueError("score must be a finite number.")
    score = float(score)
    if result_std is not None and (
        isinstance(result_std, bool)
        or not isinstance(result_std, Real)
        or not math.isfinite(float(result_std))
        or result_std < 0
    ):
        raise ValueError("result_std must be a finite, non-negative number or None.")
    if result_std is not None:
        result_std = float(result_std)
    if not results:
        raise ValueError("results must contain at least one leaderboard row.")

    valid_scores = np.asarray(
        [row["_score"] for row in results if row.get("_score") is not None], dtype=float
    )
    if valid_scores.size == 0:
        raise ValueError("No numeric leaderboard scores were found.")
    mean = float(np.mean(valid_scores))
    std = float(np.std(valid_scores, ddof=1)) if valid_scores.size > 1 else 0.0
    better = valid_scores > score if higher_is_better else valid_scores < score
    ties = valid_scores == score
    rank = int(np.count_nonzero(better)) + 1
    percentile = float(
        100 * (np.count_nonzero(~better & ~ties) + 0.5 * np.count_nonzero(ties)) / valid_scores.size
    )
    return {
        "result_name": result_name,
        "result_score": float(score),
        "result_std": None if result_std is None else float(result_std),
        "leaderboard_n": int(valid_scores.size),
        "leaderboard_mean": mean,
        "leaderboard_std": std,
        "z_score_vs_leaderboard": (float(score) - mean) / std if std > 0 else None,
        "naive_rank": rank,
        "percentile_vs_leaderboard": percentile,
        "higher_is_better": bool(higher_is_better),
        "comparison_note": comparison_note,
    }


def plot_leaderboard_comparison(
    results,
    summary,
    *,
    method_column="Method",
    title="Leaderboard comparison",
):
    """Plot each leaderboard mean ± reported standard deviation and a result."""
    metric_column = results[0]["_metric_column"]
    rows = [row for row in results if row.get("_score") is not None]
    rows.sort(key=lambda row: row["_score"], reverse=summary["higher_is_better"])
    scores = np.asarray([row["_score"] for row in rows], dtype=float)
    spreads = np.asarray(
        [row.get("_reported_std") or 0.0 for row in rows], dtype=float
    )
    ranks = np.arange(1, len(rows) + 1)
    better_count = int(
        np.count_nonzero(
            scores > summary["result_score"]
            if summary["higher_is_better"]
            else scores < summary["result_score"]
        )
    )
    result_y = better_count + 0.5

    fig, ax = plt.subplots(figsize=(11, max(5, min(14, 0.27 * len(rows) + 2))))
    mean, std = summary["leaderboard_mean"], summary["leaderboard_std"]
    ax.axvspan(mean - std, mean + std, color="#2563eb", alpha=0.10, label="Leaderboard mean ± 1 SD")
    ax.axvline(mean, color="#2563eb", linestyle="--", label=f"Leaderboard mean = {mean:.4f}")
    ax.errorbar(
        scores,
        ranks,
        xerr=spreads,
        fmt="o",
        ms=5,
        color="#64748b",
        ecolor="#cbd5e1",
        capsize=2,
        label="Leaderboard result ± reported SD",
        zorder=2,
    )
    if summary["result_std"] is not None:
        ax.errorbar(
            summary["result_score"],
            result_y,
            xerr=summary["result_std"],
            fmt="none",
            ecolor="#dc2626",
            capsize=4,
            zorder=3,
        )
    ax.scatter(
        [summary["result_score"]],
        [result_y],
        marker="*",
        s=220,
        color="#dc2626",
        edgecolor="white",
        linewidth=0.8,
        label=f"{summary['result_name']} = {summary['result_score']:.4f} (naive rank {summary['naive_rank']})",
        zorder=4,
    )
    if method_column in rows[0]:
        for index, row in enumerate(rows):
            if index < 5:
                ax.annotate(
                    str(row[method_column]),
                    (row["_score"], index + 1),
                    xytext=(5, 2),
                    textcoords="offset points",
                    fontsize=8,
                    color="#475569",
                )
    ax.set(
        xlabel=metric_column,
        ylabel="Leaderboard position (best first)",
        title=title,
        ylim=(len(rows) + 1, 0),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    note = summary.get("comparison_note")
    if note:
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#7f1d1d")
        fig.subplots_adjust(bottom=0.13)
    else:
        fig.tight_layout()
    return fig


def check_leaderboard(
    url,
    score,
    *,
    section=None,
    metric_column=None,
    result_name="Our result",
    result_std=None,
    higher_is_better=True,
    comparison_note=None,
    plot_title="Leaderboard comparison",
    timeout=20,
):
    """Fetch a leaderboard, summarize a score against it, and plot the comparison.

    Returns a dictionary with all parsed ``results``, a numeric ``summary``,
    and a matplotlib ``figure``. The reported rank is descriptive; confirm that
    dataset split and evaluation protocol match before treating it as official.
    """
    results = fetch_leaderboard(
        url,
        section=section,
        metric_column=metric_column,
        timeout=timeout,
    )
    summary = compare_leaderboard_result(
        results,
        score,
        result_name=result_name,
        result_std=result_std,
        higher_is_better=higher_is_better,
        comparison_note=comparison_note,
    )
    figure = plot_leaderboard_comparison(results, summary, title=plot_title)
    return {"results": results, "summary": summary, "figure": figure}

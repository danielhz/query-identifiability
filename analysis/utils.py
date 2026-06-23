"""
Shared helpers for analysis scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Architecture display names and plot order
# ---------------------------------------------------------------------------

ARCH_ORDER = [
    "majority_vote",
    "vanilla_overlap",
    "mlp",
    "set_transformer",
    "gnn_og",
    "closure_aware",
]

ARCH_LABEL = {
    "majority_vote": "MajVote",
    "vanilla_overlap": "VanillaOv",
    "mlp": "MLP",
    "set_transformer": "SetTransf.",
    "gnn_og": "GNN-OG",
    "closure_aware": "CA",
}

# Color-blind-safe palette (Wong 2011)
COLORS = {
    "certified": "#0072B2",  # blue
    "noncertified": "#D55E00",  # orange-red
    "neutral": "#999999",  # gray
    "highlight": "#009E73",  # green
}

LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_results(path: Path) -> tuple[list[dict], dict]:
    """Return (results, meta) from an experiment JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["results"], data.get("meta", {})


def latest_result(output_dir: Path, prefix: str) -> Path:
    """Return the most-recently written JSON matching <prefix>_*.json."""
    files = sorted(output_dir.glob(f"{prefix}_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No {prefix}_*.json found in {output_dir}. Run the corresponding experiment first."
        )
    return files[-1]


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------


def set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", **kwargs)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def grouped_stats(
    records: list[dict],
    group_keys: list[str],
    value_key: str = "test_accuracy",
) -> dict[tuple, tuple[float, float]]:
    """
    Group records by the values of group_keys and return (mean, std) of value_key.
    Returns dict keyed by tuple of group values.
    """
    from collections import defaultdict

    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(float(r[value_key]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in buckets.items()}


def to_latex_table(
    rows: list[tuple],
    col_headers: list[str],
    row_headers: list[str],
    caption: str = "",
    label: str = "",
    fmt: str = "{:.3f}",
) -> str:
    """Build a simple booktabs LaTeX table from a 2-D list of rows."""
    n_cols = len(col_headers)
    col_spec = "l" + "c" * n_cols
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    & " + " & ".join(col_headers) + r" \\",
        r"    \midrule",
    ]
    for rh, row in zip(row_headers, rows):
        cells = " & ".join(fmt.format(v) for v in row)
        lines.append(f"    {rh} & {cells} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)

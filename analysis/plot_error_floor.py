"""
E1 analysis — Error Floor and Certificate Accuracy (RQ1, RQ2).

Produces:
  fig:error-floor   Grouped bar chart: test accuracy per architecture,
                    split by query type (certified vs. non-certified).
                    A horizontal dashed line marks the 0.5 error floor.
  tab:certificate   LaTeX table of mean ± std accuracy by architecture and
                    query type (written to stdout with --latex).

Usage
-----
    # Use the most-recent E1 output in results/:
    python -m analysis.plot_error_floor

    # Explicit input file:
    python -m analysis.plot_error_floor --input results/e1_error_floor_20260520_120000.json

    # Also print the LaTeX table:
    python -m analysis.plot_error_floor --latex

    # Save figure to a custom path:
    python -m analysis.plot_error_floor --output figures/error_floor.pdf
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.utils import (
    ARCH_LABEL,
    ARCH_ORDER,
    COLORS,
    latest_result,
    load_results,
    save_fig,
    set_style,
)


def _annotate_n_tuples(records: list[dict], sources: list[dict]) -> list[dict]:
    """Annotate each record with _n_tuples from its source meta."""
    idx = 0
    annotated = list(records)
    for src in sources:
        n_a = len(src["architectures"])
        n_cert = src["n_cert_queries"]
        n_noncert = src["n_noncert_queries"]
        n_seeds = len(src["seeds"])
        n_records = n_a * (n_cert + n_noncert) * n_seeds
        for i in range(idx, idx + n_records):
            annotated[i] = dict(annotated[i], _n_tuples=src["n_tuples"])
        idx += n_records
    return annotated


def _collect(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Return acc[arch][query_type] = list of test_balanced_accuracy values (NaN excluded)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        val = r.get("test_balanced_accuracy")
        if val is not None and not math.isnan(val):
            acc[r["model"]][r["query_type"]].append(val)
    return acc


def plot(records: list[dict], output: Path) -> plt.Figure:
    set_style()
    acc = _collect(records)

    archs = [a for a in ARCH_ORDER if a in acc]
    x = np.arange(len(archs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for offset, qtype, color, label in [
        (-w / 2, "certified", COLORS["certified"], "Certified"),
        (w / 2, "noncertified", COLORS["noncertified"], "Non-certified"),
    ]:
        means = [np.mean(acc[a].get(qtype, [0.0])) for a in archs]
        stds = [np.std(acc[a].get(qtype, [0.0])) for a in archs]
        ax.bar(
            x + offset,
            means,
            w,
            yerr=stds,
            label=label,
            color=color,
            alpha=0.85,
            capsize=3,
            error_kw={"linewidth": 0.8},
        )

    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--", label="Error floor (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL.get(a, a) for a in archs], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.legend(loc="upper left", framealpha=0.8)
    fig.tight_layout()

    save_fig(fig, output)
    return fig


def latex_table(records: list[dict]) -> str:
    acc = _collect(records)
    archs = [a for a in ARCH_ORDER if a in acc]
    qtypes = ["certified", "noncertified"]
    row_headers = [ARCH_LABEL.get(a, a) for a in archs]

    rows = []
    for a in archs:
        row = []
        for qt in qtypes:
            vals = acc[a].get(qt, [])
            if vals:
                row.append(f"{np.mean(vals):.3f} \\pm {np.std(vals):.3f}")
            else:
                row.append("—")
        rows.append(row)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Architecture & Certified & Non-certified \\",
        r"    \midrule",
    ]
    for rh, row in zip(row_headers, rows):
        lines.append(f"    {rh} & {row[0]} & {row[1]} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Mean$\pm$std test accuracy by architecture and query identifiability."
        r" Certified queries are identifiable (Theorem~1); non-certified queries trigger"
        r" the $\geq\!\tfrac{1}{2}$ error floor (Theorem~5).}",
        r"  \label{tab:certificate}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot E1: error floor")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to e1_*.json (defaults to most-recent in --results-dir)",
    )
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: results/figures/error_floor.pdf)",
    )
    p.add_argument("--latex", action="store_true", help="Print LaTeX table to stdout")
    args = p.parse_args(argv)

    src = args.input or latest_result(args.results_dir, "e1_error_floor")
    out = args.output or args.results_dir / "figures" / "error_floor.pdf"

    records, meta = load_results(src)
    print(f"Loaded {len(records)} records from {src}")

    # Annotate with n_tuples from source meta and filter to n_tuples == 10
    if "sources" in meta:
        records = _annotate_n_tuples(records, meta["sources"])
        before = len(records)
        records = [r for r in records if r.get("_n_tuples") == 10]
        print(f"Filtered to n_tuples==10: {len(records)}/{before} records")
    elif "n_tuples" in meta:
        # Single-source file: n_tuples is a top-level meta field
        if meta["n_tuples"] != 10:
            print(
                f"Warning: file has n_tuples={meta['n_tuples']}, not 10 — results may not match expected subset"
            )
        else:
            print(f"Single-source file with n_tuples=10, using all {len(records)} records")

    plot(records, out)

    if args.latex:
        print("\n% --- tab:certificate ---")
        print(latex_table(records))


if __name__ == "__main__":
    main()

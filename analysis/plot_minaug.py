"""
E3 analysis — Greedy-MinAug in Practice (RQ4).

Produces fig:minaug: two-panel figure.

  Panel A — Approximation ratio CDF:
    X: approximation ratio (greedy_size / optimal_size), starting at 1.0
    Y: cumulative fraction of instances
    One curve per n_attrs value (larger n_attrs → more complex instances).
    A mass at x=1.0 corresponds to instances solved optimally.

  Panel B — Runtime scaling:
    X: n_attrs (number of attributes)
    Y: median runtime in microseconds (log scale)
    Two series: greedy (fast, polynomial) vs. brute-force (slow, exponential).
    Error bars show IQR.

Usage
-----
    python -m analysis.plot_minaug
    python -m analysis.plot_minaug --input results/e3_minaug_...json
    python -m analysis.plot_minaug --output figures/minaug.pdf
    python -m analysis.plot_minaug --latex   # print summary statistics table
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.utils import (
    COLORS,
    latest_result,
    load_results,
    save_fig,
    set_style,
)


def _collect(records: list[dict]) -> dict[int, list[dict]]:
    """Group records by n_attrs."""
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_n[r["n_attrs"]].append(r)
    return by_n


def plot(records: list[dict], output: Path) -> plt.Figure:
    set_style()
    by_n = _collect(records)
    n_attrs_sorted = sorted(by_n)

    fig, (ax_cdf, ax_rt) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # ------------------------------------------------------------------
    # Panel A: CDF of approximation ratios
    # ------------------------------------------------------------------
    cmap = plt.get_cmap("Blues")
    n_curves = len(n_attrs_sorted)

    for idx, n in enumerate(n_attrs_sorted):
        ratios = np.array([r["approx_ratio"] for r in by_n[n]])
        # Sort and compute empirical CDF
        xs = np.sort(ratios)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        # Prepend x=1 at y=0 so curve starts from the left
        xs = np.concatenate([[1.0], xs])
        ys = np.concatenate([[0.0], ys])
        color = cmap(0.35 + 0.55 * idx / max(1, n_curves - 1))
        ax_cdf.step(xs, ys, where="post", color=color, label=f"$n={n}$", linewidth=1.3)

    ax_cdf.set_xlabel("Approximation ratio")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_xlim(left=0.98)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.legend(title="Attrs", fontsize=7, title_fontsize=7, loc="lower right", framealpha=0.85)
    ax_cdf.set_title("(a) Ratio CDF")

    # ------------------------------------------------------------------
    # Panel B: Runtime scaling
    # ------------------------------------------------------------------
    greedy_med, greedy_q1, greedy_q3 = [], [], []
    bf_med, bf_q1, bf_q3 = [], [], []

    for n in n_attrs_sorted:
        g_times = np.array([r["greedy_time_us"] for r in by_n[n]])
        b_times = np.array([r["bf_time_us"] for r in by_n[n]])
        greedy_med.append(np.median(g_times))
        greedy_q1.append(np.percentile(g_times, 25))
        greedy_q3.append(np.percentile(g_times, 75))
        bf_med.append(np.median(b_times))
        bf_q1.append(np.percentile(b_times, 25))
        bf_q3.append(np.percentile(b_times, 75))

    xs = np.array(n_attrs_sorted, dtype=float)

    def _errbar(ax, x, med, q1, q3, color, label, ls="-"):
        med, q1, q3 = np.array(med), np.array(q1), np.array(q3)
        ax.plot(x, med, color=color, linestyle=ls, marker="o", markersize=3, label=label)
        ax.fill_between(x, q1, q3, alpha=0.15, color=color)

    _errbar(ax_rt, xs, greedy_med, greedy_q1, greedy_q3, COLORS["certified"], "Greedy", ls="-")
    _errbar(ax_rt, xs, bf_med, bf_q1, bf_q3, COLORS["noncertified"], "Brute-force", ls="--")

    ax_rt.set_yscale("log")
    ax_rt.set_xlabel("Number of attributes $n$")
    ax_rt.set_ylabel("Runtime (µs, log scale)")
    ax_rt.set_xticks(n_attrs_sorted)
    ax_rt.legend(fontsize=7, framealpha=0.85)
    ax_rt.set_title("(b) Runtime scaling")

    fig.tight_layout(pad=1.2)
    save_fig(fig, output)
    return fig


def latex_stats(records: list[dict]) -> str:
    """Print per-n_attrs summary statistics as a LaTeX table."""
    by_n = _collect(records)
    n_vals = sorted(by_n)

    lines = [
        r"\begin{table}[t]",
        r"  \centering \small",
        r"  \begin{tabular}{rrrrrr}",
        r"    \toprule",
        r"    $n$ & Trials & Mean ratio & Max ratio & \% Optimal"
        r" & Greedy $\mu$s \\",
        r"    \midrule",
    ]
    for n in n_vals:
        recs = by_n[n]
        ratios = [r["approx_ratio"] for r in recs]
        pct_opt = 100 * sum(r["is_optimal"] for r in recs) / len(recs)
        g_med = np.median([r["greedy_time_us"] for r in recs])
        lines.append(
            f"    {n} & {len(recs)} & {np.mean(ratios):.4f} & "
            f"{np.max(ratios):.4f} & {pct_opt:.1f}\\% & {g_med:.1f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Greedy-MinAug approximation quality and runtime by"
        r" attribute count. Mean ratio $\approx 1$ confirms near-optimal"
        r" solutions in practice (Theorem~7).}",
        r"  \label{tab:minaug}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot E3: MinAug approximation quality")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args(argv)

    src = args.input or latest_result(args.results_dir, "e3_minaug")
    out = args.output or args.results_dir / "figures" / "minaug.pdf"

    records, meta = load_results(src)
    print(f"Loaded {len(records)} records from {src}")

    plot(records, out)

    if args.latex:
        print("\n% --- tab:minaug ---")
        print(latex_stats(records))


if __name__ == "__main__":
    main()

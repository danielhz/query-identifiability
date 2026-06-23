"""
Real-world analysis — BibInteg + WDC-Product benchmark.

Produces fig:realworld: two-panel figure.

  Panel A — Accuracy by dataset+query+architecture:
    Grouped bar chart.  One group per (dataset, query).  Bars = architectures.
    Certified queries filled solid, non-certified hatched.
    Horizontal dashed line at 0.5.

  Panel B — Certified vs. non-certified accuracy summary:
    Box/scatter comparison across all certified vs. non-certified (query, arch) pairs.
    Demonstrates the theory-predicted gap in real data.

Usage
-----
    python -m analysis.plot_realworld
    python -m analysis.plot_realworld --input results/e4_realworld_...json
    python -m analysis.plot_realworld --output figures/realworld.pdf --latex
"""

from __future__ import annotations

import argparse
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

_HATCH_NONCERT = "////"


def _collect(records: list[dict]) -> dict[tuple, dict[str, list[float]]]:
    """Return acc[(dataset,query,certified)][arch] = list of accuracies."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["dataset"], r["query"], r["certified"])
        acc[key][r["model"]].append(r["test_accuracy"])
    return acc


def _group_label(ds: str, q: str) -> str:
    return f"{ds[:3].upper()}\n{q}"


def plot(records: list[dict], output: Path) -> plt.Figure:
    set_style()
    acc = _collect(records)

    # Sort groups: certified first, then non-certified; within each by dataset+query
    groups = sorted(acc.keys(), key=lambda k: (not k[2], k[0], k[1]))
    archs = [a for a in ARCH_ORDER if any(a in acc[g] for g in groups)]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # ------------------------------------------------------------------
    # Panel A: grouped bar chart — one group per query, bars = archs
    # ------------------------------------------------------------------
    n_groups = len(groups)
    n_archs = len(archs)
    w = 0.7 / n_archs  # bar width
    x_pos = np.arange(n_groups)

    for i, arch in enumerate(archs):
        offsets = (i - n_archs / 2 + 0.5) * w
        means = [np.mean(acc[g].get(arch, [np.nan])) for g in groups]
        stds = [np.std(acc[g].get(arch, [0.0])) for g in groups]
        fill_colors = [COLORS["certified"] if g[2] else COLORS["noncertified"] for g in groups]
        hatches = [None if g[2] else _HATCH_NONCERT for g in groups]

        for j, (m, s, fc, h) in enumerate(zip(means, stds, fill_colors, hatches)):
            ax_a.bar(
                x_pos[j] + offsets,
                m,
                w,
                yerr=s,
                capsize=2,
                color=fc,
                alpha=0.80,
                hatch=h,
                error_kw={"linewidth": 0.7},
                label=ARCH_LABEL.get(arch, arch) if j == 0 else "_nolegend_",
            )

    ax_a.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(
        [_group_label(g[0], g[1]) for g in groups],
        fontsize=6,
        ha="center",
    )
    ax_a.set_ylim(0, 1.10)
    ax_a.set_ylabel("Test accuracy")
    ax_a.legend(loc="upper right", fontsize=6, framealpha=0.85, ncol=2, handlelength=1.2)
    ax_a.set_title("(a) Per-query accuracy")

    # Certified / non-certified indicator
    for j, g in enumerate(groups):
        marker = "C" if g[2] else "X"
        ax_a.text(
            x_pos[j],
            -0.08,
            marker,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            transform=ax_a.get_xaxis_transform(),
            color=COLORS["certified"] if g[2] else COLORS["noncertified"],
        )

    # ------------------------------------------------------------------
    # Panel B: certified vs. non-certified accuracy scatter/box
    # ------------------------------------------------------------------
    cert_accs = [r["test_accuracy"] for r in records if r["certified"]]
    noncert_accs = [r["test_accuracy"] for r in records if not r["certified"]]

    positions = [1, 2]
    data = [cert_accs, noncert_accs]
    labels = ["Certified", "Non-cert."]
    colors = [COLORS["certified"], COLORS["noncertified"]]

    for pos, vals, lab, col in zip(positions, data, labels, colors):
        if not vals:
            continue
        bp = ax_b.boxplot(
            vals,
            positions=[pos],
            widths=0.4,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        # Overlay scatter
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax_b.scatter(np.full(len(vals), pos) + jitter, vals, color=col, alpha=0.35, s=8, zorder=3)

    ax_b.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.set_ylim(0, 1.10)
    ax_b.set_ylabel("Test accuracy")
    ax_b.set_title("(b) Theory gap (real data)")

    fig.tight_layout(pad=1.2)
    save_fig(fig, output)
    return fig


def latex_table(records: list[dict]) -> str:
    acc = _collect(records)
    groups = sorted(acc.keys(), key=lambda k: (not k[2], k[0], k[1]))
    archs = [a for a in ARCH_ORDER if any(a in acc[g] for g in groups)]

    col_headers = [f"{g[0][:3]}/{g[1]}" + (" ✓" if g[2] else " ✗") for g in groups]
    lines = [
        r"\begin{table}[t]",
        r"  \centering \small",
        r"  \begin{tabular}{l" + "c" * len(groups) + "}",
        r"    \toprule",
        "    Arch. & " + " & ".join(col_headers) + r" \\",
        r"    \midrule",
    ]
    for arch in archs:
        row_vals = []
        for g in groups:
            vs = acc[g].get(arch, [])
            row_vals.append(f"{np.mean(vs):.3f}" if vs else "—")
        lines.append(f"    {ARCH_LABEL.get(arch, arch)} & " + " & ".join(row_vals) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Mean test accuracy on real-world datasets."
        r" \checkmark = certified query (theory predicts $\geq$ chance);"
        r" $\times$ = non-certified (theory predicts $\approx 0.5$).}",
        r"  \label{tab:realworld}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot real-world benchmark results")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args(argv)

    src = args.input or latest_result(args.results_dir, "e4_realworld")
    out = args.output or args.results_dir / "figures" / "realworld.pdf"

    records, meta = load_results(src)
    print(f"Loaded {len(records)} records from {src}")

    plot(records, out)

    if args.latex:
        print("\n% --- tab:realworld ---")
        print(latex_table(records))


if __name__ == "__main__":
    main()

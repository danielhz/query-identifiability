"""
Ablation analysis — FD completeness ρ sweep.

Produces fig:ablation: two-panel figure.

  Panel A — Accuracy vs. ρ (all architectures):
    X: ρ (fraction of FDs revealed to the predictor)
    Y: mean test accuracy ± 1σ over seeds
    One line per architecture.
    Vertical dashed line marks the ρ at which the query flips to certified.
    Horizontal dashed line at 0.5 (error floor).

  Panel B — CA vs. best learned model (highlights the theory-gap):
    For each ρ: bar chart comparing CA with the single best non-CA architecture.
    Shows that CA is uniquely hurt by incomplete FD knowledge, while learned
    predictors are less affected (for this query the raw feature already
    contains some information about the label).

Usage
-----
    python -m analysis.plot_ablation
    python -m analysis.plot_ablation --input results/e1_ablation_...json
    python -m analysis.plot_ablation --output figures/ablation.pdf --latex
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
    LINE_STYLES,
    latest_result,
    load_results,
    save_fig,
    set_style,
)

# Architectures to show in Panel A (drop CA for cleaner multi-line comparison)
_PANEL_A_ARCHS = ["majority_vote", "vanilla_overlap", "mlp", "set_transformer", "gnn_og"]
_CA = "closure_aware"


def _collect(records: list[dict]) -> dict[str, dict[float, list[float]]]:
    """Return acc[arch][rho] = list of accuracies."""
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][round(r["rho"], 4)].append(r["test_accuracy"])
    return acc


def _cert_rho(records: list[dict]) -> float | None:
    """Return the smallest ρ at which certified is True."""
    rho_cert = {round(r["rho"], 4): r["certified"] for r in records}
    for rho in sorted(rho_cert):
        if rho_cert[rho]:
            return rho
    return None


def plot(records: list[dict], output: Path) -> plt.Figure:
    set_style()
    acc = _collect(records)
    rhos = sorted({round(r["rho"], 4) for r in records})
    x = np.array(rhos)
    cert_r = _cert_rho(records)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # ------------------------------------------------------------------
    # Panel A: per-architecture accuracy vs. ρ  (excluding CA)
    # ------------------------------------------------------------------
    archs_a = [a for a in _PANEL_A_ARCHS if a in acc]

    for i, arch in enumerate(archs_a):
        means = np.array([np.mean(acc[arch].get(r, [np.nan])) for r in rhos])
        stds = np.array([np.std(acc[arch].get(r, [0.0])) for r in rhos])
        c = f"C{i}"
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        ax_a.plot(
            x,
            means,
            color=c,
            linestyle=ls,
            marker="o",
            markersize=3,
            label=ARCH_LABEL.get(arch, arch),
        )
        ax_a.fill_between(x, means - stds, means + stds, alpha=0.10, color=c)

    ax_a.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    if cert_r is not None:
        ax_a.axvline(
            cert_r - 0.5 / len(_ALL_FDS_PROXY),
            color=COLORS["highlight"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="Certified →",
        )

    ax_a.set_xlabel("FD completeness ρ")
    ax_a.set_ylabel("Test accuracy")
    ax_a.set_xticks(rhos)
    ax_a.set_xticklabels([f"{r:.2f}" for r in rhos], fontsize=7)
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(loc="upper left", fontsize=7, framealpha=0.85, ncol=1, handlelength=1.5)
    ax_a.set_title("(a) All architectures vs. ρ")

    # ------------------------------------------------------------------
    # Panel B: CA vs. best non-CA architecture
    # ------------------------------------------------------------------
    w = 0.32
    ca_means, ca_stds = [], []
    best_means, best_stds = [], []
    best_arch_labels = []

    for rho in rhos:
        # CA
        ca_v = acc[_CA].get(rho, [0.0]) if _CA in acc else [0.0]
        ca_means.append(np.mean(ca_v))
        ca_stds.append(np.std(ca_v))

        # Best non-CA architecture at this ρ
        best_m, best_s, best_lab = 0.0, 0.0, "—"
        for a in _PANEL_A_ARCHS:
            if a in acc:
                vs = acc[a].get(rho, [])
                if vs and np.mean(vs) > best_m:
                    best_m = float(np.mean(vs))
                    best_s = float(np.std(vs))
                    best_lab = ARCH_LABEL.get(a, a)
        best_means.append(best_m)
        best_stds.append(best_s)
        best_arch_labels.append(best_lab)

    x_b = np.arange(len(rhos))
    ax_b.bar(
        x_b - w / 2,
        ca_means,
        w,
        yerr=ca_stds,
        label="CA",
        color=COLORS["certified"],
        alpha=0.85,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )
    ax_b.bar(
        x_b + w / 2,
        best_means,
        w,
        yerr=best_stds,
        label="Best learned",
        color=COLORS["noncertified"],
        alpha=0.85,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    ax_b.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([f"ρ={r:.2f}" for r in rhos], rotation=20, ha="right", fontsize=7)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Test accuracy")
    ax_b.legend(fontsize=7, framealpha=0.85)
    ax_b.set_title("(b) CA vs. best learned model")

    fig.tight_layout(pad=1.2)
    save_fig(fig, output)
    return fig


# Small constant used only for the vertical line position heuristic
_ALL_FDS_PROXY = [0, 1, 2, 3, 4]  # len = n_fds+1 levels


def latex_table(records: list[dict]) -> str:
    acc = _collect(records)
    rhos = sorted({round(r["rho"], 4) for r in records})
    archs = [a for a in ARCH_ORDER if a in acc]

    col_headers = [f"ρ={r:.2f}" for r in rhos]
    lines = [
        r"\begin{table}[t]",
        r"  \centering \small",
        r"  \begin{tabular}{l" + "c" * len(rhos) + "}",
        r"    \toprule",
        "    Arch. & " + " & ".join(col_headers) + r" \\",
        r"    \midrule",
    ]
    for arch in archs:
        row_vals = []
        for rho in rhos:
            vs = acc[arch].get(rho, [])
            row_vals.append(f"{np.mean(vs):.3f}" if vs else "—")
        lines.append(f"    {ARCH_LABEL.get(arch, arch)} & " + " & ".join(row_vals) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Mean test accuracy as a function of FD completeness $\rho$."
        r" The world is always generated with full $\Sigma$; only the predictor's"
        r" FD knowledge is varied. CA requires $\rho=1$ to certify the query;"
        r" at $\rho<1$ it abstains (acc $\approx 0.5$).}",
        r"  \label{tab:ablation}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot ablation: FD completeness")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args(argv)

    src = args.input or latest_result(args.results_dir, "e1_ablation")
    out = args.output or args.results_dir / "figures" / "ablation.pdf"

    records, meta = load_results(src)
    print(f"Loaded {len(records)} records from {src}")

    plot(records, out)

    if args.latex:
        print("\n% --- tab:ablation ---")
        print(latex_table(records))


if __name__ == "__main__":
    main()

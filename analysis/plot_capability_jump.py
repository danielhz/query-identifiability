"""
E2 analysis — Capability Jump Under Interface Augmentation (RQ3).

Produces:
  fig:capability-jump   One line per architecture showing test accuracy at each
                        augmentation step.  The line should be flat near 0.5
                        until the certification step, then jump abruptly.
                        Shaded ±1σ bands show variance over seeds.

Usage
-----
    python -m analysis.plot_capability_jump
    python -m analysis.plot_capability_jump --input results/e2_capability_jump_...json
    python -m analysis.plot_capability_jump --output figures/capability_jump.pdf
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


def _collect(
    records: list[dict],
) -> tuple[list[int], list[str], dict[str, dict[int, list[float]]]]:
    """
    Returns:
      steps      : sorted list of step indices
      step_labels: label for each step
      acc        : acc[arch][step] = list of test_balanced_accuracy values (NaN excluded)
    """
    import math

    step_label_map: dict[int, str] = {}
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        step = r["step"]
        step_label_map[step] = r.get("step_label", str(step))
        val = r.get("test_balanced_accuracy")
        if val is not None and not math.isnan(val):
            acc[r["model"]][step].append(val)
    steps = sorted(step_label_map)
    labels = [step_label_map[s] for s in steps]
    return steps, labels, acc


def _cert_step(records: list[dict]) -> int | None:
    """Return the first step at which certified becomes True."""
    step_cert: dict[int, bool] = {}
    for r in records:
        step_cert[r["step"]] = r["certified"]
    for s in sorted(step_cert):
        if step_cert[s]:
            return s
    return None


def plot(records: list[dict], output: Path) -> plt.Figure:
    set_style()
    steps, step_labels, acc = _collect(records)
    cert_step = _cert_step(records)

    archs = [a for a in ARCH_ORDER if a in acc]
    x = np.array(steps, dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for i, arch in enumerate(archs):
        means = np.array(
            [np.mean(acc[arch].get(s, [])) if acc[arch].get(s) else np.nan for s in steps]
        )
        stds = np.array([np.std(acc[arch].get(s, [])) if acc[arch].get(s) else 0.0 for s in steps])
        color = f"C{i}"
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        label = ARCH_LABEL.get(arch, arch)
        ax.plot(x, means, linestyle=ls, color=color, label=label, marker="o", markersize=3)
        ax.fill_between(x, means - stds, means + stds, alpha=0.12, color=color)

    # Error floor
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.7, label="Floor (0.5)")

    # Certification boundary
    if cert_step is not None:
        ax.axvline(
            cert_step - 0.5,
            color=COLORS["highlight"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="Certified →",
        )

    ax.set_xticks(steps)
    ax.set_xticklabels(step_labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Interface augmentation step")
    ax.legend(loc="upper left", framealpha=0.85, ncol=2, fontsize=7, handlelength=1.5)
    fig.tight_layout()

    save_fig(fig, output)
    return fig


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot E2: capability jump")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    src = args.input or latest_result(args.results_dir, "e2_capability_jump")
    out = args.output or args.results_dir / "figures" / "capability_jump.pdf"

    records, meta = load_results(src)
    print(f"Loaded {len(records)} records from {src}")

    plot(records, out)


if __name__ == "__main__":
    main()

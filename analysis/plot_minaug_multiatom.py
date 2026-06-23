"""
plot_minaug_multiatom.py — Analysis for multi-atom MinAug experiment.

Reads the most-recent e_minaug_multiatom_*.json from results/ and produces:
  pgfdata/minaug_cdf_k{1,2,3,4}.csv   — approximation ratio CDFs per |B_Q|
  pgfdata/minaug_sizes.csv             — mean solution sizes per |B_Q|
  figures/minaug.tikz                  — updated two-panel tikz figure

Panel (a): CDF of greedy approximation ratio per |B_Q| value.
Panel (b): Mean solution sizes — greedy vs singleton-only vs optimal — per |B_Q|.

Usage
-----
  python -m analysis.plot_minaug_multiatom
  python -m analysis.plot_minaug_multiatom --input results/e_minaug_multiatom_...json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.utils import latest_result


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("results") or data.get("records") or data


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def build_cdf(ratios: list[float]) -> list[dict]:
    """Return (ratio, cdf) pairs suitable for pgfplots const-plot."""
    sorted_r = sorted(ratios)
    n = len(sorted_r)
    rows = []
    # prepend a point at the minimum ratio with cdf=0
    rows.append({"ratio": sorted_r[0], "cdf": 0.0})
    for i, r in enumerate(sorted_r):
        rows.append({"ratio": r, "cdf": round((i + 1) / n, 6)})
    return rows


def generate_cdf_csvs(records: list[dict], pgfdata_dir: Path) -> dict[int, list[float]]:
    """Write one CDF CSV per n_atoms value. Returns {n_atoms: ratios}."""
    by_k: dict[int, list[float]] = {}
    for r in records:
        k = r["n_atoms"]
        by_k.setdefault(k, []).append(r["approx_ratio"])

    for k, ratios in sorted(by_k.items()):
        rows = build_cdf(ratios)
        path = pgfdata_dir / f"minaug_cdf_k{k}.csv"
        write_csv(path, rows, ["ratio", "cdf"])
        print(f"  Wrote {path}  ({len(ratios)} records)")

    return by_k


def generate_sizes_csv(records: list[dict], pgfdata_dir: Path) -> list[dict]:
    """Write mean solution sizes per n_atoms."""
    by_k: dict[int, list] = {}
    for r in records:
        k = r["n_atoms"]
        by_k.setdefault(k, []).append(r)

    rows = []
    for k in sorted(by_k):
        batch = by_k[k]
        rows.append(
            {
                "n_atoms": k,
                "greedy_mean": round(float(np.mean([r["greedy_size"] for r in batch])), 4),
                "singleton_mean": round(float(np.mean([r["singleton_size"] for r in batch])), 4),
                "optimal_mean": round(float(np.mean([r["optimal_size"] for r in batch])), 4),
            }
        )

    path = pgfdata_dir / "minaug_sizes.csv"
    write_csv(path, rows, ["n_atoms", "greedy_mean", "singleton_mean", "optimal_mean"])
    print(f"  Wrote {path}")
    return rows


TIKZ_TEMPLATE = r"""% MinAug — ratio CDF per |B_Q| and solution-size comparison
% Data: pgfdata/minaug_cdf_k{1..4}.csv, pgfdata/minaug_sizes.csv
\definecolor{maB0}{RGB}{198,219,239}
\definecolor{maB1}{RGB}{107,174,214}
\definecolor{maB2}{RGB}{33,113,181}
\definecolor{maB3}{RGB}{8,48,107}
\definecolor{maGy}{RGB}{102,102,102}
\definecolor{maOr}{RGB}{230,97,1}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.5cm},
]
%% Panel (a): CDF of greedy approximation ratio per |B_Q|
\nextgroupplot[
  width=0.48\columnwidth,
  height=5.5cm,
  xlabel={Approx.\ ratio},
  ylabel={Cumulative fraction},
  xmin=0.98, xmax=1.55,
  ymin=0, ymax=1.02,
  xticklabel style={font=\footnotesize},
  yticklabel style={font=\footnotesize},
  xlabel style={font=\small},
  ylabel style={font=\small},
  legend style={at={(0.97,0.08)}, anchor=south east, font=\footnotesize,
                legend cell align=left, fill=white, fill opacity=0.9,
                draw opacity=1, text opacity=1},
  grid=major, grid style={dashed, gray!30},
  tick align=outside,
  title={(a) Ratio CDF by $|\mathcal{B}_Q|$},
  title style={font=\small},
]
\addplot[maB0, thick, const plot]
  table[x=ratio, y=cdf, col sep=comma]{pgfdata/minaug_cdf_k1.csv};
\addlegendentry{$|\mathcal{B}_Q|{=}1$}
\addplot[maB1, thick, const plot]
  table[x=ratio, y=cdf, col sep=comma]{pgfdata/minaug_cdf_k2.csv};
\addlegendentry{$|\mathcal{B}_Q|{=}2$}
\addplot[maB2, thick, const plot]
  table[x=ratio, y=cdf, col sep=comma]{pgfdata/minaug_cdf_k3.csv};
\addlegendentry{$|\mathcal{B}_Q|{=}3$}
\addplot[maB3, thick, const plot]
  table[x=ratio, y=cdf, col sep=comma]{pgfdata/minaug_cdf_k4.csv};
\addlegendentry{$|\mathcal{B}_Q|{=}4$}
\draw[black, thin, dashed] (axis cs:1.0,0) -- (axis cs:1.0,1.02);
%% Panel (b): Mean solution sizes — greedy vs singleton-only vs optimal
\nextgroupplot[
  width=0.48\columnwidth,
  height=5.5cm,
  xlabel={$|\mathcal{B}_Q|$},
  ylabel={Mean actions selected},
  xmin=0.5, xmax=4.5,
  xtick={1,2,3,4},
  ymin=0,
  xticklabel style={font=\footnotesize},
  yticklabel style={font=\footnotesize},
  xlabel style={font=\small},
  ylabel style={font=\small},
  legend style={at={(0.03,0.97)}, anchor=north west, font=\footnotesize,
                legend cell align=left, fill=white, fill opacity=0.9,
                draw opacity=1, text opacity=1},
  grid=major, grid style={dashed, gray!30},
  tick align=outside,
  title={(b) Solution size vs baseline},
  title style={font=\small},
]
\addplot[maB3, thick, mark=o, mark size=2pt]
  table[x=n_atoms, y=greedy_mean, col sep=comma]{pgfdata/minaug_sizes.csv};
\addlegendentry{Greedy}
\addplot[maOr, thick, mark=square, mark size=2pt, dashed]
  table[x=n_atoms, y=singleton_mean, col sep=comma]{pgfdata/minaug_sizes.csv};
\addlegendentry{Singleton-only}
\addplot[maGy, thick, mark=triangle, mark size=2pt, dotted]
  table[x=n_atoms, y=optimal_mean, col sep=comma]{pgfdata/minaug_sizes.csv};
\addlegendentry{Optimal}
\end{groupplot}
\end{tikzpicture}
"""


def write_tikz(tikz_path: Path) -> None:
    tikz_path.parent.mkdir(parents=True, exist_ok=True)
    tikz_path.write_text(TIKZ_TEMPLATE)
    print(f"  Wrote {tikz_path}")


def print_summary(records: list[dict]) -> None:
    print("\n=== Summary by |B_Q| ===")
    by_k: dict[int, list] = {}
    for r in records:
        by_k.setdefault(r["n_atoms"], []).append(r)
    for k in sorted(by_k):
        batch = by_k[k]
        ratios = [r["approx_ratio"] for r in batch]
        s_ratios = [r["singleton_ratio"] for r in batch]
        pct_opt = 100 * sum(r["greedy_is_optimal"] for r in batch) / len(batch)
        print(
            f"  |B_Q|={k}: greedy ratio mean={np.mean(ratios):.4f} "
            f"max={np.max(ratios):.4f} | {pct_opt:.1f}% optimal | "
            f"singleton ratio mean={np.mean(s_ratios):.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Multi-atom MinAug plot generator")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--pgfdata-dir", type=Path, default=None)
    p.add_argument("--tikz-dir", type=Path, default=None)
    args = p.parse_args(argv)

    result_path = args.input or latest_result(args.results_dir, "e_minaug_multiatom")
    print(f"Reading {result_path}")
    records = load_records(result_path)
    print(f"  {len(records)} records loaded.")

    base = Path(__file__).resolve().parents[1]  # repo root
    pgfdata_dir = args.pgfdata_dir or (base / "pgfdata")
    tikz_dir = args.tikz_dir or (base / "figures")

    generate_cdf_csvs(records, pgfdata_dir)
    generate_sizes_csv(records, pgfdata_dir)
    write_tikz(tikz_dir / "minaug.tikz")
    print_summary(records)
    print("\nDone. Recompile main.tex to see updated figure.")


if __name__ == "__main__":
    main()

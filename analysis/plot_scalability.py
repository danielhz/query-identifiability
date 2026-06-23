"""
Scalability analysis — Certification and MinAug runtime across schema sizes.

Reads the most-recent e_scalability_*.json from results/ and produces:
  - pgfdata/scalability.csv   (wide format, used by scalability.tikz)
  - figures/scalability.tikz  (pgfplots two-panel figure)

Usage
-----
  python -m analysis.plot_scalability
  python -m analysis.plot_scalability --input results/e_scalability_...json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.utils import latest_result

FDS_SHOW = [10, 100, 1000]  # n_fds values to highlight as separate lines


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("records") or data.get("results") or data


def build_wide(records: list[dict]) -> dict[tuple[int, int], dict]:
    """Key: (n_attrs, n_fds_requested)."""
    out = {}
    for r in records:
        key = (r["n_attrs"], r["n_fds_requested"])
        out[key] = r
    return out


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def generate_csvs(records: list[dict], pgfdata_dir: Path) -> Path:
    """
    Build scalability.csv in wide format:
      n_attrs, cert_f10, cert_f100, cert_f1000, minaug_f10, minaug_f100, minaug_f1000

    Returns path to the written CSV.
    """
    wide = build_wide(records)
    n_attrs_vals = sorted({r["n_attrs"] for r in records})

    rows = []
    for n in n_attrs_vals:
        row: dict = {"n_attrs": n}
        for f in FDS_SHOW:
            rec = wide.get((n, f), {})
            row[f"cert_f{f}"] = rec.get("cert_median_ms", "")
            row[f"minaug_f{f}"] = rec.get("minaug_median_ms", "")
        rows.append(row)

    cols = ["n_attrs"] + [f"cert_f{f}" for f in FDS_SHOW] + [f"minaug_f{f}" for f in FDS_SHOW]
    out_path = pgfdata_dir / "scalability.csv"
    write_csv(out_path, rows, cols)
    print(f"  Wrote {out_path}")
    return out_path


TIKZ_TEMPLATE = r"""% Scalability — Certification and MinAug runtime vs schema size
% Data: pgfdata/scalability.csv
\definecolor{scBlue0}{RGB}{189,215,231}
\definecolor{scBlue1}{RGB}{107,174,214}
\definecolor{scBlue2}{RGB}{33,113,181}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.3cm},
]
%% Panel (a): Certification runtime
\nextgroupplot[
  width=0.48\columnwidth,
  height=4.8cm,
  xlabel={$|\mathit{Attr}|$},
  ylabel={Median time (ms)},
  xmin=5, xmax=1100,
  ymin=0,
  xticklabel style={font=\footnotesize},
  yticklabel style={font=\footnotesize},
  xlabel style={font=\small},
  ylabel style={font=\small},
  legend style={at={(0.97,0.97)}, anchor=north east, font=\footnotesize,
                legend cell align=left, fill=white, fill opacity=0.9,
                draw opacity=1, text opacity=1},
  grid=major, grid style={dashed, gray!30},
  tick align=outside,
  xtick={10,100,250,500,1000},
  xticklabels={10,100,250,500,1000},
  title={(a) \textsc{CheckCert}},
  title style={font=\small},
]
\addplot[color=scBlue0,mark=o,mark size=1.5pt,line width=0.8pt]
  table[col sep=comma,x=n_attrs,y=cert_f10]{pgfdata/scalability.csv};
\addlegendentry{$|\Sigma|=10$}
\addplot[color=scBlue1,mark=square,mark size=1.5pt,line width=0.8pt]
  table[col sep=comma,x=n_attrs,y=cert_f100]{pgfdata/scalability.csv};
\addlegendentry{$|\Sigma|=100$}
\addplot[color=scBlue2,mark=triangle,mark size=1.5pt,line width=0.8pt]
  table[col sep=comma,x=n_attrs,y=cert_f1000]{pgfdata/scalability.csv};
\addlegendentry{$|\Sigma|=1000$}
%% Panel (b): MinAug runtime
\nextgroupplot[
  width=0.48\columnwidth,
  height=4.8cm,
  xlabel={$|\mathit{Attr}|$},
  ylabel={Median time (ms)},
  xmin=5, xmax=1100,
  ymin=0,
  xticklabel style={font=\footnotesize},
  yticklabel style={font=\footnotesize},
  xlabel style={font=\small},
  ylabel style={font=\small},
  legend style={at={(0.97,0.97)}, anchor=north east, font=\footnotesize,
                legend cell align=left, fill=white, fill opacity=0.9,
                draw opacity=1, text opacity=1},
  grid=major, grid style={dashed, gray!30},
  tick align=outside,
  xtick={10,100,250,500,1000},
  xticklabels={10,100,250,500,1000},
  title={(b) \textsc{Greedy-MinAug}},
  title style={font=\small},
]
\addplot[color=scBlue0,mark=o,mark size=1.5pt,line width=0.8pt]
  table[col sep=comma,x=n_attrs,y=minaug_f10]{pgfdata/scalability.csv};
\addlegendentry{$|\Sigma|=10$}
\addplot[color=scBlue1,mark=square,mark size=1.5pt,line width=0.8pt]
  table[col sep=comma,x=n_attrs,y=minaug_f100]{pgfdata/scalability.csv};
\addlegendentry{$|\Sigma|=100$}
\addplot[color=scBlue2,mark=triangle,mark size=1.5pt,line width=0.8pt]
  table[col sep=comma,x=n_attrs,y=minaug_f1000]{pgfdata/scalability.csv};
\addlegendentry{$|\Sigma|=1000$}
\end{groupplot}
\end{tikzpicture}
"""


def write_tikz(tikz_path: Path) -> None:
    tikz_path.parent.mkdir(parents=True, exist_ok=True)
    tikz_path.write_text(TIKZ_TEMPLATE)
    print(f"  Wrote {tikz_path}")


def print_summary(records: list[dict]) -> None:
    wide = build_wide(records)
    n_max = max(r["n_attrs"] for r in records)
    f_max = max(r["n_fds_requested"] for r in records)
    worst = wide.get((n_max, f_max), {})
    print(f"\nWorst-case (n_attrs={n_max}, n_fds={f_max}):")
    print(
        f"  cert  median = {worst.get('cert_median_ms')} ms, max = {worst.get('cert_max_ms')} ms"
    )
    print(
        f"  minaug median = {worst.get('minaug_median_ms')} ms, "
        f"max = {worst.get('minaug_max_ms')} ms"
    )
    print(f"  minaug feasible = {worst.get('minaug_n')}/100 queries")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Scalability plot generator")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to e_scalability_*.json (default: latest in results/)",
    )
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--pgfdata-dir",
        type=Path,
        default=None,
        help="Where to write CSV (default: pgfdata/ under the repo root)",
    )
    p.add_argument(
        "--tikz-dir",
        type=Path,
        default=None,
        help="Where to write tikz (default: figures/ under the repo root)",
    )
    args = p.parse_args(argv)

    result_path = args.input or latest_result(args.results_dir, "e_scalability")
    print(f"Reading {result_path}")
    records = load_records(result_path)
    print(f"  {len(records)} records loaded.")

    # Default output dirs: pgfdata/ and figures/ under the repo root
    base = Path(__file__).resolve().parents[1]  # repo root
    pgfdata_dir = args.pgfdata_dir or (base / "pgfdata")
    tikz_dir = args.tikz_dir or (base / "figures")

    generate_csvs(records, pgfdata_dir)
    write_tikz(tikz_dir / "scalability.tikz")
    print_summary(records)
    print("\nDone. Now recompile main.tex to include the new figure.")


if __name__ == "__main__":
    main()

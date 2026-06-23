"""
Export CSV data files for pgfplots/TikZ figures.

Reads canonical combined result files from results/ and writes CSVs to
pgfdata/ under the repo root (override with --output-dir).

Usage
-----
    python -m analysis.export_pgfdata
    python -m analysis.export_pgfdata --results-dir results --output-dir pgfdata
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Architecture order (for error_floor.csv)
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

# Architecture order for capability_jump (column order in CSV)
CAP_JUMP_ARCHS = [
    "mlp",
    "set_transformer",
    "gnn_og",
    "closure_aware",
    "vanilla_overlap",
    "majority_vote",
]
CAP_JUMP_SHORT = ["mlp", "st", "gnn", "ca", "vov", "mjv"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def annotate_n_tuples(records: list[dict], sources: list[dict]) -> list[dict]:
    """Annotate each record with _n_tuples from its source meta (E1 combined)."""
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    """Return (mean, std) excluding NaN values."""
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        return (float("nan"), float("nan"))
    return (float(np.mean(clean)), float(np.std(clean)))


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_capability_jump(results_dir: Path, output_dir: Path) -> None:
    """Export pgfdata/capability_jump.csv."""
    path = results_dir / "e2_capability_jump_combined.json"
    data = load_json(path)
    records = data["results"]
    print(f"[capability_jump] Loaded {len(records)} records from {path}")

    # Collect test_balanced_accuracy by (step, arch)
    acc: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in records:
        val = r.get("test_balanced_accuracy")
        if val is not None and not math.isnan(val):
            acc[(r["step"], r["model"])].append(val)

    steps = sorted(set(s for s, _ in acc))

    fieldnames = ["step"]
    for short in CAP_JUMP_SHORT:
        fieldnames += [f"{short}_mean", f"{short}_std"]

    rows = []
    for step in steps:
        row: dict = {"step": step}
        for arch, short in zip(CAP_JUMP_ARCHS, CAP_JUMP_SHORT):
            vals = acc.get((step, arch), [])
            mean, std = _mean_std(vals)
            row[f"{short}_mean"] = f"{mean:.6f}"
            row[f"{short}_std"] = f"{std:.6f}"
        rows.append(row)

    out = output_dir / "capability_jump.csv"
    write_csv(out, fieldnames, rows)

    # Print head
    print("  Head:")
    for r in rows[:3]:
        print("   ", r)


def export_error_floor(results_dir: Path, output_dir: Path) -> None:
    """Export pgfdata/error_floor.csv."""
    path = results_dir / "e1_error_floor_combined.json"
    data = load_json(path)
    records = data["results"]
    meta = data.get("meta", {})
    print(f"[error_floor] Loaded {len(records)} records from {path}")

    # Annotate with n_tuples and filter to n_tuples == 10
    if "sources" in meta:
        records = annotate_n_tuples(records, meta["sources"])
        before = len(records)
        records = [r for r in records if r.get("_n_tuples") == 10]
        print(f"  Filtered to n_tuples==10: {len(records)}/{before} records")

    # Collect test_balanced_accuracy by (arch, query_type)
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        val = r.get("test_balanced_accuracy")
        if val is not None and not math.isnan(val):
            acc[(r["model"], r["query_type"])].append(val)

    fieldnames = ["arch_idx", "arch_label", "cert_mean", "cert_std", "noncert_mean", "noncert_std"]
    rows = []
    for idx, arch in enumerate(ARCH_ORDER):
        cert_vals = acc.get((arch, "certified"), [])
        noncert_vals = acc.get((arch, "noncertified"), [])
        cert_mean, cert_std = _mean_std(cert_vals)
        noncert_mean, noncert_std = _mean_std(noncert_vals)
        rows.append(
            {
                "arch_idx": idx,
                "arch_label": ARCH_LABEL.get(arch, arch),
                "cert_mean": f"{cert_mean:.6f}",
                "cert_std": f"{cert_std:.6f}",
                "noncert_mean": f"{noncert_mean:.6f}",
                "noncert_std": f"{noncert_std:.6f}",
            }
        )

    out = output_dir / "error_floor.csv"
    write_csv(out, fieldnames, rows)

    print("  Head:")
    for r in rows:
        print("   ", r)


def export_minaug_cdf(
    results_dir: Path, output_dir: Path, n_attrs_list: list[int] | None = None
) -> None:
    """Export pgfdata/minaug_cdf_n{N}.csv for each n_attrs."""
    path = results_dir / "e3_minaug_20260523_080516.json"
    data = load_json(path)
    records = data["results"]
    print(f"[minaug_cdf] Loaded {len(records)} records from {path}")

    if n_attrs_list is None:
        n_attrs_list = [4, 6, 8, 10]

    by_n: dict[int, list[float]] = defaultdict(list)
    for r in records:
        n = r["n_attrs"]
        if n in n_attrs_list:
            by_n[n].append(r["approx_ratio"])

    max_points = 200

    for n in n_attrs_list:
        ratios = np.sort(by_n[n])
        total = len(ratios)

        # Empirical CDF: ratio[i] -> (i+1)/total
        cdf_vals = (np.arange(1, total + 1)) / total

        # Prepend a point at (min_ratio, 0) so the step curve starts correctly
        ratios_full = np.concatenate([[ratios[0]], ratios])
        cdf_full = np.concatenate([[0.0], cdf_vals])

        # Downsample the main body (excluding first anchor point)
        if total > max_points:
            k = total // max_points
            indices: np.ndarray = np.arange(0, total, k)
            # Always include last point
            if indices[-1] != total - 1:
                indices = np.append(indices, total - 1)
            ratios_ds = np.concatenate([[ratios[0]], ratios[indices]])
            cdf_ds = np.concatenate([[0.0], cdf_vals[indices]])
        else:
            ratios_ds = ratios_full
            cdf_ds = cdf_full

        fieldnames = ["ratio", "cdf"]
        rows = [{"ratio": f"{r:.6f}", "cdf": f"{c:.6f}"} for r, c in zip(ratios_ds, cdf_ds)]

        out = output_dir / f"minaug_cdf_n{n}.csv"
        write_csv(out, fieldnames, rows)
        print(
            f"  n={n}: {len(ratios)} -> {len(rows)} points (downsampled k={total // max_points if total > max_points else 1})"
        )
        print(f"  Head: {rows[:3]}")


def export_minaug_runtime(results_dir: Path, output_dir: Path) -> None:
    """Export pgfdata/minaug_runtime.csv."""
    path = results_dir / "e3_minaug_20260523_080516.json"
    data = load_json(path)
    records = data["results"]
    print(f"[minaug_runtime] Loaded {len(records)} records from {path}")

    by_n: dict[int, list[float]] = defaultdict(list)
    for r in records:
        by_n[r["n_attrs"]].append(r["greedy_time_us"])

    fieldnames = ["n_attrs", "median_us", "q25_us", "q75_us"]
    rows = []
    for n in sorted(by_n):
        times = np.array(by_n[n])
        rows.append(
            {
                "n_attrs": n,
                "median_us": f"{np.median(times):.4f}",
                "q25_us": f"{np.percentile(times, 25):.4f}",
                "q75_us": f"{np.percentile(times, 75):.4f}",
            }
        )

    out = output_dir / "minaug_runtime.csv"
    write_csv(out, fieldnames, rows)

    print("  Head:")
    for r in rows:
        print("   ", r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export pgfplots CSV data files")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing combined JSON result files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pgfdata"),
        help="Output directory for CSV files",
    )
    args = p.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir

    print(f"Results dir : {results_dir.resolve()}")
    print(f"Output dir  : {output_dir.resolve()}")
    print()

    export_capability_jump(results_dir, output_dir)
    print()
    export_error_floor(results_dir, output_dir)
    print()
    export_minaug_cdf(results_dir, output_dir)
    print()
    export_minaug_runtime(results_dir, output_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()

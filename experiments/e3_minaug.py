"""
E3 — Greedy-MinAug in Practice  (RQ4)

Evaluates Algorithm 1 (Greedy-MinAug) by comparing its output size to the
brute-force optimal across randomly generated instances and measuring runtime.

Key questions:
  - How close is the greedy approximation ratio to 1 in practice?
  - How does runtime scale with footprint size and number of candidates?

Instance generation
-------------------
For each trial:
  - n_attrs attributes, footprint S ⊆ {0,...,n_attrs-1}
  - Random FDs (lhs ⊆ attrs, rhs ∈ attrs \\ lhs)
  - Candidate actions: all singletons {a} plus random subsets of size 2-3
  - Greedy-MinAug is run and timed
  - Brute-force optimal is computed by exhaustive enumeration (feasible for ≤ 20 candidates)

Approximation ratio = greedy_size / optimal_size  (≥ 1 always; 1 = optimal).

Quick test (< 2 s on CPU):
    python -m experiments.e3_minaug --mini

Full sweep:
    python -m experiments.e3_minaug --n-trials 2000 --n-attrs-range 4 8 12 \\
        --n-fds-range 0 3 6 10 --seed 0
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from data.utils import FD, action_coverage, greedy_minaug
from experiments.runner import save_results

# ---------------------------------------------------------------------------
# Brute-force optimal (exhaustive set cover)
# ---------------------------------------------------------------------------


def brute_force_optimal(
    footprint: frozenset[int],
    candidates: list[frozenset[int]],
    fds: list[FD],
) -> int | None:
    """
    Return the minimum number of candidates required to Σ-cover footprint.
    Returns None if the instance is infeasible.
    O(2^|candidates|) — only use for |candidates| ≤ ~20.
    """
    coverages = [action_coverage(a, footprint, fds) for a in candidates]
    for size in range(0, len(candidates) + 1):
        for subset_idx in combinations(range(len(candidates)), size):
            covered: frozenset[int] = frozenset()
            for i in subset_idx:
                covered |= coverages[i]
            if covered >= footprint:
                return size
    return None  # infeasible


# ---------------------------------------------------------------------------
# Random instance generator
# ---------------------------------------------------------------------------


def random_instance(
    n_attrs: int,
    n_fds: int,
    footprint_size: int,
    rng: np.random.Generator,
) -> tuple[frozenset[int], list[frozenset[int]], list[FD]]:
    """
    Generate a random MinAug instance.
    Returns (footprint, candidates, fds).
    Candidates = all singletons + random pairs and triples.
    """
    attrs = list(range(n_attrs))
    footprint_size = min(footprint_size, n_attrs)
    footprint = frozenset(map(int, rng.choice(attrs, size=footprint_size, replace=False)))

    # Random FDs: lhs is a small subset, rhs ∉ lhs
    fds: list[FD] = []
    seen: set = set()
    attempts = 0
    while len(fds) < n_fds and attempts < n_fds * 10:
        attempts += 1
        lhs_size = int(rng.integers(1, min(3, n_attrs - 1) + 1))
        lhs = frozenset(map(int, rng.choice(attrs, size=lhs_size, replace=False)))
        remaining = [a for a in attrs if a not in lhs]
        if not remaining:
            continue
        rhs = int(rng.choice(remaining))
        key = (lhs, rhs)
        if key not in seen:
            seen.add(key)
            fds.append((lhs, rhs))

    # Candidates: singletons + random pairs + random triples
    candidates: list[frozenset[int]] = [frozenset({a}) for a in attrs]
    for _ in range(n_attrs):
        size = int(rng.integers(2, min(4, n_attrs) + 1))
        c = frozenset(map(int, rng.choice(attrs, size=size, replace=False)))
        if c not in candidates:
            candidates.append(c)

    return footprint, candidates, fds


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_minaug_trial(
    n_attrs: int,
    n_fds: int,
    footprint_size: int,
    rng: np.random.Generator,
) -> dict | None:
    """
    Run one MinAug trial.  Returns None if the instance is infeasible.
    """
    footprint, candidates, fds = random_instance(n_attrs, n_fds, footprint_size, rng)

    # Greedy
    t0 = time.perf_counter()
    greedy = greedy_minaug(footprint, candidates, fds)
    greedy_time = time.perf_counter() - t0
    if greedy is None:
        return None  # infeasible

    # Brute-force (skip if too many candidates to keep tests fast)
    if len(candidates) > 20:
        return None
    t0 = time.perf_counter()
    optimal = brute_force_optimal(footprint, candidates, fds)
    bf_time = time.perf_counter() - t0
    if optimal is None:
        return None

    ratio = len(greedy) / optimal if optimal > 0 else 1.0

    return {
        "n_attrs": n_attrs,
        "n_fds": n_fds,
        "footprint_size": len(footprint),
        "n_candidates": len(candidates),
        "greedy_size": len(greedy),
        "optimal_size": optimal,
        "approx_ratio": round(ratio, 6),
        "greedy_time_us": round(greedy_time * 1e6, 3),
        "bf_time_us": round(bf_time * 1e6, 3),
        "is_optimal": len(greedy) == optimal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="E3: Greedy-MinAug approximation quality")
    p.add_argument("--mini", action="store_true", help="Fast smoke-test: small sweep")
    p.add_argument(
        "--n-trials", type=int, default=500, help="Trials per (n_attrs, n_fds) combination"
    )
    p.add_argument(
        "--n-attrs-range",
        nargs="+",
        type=int,
        default=[4, 6, 8, 10],
        help="Attribute counts to sweep over",
    )
    p.add_argument(
        "--n-fds-range", nargs="+", type=int, default=[0, 2, 4, 6], help="FD counts to sweep over"
    )
    p.add_argument(
        "--footprint-frac", type=float, default=0.6, help="Footprint size as fraction of n_attrs"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    if args.mini:
        args.n_trials = 20
        args.n_attrs_range = [4, 6]
        args.n_fds_range = [0, 3]

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    infeasible = 0

    combos = [(n, f) for n in args.n_attrs_range for f in args.n_fds_range]
    total = len(combos) * args.n_trials

    print(f"Running {total} trials across {len(combos)} (n_attrs, n_fds) combinations...")

    for n_attrs, n_fds in combos:
        fp_size = max(2, round(n_attrs * args.footprint_frac))
        batch_ok = 0
        for _ in range(args.n_trials):
            rec = run_minaug_trial(n_attrs, n_fds, fp_size, rng)
            if rec is None:
                infeasible += 1
            else:
                records.append(rec)
                batch_ok += 1

        if records:
            batch = [r for r in records if r["n_attrs"] == n_attrs and r["n_fds"] == n_fds]
            if batch:
                ratios = [r["approx_ratio"] for r in batch]
                pct_opt = 100 * sum(r["is_optimal"] for r in batch) / len(batch)
                print(
                    f"  n_attrs={n_attrs} n_fds={n_fds:2d} | "
                    f"n={batch_ok:4d} | "
                    f"ratio mean={np.mean(ratios):.4f} max={np.max(ratios):.4f} | "
                    f"optimal {pct_opt:.1f}%"
                )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e3_minaug_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e3_minaug",
            "timestamp": ts,
            "n_trials": args.n_trials,
            "n_attrs_range": args.n_attrs_range,
            "n_fds_range": args.n_fds_range,
            "footprint_frac": args.footprint_frac,
            "seed": args.seed,
            "n_infeasible": infeasible,
            "n_feasible": len(records),
        },
    )

    if records:
        ratios = [r["approx_ratio"] for r in records]
        pct_opt = 100 * sum(r["is_optimal"] for r in records) / len(records)
        print(f"\n=== Overall MinAug statistics ({len(records)} feasible instances) ===")
        print(
            f"  Approximation ratio: mean={np.mean(ratios):.4f}  "
            f"max={np.max(ratios):.4f}  p99={np.percentile(ratios, 99):.4f}"
        )
        print(f"  Optimal solution found: {pct_opt:.1f}% of instances")
        print(f"  Infeasible instances skipped: {infeasible}")


if __name__ == "__main__":
    main()

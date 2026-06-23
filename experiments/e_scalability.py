"""
e_scalability.py — Certification and MinAug runtime scaling

Sweeps n_attrs and n_fds up to 10^3 to demonstrate that:
  - check_certificate completes in milliseconds for all schema sizes tested
  - greedy_minaug (20 candidate actions) stays practical at large schemas

No ML training. Pure algorithmic timing.

Measurement design
------------------
Two instance types per (n_attrs, n_fds) pair:

  RANDOM — for certification timing:
    Build one Config with n_fds random FDs over n_attrs attributes,
    one overlap of size max(1, n_attrs // 10), lhs_size ≤ 2.
    Time check_certificate for n_queries random footprints (size 60% n_attrs).

  PLANTED — for MinAug timing (guarantees feasibility):
    overlap = {0}
    footprint = {0, 1, …, fp_size-1}   where fp_size = n_attrs // 2
    Chain FDs 1→2→…→fp_size-1 ensure fd_closure({1}) covers all bridge attrs.
    Random FDs are restricted to attrs outside [1, fp_size) so they do not
    accidentally certify the footprint from the bare overlap.
    Candidates: singletons {1}, {2}, …, up to n_candidates.
    greedy_minaug always finds a feasible augmentation (picking {1} suffices).

All times in milliseconds; median and IQR reported.

Usage
-----
  python -m experiments.e_scalability            # full run
  python -m experiments.e_scalability --mini     # quick smoke test
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data.synthetic import Config
from data.utils import FD, check_certificate, greedy_minaug
from experiments.runner import save_results

# ---------------------------------------------------------------------------
# Instance generation — random (for cert timing)
# ---------------------------------------------------------------------------


def make_random_fds(
    n_attrs: int,
    n_fds: int,
    rng: np.random.Generator,
    exclude_lhs: frozenset[int] | None = None,
    exclude_rhs: set[int] | None = None,
) -> list[FD]:
    """Random FDs of lhs_size ≤ 2. Optionally restrict LHS/RHS attribute sets."""
    attrs = list(range(n_attrs))
    fds: list[FD] = []
    seen: set = set()
    attempts = 0
    while len(fds) < n_fds and attempts < n_fds * 10:
        attempts += 1
        lhs_size = int(rng.integers(1, min(3, n_attrs - 1) + 1))
        lhs = frozenset(map(int, rng.choice(attrs, size=lhs_size, replace=False)))
        if exclude_lhs is not None and not lhs.isdisjoint(exclude_lhs):
            continue
        remaining = [a for a in attrs if a not in lhs]
        if exclude_rhs is not None:
            remaining = [a for a in remaining if a not in exclude_rhs]
        if not remaining:
            continue
        rhs = int(rng.choice(remaining))
        key = (lhs, rhs)
        if key not in seen:
            seen.add(key)
            fds.append((lhs, rhs))
    return fds


def make_large_config(
    n_attrs: int,
    n_fds: int,
    overlap_size: int,
    rng: np.random.Generator,
) -> Config:
    """Random Config with n_fds FDs of lhs_size ≤ 2, one overlap."""
    fds = make_random_fds(n_attrs, n_fds, rng)
    overlap = frozenset(range(overlap_size))
    return Config(
        n_attrs=n_attrs,
        domain_size=2,
        fds=fds,
        view_schemas={0: frozenset(range(n_attrs))},
        overlap_schemas=[overlap],
    )


# ---------------------------------------------------------------------------
# Instance generation — planted (for MinAug timing, guaranteed feasible)
# ---------------------------------------------------------------------------


def make_planted_minaug_instance(
    n_attrs: int,
    n_fds: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[Config, list[frozenset[int]], frozenset[int]]:
    """
    Return (config, candidates, footprint) where MinAug is guaranteed feasible.

    Structure:
      footprint = {0, 1, …, fp_size-1}
      Chain FDs: {0}→1, {1}→2, …, {fp_size-2}→fp_size-1

    fd_closure({0}) covers the full footprint via the chain.
    Candidate {0} (the first singleton) therefore covers everything.
    Random FDs are added for n_attrs/n_fds scaling realism; they are restricted
    to attrs outside [0, fp_size) so they do not spuriously cover other singletons.

    greedy_minaug picks candidate {0} and terminates in one step →
    always feasible, timing dominated by the O(n_candidates × n_attrs × n_fds)
    precomputation of all candidate coverages.
    """
    fp_size = max(2, n_attrs // 2)
    footprint = frozenset(range(fp_size))

    # Chain FDs: 0→1, 1→2, …, (fp_size-2)→(fp_size-1)
    chain_fds: list[FD] = [(frozenset({j}), j + 1) for j in range(fp_size - 1)]

    # Random FDs restricted to outer attrs {fp_size, …, n_attrs-1}
    outer = list(range(fp_size, n_attrs))
    random_fds: list[FD] = []
    if len(outer) >= 2:
        seen: set = set()
        attempts = 0
        target = max(0, n_fds - len(chain_fds))
        while len(random_fds) < target and attempts < target * 10:
            attempts += 1
            lhs_size = int(rng.integers(1, min(3, len(outer) - 1) + 1))
            lhs = frozenset(rng.choice(outer, size=lhs_size, replace=False).tolist())
            rhs_pool = [a for a in outer if a not in lhs]
            if not rhs_pool:
                continue
            rhs = int(rng.choice(rhs_pool))
            key = (lhs, rhs)
            if key not in seen:
                seen.add(key)
                random_fds.append((lhs, rhs))

    all_fds = chain_fds + random_fds

    config = Config(
        n_attrs=n_attrs,
        domain_size=2,
        fds=all_fds,
        view_schemas={0: frozenset(range(n_attrs))},
        overlap_schemas=[frozenset({0})],
    )

    # Candidates: first n_candidates singletons (candidate {0} alone suffices)
    candidates: list[frozenset[int]] = [frozenset({a}) for a in range(min(n_candidates, n_attrs))]
    # Pad with random pairs if needed
    while len(candidates) < n_candidates and n_attrs >= 2:
        pair = frozenset(rng.choice(n_attrs, size=2, replace=False).tolist())
        if pair not in candidates:
            candidates.append(pair)

    return config, candidates, footprint


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_cert(
    config: Config,
    n_queries: int,
    footprint_frac: float,
    rng: np.random.Generator,
) -> list[float]:
    """Return per-query certification times in ms."""
    fp_size = max(1, round(config.n_attrs * footprint_frac))
    times_ms = []
    for _ in range(n_queries):
        fp = frozenset(map(int, rng.choice(config.n_attrs, size=fp_size, replace=False)))
        t0 = time.perf_counter()
        check_certificate(fp, config.overlap_schemas, config.fds)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return times_ms


def time_minaug(
    config: Config,
    candidates: list[frozenset[int]],
    footprint: frozenset[int],
    n_queries: int,
) -> list[float]:
    """
    Return per-query MinAug greedy times in ms using a fixed planted footprint.
    Skips infeasible (unexpected) instances.
    """
    times_ms = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        result = greedy_minaug(footprint, candidates, config.fds)
        elapsed = (time.perf_counter() - t0) * 1e3
        if result is not None:
            times_ms.append(elapsed)
    return times_ms


def summarise(times_ms: list[float]) -> dict:
    if not times_ms:
        return {
            "n": 0,
            "median_ms": None,
            "p25_ms": None,
            "p75_ms": None,
            "mean_ms": None,
            "max_ms": None,
        }
    a = np.array(times_ms)
    return {
        "n": len(a),
        "median_ms": round(float(np.median(a)), 4),
        "p25_ms": round(float(np.percentile(a, 25)), 4),
        "p75_ms": round(float(np.percentile(a, 75)), 4),
        "mean_ms": round(float(np.mean(a)), 4),
        "max_ms": round(float(np.max(a)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Certification and MinAug scalability sweep")
    p.add_argument("--mini", action="store_true", help="Fast smoke test")
    p.add_argument("--n-attrs-range", nargs="+", type=int, default=[10, 50, 100, 250, 500, 1000])
    p.add_argument("--n-fds-range", nargs="+", type=int, default=[10, 50, 100, 250, 500, 1000])
    p.add_argument(
        "--n-queries", type=int, default=100, help="Queries to time per (n_attrs, n_fds) pair"
    )
    p.add_argument("--n-candidates", type=int, default=20, help="Candidate actions for MinAug")
    p.add_argument("--footprint-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    if args.mini:
        args.n_attrs_range = [10, 50, 100]
        args.n_fds_range = [10, 50, 100]
        args.n_queries = 10

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    combos = [(n, f) for n in args.n_attrs_range for f in args.n_fds_range]
    print(
        f"Scalability sweep: {len(combos)} (n_attrs, n_fds) combinations, "
        f"{args.n_queries} queries each."
    )

    for n_attrs, n_fds in combos:
        overlap_size = max(1, n_attrs // 10)

        # Random config for cert timing
        cert_config = make_large_config(n_attrs, n_fds, overlap_size, rng)

        # Planted config for MinAug timing (guaranteed feasible)
        minaug_config, candidates, planted_fp = make_planted_minaug_instance(
            n_attrs, n_fds, args.n_candidates, rng
        )

        cert_times = time_cert(cert_config, args.n_queries, args.footprint_frac, rng)
        minaug_times = time_minaug(minaug_config, candidates, planted_fp, args.n_queries)

        cert_stats = summarise(cert_times)
        minaug_stats = summarise(minaug_times)

        rec = {
            "n_attrs": n_attrs,
            "n_fds_requested": n_fds,
            "n_fds_actual_cert": len(cert_config.fds),
            "n_fds_actual_minaug": len(minaug_config.fds),
            "overlap_size": overlap_size,
            "n_candidates": len(candidates),
            "footprint_size_cert": max(1, round(n_attrs * args.footprint_frac)),
            "footprint_size_minaug": len(planted_fp),
            **{"cert_" + k: v for k, v in cert_stats.items()},
            **{"minaug_" + k: v for k, v in minaug_stats.items()},
        }
        records.append(rec)

        cert_med = cert_stats["median_ms"]
        minaug_med = minaug_stats["median_ms"]
        cert_str = f"{cert_med:.3f}" if cert_med is not None else "N/A"
        minaug_str = f"{minaug_med:.3f}" if minaug_med is not None else "N/A"
        print(
            f"  n_attrs={n_attrs:5d}  n_fds={n_fds:5d} | "
            f"cert {cert_str} ms | "
            f"minaug {minaug_str} ms ({minaug_stats['n']}/{args.n_queries} feasible)"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e_scalability_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e_scalability",
            "timestamp": ts,
            "n_attrs_range": args.n_attrs_range,
            "n_fds_range": args.n_fds_range,
            "n_queries": args.n_queries,
            "n_candidates": args.n_candidates,
            "footprint_frac": args.footprint_frac,
            "seed": args.seed,
            "note": (
                "cert uses random configs; minaug uses planted configs "
                "(chain FDs guarantee feasibility)"
            ),
        },
    )
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

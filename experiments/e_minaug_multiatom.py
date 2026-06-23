"""
e_minaug_multiatom.py — Greedy-MinAug with multi-atom queries (|B_Q| > 1)

Models queries with k ∈ {1,2,3,4} atom obligations (B_Q = {U_1,...,U_k}).
Each atom has atom_size=5 dedicated attributes; the footprint is their union.
Within-atom chain FDs (root_i → root_i+1 → ... → end_i) model local FD structure.
Random cross-atom FDs allow one action to cover multiple atoms' attributes.

Three solutions are measured per instance:
  greedy     — Greedy-MinAug (Algorithm 1) with singletons + cross-atom pairs.
  singleton  — Same greedy restricted to singleton candidates only.
  optimal    — Brute-force optimal over all candidates (≤ 16 total, feasible).

Key comparisons
---------------
RQ3-a: As |B_Q| grows, what fraction of greedy solutions are optimal?
RQ3-b: How much does allowing cross-atom pair candidates reduce augmentation size
        compared to singleton-only?

Usage
-----
  python -m experiments.e_minaug_multiatom            # full run
  python -m experiments.e_minaug_multiatom --mini     # 50-trial smoke test
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

ATOM_SIZE = 5  # attrs per atom (fixed)
MAX_CANDIDATES = 16  # brute-force feasibility cap (2^16 ≈ 65 K iterations)


# ---------------------------------------------------------------------------
# Brute-force optimal
# ---------------------------------------------------------------------------


def brute_force_optimal(
    footprint: frozenset[int],
    candidates: list[frozenset[int]],
    fds: list[FD],
) -> int | None:
    """Minimum candidates needed to Σ-cover footprint. None if infeasible."""
    coverages = [action_coverage(a, footprint, fds) for a in candidates]
    for size in range(0, len(candidates) + 1):
        for subset_idx in combinations(range(len(candidates)), size):
            covered: frozenset[int] = frozenset()
            for i in subset_idx:
                covered |= coverages[i]
            if covered >= footprint:
                return size
    return None


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def make_multiatom_instance(
    n_atoms: int,
    n_cross_fds: int,
    rng: np.random.Generator,
) -> tuple[frozenset[int], list[frozenset[int]], list[frozenset[int]], list[FD]]:
    """
    Build one multi-atom MinAug instance.

    Returns
    -------
    footprint       : union of all atom attrs
    all_cands       : singletons (root + non-root) + random cross-atom pairs (≤ MAX_CANDIDATES)
    singleton_cands : root singleton per atom only (subset of all_cands)
    fds             : within-atom chains + random cross-atom FDs
    """
    n_attrs = n_atoms * ATOM_SIZE
    footprint = frozenset(range(n_attrs))

    # Within-atom chain FDs: root_i → root_i+1 → ... → root_i+ATOM_SIZE-2 → root_i+ATOM_SIZE-1
    fds: list[FD] = []
    for i in range(n_atoms):
        base = i * ATOM_SIZE
        for j in range(ATOM_SIZE - 1):
            fds.append((frozenset({base + j}), base + j + 1))

    # Random cross-atom FDs: source in one atom, target in another
    seen_cross: set = set()
    attempts = 0
    n_added = 0
    while n_added < n_cross_fds and attempts < n_cross_fds * 30:
        attempts += 1
        sa = int(rng.integers(0, n_atoms))
        ta = int(rng.integers(0, n_atoms))
        if sa == ta:
            continue
        src = int(rng.integers(sa * ATOM_SIZE, (sa + 1) * ATOM_SIZE))
        tgt = int(rng.integers(ta * ATOM_SIZE, (ta + 1) * ATOM_SIZE))
        key = (src, tgt)
        if key not in seen_cross:
            seen_cross.add(key)
            fds.append((frozenset({src}), tgt))
            n_added += 1

    # Root singletons: one per atom (each covers all ATOM_SIZE attrs via chain)
    root_cands = [frozenset({i * ATOM_SIZE}) for i in range(n_atoms)]
    # Singleton baseline: just the root singletons (always feasible in k actions)
    singleton_cands = list(root_cands)

    # Remaining candidate budget for pairs
    n_pairs = MAX_CANDIDATES - n_atoms
    pair_cands: list[frozenset[int]] = []
    seen_pairs: set = set()
    for _ in range(n_pairs * 5):
        if len(pair_cands) >= n_pairs:
            break
        a, b = int(rng.integers(0, n_attrs)), int(rng.integers(0, n_attrs))
        if a == b:
            continue
        pair_key = frozenset({a, b})
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            pair_cands.append(pair_key)

    # Shuffle so greedy doesn't have a positional bias
    all_cands: list[frozenset[int]] = root_cands + pair_cands
    idx = list(range(len(all_cands)))
    rng.shuffle(idx)
    all_cands = [all_cands[i] for i in idx]

    return footprint, all_cands, singleton_cands, fds


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_trial(
    n_atoms: int,
    n_cross_fds: int,
    rng: np.random.Generator,
) -> dict | None:
    """
    One trial. Returns None if the instance is infeasible for either strategy.
    """
    footprint, all_cands, singleton_cands, fds = make_multiatom_instance(n_atoms, n_cross_fds, rng)

    # --- Greedy (full candidates) ---
    t0 = time.perf_counter()
    g_result = greedy_minaug(footprint, all_cands, fds)
    g_time = time.perf_counter() - t0
    if g_result is None:
        return None

    # --- Greedy (singleton-only baseline) ---
    s_result = greedy_minaug(footprint, singleton_cands, fds)
    if s_result is None:
        return None  # singleton-only infeasible (unexpected with root singletons)

    # --- Brute-force optimal ---
    if len(all_cands) > MAX_CANDIDATES:
        return None  # safety cap
    optimal = brute_force_optimal(footprint, all_cands, fds)
    if optimal is None:
        return None

    g_size = len(g_result)
    s_size = len(s_result)
    ratio = g_size / optimal if optimal > 0 else 1.0
    singleton_ratio = s_size / optimal if optimal > 0 else 1.0

    return {
        "n_atoms": n_atoms,
        "n_cross_fds": n_cross_fds,
        "n_attrs": n_atoms * ATOM_SIZE,
        "footprint_size": n_atoms * ATOM_SIZE,
        "n_all_cands": len(all_cands),
        "n_singleton_cands": len(singleton_cands),
        "greedy_size": g_size,
        "singleton_size": s_size,
        "optimal_size": optimal,
        "approx_ratio": round(ratio, 6),
        "singleton_ratio": round(singleton_ratio, 6),
        "greedy_is_optimal": g_size == optimal,
        "singleton_is_optimal": s_size == optimal,
        "greedy_time_us": round(g_time * 1e6, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MinAug with multi-atom queries (|B_Q| in {1,2,3,4})")
    p.add_argument("--mini", action="store_true", help="50-trial smoke test")
    p.add_argument("--n-trials", type=int, default=500)
    p.add_argument("--n-atoms-range", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument(
        "--n-cross-fds", type=int, default=None, help="Cross-atom FDs per trial (default: n_atoms)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    if args.mini:
        args.n_trials = 50

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    infeasible = 0

    total = len(args.n_atoms_range) * args.n_trials
    print(
        f"Multi-atom MinAug: {total} trials "
        f"({args.n_trials} × {len(args.n_atoms_range)} n_atoms values)."
    )

    for n_atoms in args.n_atoms_range:
        n_cross = args.n_cross_fds if args.n_cross_fds is not None else n_atoms
        batch: list[dict] = []
        for _ in range(args.n_trials):
            rec = run_trial(n_atoms, n_cross, rng)
            if rec is None:
                infeasible += 1
            else:
                batch.append(rec)

        records.extend(batch)
        if batch:
            ratios = [r["approx_ratio"] for r in batch]
            s_ratios = [r["singleton_ratio"] for r in batch]
            pct_opt = 100 * sum(r["greedy_is_optimal"] for r in batch) / len(batch)
            mean_g = np.mean([r["greedy_size"] for r in batch])
            mean_s = np.mean([r["singleton_size"] for r in batch])
            mean_o = np.mean([r["optimal_size"] for r in batch])
            print(
                f"  |B_Q|={n_atoms} | n={len(batch):4d} | "
                f"ratio mean={np.mean(ratios):.4f} max={np.max(ratios):.4f} | "
                f"optimal {pct_opt:.1f}% | "
                f"mean sizes: greedy={mean_g:.2f} single={mean_s:.2f} opt={mean_o:.2f} | "
                f"singleton_ratio mean={np.mean(s_ratios):.3f}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e_minaug_multiatom_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e_minaug_multiatom",
            "timestamp": ts,
            "n_trials": args.n_trials,
            "n_atoms_range": args.n_atoms_range,
            "atom_size": ATOM_SIZE,
            "max_candidates": MAX_CANDIDATES,
            "seed": args.seed,
            "n_infeasible": infeasible,
            "n_feasible": len(records),
        },
    )

    if records:
        ratios = [r["approx_ratio"] for r in records]
        s_ratios = [r["singleton_ratio"] for r in records]
        pct_opt = 100 * sum(r["greedy_is_optimal"] for r in records) / len(records)
        print(f"\n=== Multi-atom MinAug ({len(records)} feasible) ===")
        print(
            f"  Greedy ratio  : mean={np.mean(ratios):.4f}  "
            f"max={np.max(ratios):.4f}  p99={np.percentile(ratios, 99):.4f}"
        )
        print(
            f"  Singleton ratio: mean={np.mean(s_ratios):.4f}  "
            f"max={np.max(s_ratios):.4f}  p99={np.percentile(s_ratios, 99):.4f}"
        )
        print(f"  Greedy optimal: {pct_opt:.1f}%")
        print(f"  Infeasible skipped: {infeasible}")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

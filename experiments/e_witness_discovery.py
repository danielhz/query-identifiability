"""
e_witness_discovery.py — Empirical witness discovery rate vs. corpus size (RQ2 quantification).

For each non-certified WDC query (Q_highly_rated, Q_reviewed, Q_popular), performs
N random trials in which the dataset records are shuffled and scanned one at a time.
The scan stops as soon as the first witness pair is found (two records with identical
Σ-closed overlap projection but different query answers).

Records the scan position at which the first witness was discovered across all trials,
then reports: median, 90th-percentile, and fraction found within 50/100/200/500 records.

This replaces the qualitative claim 'witnesses are found within the first few hundred
records' (§5.2) with empirical CDF statistics.

Two datasets are supported via --dataset:
  * wdc      (default): the synthetic WDC corpus — witnesses on synthesized attributes.
  * crosskg  : OpenAlex × DBLP — REAL witnesses (the two sources genuinely disagree on
               author counts).  Defaults to the real joined corpus; --mock uses synthetic.

Design notes
------------
* Records are treated as single-tuple worlds (n_tuples=1).  Two records r1, r2 are a
  witness pair when observation_key([r1]) == observation_key([r2]) AND Q([r1]) ≠ Q([r2]).
* For the WDC mock data, records are grouped in clusters of 3 (same brand+model), all
  sharing the same augmented-overlap projection {0,1,2,3,7}.  Witness pairs therefore
  require two records from the same product cluster whose query-answer differs.
* evaluate_cq is called once per record and pre-cached; the inner trial loop only
  permutes and scans pre-computed arrays.

Usage
-----
  python -m experiments.e_witness_discovery             # 1000 trials, 3000 mock records
  python -m experiments.e_witness_discovery --n-trials 200 --mini
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import data.crosskg_dblp as ckg
import data.wdc as wdc
from data.exhaustive import evaluate_cq
from data.synthetic import BooleanCQ, Config
from data.utils import augmented_overlap
from data.witness import observation_key
from experiments.runner import save_results

# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------


def _precompute(
    records: list,
    cq: BooleanCQ,
    aug_overlaps: list[frozenset[int]],
    config: Config,
) -> tuple[list[tuple], np.ndarray]:
    """
    Compute obs_key and Q value for every record once.

    Returns
    -------
    keys : list of observation_key tuples (length = len(records))
    q_vals : bool array of shape (len(records),)
    """
    keys: list[tuple] = []
    q_vals: list[bool] = []
    for rec in records:
        world = np.array([rec.to_world_row()], dtype=np.int32)
        keys.append(observation_key(world, aug_overlaps))
        q_vals.append(evaluate_cq(cq, world, config))
    return keys, np.array(q_vals, dtype=bool)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def _scan_trial(
    keys: list[tuple],
    q_vals: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Scan records in random order; return 1-indexed position of first witness.
    Returns len(keys)+1 when no witness is found within the corpus.
    """
    order = rng.permutation(len(keys))
    # obs_key → [has_True, has_False]
    obs_groups: dict[tuple, list[bool]] = {}

    for pos0, idx in enumerate(order):
        k = keys[idx]
        q = bool(q_vals[idx])

        if k not in obs_groups:
            obs_groups[k] = [False, False]
        state = obs_groups[k]

        if q:
            state[0] = True
        else:
            state[1] = True

        if state[0] and state[1]:
            return pos0 + 1  # 1-indexed

    return len(keys) + 1  # not found


# ---------------------------------------------------------------------------
# Per-query experiment
# ---------------------------------------------------------------------------


def run_discovery_experiment(
    query_name: str,
    cq: BooleanCQ,
    records: list,
    config: Config,
    n_trials: int,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Run n_trials random-order scans; return a result record with CDF statistics.
    """
    aug_overlaps = [augmented_overlap(o, config.fds) for o in config.overlap_schemas]
    n = len(records)

    t0 = time.perf_counter()
    keys, q_vals = _precompute(records, cq, aug_overlaps, config)
    precompute_s = time.perf_counter() - t0

    pos_rate = float(np.mean(q_vals))

    rng = np.random.default_rng(seed)

    t1 = time.perf_counter()
    positions = [_scan_trial(keys, q_vals, rng) for _ in range(n_trials)]
    scan_s = time.perf_counter() - t1

    pos_arr = np.array(positions, dtype=np.int32)
    found_mask = pos_arr <= n
    found_arr = pos_arr[found_mask]
    n_found = int(found_mask.sum())

    result: dict = {
        "query": query_name,
        "n_records": n,
        "n_trials": n_trials,
        "positive_rate": round(pos_rate, 4),
        "n_found": n_found,
        "n_not_found": n_trials - n_found,
        "frac_found": round(n_found / n_trials, 4),
        "median_position": int(np.median(found_arr)) if n_found > 0 else None,
        "pct10_position": int(np.percentile(found_arr, 10)) if n_found > 0 else None,
        "pct90_position": int(np.percentile(found_arr, 90)) if n_found > 0 else None,
        "frac_within_50": round(float(np.mean(pos_arr <= 50)), 4),
        "frac_within_100": round(float(np.mean(pos_arr <= 100)), 4),
        "frac_within_200": round(float(np.mean(pos_arr <= 200)), 4),
        "frac_within_500": round(float(np.mean(pos_arr <= 500)), 4),
        "frac_within_1000": round(float(np.mean(pos_arr <= 1000)), 4),
        "cdf_positions": [int(p) for p in sorted(found_arr)] if n_found > 0 else [],
        "precompute_s": round(precompute_s, 4),
        "scan_s": round(scan_s, 4),
    }

    if verbose:
        tag = "U"
        print(f"\n  [{tag}] {query_name}")
        print(f"    records          : {n}")
        print(f"    positive_rate    : {pos_rate:.3f}")
        print(f"    found/trials     : {n_found}/{n_trials} ({result['frac_found']:.2%})")
        if n_found > 0:
            print(f"    median pos       : {result['median_position']}")
            print(f"    90th-pct pos     : {result['pct90_position']}")
            print(
                f"    within 50/100/200/500 records: "
                f"{result['frac_within_50']:.2%} / "
                f"{result['frac_within_100']:.2%} / "
                f"{result['frac_within_200']:.2%} / "
                f"{result['frac_within_500']:.2%}"
            )
        print(f"    precompute       : {precompute_s:.3f}s  |  scan: {scan_s:.3f}s")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Empirical witness discovery rate vs. corpus size (RQ2)"
    )
    p.add_argument(
        "--dataset",
        choices=["wdc", "crosskg"],
        default="wdc",
        help="wdc (synthetic, default) or crosskg (OpenAlex × DBLP, real witnesses)",
    )
    p.add_argument(
        "--n-trials", type=int, default=1000, help="Number of random-shuffle trials per query"
    )
    p.add_argument(
        "--n-records",
        type=int,
        default=3000,
        help="Number of mock records to generate (WDC: multiple of 3)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mini", action="store_true", help="Quick smoke-test: small corpus, 50 trials")
    p.add_argument(
        "--mock",
        action="store_true",
        help="crosskg only: use synthetic mock instead of the real joined corpus",
    )
    p.add_argument("--data-dir-crosskg", type=Path, default=Path("data/raw/crosskg_dblp"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    verbose = not args.quiet
    n_trials = 50 if args.mini else args.n_trials

    # ------------------------------------------------------------------
    # Build the corpus and the non-certified queries for the chosen dataset
    # ------------------------------------------------------------------
    records: list[Any]  # WDC ProductRecord or CrossKG CrossKGRecord, depending on dataset
    if args.dataset == "wdc":
        # WDC clusters have 3 records each; keep n_records a multiple of 3.
        n_records = 300 if args.mini else args.n_records
        n_records = (n_records // 3) * 3
        records = wdc.make_mock_dataset(n=n_records, seed=args.seed)
        config = wdc.CONFIG
        non_certified_queries = [
            ("Q_highly_rated", wdc.Q_HIGHLY_RATED),
            ("Q_reviewed", wdc.Q_REVIEWED),
            ("Q_popular", wdc.Q_POPULAR),
        ]
        data_source = "mock"
        meta_extra = {"n_clusters": len(records) // 3}
        corpus_desc = f"{len(records)} records ({len(records) // 3} product clusters)"
    else:  # crosskg — OpenAlex × DBLP (real witnesses)
        if args.mock or args.mini:
            n_papers = 100 if args.mini else max(1, args.n_records // 2)
            records = ckg.make_mock_dataset(n=n_papers, seed=args.seed)
            data_source = "mock"
        else:
            records = ckg.load_dataset(args.data_dir_crosskg)
            data_source = "real"
        config = ckg.CONFIG
        # Q_publisher is certified (no witness); only Q_large_team is scanned.
        non_certified_queries = [("Q_large_team", ckg.Q_LARGE_TEAM)]
        meta_extra = {"n_papers": len(records) // 2}
        corpus_desc = (
            f"{len(records)} records ({len(records) // 2} papers × 2 sources, {data_source})"
        )

    if verbose:
        print(f"\n=== Witness Discovery Rate (RQ2 quantification) — {args.dataset} ===")
        print(f"  Corpus      : {corpus_desc}")
        print(f"  Trials/query: {n_trials}")
        print(f"  Seed        : {args.seed}")

    results: list[dict] = []
    for q_name, cq in non_certified_queries:
        if verbose:
            print(f"\n  Running {q_name} ({n_trials} trials)…", end="", flush=True)
        rec = run_discovery_experiment(
            query_name=q_name,
            cq=cq,
            records=records,
            config=config,
            n_trials=n_trials,
            seed=args.seed,
            verbose=verbose,
        )
        results.append(rec)
        if not verbose:
            status = (
                f"median={rec['median_position']}, "
                f"p90={rec['pct90_position']}, "
                f"found={rec['frac_found']:.1%}"
            )
            print(f"  {q_name:20s}: {status}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"\n{'Query':20s}  {'pos_rate':9s}  {'found%':8s}  {'median':8s}  "
            f"{'p90':8s}  {'<50':6s}  {'<100':6s}  {'<200':6s}  {'<500':6s}"
        )
        print("-" * 90)
        for r in results:
            med = str(r["median_position"]) if r["median_position"] else "n/a"
            p90 = str(r["pct90_position"]) if r["pct90_position"] else "n/a"
            print(
                f"{r['query']:20s}  {r['positive_rate']:9.3f}  "
                f"{r['frac_found']:8.2%}  {med:8s}  {p90:8s}  "
                f"{r['frac_within_50']:6.2%}  {r['frac_within_100']:6.2%}  "
                f"{r['frac_within_200']:6.2%}  {r['frac_within_500']:6.2%}"
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    infix = "" if args.dataset == "wdc" else f"{args.dataset}_"
    out = args.output_dir / f"e_witness_discovery_{infix}{ts}.json"
    save_results(
        results,
        out,
        meta={
            "experiment": "e_witness_discovery",
            "dataset": args.dataset,
            "data_source": data_source,
            "timestamp": ts,
            "n_records": len(records),
            "n_trials": n_trials,
            "seed": args.seed,
            "queries": [r["query"] for r in results],
            **meta_extra,
        },
    )


if __name__ == "__main__":
    main()

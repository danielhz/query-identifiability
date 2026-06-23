"""
Exhaustive identifiability verification  (companion to e_cert_benchmark)

Re-generates the same (config, query) pairs as e_cert_benchmark (same seed,
same arguments) and runs the exhaustive verifier from data/exhaustive.py
alongside the sampling-based witness finder.

Two modes:
  --mode failing   Run exhaustive check only for records where the sampling
                   finder reported witness_found=False (fast; targets the 12
                   ambiguous cases).
  --mode all       Run exhaustive check for every record (slower; provides
                   a complete comparison).

The exhaustive check uses relational FD semantics (not the resolver model):
for n_tuples=1 it enumerates all d^n single-row matrices.  This tests
UNIVERSAL identifiability as defined in the paper.

Timing guidance
---------------
  n_attrs ≤ 8 : single-threaded, < 10 s for 841 records
  n_attrs = 10: use --n-workers 4-8, ~ 30 s
  n_attrs ≥ 12: definitely use --n-workers

Usage
-----
  # Check only the 12 no-witness cases from the default full run:
  python -m experiments.e_exhaustive_check \\
      --sampling-results results/e_cert_benchmark_20260531_054739.json \\
      --mode failing --seed 0 --n-configs 200 --queries-per-config 5

  # Check all 841 records:
  python -m experiments.e_exhaustive_check \\
      --sampling-results results/e_cert_benchmark_20260531_054739.json \\
      --mode all --seed 0 --n-configs 200 --queries-per-config 5 --n-workers 1

  # Fast smoke test:
  python -m experiments.e_exhaustive_check --mini
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data.exhaustive import (
    _config_to_dict,
    _cq_to_dict,
    exhaustive_check_batch,
)
from data.synthetic import SyntheticBenchmark
from data.witness import check_and_witness
from experiments.e_cert_benchmark import (
    generate_queries,
    is_degenerate,
    random_config,
)
from experiments.runner import save_results

# ---------------------------------------------------------------------------
# Re-generate (config, query) pairs from seed (mirrors e_cert_benchmark main)
# ---------------------------------------------------------------------------


def regenerate_records(
    n_configs: int,
    queries_per_config: int,
    n_attrs_range: list[int],
    n_fds_range: list[int],
    domain_size: int,
    n_witness_samples: int,
    n_tuples: int,
    seed: int,
) -> list[dict]:
    """
    Reproduce the same (config, query, sampling_result) sequence as
    e_cert_benchmark with identical arguments and seed.  Returns list of
    records augmented with serialised config and query dicts.
    """
    rng = np.random.default_rng(seed)
    n_certified_target = queries_per_config // 2
    n_noncert_target = queries_per_config - n_certified_target

    records: list[dict] = []
    record_id = 0

    for config_i in range(n_configs):
        n_attrs = int(rng.choice(n_attrs_range))
        n_fds = int(rng.choice(n_fds_range))
        n_views = int(rng.integers(2, max(3, n_attrs // 2) + 1))
        n_overlaps = int(rng.integers(1, max(2, n_views // 2) + 1))

        config = random_config(
            n_attrs=n_attrs,
            n_views=n_views,
            n_overlaps=n_overlaps,
            n_fds=n_fds,
            domain_size=domain_size,
            rng=rng,
        )

        queries = generate_queries(
            config=config,
            n_certified=n_certified_target,
            n_noncertified=n_noncert_target,
            rng=rng,
        )

        for q_i, cq in enumerate(queries):
            seed_i = int(rng.integers(0, 2**31))
            bench = SyntheticBenchmark(config, seed=seed_i)

            if is_degenerate(cq, bench, n_tuples):
                continue

            sampling = check_and_witness(
                cq=cq,
                config=config,
                bench=bench,
                n_samples=n_witness_samples,
                n_tuples=n_tuples,
            )
            sampling["n_attrs"] = config.n_attrs
            sampling["n_fds"] = len(config.fds)
            sampling["n_views"] = len(config.view_schemas)
            sampling["n_overlaps"] = len(config.overlap_schemas)
            sampling["domain_size"] = config.domain_size
            sampling["n_tuples"] = n_tuples
            sampling["n_atoms"] = len(cq.atoms)
            sampling["seed"] = seed_i
            sampling["config_id"] = config_i
            sampling["query_id"] = q_i
            sampling["record_id"] = record_id

            # Serialise config and query for exhaustive check
            sampling["_config"] = _config_to_dict(config)
            sampling["_query"] = _cq_to_dict(cq)

            records.append(sampling)
            record_id += 1

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Exhaustive identifiability verification")
    p.add_argument("--mini", action="store_true", help="Fast smoke test")
    p.add_argument(
        "--sampling-results",
        type=Path,
        default=None,
        help="Existing e_cert_benchmark JSON (optional; used only to cross-check record counts)",
    )
    p.add_argument(
        "--mode",
        choices=["failing", "all"],
        default="failing",
        help="failing: exhaustive check only for witness_found=False; all: check every record",
    )
    p.add_argument("--n-configs", type=int, default=200)
    p.add_argument("--queries-per-config", type=int, default=5)
    p.add_argument("--n-attrs-range", nargs="+", type=int, default=[4, 6, 8])
    p.add_argument("--domain-size", type=int, default=3)
    p.add_argument("--n-fds-range", nargs="+", type=int, default=[0, 2, 4])
    p.add_argument("--n-witness-samples", type=int, default=50_000)
    p.add_argument("--n-tuples", type=int, default=1)
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Worker processes for exhaustive check. "
        "Use 1 for n_attrs≤8; increase for n_attrs≥10.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    if args.mini:
        args.n_configs = 10
        args.queries_per_config = 4
        args.n_attrs_range = [4, 6]
        args.n_fds_range = [0, 2]
        args.n_witness_samples = 5_000
        args.mode = "all"

    print(
        f"Regenerating records (seed={args.seed}, n_configs={args.n_configs}, "
        f"q/config={args.queries_per_config}) …"
    )
    t0 = time.perf_counter()
    all_records = regenerate_records(
        n_configs=args.n_configs,
        queries_per_config=args.queries_per_config,
        n_attrs_range=args.n_attrs_range,
        n_fds_range=args.n_fds_range,
        domain_size=args.domain_size,
        n_witness_samples=args.n_witness_samples,
        n_tuples=args.n_tuples,
        seed=args.seed,
    )
    regen_time = time.perf_counter() - t0
    print(f"  {len(all_records)} records regenerated in {regen_time:.1f}s")

    # Cross-check with sampling results file if provided
    if args.sampling_results and args.sampling_results.exists():
        with open(args.sampling_results) as f:
            ref = json.load(f)
        ref_total = len(ref["results"])
        print(
            f"  Reference file has {ref_total} records "
            f"({'match' if ref_total == len(all_records) else 'MISMATCH'})"
        )

    # Select records for exhaustive check
    if args.mode == "failing":
        targets = [
            r for r in all_records if not r["certified"] and not r.get("witness_found", False)
        ]
        print(f"\nMode=failing: {len(targets)} records with witness_found=False")
    else:
        targets = all_records
        print(f"\nMode=all: {len(targets)} records")

    if not targets:
        print("No records to check.")
        return

    # Build task list for exhaustive check
    tasks = [(r["_config"], r["_query"], args.n_tuples, r["record_id"]) for r in targets]

    # Parallelism advisory
    max_worlds = args.domain_size ** max(args.n_attrs_range)
    if args.n_workers == 1 and max_worlds > 10_000:
        print(f"  Note: max_worlds={max_worlds:,} — consider --n-workers > 1")

    print(
        f"Running exhaustive check on {len(tasks)} tasks "
        f"({'parallel ×' + str(args.n_workers) if args.n_workers > 1 else 'single-threaded'}) …"
    )

    t1 = time.perf_counter()
    exhaustive_results = exhaustive_check_batch(
        tasks,
        n_workers=args.n_workers,
        chunksize=max(1, len(tasks) // max(1, args.n_workers * 4)),
    )
    exhaustive_time = time.perf_counter() - t1
    print(f"  Exhaustive check done in {exhaustive_time:.2f}s")

    # Merge exhaustive results back into records
    ex_by_id = {er["record_id"]: er for er in exhaustive_results}
    for r in all_records:
        rid = r["record_id"]
        if rid in ex_by_id:
            er = ex_by_id[rid]
            r["exhaustive_run"] = True
            r["exhaustive_universally_identifiable"] = er["exhaustive_universally_identifiable"]
            r["exhaustive_n_legal_worlds"] = er["exhaustive_n_legal_worlds"]
            r["exhaustive_n_obs_groups"] = er["exhaustive_n_obs_groups"]
            r["exhaustive_witness_found"] = er["exhaustive_witness_found"]
            r["exhaustive_witness"] = er["exhaustive_witness"]
        else:
            r["exhaustive_run"] = False

    # Strip internal fields before saving
    for r in all_records:
        r.pop("_config", None)
        r.pop("_query", None)

    # Summary statistics
    checked = [r for r in all_records if r.get("exhaustive_run")]
    cert = [r for r in all_records if r["certified"]]
    nc = [r for r in all_records if not r["certified"]]
    sampling_found = [r for r in nc if r.get("witness_found")]
    sampling_not_found = [r for r in nc if not r.get("witness_found")]
    ex_univ_id = [r for r in checked if r.get("exhaustive_universally_identifiable")]
    ex_witness = [r for r in checked if r.get("exhaustive_witness_found")]

    print("\n=== Exhaustive verification results ===")
    print(f"  Total records:      {len(all_records)}")
    print(f"  Certified:          {len(cert)}")
    print(f"  Non-certified:      {len(nc)}")
    print(
        f"    Sampling witness: {len(sampling_found)}/{len(nc)} "
        f"({100 * len(sampling_found) / max(1, len(nc)):.1f}%)"
    )
    if sampling_not_found:
        print(f"    No sampling witness: {len(sampling_not_found)}")
    print(f"\n  Exhaustive checked: {len(checked)} records")
    if checked:
        print(f"    Universally identifiable: {len(ex_univ_id)}/{len(checked)}")
        print(f"    Exhaustive witness found: {len(ex_witness)}/{len(checked)}")

    # Specifically for the failing cases
    if args.mode == "failing" and sampling_not_found:
        print(f"\n  === Failing-case breakdown ({len(sampling_not_found)} records) ===")
        for r in sampling_not_found:
            univ_id = r.get("exhaustive_universally_identifiable", "N/A")
            ex_w = r.get("exhaustive_witness_found", "N/A")
            nw = r.get("exhaustive_n_legal_worlds", "N/A")
            ng = r.get("exhaustive_n_obs_groups", "N/A")
            print(
                f"    n_attrs={r['n_attrs']} n_fds={r['n_fds']} "
                f"n_truly_free={r['n_truly_free']} | "
                f"univ_identifiable={univ_id} ex_witness={ex_w} "
                f"worlds={nw} groups={ng}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e_exhaustive_check_{ts}.json"
    save_results(
        all_records,
        out,
        meta={
            "experiment": "e_exhaustive_check",
            "timestamp": ts,
            "mode": args.mode,
            "n_configs": args.n_configs,
            "queries_per_config": args.queries_per_config,
            "n_attrs_range": args.n_attrs_range,
            "n_fds_range": args.n_fds_range,
            "domain_size": args.domain_size,
            "n_witness_samples": args.n_witness_samples,
            "n_tuples": args.n_tuples,
            "n_workers": args.n_workers,
            "seed": args.seed,
            "regen_time_s": round(regen_time, 2),
            "exhaustive_time_s": round(exhaustive_time, 2),
            "n_exhaustive_checked": len(checked),
            "n_universally_identifiable": len(ex_univ_id),
            "n_exhaustive_witness_found": len(ex_witness),
        },
    )


if __name__ == "__main__":
    main()

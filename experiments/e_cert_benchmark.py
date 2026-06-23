"""
Finite exact certification benchmark  (RQ1 in the redesigned evaluation)

For each sampled (config, query) pair this experiment:
  1. Runs the certification check (sufficient certificate from Theorem 1).
  2. If certified: samples worlds to confirm no observation conflict exists.
  3. If non-certified:
       n_attrs ≤ EXHAUSTIVE_THRESHOLD (default 8): exhaustive enumerator over
         all d^n legal worlds under relational FD semantics — the correct oracle
         for universal identifiability (no false negatives from resolver bias).
       n_attrs > EXHAUSTIVE_THRESHOLD: sampling-based witness finder (fallback).

Ground truth is exact for certified queries (the certificate is both
sufficient and necessary for the query class tested here, per Thm:iv-cq-iff).
For non-certified queries with small schemas the exhaustive verifier is the
ground-truth oracle: it tests universal identifiability, not per-instance
identifiability restricted to a fixed resolver.

Output: JSON file in results/ with ≥500 records (≥50 non-certified with witnesses).

Quick test (< 15 s on CPU):
    python -m experiments.e_cert_benchmark --mini

Full run:
    python -m experiments.e_cert_benchmark --n-configs 200 --queries-per-config 5 --seed 0
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data.exhaustive import exhaustive_check
from data.synthetic import Atom, BooleanCQ, Config, SyntheticBenchmark
from data.utils import FD, augmented_overlap, check_certificate
from data.witness import check_and_witness, truly_free_attrs
from experiments.runner import save_results

# Exhaustive enumeration is feasible (< 1 ms/query) for d^n ≤ 3^8 = 6561 worlds.
EXHAUSTIVE_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Random schema / config generator
# ---------------------------------------------------------------------------


def _random_fds(n_attrs: int, n_fds: int, rng: np.random.Generator) -> list[FD]:
    fds: list[FD] = []
    seen: set = set()
    attrs = list(range(n_attrs))
    for _ in range(n_fds * 20):
        if len(fds) >= n_fds:
            break
        lhs_size = int(rng.integers(1, min(3, n_attrs - 1) + 1))
        lhs = frozenset(map(int, rng.choice(attrs, size=lhs_size, replace=False)))
        rest = [a for a in attrs if a not in lhs]
        if not rest:
            continue
        rhs = int(rng.choice(rest))
        key = (lhs, rhs)
        if key not in seen:
            seen.add(key)
            fds.append((lhs, rhs))
    return fds


def random_config(
    n_attrs: int,
    n_views: int,
    n_overlaps: int,
    n_fds: int,
    domain_size: int,
    rng: np.random.Generator,
) -> Config:
    """
    Sample a random configuration with guaranteed certified-view support.

    Construction order:
      1. Generate FDs and overlaps.
      2. Compute the observable universe (Σ-closure of overlaps).
      3. Create at least one view with schema ⊆ observable universe
         (enables certified queries).
      4. Create the remaining views randomly (may extend outside the observable
         universe, enabling non-certified queries).
    """
    attrs = list(range(n_attrs))

    fds = _random_fds(n_attrs, n_fds, rng)

    # Overlaps: small random subsets
    overlap_schemas: list[frozenset[int]] = []
    for _ in range(n_overlaps):
        size = int(rng.integers(1, max(2, n_attrs // 3) + 1))
        o = frozenset(map(int, rng.choice(attrs, size=min(size, n_attrs), replace=False)))
        if o not in overlap_schemas:
            overlap_schemas.append(o)
    if not overlap_schemas:
        overlap_schemas.append(frozenset({0}))

    # Σ-closure of each individual overlap (certification requires subset of a SINGLE closure)
    individual_closures: list[frozenset[int]] = [
        augmented_overlap(o, fds) for o in overlap_schemas
    ]

    max_view_size = max(2, n_attrs // 2 + 1)
    view_schemas: dict[int, frozenset[int]] = {}

    # View 0: guaranteed to be inside one specific Σ-closure (certified view)
    # Pick the largest closure to maximise the chance of generating interesting certified queries.
    largest_closure = max(individual_closures, key=len)
    best_closure = sorted(largest_closure)
    if len(best_closure) >= 2:
        cert_size = int(rng.integers(2, min(len(best_closure), max_view_size) + 1))
        view_schemas[0] = frozenset(
            map(int, rng.choice(best_closure, size=cert_size, replace=False))
        )
    elif best_closure:
        view_schemas[0] = frozenset(best_closure)
    else:
        view_schemas[0] = frozenset({0})

    # Remaining views: random (may extend outside observable universe)
    for v in range(1, n_views):
        size = int(rng.integers(2, max_view_size + 1))
        view_schemas[v] = frozenset(
            map(int, rng.choice(attrs, size=min(size, n_attrs), replace=False))
        )

    return Config(
        n_attrs=n_attrs,
        domain_size=domain_size,
        fds=fds,
        view_schemas=view_schemas,
        overlap_schemas=overlap_schemas,
    )


# ---------------------------------------------------------------------------
# Query generator
# ---------------------------------------------------------------------------


def _individual_closures(config: Config) -> list[frozenset[int]]:
    """Σ-closure of each individual overlap (not union)."""
    return [augmented_overlap(o, config.fds) for o in config.overlap_schemas]


def _closed_universe(config: Config) -> frozenset[int]:
    """Union of all individual Σ-closures (for reference only; do not use for certification)."""
    universe: set[int] = set()
    for o in config.overlap_schemas:
        universe.update(augmented_overlap(o, config.fds))
    return frozenset(universe)


def _make_single_atom_query(
    view_id: int,
    config: Config,
    n_constants: int,
    rng: np.random.Generator,
    prefer_free_attrs: frozenset[int] | None = None,
) -> BooleanCQ:
    """
    Single-atom query: ∃ free_vars. R_{view_id}(pattern).
    Exactly n_constants attributes get fixed constants; the rest are existential variables.
    If prefer_free_attrs is given, at least one constant is placed on a truly-free
    attribute (required for the directed witness construction to succeed).
    """
    attrs = sorted(config.view_schemas[view_id])
    n = len(attrs)
    n_const = min(max(n_constants, 1), n)

    # Prefer to place at least one constant on a truly-free attribute
    free_positions = (
        [i for i, a in enumerate(attrs) if a in prefer_free_attrs] if prefer_free_attrs else []
    )

    const_positions: list[int] = []
    if free_positions and n_const >= 1:
        # Guarantee one constant on a truly-free attribute
        forced = int(rng.choice(free_positions))
        const_positions.append(forced)
        remaining_positions = [i for i in range(n) if i != forced]
        extra = min(n_const - 1, len(remaining_positions))
        if extra > 0:
            const_positions += sorted(
                rng.choice(remaining_positions, size=extra, replace=False).tolist()
            )
    else:
        const_positions = sorted(rng.choice(n, size=n_const, replace=False).tolist())

    const_set = set(const_positions)
    pattern: list[int | str] = []
    var_idx = 0
    for i in range(n):
        if i in const_set:
            pattern.append(int(rng.integers(0, config.domain_size)))
        else:
            pattern.append(f"x{var_idx}")
            var_idx += 1
    return BooleanCQ(atoms=[Atom(view_id=view_id, pattern=pattern)])


def _make_join_query(
    view_ids: list[int],
    config: Config,
    rng: np.random.Generator,
) -> BooleanCQ:
    """
    Multi-atom join query: share a variable between adjacent atoms when they
    have overlapping attribute positions.
    """
    atoms: list[Atom] = []
    # shared_var[pos] -> variable name for position p in a view's sorted attrs
    # We link view i and view i+1 by sharing a variable on a random shared attribute.
    global_var: dict[int, str] = {}  # attr_index -> variable name
    var_counter = [0]

    def fresh() -> str:
        name = f"x{var_counter[0]}"
        var_counter[0] += 1
        return name

    for view_id in view_ids:
        attrs = sorted(config.view_schemas[view_id])
        pattern: list[int | str] = []
        for a in attrs:
            if a in global_var:
                pattern.append(global_var[a])
            else:
                v = fresh()
                global_var[a] = v
                pattern.append(v)
        atoms.append(Atom(view_id=view_id, pattern=pattern))

    return BooleanCQ(atoms=atoms)


def generate_queries(
    config: Config,
    n_certified: int,
    n_noncertified: int,
    rng: np.random.Generator,
) -> list[BooleanCQ]:
    """
    Generate a mix of certified and non-certified queries for a config.

    Certified queries: views whose schemas are contained in a SINGLE Σ-closure
    (checked via check_certificate, which is the exact same predicate used at
    evaluation time).  Non-certified queries: views with at least one truly-free
    attribute, with a constant placed on that attribute to enable directed
    witness construction.
    """
    free = truly_free_attrs(config)
    n_views = len(config.view_schemas)

    # Use check_certificate on the full view schema to determine certified views
    certified_views = [
        v
        for v in range(n_views)
        if check_certificate(config.view_schemas[v], config.overlap_schemas, config.fds)
    ]
    # Non-certified view: has at least one truly-free attribute in its schema
    noncert_views = [v for v in range(n_views) if not config.view_schemas[v].isdisjoint(free)]

    queries: list[BooleanCQ] = []

    # --- Certified queries ---
    cert_pool = certified_views if certified_views else list(range(n_views))
    for _ in range(n_certified):
        v = int(rng.choice(cert_pool))
        n_const = int(rng.integers(1, len(config.view_schemas[v]) + 1))
        # No need to prefer free attrs here; we want attrs in the closed universe
        q = _make_single_atom_query(v, config, n_const, rng)
        queries.append(q)

    # --- Non-certified queries ---
    # Use single-atom queries with a constant on a truly-free attribute so the
    # directed witness constructor can always find a witness efficiently.
    # Multi-atom non-certified queries (joins) are tested separately in the
    # join-query section of the paper.
    if noncert_views:
        for _ in range(n_noncertified):
            nc_v = int(rng.choice(noncert_views))
            free_in_view = free & config.view_schemas[nc_v]
            n_const = int(rng.integers(1, len(config.view_schemas[nc_v]) + 1))
            q = _make_single_atom_query(
                nc_v,
                config,
                n_const,
                rng,
                prefer_free_attrs=free_in_view,
            )
            queries.append(q)
    else:
        # Fallback: no views with truly-free attrs; generate multi-atom queries
        # (witness may require sampling rather than directed construction)
        for _ in range(n_noncertified):
            v1 = int(rng.integers(0, n_views))
            v2 = int(rng.integers(0, n_views))
            q = _make_join_query([v1, v2], config, rng)
            queries.append(q)

    return queries


# ---------------------------------------------------------------------------
# Single benchmark record
# ---------------------------------------------------------------------------


def is_degenerate(
    cq: BooleanCQ,
    bench: SyntheticBenchmark,
    n_tuples: int,
    n_probe: int = 300,
) -> bool:
    """
    Return True if Q is always True or always False over n_probe sampled worlds.
    Degenerate queries are skipped: they have no interesting non-identifiability
    (constant queries give no witness; always-True queries have nothing to flip).
    """
    seen_true = seen_false = False
    for _ in range(n_probe):
        w = bench.generate_world(n_tuples)
        q = bench.evaluate(cq, w)
        if q:
            seen_true = True
        else:
            seen_false = True
        if seen_true and seen_false:
            return False
    return True  # never saw both values


def _exhaustive_record(cq: BooleanCQ, config: Config, n_tuples: int) -> dict:
    """
    Run exhaustive check and return a dict compatible with check_and_witness output.

    Uses relational FD semantics (not the resolver model): for n_tuples=1 all
    d^n single-row matrices are legal, so this tests UNIVERSAL identifiability.
    """
    result = exhaustive_check(cq, config, n_tuples=n_tuples, fast_exit=True)
    free = truly_free_attrs(config)
    return {
        "certified": False,
        "footprint": sorted(cq.footprint(config)),
        "n_atoms": len(cq.atoms),
        "n_truly_free": len(free & cq.footprint(config)),
        "witness_found": result.witness_w_true is not None,
        "witness_method": "exhaustive",
        "witness_valid": result.witness_w_true is not None,
        "witness_checks": None,
        "n_samples_to_find": result.n_legal_worlds,
        "witness": (
            {"w_true": result.witness_w_true, "w_false": result.witness_w_false}
            if result.witness_w_true is not None
            else None
        ),
        "exhaustive_n_legal_worlds": result.n_legal_worlds,
        "exhaustive_n_obs_groups": result.n_obs_groups,
    }


def run_benchmark_record(
    config: Config,
    cq: BooleanCQ,
    seed: int,
    n_witness_samples: int,
    n_verify_samples: int,
    n_tuples: int,
) -> dict | None:
    """
    Returns None for degenerate queries (Q always True or always False).

    Oracle selection for non-certified queries:
      n_attrs ≤ EXHAUSTIVE_THRESHOLD and n_tuples == 1:
        exhaustive check (relational semantics, zero false negatives)
      otherwise:
        sampling-based witness finder (via check_and_witness)
    """
    bench = SyntheticBenchmark(config, seed=seed)

    if is_degenerate(cq, bench, n_tuples):
        return None

    certified = cq.is_certified(config)
    use_exhaustive = not certified and n_tuples == 1 and config.n_attrs <= EXHAUSTIVE_THRESHOLD

    t0 = time.perf_counter()
    if use_exhaustive:
        record = _exhaustive_record(cq, config, n_tuples)
    else:
        record = check_and_witness(
            cq=cq,
            config=config,
            bench=bench,
            n_samples=n_witness_samples,
            n_tuples=n_tuples,
        )
    elapsed = time.perf_counter() - t0

    record["elapsed_s"] = round(elapsed, 4)
    record["n_tuples"] = n_tuples
    record["n_attrs"] = config.n_attrs
    record["n_fds"] = len(config.fds)
    record["n_views"] = len(config.view_schemas)
    record["n_overlaps"] = len(config.overlap_schemas)
    record["domain_size"] = config.domain_size
    record["seed"] = seed
    record["n_atoms"] = len(cq.atoms)

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Finite exact certification benchmark (RQ1)")
    p.add_argument("--mini", action="store_true", help="Fast smoke-test (< 15 s)")
    p.add_argument(
        "--n-configs", type=int, default=200, help="Number of random schemas to generate"
    )
    p.add_argument(
        "--queries-per-config",
        type=int,
        default=5,
        help="Queries per schema (split evenly between certified/non-certified)",
    )
    p.add_argument(
        "--n-attrs-range", nargs="+", type=int, default=[4, 6, 8], help="Attribute counts to sweep"
    )
    p.add_argument("--domain-size", type=int, default=3)
    p.add_argument(
        "--n-fds-range", nargs="+", type=int, default=[0, 2, 4], help="FD counts to sweep"
    )
    p.add_argument(
        "--n-witness-samples",
        type=int,
        default=50_000,
        help="Max samples for witness search per query",
    )
    p.add_argument(
        "--n-verify-samples",
        type=int,
        default=5_000,
        help="Samples for certified-query verification",
    )
    p.add_argument(
        "--n-tuples",
        type=int,
        default=1,
        help="World size (m); 1 = degenerate single-tuple, >1 = relational",
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
        args.n_verify_samples = 500

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    n_certified_target = args.queries_per_config // 2
    n_noncert_target = args.queries_per_config - n_certified_target

    print(f"Generating {args.n_configs} configs × {args.queries_per_config} queries …")

    stats = {
        "total": 0,
        "certified": 0,
        "noncertified": 0,
        "witness_found": 0,
        "witness_not_found": 0,
        "cert_violations": 0,
        "skipped_degenerate": 0,
        "witness_exhaustive": 0,
        "witness_directed": 0,
        "witness_sampling": 0,
    }

    for config_i in range(args.n_configs):
        n_attrs = int(rng.choice(args.n_attrs_range))
        n_fds = int(rng.choice(args.n_fds_range))
        n_views = int(rng.integers(2, max(3, n_attrs // 2) + 1))
        n_overlaps = int(rng.integers(1, max(2, n_views // 2) + 1))

        config = random_config(
            n_attrs=n_attrs,
            n_views=n_views,
            n_overlaps=n_overlaps,
            n_fds=n_fds,
            domain_size=args.domain_size,
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
            rec = run_benchmark_record(
                config=config,
                cq=cq,
                seed=seed_i,
                n_witness_samples=args.n_witness_samples,
                n_verify_samples=args.n_verify_samples,
                n_tuples=args.n_tuples,
            )

            if rec is None:
                stats["skipped_degenerate"] += 1
                continue

            rec["config_id"] = config_i
            rec["query_id"] = q_i
            records.append(rec)

            stats["total"] += 1
            if rec["certified"]:
                stats["certified"] += 1
                if rec.get("n_violations", 0) > 0:
                    stats["cert_violations"] += 1
            else:
                stats["noncertified"] += 1
                if rec.get("witness_found"):
                    stats["witness_found"] += 1
                    method = rec.get("witness_method", "")
                    if method == "exhaustive":
                        stats["witness_exhaustive"] += 1
                    elif method == "directed":
                        stats["witness_directed"] += 1
                    elif method == "sampling":
                        stats["witness_sampling"] += 1
                else:
                    stats["witness_not_found"] += 1

        if (config_i + 1) % max(1, args.n_configs // 10) == 0:
            c = stats["certified"]
            nc = stats["noncertified"]
            wf = stats["witness_found"]
            print(
                f"  config {config_i + 1:4d}/{args.n_configs} | "
                f"records={stats['total']} | "
                f"certified={c} non-cert={nc} | "
                f"witnesses={wf}/{nc}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e_cert_benchmark_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e_cert_benchmark",
            "timestamp": ts,
            "n_configs": args.n_configs,
            "queries_per_config": args.queries_per_config,
            "n_attrs_range": args.n_attrs_range,
            "domain_size": args.domain_size,
            "n_fds_range": args.n_fds_range,
            "n_witness_samples": args.n_witness_samples,
            "n_verify_samples": args.n_verify_samples,
            "n_tuples": args.n_tuples,
            "seed": args.seed,
            **stats,
        },
    )

    print(f"\n=== Certification benchmark ({stats['total']} records) ===")
    c, nc = stats["certified"], stats["noncertified"]
    print(f"  Certified:      {c} ({100 * c / max(1, stats['total']):.1f}%)")
    print(f"  Non-certified:  {nc} ({100 * nc / max(1, stats['total']):.1f}%)")
    print(f"  Skipped (degenerate Q): {stats['skipped_degenerate']}")
    if nc > 0:
        wf = stats["witness_found"]
        print(f"  Witnesses found: {wf}/{nc} ({100 * wf / nc:.1f}%)")
        if stats["witness_exhaustive"]:
            print(f"    exhaustive (relational):  {stats['witness_exhaustive']}")
        if stats["witness_directed"]:
            print(f"    directed (resolver):       {stats['witness_directed']}")
        if stats["witness_sampling"]:
            print(f"    sampling (resolver):       {stats['witness_sampling']}")
        if stats["witness_not_found"]:
            print(
                f"  WARNING: {stats['witness_not_found']} non-certified queries "
                f"yielded no witness (increase --n-witness-samples)"
            )
    if stats["cert_violations"]:
        print(
            f"  CRITICAL: {stats['cert_violations']} certified queries had "
            f"observation conflicts (theory violation)"
        )
    else:
        print(
            f"  Certified consistency: all {c} certified queries passed "
            f"empirical observation-consistency check"
        )


if __name__ == "__main__":
    main()

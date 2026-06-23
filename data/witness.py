"""
Exact witness finder for query non-identifiability.

A witness for a non-certifiable Boolean CQ Q under a configuration is a pair
(w, w') of worlds such that:
  - Obs(w) = Obs(w')   [same Σ-closed overlap projections]
  - Q(w)  ≠  Q(w')    [different query answers]

Two strategies are implemented:

1. **Directed** (fast, single-atom queries): given a query with a constant
   constraint on a truly-free attribute, construct w with the constant
   satisfied, then w' with a different value.  Truly-free attributes are those
   outside every Σ-closure AND not transitively determining any observable
   attribute; changing them cannot affect Obs(w).

2. **Sampling** (general): sample worlds, group by observation key, return the
   first group containing both a True and a False instance.  Fast when the
   observation space is small; falls back gracefully otherwise.

Usage
-----
    bench = SyntheticBenchmark(config, seed=0)
    result = find_witness(cq, config, bench, n_samples=50_000, n_tuples=1)
    if result is not None:
        w, w_prime = result.w_true, result.w_false
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .synthetic import BooleanCQ, Config, SyntheticBenchmark
from .utils import augmented_overlap, fd_closure

# ---------------------------------------------------------------------------
# Observation key
# ---------------------------------------------------------------------------


def observation_key(
    world: np.ndarray,
    augmented_overlaps: Sequence[frozenset[int]],
) -> tuple:
    """
    Canonical observation of a world: tuple of sorted (attrs, values) pairs for
    each Σ-closed overlap, representing w|_H for each H ∈ ObsFam.

    Rows are sorted for order-independence (bag semantics).
    """
    parts: list[tuple] = []
    for aug in augmented_overlaps:
        attrs = sorted(aug)
        proj = world[:, attrs]
        rows = tuple(sorted(map(tuple, proj.tolist())))
        parts.append((tuple(attrs), rows))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Truly-free attributes
# ---------------------------------------------------------------------------


def truly_free_attrs(config: Config) -> frozenset[int]:
    """
    Attributes that can be changed in one world to produce another world with
    the same observation.

    An attribute a is truly free if:
      (a) a ∉ Σ-closure(O) for any overlap O  (a is not observable), AND
      (b) Σ-closure({a}) ∩ observable = ∅     (a does not determine any
          observable attribute through FD chains).

    Changing a truly-free attribute and then re-applying FD resolvers leaves
    all observable projections unchanged.
    """
    observable: set[int] = set()
    for o in config.overlap_schemas:
        observable.update(augmented_overlap(o, config.fds))

    safe: set[int] = set()
    for a in range(config.n_attrs):
        if a in observable:
            continue
        if fd_closure(frozenset({a}), config.fds).isdisjoint(observable):
            safe.add(a)
    return frozenset(safe)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class WitnessResult:
    """A confirmed non-identifiability witness."""

    w_true: np.ndarray  # world where Q = True
    w_false: np.ndarray  # world where Q = False; Obs(w_false) = Obs(w_true)
    obs_key: tuple  # shared observation
    n_samples_tried: int
    method: str  # "directed" or "sampling"


# ---------------------------------------------------------------------------
# Strategy 1: Directed witness (single-atom queries)
# ---------------------------------------------------------------------------


def directed_witness(
    cq: BooleanCQ,
    bench: SyntheticBenchmark,
    n_tuples: int = 1,
    max_retries: int = 500,
) -> WitnessResult | None:
    """
    Directed witness construction for single-atom queries.

    Algorithm:
      1. Sample worlds until Q(w) = True.
      2. Identify a "mutable free attribute": truly-free AND not the RHS of any
         FD (so that modifying it survives re-application of the resolvers).
      3. Build w' by changing the mutable free attribute's constant to a
         different value; apply resolvers (the change persists since the attr
         is not a resolver RHS); verify Q(w') = False and Obs(w') = Obs(w).

    Returns None if the query doesn't qualify or no witness is found.
    """
    if len(cq.atoms) != 1:
        return None

    atom = cq.atoms[0]
    config = bench.config
    free = truly_free_attrs(config)
    view_attrs = sorted(config.view_schemas[atom.view_id])

    # Attributes that are truly-free AND not the RHS of any FD.
    # Only these can be directly mutated: after _apply_resolvers the change persists.
    rhs_of_fd: frozenset[int] = frozenset(rhs for _, rhs in config.fds)
    mutable_free = free - rhs_of_fd

    # Find constant constraints on mutable-free attributes
    mutable_constants: list[tuple[int, int]] = []  # (attr_index, required_value)
    for attr, pat in zip(view_attrs, atom.pattern):
        if isinstance(pat, int) and attr in mutable_free:
            mutable_constants.append((attr, pat))

    if not mutable_constants:
        return None

    aug_overlaps = bench._augmented_overlaps
    rng = bench.rng
    d = config.domain_size

    # Precompute: for the first mutable constant, what alternative values exist?
    attr0, val0 = mutable_constants[0]
    alt_vals = [v for v in range(d) if v != val0]
    if not alt_vals:
        return None  # domain_size == 1, degenerate

    for attempt in range(max_retries):
        # Sample a world and check Q = True (no forcing; resolvers applied once)
        w = bench.generate_world(n_tuples)
        if not bench.evaluate(cq, w):
            continue

        # Build w' by mutating attr0 to an alternative value.
        # Since attr0 ∉ rhs_of_fd, the change survives _apply_resolvers.
        alt_val = alt_vals[int(rng.integers(0, len(alt_vals)))]
        w_prime = w.copy()
        w_prime[:, attr0] = alt_val
        bench._apply_resolvers(w_prime)

        obs_w = observation_key(w, aug_overlaps)
        obs_wp = observation_key(w_prime, aug_overlaps)

        if obs_w != obs_wp:
            continue  # guard: mutable free attrs should not affect obs

        if not bench.evaluate(cq, w_prime):
            return WitnessResult(
                w_true=w,
                w_false=w_prime,
                obs_key=obs_w,
                n_samples_tried=attempt + 1,
                method="directed",
            )

    return None


# ---------------------------------------------------------------------------
# Strategy 2: Sampling-based witness
# ---------------------------------------------------------------------------


def find_witness(
    cq: BooleanCQ,
    bench: SyntheticBenchmark,
    n_samples: int = 100_000,
    n_tuples: int = 1,
) -> WitnessResult | None:
    """
    Sample worlds until finding (w, w') with same Obs but Q(w)≠Q(w').

    Tries directed construction first (fast for qualifying single-atom queries),
    then falls back to random sampling.
    """
    # Try directed first
    dr = directed_witness(cq, bench, n_tuples=n_tuples)
    if dr is not None:
        return dr

    # Fall back to sampling
    aug_overlaps = bench._augmented_overlaps
    groups: dict[tuple, dict[bool, np.ndarray]] = {}

    for i in range(n_samples):
        w = bench.generate_world(n_tuples)
        key = observation_key(w, aug_overlaps)
        q = bench.evaluate(cq, w)

        if key not in groups:
            groups[key] = {}
        group = groups[key]

        if q not in group:
            group[q] = w

        if True in group and False in group:
            return WitnessResult(
                w_true=group[True],
                w_false=group[False],
                obs_key=key,
                n_samples_tried=i + 1,
                method="sampling",
            )

    return None


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_witness(
    result: WitnessResult,
    cq: BooleanCQ,
    bench: SyntheticBenchmark,
) -> dict[str, bool]:
    """
    Check that result is a valid witness:
      (a) Obs(w_true) == Obs(w_false)
      (b) Q(w_true) == True
      (c) Q(w_false) == False
    """
    aug_overlaps = bench._augmented_overlaps
    obs_true = observation_key(result.w_true, aug_overlaps)
    obs_false = observation_key(result.w_false, aug_overlaps)
    return {
        "obs_match": obs_true == obs_false,
        "q_true_correct": bench.evaluate(cq, result.w_true) is True,
        "q_false_correct": bench.evaluate(cq, result.w_false) is False,
    }


def verify_certified(
    cq: BooleanCQ,
    bench: SyntheticBenchmark,
    n_samples: int = 5_000,
    n_tuples: int = 1,
) -> dict[str, Any]:
    """
    Empirically verify certified behavior: all sampled worlds with the same
    observation should give the same query answer.

    Returns {"consistent": bool, "n_groups": int, "n_violations": int}.
    A violation means two worlds with the same Obs gave different Q answers.
    """
    aug_overlaps = bench._augmented_overlaps
    groups: dict[tuple, bool] = {}
    violations = 0

    for _ in range(n_samples):
        w = bench.generate_world(n_tuples)
        key = observation_key(w, aug_overlaps)
        q = bench.evaluate(cq, w)
        if key in groups:
            if groups[key] != q:
                violations += 1
        else:
            groups[key] = q

    return {
        "consistent": violations == 0,
        "n_groups": len(groups),
        "n_violations": violations,
    }


# ---------------------------------------------------------------------------
# Convenience: find or verify depending on certification status
# ---------------------------------------------------------------------------


def check_and_witness(
    cq: BooleanCQ,
    config: Config,
    bench: SyntheticBenchmark,
    n_samples: int = 50_000,
    n_tuples: int = 1,
) -> dict:
    """
    Run certification check and, for non-certified queries, attempt to find
    an explicit witness.  Returns a result dict suitable for JSON serialization.
    """
    certified = cq.is_certified(config)
    record: dict = {
        "certified": certified,
        "footprint": sorted(cq.footprint(config)),
        "n_atoms": len(cq.atoms),
        "n_truly_free": len(truly_free_attrs(config) & cq.footprint(config)),
    }

    if certified:
        v = verify_certified(cq, bench, n_samples=min(n_samples, 5_000), n_tuples=n_tuples)
        record.update(
            {
                "verification": "consistent" if v["consistent"] else "VIOLATION",
                "n_obs_groups": v["n_groups"],
                "n_violations": v["n_violations"],
                "witness": None,
            }
        )
    else:
        result = find_witness(cq, bench, n_samples=n_samples, n_tuples=n_tuples)
        if result is not None:
            checks = verify_witness(result, cq, bench)
            record.update(
                {
                    "witness_found": True,
                    "witness_method": result.method,
                    "witness_valid": all(checks.values()),
                    "witness_checks": checks,
                    "n_samples_to_find": result.n_samples_tried,
                    "witness": {
                        "w_true": result.w_true.tolist(),
                        "w_false": result.w_false.tolist(),
                    },
                }
            )
        else:
            # No witness found: run verify_certified to check if query is actually
            # identifiable (certificate false negative) or just hard to witness.
            v = verify_certified(cq, bench, n_samples=min(n_samples, 5_000), n_tuples=n_tuples)
            likely_identifiable = v["consistent"] and v["n_groups"] >= 3
            record.update(
                {
                    "witness_found": False,
                    "witness_method": None,
                    "witness_valid": None,
                    "witness_checks": None,
                    "n_samples_to_find": n_samples,
                    "witness": None,
                    # Diagnostic fields for false-negative analysis
                    "no_witness_obs_consistent": v["consistent"],
                    "no_witness_n_groups": v["n_groups"],
                    "no_witness_likely_identifiable": likely_identifiable,
                }
            )

    return record

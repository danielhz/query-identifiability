"""
e_minaug_realworld.py — Greedy-MinAug applied to real WDC and BibInteg schemas.

Closes the RQ2--RQ3 narrative gap: RQ2 identifies which WDC queries are non-certified;
this experiment shows Greedy-MinAug automatically prescribing the minimum overlap
augmentation needed to certify each one.

Problem formulation
-------------------
Given overlap Ω, FDs Σ, and a query Q with atom obligations B_Q = {U_1,...,U_m}:
  - An *action* A is a frozenset of attributes added to the overlap.
  - The obligation coverage of A (w.r.t. existing overlap) is:
        coverage(A) = {j | attr(U_j) ⊆ fd_closure(Ω ∪ A, Σ)}
    (note: uses Ω ∪ A as the seed, so the existing overlap contributes for free)
  - Greedy-MinAug selects actions greedily until all obligations are covered.

For the WDC schema, candidate actions are subsets of uncovered attributes
{4, 5, 6} (rating_bucket, n_reviews_log, source_id).  For BibInteg, all queries
are already certified so the algorithm returns an empty augmentation immediately.

Expected results (WDC, tight instance)
---------------------------------------
  Q_highly_rated   : add {4}      (expose rating_bucket in the shared overlap)
  Q_reviewed       : add {5, 6}   (expose n_reviews_log + source_id)
  Q_popular        : add {4} + {5, 6}  (covers both atoms in two steps)
  Q_available      : already certified → empty augmentation
  Q_cheap          : already certified → empty augmentation

CrossKG-DBLP (OpenAlex × DBLP, the real-witness dataset):
  Q_publisher  : already certified (publisher is in the shared DOI) → empty
  Q_large_team : add {2}  (expose large_team_bit — i.e. the two sources must
                 reconcile author counts for the query to become identifiable)

Usage
-----
  python -m experiments.e_minaug_realworld
  python -m experiments.e_minaug_realworld --verbose
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import data.amazon_google as ag
import data.bibinteg as bib
import data.crosskg_dblp as ckg
import data.fodors_zagat as fz
import data.wdc as wdc
from data.utils import FD, augmented_overlap, fd_closure
from experiments.runner import save_results

# ---------------------------------------------------------------------------
# Obligation-index MinAug (correct real-world formulation)
# ---------------------------------------------------------------------------


def obligation_coverage(
    action: frozenset[int],
    atom_footprints: list[frozenset[int]],
    existing_overlap: frozenset[int],
    fds: list[FD],
) -> frozenset[int]:
    """
    Return set of obligation indices covered when action A is added to the overlap.

    Uses fd_closure(Ω ∪ A, Σ) so that attributes already derivable from the
    existing overlap Ω are credited for free — only the residual gap need be covered.
    """
    aug = fd_closure(existing_overlap | action, fds)
    return frozenset(j for j, fp in enumerate(atom_footprints) if fp <= aug)


def greedy_minaug_obligations(
    atom_footprints: list[frozenset[int]],
    existing_overlap: frozenset[int],
    candidate_actions: list[frozenset[int]],
    fds: list[FD],
) -> list[frozenset[int]] | None:
    """
    Greedy-MinAug over obligation indices.

    Returns the greedy-selected list of actions or None if infeasible.
    Already-certified obligations are credited immediately (no action needed).
    """
    all_obs = frozenset(range(len(atom_footprints)))

    # Pre-compute per-candidate coverage
    coverages: list[frozenset[int]] = [
        obligation_coverage(A, atom_footprints, existing_overlap, fds) for A in candidate_actions
    ]

    # Initially covered by the existing overlap (no action taken yet)
    aug0 = fd_closure(existing_overlap, fds)
    covered = frozenset(j for j, fp in enumerate(atom_footprints) if fp <= aug0)

    selected: list[frozenset[int]] = []
    remaining = list(range(len(candidate_actions)))

    while covered < all_obs:
        best_i, best_gain = None, 0
        for i in remaining:
            gain = len(coverages[i] - covered)
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None:
            return None  # infeasible
        selected.append(candidate_actions[best_i])
        covered |= coverages[best_i]
        remaining.remove(best_i)

    return selected


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(
    uncovered_attrs: frozenset[int],
    max_action_size: int = 2,
) -> list[frozenset[int]]:
    """
    Enumerate all subsets of uncovered_attrs up to max_action_size.
    Singletons + pairs covers all practical real-world cases.
    """
    candidates: list[frozenset[int]] = []
    attrs = sorted(uncovered_attrs)
    for size in range(1, min(max_action_size, len(attrs)) + 1):
        for combo in combinations(attrs, size):
            candidates.append(frozenset(combo))
    return candidates


# ---------------------------------------------------------------------------
# Per-query analysis
# ---------------------------------------------------------------------------


def _attr_name(dataset: str, attr: int) -> str:
    """Human-readable attribute label for reporting."""
    wdc_names = {
        0: "brand_hash",
        1: "model_hash",
        2: "category_id",
        3: "price_bucket",
        4: "rating_bucket",
        5: "n_reviews_log",
        6: "source_id",
        7: "in_stock",
    }
    bib_names = {
        0: "title_hash",
        1: "author_hash",
        2: "year",
        3: "venue_hash",
        4: "doi_prefix",
        5: "n_authors",
        6: "decade",
    }
    crosskg_names = {
        0: "doi_id",
        1: "publisher_bit",
        2: "large_team_bit",
        3: "source_id",
    }
    amazon_google_names = {0: "pair_id", 1: "catalog_bit", 2: "expensive_bit", 3: "source_id"}
    fodors_zagat_names = {0: "pair_id", 1: "segment_bit", 2: "cuisine_bit", 3: "source_id"}
    table = {
        "wdc": wdc_names,
        "crosskg_dblp": crosskg_names,
        "amazon_google": amazon_google_names,
        "fodors_zagat": fodors_zagat_names,
    }.get(dataset, bib_names)
    return table.get(attr, f"attr_{attr}")


def analyse_query(
    *,
    dataset: str,
    query_name: str,
    query,
    config,
    overlap: frozenset[int],
    fds: list[FD],
    verbose: bool = False,
) -> dict:
    """Run MinAug on one query and return a result record."""
    atom_footprints = [frozenset(config.view_schemas[atom.view_id]) for atom in query.atoms]
    is_certified = query.is_certified(config)

    aug0 = augmented_overlap(overlap, fds)
    # Uncovered = footprint attributes not reachable from current overlap
    full_fp = frozenset(a for fp in atom_footprints for a in fp)
    uncovered_attrs = full_fp - aug0

    candidates = generate_candidates(uncovered_attrs, max_action_size=2)

    t0 = time.perf_counter()
    result = greedy_minaug_obligations(atom_footprints, overlap, candidates, fds)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    if is_certified:
        # Greedy should immediately return empty list (already covered)
        assert result == [], f"Certified query {query_name} returned non-empty augmentation"
        n_actions = 0
        selected_attrs: list[list[int]] = []
    elif result is None:
        n_actions = -1  # infeasible with candidate pool
        selected_attrs = []
    else:
        n_actions = len(result)
        selected_attrs = [sorted(a) for a in result]

    # Verify: with selected actions, is the query certified?
    verified = False
    if result is not None:
        aug_new = aug0
        for act in result:
            aug_new = fd_closure(aug_new | act, fds)
        verified = all(fp <= aug_new for fp in atom_footprints)

    rec = {
        "dataset": dataset,
        "query": query_name,
        "certified_before": is_certified,
        "n_atoms": len(query.atoms),
        "full_footprint": sorted(full_fp),
        "aug_overlap_before": sorted(aug0),
        "uncovered_attrs": sorted(uncovered_attrs),
        "n_candidates": len(candidates),
        "n_actions_greedy": n_actions,
        "selected_actions": selected_attrs,
        "verified_certified_after": verified,
        "greedy_time_us": round(elapsed_us, 3),
    }

    if verbose:
        tag = "C" if is_certified else "U"
        print(f"\n  [{dataset}] {query_name} [{tag}]")
        print(f"    footprint         : {sorted(full_fp)}")
        print(f"    aug_overlap       : {sorted(aug0)}")
        print(f"    uncovered attrs   : {sorted(uncovered_attrs)}")
        if is_certified:
            print("    → already certified; no augmentation needed")
        elif result is None:
            print("    → INFEASIBLE with candidate pool")
        else:
            print(f"    → {n_actions} action(s) needed:")
            for act in result:
                names = [_attr_name(dataset, a) for a in sorted(act)]
                print(f"       + {{{', '.join(names)}}}")
            print(f"    verified certified: {verified}")
        print(f"    runtime: {elapsed_us:.2f} µs")

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MinAug on real WDC and BibInteg schemas")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    records: list[dict] = []

    # ------------------------------------------------------------------
    # WDC-Product
    # ------------------------------------------------------------------
    wdc_overlap = wdc.OVERLAP_SCHEMAS[0]  # {0, 1, 2}
    wdc_queries = [
        ("Q_available", wdc.Q_AVAILABLE),
        ("Q_cheap", wdc.Q_CHEAP),
        ("Q_highly_rated", wdc.Q_HIGHLY_RATED),
        ("Q_reviewed", wdc.Q_REVIEWED),
        ("Q_popular", wdc.Q_POPULAR),
    ]

    if args.verbose:
        aug_wdc = augmented_overlap(wdc_overlap, wdc.FDS)
        print("\n=== WDC-Product ===")
        print(f"  Base overlap  : {sorted(wdc_overlap)}")
        print(f"  Augmented Ω̃  : {sorted(aug_wdc)}")
        print(f"  FDs           : {[(sorted(lhs), rhs) for lhs, rhs in wdc.FDS]}")

    for q_name, query in wdc_queries:
        rec = analyse_query(
            dataset="wdc",
            query_name=q_name,
            query=query,
            config=wdc.CONFIG,
            overlap=wdc_overlap,
            fds=wdc.FDS,
            verbose=args.verbose,
        )
        records.append(rec)

    # ------------------------------------------------------------------
    # BibInteg
    # ------------------------------------------------------------------
    bib_overlap = bib.OVERLAP_SCHEMAS[0]
    bib_queries = [
        ("Q_venue", bib.Q_VENUE),
        ("Q_doi", bib.Q_DOI),
        ("Q_large_team", bib.Q_LARGE_TEAM),
    ]

    if args.verbose:
        aug_bib = augmented_overlap(bib_overlap, bib.FDS)
        print("\n=== BibInteg ===")
        print(f"  Base overlap  : {sorted(bib_overlap)}")
        print(f"  Augmented Ω̃  : {sorted(aug_bib)}")
        print(f"  FDs           : {[(sorted(lhs), rhs) for lhs, rhs in bib.FDS]}")

    for q_name, query in bib_queries:
        rec = analyse_query(
            dataset="bibinteg",
            query_name=q_name,
            query=query,
            config=bib.CONFIG,
            overlap=bib_overlap,
            fds=bib.FDS,
            verbose=args.verbose,
        )
        records.append(rec)

    # ------------------------------------------------------------------
    # CrossKG-DBLP (OpenAlex × DBLP) — the real-witness dataset.
    # Q_publisher is certified (no augmentation). For Q_large_team, MinAug
    # prescribes adding the author-count attribute to the overlap, i.e. the two
    # sources must reconcile author counts for the query to become identifiable.
    # ------------------------------------------------------------------
    ckg_overlap = ckg.OVERLAP_SCHEMAS[0]  # {0} (doi)
    ckg_queries = [
        ("Q_publisher", ckg.Q_PUBLISHER),
        ("Q_large_team", ckg.Q_LARGE_TEAM),
    ]

    if args.verbose:
        aug_ckg = augmented_overlap(ckg_overlap, ckg.FDS)
        print("\n=== CrossKG-DBLP (OpenAlex × DBLP) ===")
        print(f"  Base overlap  : {sorted(ckg_overlap)}")
        print(f"  Augmented Ω̃  : {sorted(aug_ckg)}")
        print(f"  FDs           : {[(sorted(lhs), rhs) for lhs, rhs in ckg.FDS]}")

    for q_name, query in ckg_queries:
        rec = analyse_query(
            dataset="crosskg_dblp",
            query_name=q_name,
            query=query,
            config=ckg.CONFIG,
            overlap=ckg_overlap,
            fds=ckg.FDS,
            verbose=args.verbose,
        )
        records.append(rec)

    # ------------------------------------------------------------------
    # Amazon-Google and Fodors-Zagat (matched-pair, same model as CrossKG):
    # the uncertified query needs the disagreed attribute exposed in the overlap.
    # ------------------------------------------------------------------
    for ds_name, mod, ds_queries in (
        ("amazon_google", ag, [("Q_catalog", ag.Q_CATALOG), ("Q_expensive", ag.Q_EXPENSIVE)]),
        ("fodors_zagat", fz, [("Q_segment", fz.Q_SEGMENT), ("Q_cuisine", fz.Q_CUISINE)]),
    ):
        overlap = mod.OVERLAP_SCHEMAS[0]
        if args.verbose:
            print(f"\n=== {ds_name} ===")
            print(f"  Base overlap  : {sorted(overlap)}")
            print(f"  Augmented Ω̃  : {sorted(augmented_overlap(overlap, mod.FDS))}")
            print(f"  FDs           : {[(sorted(lhs), rhs) for lhs, rhs in mod.FDS]}")
        for q_name, query in ds_queries:
            records.append(
                analyse_query(
                    dataset=ds_name,
                    query_name=q_name,
                    query=query,
                    config=mod.CONFIG,
                    overlap=overlap,
                    fds=mod.FDS,
                    verbose=args.verbose,
                )
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e_minaug_realworld_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e_minaug_realworld",
            "timestamp": ts,
            "datasets": ["wdc", "bibinteg", "crosskg_dblp", "amazon_google", "fodors_zagat"],
        },
    )
    if args.verbose:
        print(f"\nSaved {len(records)} records → {out}")

    # Summary table
    if args.verbose:
        print("\n=== Summary ===")
        print(
            f"{'Dataset':10s}  {'Query':20s}  {'Cert':4s}  {'Actions':7s}  {'Added attrs':30s}  {'µs':8s}"
        )
        print("-" * 88)
        for r in records:
            cert = "C" if r["certified_before"] else "U"
            acts = str(r["n_actions_greedy"]) if r["n_actions_greedy"] >= 0 else "inf"
            attrs_str = (
                " | ".join(
                    "{" + ",".join(str(a) for a in act) + "}" for act in r["selected_actions"]
                )
                or "—"
            )
            print(
                f"{r['dataset']:10s}  {r['query']:20s}  {cert:4s}  {acts:7s}  "
                f"{attrs_str:30s}  {r['greedy_time_us']:8.2f}"
            )


if __name__ == "__main__":
    main()

"""
Data-driven witness scan for real-world datasets (RQ2 complement).

For each non-certified query in BibInteg and WDC-Product, scans the actual
dataset records for a non-identifiability witness: two records (w, w') with
  - Obs(w) = Obs(w')   (same Σ-closed overlap projection)
  - Q(w) ≠ Q(w')       (different query answers)

This complements the synthetic exhaustive benchmark (e_cert_benchmark) with
concrete real-world examples drawn from the actual dataset.

For certified queries the scan is skipped: the certificate guarantees no
witness exists.

Datasets
--------
BibInteg  — all three queries (Q_VENUE, Q_DOI, Q_LARGE_TEAM) are certified.
             No witness scan needed.
WDC-Product — Q_AVAILABLE and Q_CHEAP are certified.
               Q_HIGHLY_RATED, Q_REVIEWED, Q_POPULAR are not: scan finds witnesses.
CrossKG-DBLP (OpenAlex × DBLP) — the real-witness dataset. Q_publisher is
               certified (both sources share the DOI). Q_large_team is not: the
               two independent sources genuinely disagree on author counts, so
               the scan finds *real* (not synthesized) non-identifiability witnesses.
Amazon-Google — product-domain real-witness dataset. Q_catalog is certified
               (both sources share the matched-pair identity). Q_expensive is
               not: the two sources genuinely disagree on price → real witnesses.
Fodors-Zagat — restaurant-domain real-witness dataset (breadth). Q_segment is
               certified; Q_cuisine is not: the two guides categorize cuisine
               differently → real witnesses.

Usage
-----
  # Real data (requires downloaded CSVs in data/raw/)
  python -m experiments.e_realworld_witnesses

  # Mock data (no download required — good for CI / quick check)
  python -m experiments.e_realworld_witnesses --mock
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import data.amazon_google as ag
import data.bibinteg as bib
import data.crosskg_dblp as ckg
import data.fodors_zagat as fz
import data.wdc as wdc
from data.exhaustive import evaluate_cq
from data.synthetic import BooleanCQ, Config
from data.utils import augmented_overlap
from data.witness import observation_key
from experiments.runner import save_results

# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------


def data_witness_scan(
    records: list[Any],
    cq: BooleanCQ,
    config: Config,
) -> dict:
    """
    Scan dataset records for a non-identifiability witness.

    Each record is treated as a 1-tuple world: world = record.to_world_row()
    reshaped to (1, n_attrs).  Returns a result dict with witness_found,
    n_records_scanned, n_obs_groups, and the witness pair if found.
    """
    aug_overlaps = [augmented_overlap(o, config.fds) for o in config.overlap_schemas]
    groups: dict[tuple, dict[bool, np.ndarray]] = {}

    for i, rec in enumerate(records):
        world = np.array([rec.to_world_row()], dtype=np.int32)
        key = observation_key(world, aug_overlaps)
        q = evaluate_cq(cq, world, config)

        if key not in groups:
            groups[key] = {}
        grp = groups[key]
        if q not in grp:
            grp[q] = world.copy()

        if True in grp and False in grp:
            return {
                "witness_found": True,
                "n_records_scanned": i + 1,
                "n_obs_groups": len(groups),
                "witness": {
                    "w_true": grp[True].tolist(),
                    "w_false": grp[False].tolist(),
                },
            }

    return {
        "witness_found": False,
        "n_records_scanned": len(records),
        "n_obs_groups": len(groups),
        "witness": None,
    }


# ---------------------------------------------------------------------------
# Per-query record builder
# ---------------------------------------------------------------------------


def run_query(
    dataset_name: str,
    query_name: str,
    cq: BooleanCQ,
    config: Config,
    records: list[Any],
) -> dict:
    certified = cq.is_certified(config)
    record: dict = {
        "dataset": dataset_name,
        "query": query_name,
        "certified": certified,
        "footprint": sorted(cq.footprint(config)),
        "n_records": len(records),
    }

    if certified:
        record.update(
            {
                "scan_run": False,
                "witness_found": None,
                "n_records_scanned": None,
                "n_obs_groups": None,
                "witness": None,
                "elapsed_s": None,
                "note": "certified — witness provably absent",
            }
        )
    else:
        t0 = time.perf_counter()
        scan = data_witness_scan(records, cq, config)
        elapsed = time.perf_counter() - t0
        record.update(
            {
                "scan_run": True,
                "elapsed_s": round(elapsed, 4),
                **scan,
            }
        )

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Data-driven witness scan for real-world datasets")
    p.add_argument(
        "--mock", action="store_true", help="Use synthetic mock data instead of real CSVs"
    )
    p.add_argument("--data-dir-bibinteg", type=Path, default=Path("data/raw/bibinteg"))
    p.add_argument("--data-dir-wdc", type=Path, default=Path("data/raw/wdc"))
    p.add_argument("--data-dir-crosskg", type=Path, default=Path("data/raw/crosskg_dblp"))
    p.add_argument("--data-dir-amazon-google", type=Path, default=Path("data/raw/amazon_google"))
    p.add_argument("--data-dir-fodors-zagat", type=Path, default=Path("data/raw/fodors_zagat"))
    p.add_argument("--mock-n", type=int, default=1000, help="Records per dataset in mock mode")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    # ----- Load datasets -----
    if args.mock:
        print("Mock mode: using synthetic records that respect FDs.")
        bib_records = bib.make_mock_dataset(n=args.mock_n, seed=0)
        wdc_records = wdc.make_mock_dataset(n=args.mock_n, seed=0)
        ckg_records = ckg.make_mock_dataset(n=args.mock_n, seed=0)
        ag_records = ag.make_mock_dataset(n=args.mock_n, seed=0)
        fz_records = fz.make_mock_dataset(n=args.mock_n, seed=0)
        data_source = "mock"
    else:
        print(f"Loading BibInteg from {args.data_dir_bibinteg} …")
        bib_records = bib.load_dataset(args.data_dir_bibinteg)
        print(f"  {len(bib_records)} records loaded.")
        print(f"Loading WDC-Product from {args.data_dir_wdc} …")
        wdc_records = wdc.load_dataset(args.data_dir_wdc)
        print(f"  {len(wdc_records)} records loaded.")
        print(f"Loading CrossKG-DBLP (OpenAlex × DBLP) from {args.data_dir_crosskg} …")
        ckg_records = ckg.load_dataset(args.data_dir_crosskg)
        print(f"  {len(ckg_records)} records loaded ({len(ckg_records) // 2} papers × 2 sources).")
        print(f"Loading Amazon-Google from {args.data_dir_amazon_google} …")
        ag_records = ag.load_dataset(args.data_dir_amazon_google)
        print(
            f"  {len(ag_records)} records loaded ({len(ag_records) // 2} matched pairs × 2 sources)."
        )
        print(f"Loading Fodors-Zagat from {args.data_dir_fodors_zagat} …")
        fz_records = fz.load_dataset(args.data_dir_fodors_zagat)
        print(
            f"  {len(fz_records)} records loaded ({len(fz_records) // 2} matched pairs × 2 sources)."
        )
        data_source = "real"

    # ----- Run queries -----
    results: list[dict] = []

    bib_queries = [
        ("Q_venue", bib.Q_VENUE),
        ("Q_doi", bib.Q_DOI),
        ("Q_large_team", bib.Q_LARGE_TEAM),
    ]
    for q_name, cq in bib_queries:
        print(f"  BibInteg / {q_name} …", end=" ", flush=True)
        rec = run_query("bibinteg", q_name, cq, bib.CONFIG, bib_records)
        results.append(rec)
        status = (
            "certified"
            if rec["certified"]
            else ("WITNESS FOUND" if rec.get("witness_found") else "no witness")
        )
        print(status)

    wdc_queries = [
        ("Q_available", wdc.Q_AVAILABLE),
        ("Q_cheap", wdc.Q_CHEAP),
        ("Q_highly_rated", wdc.Q_HIGHLY_RATED),
        ("Q_reviewed", wdc.Q_REVIEWED),
        ("Q_popular", wdc.Q_POPULAR),
    ]
    for q_name, cq in wdc_queries:
        print(f"  WDC / {q_name} …", end=" ", flush=True)
        rec = run_query("wdc", q_name, cq, wdc.CONFIG, wdc_records)
        results.append(rec)
        status = (
            "certified"
            if rec["certified"]
            else ("WITNESS FOUND" if rec.get("witness_found") else "no witness")
        )
        print(status)

    # CrossKG-DBLP (OpenAlex × DBLP): the real-witness dataset. Q_publisher is
    # certified (both sources share the DOI); Q_large_team is not — the two
    # sources genuinely disagree on author counts, yielding real witnesses.
    crosskg_queries = [
        ("Q_publisher", ckg.Q_PUBLISHER),
        ("Q_large_team", ckg.Q_LARGE_TEAM),
    ]
    for q_name, cq in crosskg_queries:
        print(f"  CrossKG / {q_name} …", end=" ", flush=True)
        rec = run_query("crosskg_dblp", q_name, cq, ckg.CONFIG, ckg_records)
        results.append(rec)
        status = (
            "certified"
            if rec["certified"]
            else ("WITNESS FOUND" if rec.get("witness_found") else "no witness")
        )
        print(status)

    # Amazon-Google: the product-domain real-witness dataset. Q_catalog is
    # certified (both sources share the matched-pair identity); Q_expensive is
    # not — the two sources genuinely disagree on price, yielding real witnesses.
    amazon_google_queries = [
        ("Q_catalog", ag.Q_CATALOG),
        ("Q_expensive", ag.Q_EXPENSIVE),
    ]
    for q_name, cq in amazon_google_queries:
        print(f"  AmazonGoogle / {q_name} …", end=" ", flush=True)
        rec = run_query("amazon_google", q_name, cq, ag.CONFIG, ag_records)
        results.append(rec)
        status = (
            "certified"
            if rec["certified"]
            else ("WITNESS FOUND" if rec.get("witness_found") else "no witness")
        )
        print(status)

    # Fodors-Zagat: restaurant-domain real-witness dataset (breadth). Q_segment is
    # certified (both guides share the matched-pair identity); Q_cuisine is not —
    # the two guides genuinely categorize cuisine differently, yielding witnesses.
    fodors_zagat_queries = [
        ("Q_segment", fz.Q_SEGMENT),
        ("Q_cuisine", fz.Q_CUISINE),
    ]
    for q_name, cq in fodors_zagat_queries:
        print(f"  FodorsZagat / {q_name} …", end=" ", flush=True)
        rec = run_query("fodors_zagat", q_name, cq, fz.CONFIG, fz_records)
        results.append(rec)
        status = (
            "certified"
            if rec["certified"]
            else ("WITNESS FOUND" if rec.get("witness_found") else "no witness")
        )
        print(status)

    # ----- Summary -----
    print("\n=== Real-world witness scan ===")
    certified_n = sum(1 for r in results if r["certified"])
    scanned = [r for r in results if r.get("scan_run")]
    witnesses = [r for r in scanned if r.get("witness_found")]
    print(
        f"  Queries: {len(results)} total, {certified_n} certified, "
        f"{len(scanned)} non-certified scanned"
    )
    if scanned:
        print(f"  Witnesses found: {len(witnesses)}/{len(scanned)}")
    for r in witnesses:
        w = r["witness"]
        print(
            f"    {r['dataset']}/{r['query']}: "
            f"scanned {r['n_records_scanned']} records, "
            f"{r['n_obs_groups']} obs-groups, "
            f"elapsed {r['elapsed_s']:.3f}s"
        )
        print(f"      w_true  = {w['w_true']}")
        print(f"      w_false = {w['w_false']}")

    # ----- Save -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e_realworld_witnesses_{ts}.json"
    save_results(
        results,
        out,
        meta={
            "experiment": "e_realworld_witnesses",
            "timestamp": ts,
            "data_source": data_source,
            "n_bib_records": len(bib_records),
            "n_wdc_records": len(wdc_records),
            "n_crosskg_records": len(ckg_records),
            "n_amazon_google_records": len(ag_records),
            "n_fodors_zagat_records": len(fz_records),
            "n_queries": len(results),
            "n_certified": certified_n,
            "n_scanned": len(scanned),
            "n_witnesses": len(witnesses),
        },
    )


if __name__ == "__main__":
    main()

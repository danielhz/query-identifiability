"""
CrossKG-DBLP dataset loader — OpenAlex × DBLP, two independent CS bibliographies
for the same papers, joined on DOI.  This is the *real-witness* dataset: unlike
BibInteg (one source projected three ways, FDs hold by construction) and WDC
(query-critical attributes synthesized), here the non-identifiability is genuine
— the two sources actually disagree on author counts in the wild.

Build the raw corpus first (see data/build_crosskg_dblp.py):
    make crosskg-data        # DBLP dump + OpenAlex DOI lookup → data/raw/crosskg_dblp/

Modelling (the witness mechanics)
---------------------------------
Each paper (one DOI) is observed by two sources.  We emit ONE record per
(paper, source); the two records for a paper share the DOI (the overlap) but
may disagree on author count.  The witness scan groups records by their
Σ-closed overlap projection, so the two source-views of one paper land in the
same group — and a query that depends on the disagreed attribute splits them.

Schema (4 attributes, 2 views):
  attr 0: doi_id          (int; unique per DOI — the join key / overlap Ω)
  attr 1: publisher_bit   (int 0/1; deterministic from the DOI prefix, so both
                           sources agree — the publisher is encoded in the DOI)
  attr 2: large_team_bit  (int 0/1; 1 iff n_authors ≥ LARGE_TEAM_THRESHOLD —
                           sources DISAGREE here ~1-2% of the time)
  attr 3: source_id       (int 0=DBLP, 1=OpenAlex; for interpretability only)

Views (relation = one source's projection used by a query):
  V0 = bibliographic core: attrs {0, 1}     (doi + publisher)
  V1 = authorship:         attrs {0, 2}     (doi + team size)

Overlap and FDs (cross-source interface laws):
  Ω = {0}                       (the DOI — the only attribute both sources share
                                 reliably, because we joined on it)
  {0} → 1                       (DOI determines publisher; holds by construction)
  ⇒ augmented overlap Õ = {0, 1}

Queries:
  Q_PUBLISHER  (CERTIFIED):    footprint {0,1} ⊆ Õ — both sources agree on the
                               publisher (it's in the DOI), so no witness exists.
  Q_LARGE_TEAM (NOT CERTIFIED): footprint {0,2}, attr 2 ∉ Õ — the two sources
                               disagree on author count, producing real witnesses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from data.synthetic import Atom, BooleanCQ, Config

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

N_ATTRS = 4
# domain_size is a nominal upper bound; CrossKG is used only by the witness scan
# (observation_key / evaluate_cq), which read actual values, never domain_size.
DOMAIN_SIZE = 2**20
LARGE_TEAM_THRESHOLD = 4  # "large team" = >= 4 authors (≈ balanced on CS corpus)

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 1}),  # bibliographic core (doi + publisher)
    1: frozenset({0, 2}),  # authorship (doi + team size)
}

OVERLAP_SCHEMAS: list[frozenset[int]] = [
    frozenset({0}),  # the DOI — shared join key
]

FDS: list[tuple[frozenset[int], int]] = [
    (frozenset({0}), 1),  # DOI determines publisher (encoded in the DOI prefix)
]

CONFIG = Config(
    n_attrs=N_ATTRS,
    domain_size=DOMAIN_SIZE,
    fds=FDS,
    view_schemas=VIEW_SCHEMAS,
    overlap_schemas=OVERLAP_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Canonical queries
# ---------------------------------------------------------------------------

# "Is this paper from a publisher whose DOI prefix hashes to 1?" — footprint {0,1}
# CERTIFIED: {0}→1 so footprint ⊆ Õ={0,1}.  Both sources share the DOI (hence the
# publisher), so the certificate's "no witness" guarantee genuinely holds.
Q_PUBLISHER = BooleanCQ(atoms=[Atom(view_id=0, pattern=["doi0", 1])])

# "Does this paper have a large author team (≥ threshold)?" — footprint {0,2}
# NOT CERTIFIED: attr 2 (large_team_bit) ∉ Õ.  DBLP and OpenAlex genuinely disagree
# on author counts for some papers, so straddling papers are real witnesses.
Q_LARGE_TEAM = BooleanCQ(atoms=[Atom(view_id=1, pattern=["doi0", 1])])


# ---------------------------------------------------------------------------
# Data record type
# ---------------------------------------------------------------------------


class CrossKGRecord:
    """One (paper, source) observation as a length-4 int array."""

    __slots__ = ("attrs",)

    def __init__(self, attrs: list[int] | np.ndarray):
        self.attrs = np.asarray(attrs, dtype=np.int32)

    def __getitem__(self, i: int) -> int:
        return int(self.attrs[i])

    def to_world_row(self) -> list[int]:
        return [int(v) for v in self.attrs]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _publisher_bit(doi: str) -> int:
    """Deterministic 0/1 from the DOI registrant prefix (e.g. '10.1145')."""
    prefix = doi.split("/", 1)[0].strip().lower()
    return int(hashlib.sha1(prefix.encode()).hexdigest(), 16) % 2


def _team_bit(n_authors: int) -> int:
    return 1 if n_authors >= LARGE_TEAM_THRESHOLD else 0


# ---------------------------------------------------------------------------
# Mock dataset (for tests / CI — no download required)
# ---------------------------------------------------------------------------


def make_mock_dataset(n: int = 200, seed: int = 42) -> list[CrossKGRecord]:
    """
    Return ~2n CrossKGRecords: n papers, each observed by both sources.

    Both source-records of a paper share doi_id and publisher_bit; their team
    counts agree most of the time but differ across the threshold for a small
    fraction (~15 %), reproducing the real cross-source disagreement so the
    witness scan reliably finds witnesses on mock data.
    """
    rng = np.random.default_rng(seed)
    records: list[CrossKGRecord] = []
    for doi_id in range(n):
        pub = int(rng.integers(0, 2))
        base = int(rng.integers(1, 9))  # 1..8 authors
        if rng.random() < 0.15:
            # force a straddle of the threshold → a real witness
            oa_n = (
                LARGE_TEAM_THRESHOLD - 1 if base >= LARGE_TEAM_THRESHOLD else LARGE_TEAM_THRESHOLD
            )
        else:
            oa_n = base
        records.append(CrossKGRecord([doi_id, pub, _team_bit(base), 0]))  # DBLP
        records.append(CrossKGRecord([doi_id, pub, _team_bit(oa_n), 1]))  # OpenAlex
    return records


# ---------------------------------------------------------------------------
# CSV-based loader (requires the joined corpus from build_crosskg_dblp)
# ---------------------------------------------------------------------------


def _read(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["doi"].strip().lower(): r for r in csv.DictReader(f)}


def _n_authors(row: dict) -> int:
    try:
        return max(0, int(row.get("n_authors", 0) or 0))
    except ValueError:
        return 0


def load_dataset(data_dir: Path | str) -> list[CrossKGRecord]:
    """
    Load the OpenAlex × DBLP corpus and emit two records per joined DOI.

    Expects dblp.csv and openalex.csv (columns: doi,title,venue,year,n_authors,type)
    in data_dir, produced by `python -m data.build_crosskg_dblp`.
    """
    data_dir = Path(data_dir)
    dblp_p, oa_p = data_dir / "dblp.csv", data_dir / "openalex.csv"
    for p in (dblp_p, oa_p):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {p}.  Run: make crosskg-data  (or python -m data.build_crosskg_dblp)"
            )
    dblp, oa = _read(dblp_p), _read(oa_p)

    records: list[CrossKGRecord] = []
    for doi_id, doi in enumerate(sorted(set(dblp) & set(oa))):
        # drop OpenAlex front-matter (paratext, 0 authors) — not real papers
        if oa[doi].get("type", "") == "paratext":
            continue
        n_d, n_o = _n_authors(dblp[doi]), _n_authors(oa[doi])
        if n_o == 0:  # missing author list on the OpenAlex side — skip
            continue
        pub = _publisher_bit(doi)
        records.append(CrossKGRecord([doi_id, pub, _team_bit(n_d), 0]))  # DBLP
        records.append(CrossKGRecord([doi_id, pub, _team_bit(n_o), 1]))  # OpenAlex
    return records


def records_to_worlds(records: list[CrossKGRecord]) -> list[list[list[int]]]:
    """Convert flat records to the 1-tuple-world format expected by the scan."""
    return [[r.to_world_row()] for r in records]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CrossKG-DBLP (OpenAlex × DBLP) dataset utility")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/crosskg_dblp"))
    p.add_argument("--info", action="store_true", help="Print schema and FD summary")
    p.add_argument("--mock", action="store_true", help="Summarize the mock dataset")
    args = p.parse_args(argv)

    if args.info:
        from data.utils import augmented_overlap

        aug = augmented_overlap(OVERLAP_SCHEMAS[0], FDS)
        print(f"CrossKG-DBLP: {N_ATTRS} attributes, {len(VIEW_SCHEMAS)} views")
        print(f"Views:    {VIEW_SCHEMAS}")
        print(f"Overlap:  {OVERLAP_SCHEMAS}")
        print(f"FDs:      {FDS}")
        print(f"Õ (augmented overlap): {sorted(aug)}")
        print(f"Q_PUBLISHER  certified: {Q_PUBLISHER.is_certified(CONFIG)}")
        print(f"Q_LARGE_TEAM certified: {Q_LARGE_TEAM.is_certified(CONFIG)}")
        return

    if args.mock:
        recs = make_mock_dataset()
        print(f"Mock CrossKG-DBLP: {len(recs)} records ({len(recs) // 2} papers × 2 sources)")
        print(f"  Example: {recs[0].to_world_row()} / {recs[1].to_world_row()}")
        return

    records = load_dataset(args.data_dir)
    n_papers = len(records) // 2
    print(f"Loaded {len(records)} records ({n_papers} papers × 2 sources) from {args.data_dir}")


if __name__ == "__main__":
    main()

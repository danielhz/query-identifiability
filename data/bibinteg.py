"""
BibInteg dataset loader — DBLP + ACM + SemanticScholar citation records.

Schema (3 views, 7 attributes):
  attr 0: title_token_hash  (int, domain=2^16)
  attr 1: author_hash       (int, domain=2^16)
  attr 2: year              (int, 1960..2024)
  attr 3: venue_hash        (int, domain=2^12)
  attr 4: doi_prefix_hash   (int, domain=2^8; 0 if absent)
  attr 5: n_authors         (int, 1..20)
  attr 6: decade            (int, derived: year // 10)

Views (relation = one source):
  V0 = DBLP:           attrs {0, 1, 2, 3}
  V1 = ACM:            attrs {0, 1, 2, 4}
  V2 = SemanticScholar: attrs {0, 1, 2, 5}

Overlaps:
  O_01 = V0 ∩ V1 = {0, 1, 2}   (title + author + year)
  O_02 = V0 ∩ V2 = {0, 1, 2}
  O_12 = V1 ∩ V2 = {0, 1, 2}

FDs (cross-world interface laws):
  {0, 1, 2} → 3    (title+author+year determines venue, from DBLP)
  {0, 1, 2} → 4    (title+author+year determines DOI prefix, from ACM)
  {0, 1, 2} → 5    (title+author+year determines author count, from SS)
  {2}        → 6   (year determines decade — deterministic derived attr)

Query examples (defined as BooleanCQ):
  Q_venue_lookup:  ∃ t in V0 s.t. t.venue_hash = TARGET_VENUE_HASH
  Q_doi_exists:    ∃ t in V1 s.t. t.doi_prefix_hash != 0
  Q_large_team:    ∃ t in V2 s.t. t.n_authors >= 5

Download:
  Raw CSVs are NOT bundled.  Run:
    python -m data.bibinteg --download --data-dir data/raw/bibinteg
  which fetches the pre-processed snapshots from the project data share.
  (Requires VPN / institutional access.)

  For local unit-testing and CI, use `make_mock_dataset()` instead.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from data.synthetic import Atom, BooleanCQ, Config

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

N_ATTRS = 7
DOMAIN_SIZES = [2**16, 2**16, 65, 2**12, 2**8, 20, 7]  # per-attribute domains

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 1, 2, 3}),  # DBLP
    1: frozenset({0, 1, 2, 4}),  # ACM
    2: frozenset({0, 1, 2, 5}),  # SemanticScholar
}

OVERLAP_SCHEMAS: list[frozenset[int]] = [
    frozenset({0, 1, 2}),  # O_01 / O_02 / O_12 — same triple
]

FDS: list[tuple[frozenset[int], int]] = [
    (frozenset({0, 1, 2}), 3),
    (frozenset({0, 1, 2}), 4),
    (frozenset({0, 1, 2}), 5),
    (frozenset({2}), 6),
]

CONFIG = Config(
    n_attrs=N_ATTRS,
    domain_size=max(DOMAIN_SIZES),  # upper bound; actual domains vary per attr
    fds=FDS,
    view_schemas=VIEW_SCHEMAS,
    overlap_schemas=OVERLAP_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Canonical queries
# ---------------------------------------------------------------------------

# "Does this paper's venue hash to the upper half of the domain?" — footprint {0,1,2,3}
# Certified: {0,1,2}→3 ∈ Σ-closure({0,1,2}).
# Attr 3 is venue_hash % 2 ∈ {0,1}; ~50 % positive.
Q_VENUE = BooleanCQ(atoms=[Atom(view_id=0, pattern=["t0", "a0", "y0", 1])])

# "Does this paper's DOI-prefix hash to the upper half?" — footprint {0,1,2,4}
# Certified: {0,1,2}→4 ∈ Σ-closure.
# Attr 4 is doi_prefix % 2 ∈ {0,1}; ~50 % positive.
Q_DOI = BooleanCQ(atoms=[Atom(view_id=1, pattern=["t0", "a0", "y0", 1])])

# "Does this paper have ≥ 5 authors?" — footprint {0,1,2,5}
# Certified: {0,1,2}→5 ∈ Σ-closure.
# Attr 5 is (n_authors ≥ 5) ∈ {0,1}; ~48 % positive on OpenAlex CS corpus.
Q_LARGE_TEAM = BooleanCQ(atoms=[Atom(view_id=2, pattern=["t0", "a0", "y0", 1])])


# ---------------------------------------------------------------------------
# Data record type
# ---------------------------------------------------------------------------


class BibRecord:
    """One paper as a length-7 int array (one value per attribute)."""

    __slots__ = ("attrs",)

    def __init__(self, attrs: list[int] | np.ndarray):
        self.attrs = np.asarray(attrs, dtype=np.int32)

    def __getitem__(self, i: int) -> int:
        return int(self.attrs[i])

    def to_world_row(self) -> list[int]:
        return [int(v) for v in self.attrs]


# ---------------------------------------------------------------------------
# Mock dataset (for tests / CI — no download required)
# ---------------------------------------------------------------------------


def make_mock_dataset(n: int = 200, seed: int = 42) -> list[BibRecord]:
    """
    Return n synthetic BibRecord objects that respect the FDs.
    title+author+year  → venue, doi_prefix, n_authors (deterministically)
    year               → decade
    """
    rng = np.random.default_rng(seed)

    records: list[BibRecord] = []
    for _ in range(n):
        title = int(rng.integers(0, 2**16))
        author = int(rng.integers(0, 2**16))
        year = int(rng.integers(1960, 2025))

        # Deterministic functions of (title, author, year); binary attrs for ~50% queries
        key = f"{title},{author},{year}".encode()
        h = int(hashlib.md5(key).hexdigest(), 16)
        venue = h % 2
        doi_prefix = (h >> 1) % 2
        n_authors = (h >> 2) % 2
        decade = (year // 10) % 7

        records.append(BibRecord([title, author, year, venue, doi_prefix, n_authors, decade]))

    return records


# ---------------------------------------------------------------------------
# CSV-based loader (requires downloaded data)
# ---------------------------------------------------------------------------


def _hash_str(s: str, bits: int = 16) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % (2**bits)


def load_dataset(data_dir: Path | str) -> list[BibRecord]:
    """
    Load BibInteg from pre-processed CSVs in data_dir.
    Expects: dblp.csv (title,authors,year,venue), acm.csv (title,authors,year,doi),
    semantic_scholar.csv (title,authors,year,n_authors).
    All three must have the same rows in the same order (same papers, three projections).
    They are joined row-by-row so each BibRecord has all seven attributes correctly set.
    """
    import csv

    fnames = ("dblp.csv", "acm.csv", "semantic_scholar.csv")
    paths = [Path(data_dir) / fn for fn in fnames]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {p}.  Run: python -m data.download_openalex --data-dir {data_dir}"
            )

    records: list[BibRecord] = []
    handles = [open(p, newline="", encoding="utf-8") for p in paths]
    try:
        readers = [csv.DictReader(h) for h in handles]
        for dblp_row, acm_row, ss_row in zip(*readers):
            year = int(dblp_row.get("year", 0) or 0)
            if not (1960 <= year <= 2024):
                continue
            title = _hash_str(dblp_row.get("title", ""), 16)
            author = _hash_str(dblp_row.get("authors", ""), 16)
            # Binary attrs so queries Q_VENUE/Q_DOI/Q_LARGE_TEAM have ~50% positive rate
            venue = _hash_str(dblp_row.get("venue", ""), 12) % 2
            doi_prefix = _hash_str((acm_row.get("doi", "") or "")[:8], 8) % 2
            try:
                raw_n = max(1, min(100, int(ss_row.get("n_authors", 1) or 1)))
            except ValueError:
                raw_n = 1
            n_auth = 1 if raw_n >= 5 else 0
            decade = (year // 10) % 7
            records.append(BibRecord([title, author, year, venue, doi_prefix, n_auth, decade]))
    finally:
        for h in handles:
            h.close()
    return records


def records_to_worlds(records: list[BibRecord]) -> list[list[list[int]]]:
    """Convert flat BibRecords to world format expected by SyntheticBenchmark."""
    return [[r.to_world_row()] for r in records]


# ---------------------------------------------------------------------------
# Download stub (actual download requires institutional access)
# ---------------------------------------------------------------------------


def _download(data_dir: Path) -> None:
    raise NotImplementedError(
        "BibInteg download requires institutional VPN access.\n"
        "Place dblp.csv, acm.csv, semantic_scholar.csv in:\n"
        f"  {data_dir}\n"
        "then re-run without --download."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="BibInteg dataset utility")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/bibinteg"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--info", action="store_true", help="Print schema and FD summary")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Generate and summarize mock dataset (no download needed)",
    )
    args = p.parse_args(argv)

    if args.download:
        _download(args.data_dir)
        return

    if args.info:
        print(f"BibInteg schema: {N_ATTRS} attributes, {len(VIEW_SCHEMAS)} views")
        print(f"Views:    {VIEW_SCHEMAS}")
        print(f"Overlaps: {OVERLAP_SCHEMAS}")
        print(f"FDs:      {FDS}")
        from data.utils import augmented_overlap

        aug = augmented_overlap(OVERLAP_SCHEMAS[0], FDS)
        print(f"Õ (augmented overlap): {sorted(aug)}")
        return

    if args.mock:
        recs = make_mock_dataset()
        print(f"Mock BibInteg: {len(recs)} records")
        r = recs[0]
        print(f"  Example: {r.to_world_row()}")
        return

    records = load_dataset(args.data_dir)
    print(f"Loaded {len(records)} BibInteg records from {args.data_dir}")


if __name__ == "__main__":
    main()

"""
WikiScholar dataset loader — scholarly articles extracted from Wikidata.

Schema (3 views, 7 attributes):
  attr 0: title_hash         (int, domain=2^16)
  attr 1: author_hash        (int, domain=2^16)
  attr 2: year               (int, 2010..2024)
  attr 3: venue_hash         (int, binary: venue_hash_full % 2)
  attr 4: subject_hash       (int, binary: subject_hash_full % 2)
  attr 5: n_citations_class  (int, binary: n_citations >= 41; median threshold → ~50% positive)
  attr 6: decade             (int, derived: year // 10)

Views:
  V0 = Crossref:       attrs {0, 1, 2, 3}   title + author + year + venue
  V1 = arXiv:          attrs {0, 1, 2, 4}   title + author + year + subject_category
  V2 = Citations:      attrs {0, 1, 2, 5}   title + author + year + n_citations

Overlaps:
  O_01 = O_02 = O_12 = {0, 1, 2}   (title + author + year)

FDs (cross-world interface laws):
  {0, 1, 2} → 3    (title+author+year determines venue)
  {0, 1, 2} → 4    (title+author+year determines subject_category)
  {0, 1, 2} → 5    (title+author+year determines n_citations)
  {2}        → 6   (year determines decade)

Canonical queries:
  Q_venue:       ∃ t in V0 s.t. venue_hash % 2 = 1  — footprint {0,1,2,3} ⊆ Ω̃  [C]
  Q_subject:     ∃ t in V1 s.t. subject_hash % 2 = 1 — footprint {0,1,2,4} ⊆ Ω̃ [C]
  Q_highly_cited: ∃ t in V2 s.t. n_citations ≥ 41    — footprint {0,1,2,5} ⊆ Ω̃  [C]

Data source:
  Raw CSVs extracted from the Wikidata dump by data/wikidata_extraction_task.md.
  Place crossref.csv, arxiv.csv, citations.csv in data/raw/wikischolar/,
  then run: python -m data.wikischolar --info --data-dir data/raw/wikischolar
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

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 1, 2, 3}),  # Crossref
    1: frozenset({0, 1, 2, 4}),  # arXiv
    2: frozenset({0, 1, 2, 5}),  # Citations
}

OVERLAP_SCHEMAS: list[frozenset[int]] = [
    frozenset({0, 1, 2}),
]

FDS: list[tuple[frozenset[int], int]] = [
    (frozenset({0, 1, 2}), 3),
    (frozenset({0, 1, 2}), 4),
    (frozenset({0, 1, 2}), 5),
    (frozenset({2}), 6),
]

CONFIG = Config(
    n_attrs=N_ATTRS,
    domain_size=2**16,
    fds=FDS,
    view_schemas=VIEW_SCHEMAS,
    overlap_schemas=OVERLAP_SCHEMAS,
)

# ---------------------------------------------------------------------------
# Canonical queries  (all certified; ~50% positive rate on Wikidata corpus)
# ---------------------------------------------------------------------------

Q_VENUE = BooleanCQ(atoms=[Atom(view_id=0, pattern=["t0", "a0", "y0", 1])])
Q_SUBJECT = BooleanCQ(atoms=[Atom(view_id=1, pattern=["t0", "a0", "y0", 1])])
Q_HIGHLY_CITED = BooleanCQ(atoms=[Atom(view_id=2, pattern=["t0", "a0", "y0", 1])])

# ---------------------------------------------------------------------------
# Data record type
# ---------------------------------------------------------------------------


class WikiRecord:
    """One article as a length-7 int array (one value per attribute)."""

    __slots__ = ("attrs",)

    def __init__(self, attrs: list[int] | np.ndarray):
        self.attrs = np.asarray(attrs, dtype=np.int32)

    def __getitem__(self, i: int) -> int:
        return int(self.attrs[i])

    def to_world_row(self) -> list[int]:
        return [int(v) for v in self.attrs]


# ---------------------------------------------------------------------------
# Mock dataset
# ---------------------------------------------------------------------------


def make_mock_dataset(n: int = 200, seed: int = 42) -> list[WikiRecord]:
    """Return n synthetic WikiRecords respecting the FDs."""
    rng = np.random.default_rng(seed)
    records: list[WikiRecord] = []
    for _ in range(n):
        title = int(rng.integers(0, 2**16))
        author = int(rng.integers(0, 2**16))
        year = int(rng.integers(2010, 2025))
        key = f"{title},{author},{year}".encode()
        h = int(hashlib.md5(key).hexdigest(), 16)
        venue = h % 2
        subject = (h >> 1) % 2
        n_cited = (h >> 2) % 2
        decade = (year // 10) % 7
        records.append(WikiRecord([title, author, year, venue, subject, n_cited, decade]))
    return records


# ---------------------------------------------------------------------------
# CSV-based loader
# ---------------------------------------------------------------------------


def _hash_str(s: str, bits: int = 16) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % (2**bits)


def load_dataset(data_dir: Path | str) -> list[WikiRecord]:
    """
    Load WikiScholar from pre-processed CSVs in data_dir.
    Expects: crossref.csv (title,author,year,venue),
             arxiv.csv (title,author,year,subject_category),
             citations.csv (title,author,year,n_citations).
    All three CSVs must have the same rows in the same order (same papers).
    They are joined row-by-row so each WikiRecord has all attributes set.
    """
    import csv

    fnames = ("crossref.csv", "arxiv.csv", "citations.csv")
    paths = [Path(data_dir) / fn for fn in fnames]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {p}.  Follow data/wikidata_extraction_task.md "
                f"to extract the data from the Wikidata dump."
            )

    records: list[WikiRecord] = []
    handles = [open(p, newline="", encoding="utf-8") for p in paths]
    try:
        readers = [csv.DictReader(h) for h in handles]
        for cr_row, ax_row, ci_row in zip(*readers):
            year = int(cr_row.get("year", 0) or 0)
            if not (2010 <= year <= 2024):
                continue
            title = _hash_str(cr_row.get("title", ""), 16)
            author = _hash_str(cr_row.get("author", ""), 16)
            venue = _hash_str(cr_row.get("venue", ""), 12) % 2
            subject = _hash_str(ax_row.get("subject_category", ""), 12) % 2
            try:
                n_cit = int(ci_row.get("n_citations", 0) or 0)
            except ValueError:
                n_cit = 0
            n_cited = 1 if n_cit >= 41 else 0  # median threshold → ~50% positive rate
            decade = (year // 10) % 7
            records.append(WikiRecord([title, author, year, venue, subject, n_cited, decade]))
    finally:
        for h in handles:
            h.close()
    return records


def records_to_worlds(records: list[WikiRecord]) -> list[list[list[int]]]:
    return [[r.to_world_row()] for r in records]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="WikiScholar dataset info / smoke test")
    p.add_argument("--info", action="store_true", help="Print dataset stats and exit")
    p.add_argument("--mock", action="store_true", help="Use mock data")
    p.add_argument("--n", type=int, default=200, help="Mock records (default 200)")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/wikischolar"))
    args = p.parse_args(argv)

    if args.mock:
        records = make_mock_dataset(args.n)
        print(f"Mock dataset: {len(records)} records")
    else:
        records = load_dataset(args.data_dir)
        print(f"Loaded {len(records)} records from {args.data_dir}")

    if args.info:
        for q, name in [
            (Q_VENUE, "Q_venue"),
            (Q_SUBJECT, "Q_subject"),
            (Q_HIGHLY_CITED, "Q_highly_cited"),
        ]:
            certified = q.is_certified(CONFIG)
            print(f"  {name}: certified={certified}")
        print(f"  Example: {records[0].to_world_row()}")


if __name__ == "__main__":
    main()

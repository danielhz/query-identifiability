"""
Fodors-Zagat dataset loader — the restaurant-domain real-witness dataset (breadth).

Two independent restaurant guides (Fodors, Zagat) with a gold set of matched
pairs (same restaurant). Like CrossKG-DBLP and Amazon-Google, the
non-identifiability is genuine: the two guides categorize *cuisine* differently
(e.g. "asian" vs "japanese", "french" vs "seafood"), so a cuisine query splits
the two source-views. Tiny and clean — a third domain, treated as breadth.

Build the raw corpus first (git-ignored, not redistributed):
    python -m data.download_fodors_zagat       # -> data/raw/fodors_zagat/

Modelling (mirrors data/amazon_google.py / data/crosskg_dblp.py)
----------------------------------------------------------------
Each matched pair is one restaurant observed by two guides. We emit ONE record
per (pair, source); the two records share the matched-pair id (the overlap) but
may disagree on cuisine.

Schema (4 attributes, 2 views):
  attr 0: pair_id      (int; unique per matched pair — the join key / overlap Ω)
  attr 1: segment_bit  (int 0/1; deterministic from the matched-pair identity,
                        so both guides agree — analogous to Amazon-Google catalog_bit)
  attr 2: cuisine_bit  (int 0/1; hash of the cuisine type — the two guides
                        DISAGREE here when their cuisine labels hash differently)
  attr 3: source_id    (int 0=Fodors, 1=Zagat; for interpretability only)

Views:  V0 = {0, 1} (listing core)   V1 = {0, 2} (cuisine)
Overlap Ω = {0}; FD {0} -> 1; augmented overlap Õ = {0, 1}.

Queries:
  Q_segment (CERTIFIED):     footprint {0,1} ⊆ Õ — both guides share the pair
                             identity, so no witness exists.
  Q_cuisine (NOT CERTIFIED): footprint {0,2}, attr 2 ∉ Õ — the two guides
                             genuinely disagree on cuisine, producing real witnesses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np

from data.synthetic import Atom, BooleanCQ, Config

N_ATTRS = 4
DOMAIN_SIZE = 2**20  # nominal; the witness scan reads actual values, never domain_size

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 1}),  # listing core (pair + segment)
    1: frozenset({0, 2}),  # cuisine
}
OVERLAP_SCHEMAS: list[frozenset[int]] = [frozenset({0})]
FDS: list[tuple[frozenset[int], int]] = [(frozenset({0}), 1)]

CONFIG = Config(
    n_attrs=N_ATTRS,
    domain_size=DOMAIN_SIZE,
    fds=FDS,
    view_schemas=VIEW_SCHEMAS,
    overlap_schemas=OVERLAP_SCHEMAS,
)

# "Is this listing in segment 1?" — footprint {0,1} ⊆ Õ → CERTIFIED.
Q_SEGMENT = BooleanCQ(atoms=[Atom(view_id=0, pattern=["pair0", 1])])
# "Does this restaurant's cuisine hash to 1?" — footprint {0,2}, 2 ∉ Õ → NOT certified.
Q_CUISINE = BooleanCQ(atoms=[Atom(view_id=1, pattern=["pair0", 1])])


class FodorsZagatRecord:
    """One (matched-pair, source) observation as a length-4 int array."""

    __slots__ = ("attrs",)

    def __init__(self, attrs: list[int] | np.ndarray):
        self.attrs = np.asarray(attrs, dtype=np.int32)

    def __getitem__(self, i: int) -> int:
        return int(self.attrs[i])

    def to_world_row(self) -> list[int]:
        return [int(v) for v in self.attrs]


def _segment_bit(fodors_id: str, zagat_id: str) -> int:
    """Deterministic 0/1 from the matched-pair identity (shared by both guides)."""
    return int(hashlib.sha1(f"{fodors_id}|{zagat_id}".encode()).hexdigest(), 16) % 2


def _cuisine_bit(cuisine: str) -> int:
    """0/1 partition of the cuisine label (normalized). Two guides assigning
    different cuisines that fall on opposite sides of the partition disagree."""
    norm = re.sub(r"[^a-z0-9]", "", (cuisine or "").lower())
    return int(hashlib.sha1(norm.encode()).hexdigest(), 16) % 2


def make_mock_dataset(n: int = 200, seed: int = 42) -> list[FodorsZagatRecord]:
    """~2n records: n matched pairs, each observed by both guides; ~45% disagree
    on the cuisine bit (a real witness), matching the observed Fodors-Zagat rate."""
    rng = np.random.default_rng(seed)
    records: list[FodorsZagatRecord] = []
    for pair_id in range(n):
        seg = int(rng.integers(0, 2))
        cf = int(rng.integers(0, 2))  # Fodors cuisine bit
        cz = 1 - cf if rng.random() < 0.45 else cf  # Zagat disagrees ~45% (witness)
        records.append(FodorsZagatRecord([pair_id, seg, cf, 0]))  # Fodors
        records.append(FodorsZagatRecord([pair_id, seg, cz, 1]))  # Zagat
    return records


def load_dataset(data_dir: Path | str) -> list[FodorsZagatRecord]:
    """Load matched Fodors/Zagat restaurants and emit two records per matched pair."""
    data_dir = Path(data_dir)
    need = ("fodors.csv", "zagat.csv", "matches.csv")
    for fn in need:
        if not (data_dir / fn).exists():
            raise FileNotFoundError(
                f"Expected {data_dir / fn}.  Run: python -m data.download_fodors_zagat"
            )

    def index(fn: str) -> dict[str, dict]:
        with open(data_dir / fn, newline="", encoding="utf-8", errors="ignore") as f:
            return {r["id"]: r for r in csv.DictReader(f)}

    fodors, zagat = index("fodors.csv"), index("zagat.csv")
    records: list[FodorsZagatRecord] = []
    with open(data_dir / "matches.csv", newline="", encoding="utf-8") as f:
        for pair_id, m in enumerate(csv.DictReader(f)):
            fr, zr = fodors.get(m["fodors_id"]), zagat.get(m["zagat_id"])
            if fr is None or zr is None:
                continue
            seg = _segment_bit(m["fodors_id"], m["zagat_id"])
            records.append(FodorsZagatRecord([pair_id, seg, _cuisine_bit(fr["type"]), 0]))
            records.append(FodorsZagatRecord([pair_id, seg, _cuisine_bit(zr["type"]), 1]))
    return records


def records_to_worlds(records: list[FodorsZagatRecord]) -> list[list[list[int]]]:
    """Convert flat records to the 1-tuple-world format expected by the scan."""
    return [[r.to_world_row()] for r in records]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fodors-Zagat dataset utility")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/fodors_zagat"))
    p.add_argument("--info", action="store_true", help="Print schema and FD summary")
    p.add_argument("--mock", action="store_true", help="Summarize the mock dataset")
    args = p.parse_args(argv)

    if args.info:
        from data.utils import augmented_overlap

        aug = augmented_overlap(OVERLAP_SCHEMAS[0], FDS)
        print(f"Fodors-Zagat: {N_ATTRS} attributes, {len(VIEW_SCHEMAS)} views")
        print(f"Views:    {VIEW_SCHEMAS}")
        print(f"Overlap:  {OVERLAP_SCHEMAS}")
        print(f"FDs:      {FDS}")
        print(f"Õ (augmented overlap): {sorted(aug)}")
        print(f"Q_segment certified: {Q_SEGMENT.is_certified(CONFIG)}")
        print(f"Q_cuisine certified: {Q_CUISINE.is_certified(CONFIG)}")
        return

    if args.mock:
        recs = make_mock_dataset()
        print(f"Mock Fodors-Zagat: {len(recs)} records ({len(recs) // 2} pairs × 2 sources)")
        print(f"  Example: {recs[0].to_world_row()} / {recs[1].to_world_row()}")
        return

    records = load_dataset(args.data_dir)
    print(f"Loaded {len(records)} records ({len(records) // 2} matched pairs × 2 sources)")


if __name__ == "__main__":
    main()

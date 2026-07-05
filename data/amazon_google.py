"""
Amazon-Google dataset loader — the product-domain real-witness dataset.

Two independent product sources (Amazon, Google) with a gold set of matched
pairs (same product, both sources). Like CrossKG-DBLP, the non-identifiability
is genuine: the two sources list *different prices* for the same product, so a
price-threshold query splits the two source-views — a real witness in a product
(non-bibliographic) domain, with no synthesized columns.

Build the raw corpus first (git-ignored, not redistributed):
    python -m data.download_amazon_google     # -> data/raw/amazon_google/

Modelling (mirrors data/crosskg_dblp.py)
----------------------------------------
Each matched pair is one product observed by two sources. We emit ONE record per
(pair, source); the two records share the matched-pair id (the overlap) but may
disagree on price.

Schema (4 attributes, 2 views):
  attr 0: pair_id        (int; unique per matched pair — the join key / overlap Ω)
  attr 1: catalog_bit    (int 0/1; deterministic from the matched-pair identity,
                          so both sources agree — analogous to CrossKG publisher_bit)
  attr 2: expensive_bit  (int 0/1; 1 iff price >= EXPENSIVE_THRESHOLD — the two
                          sources DISAGREE here when their prices straddle it)
  attr 3: source_id      (int 0=Amazon, 1=Google; for interpretability only)

Views:  V0 = {0, 1} (catalog core)   V1 = {0, 2} (pricing)
Overlap Ω = {0}; FD {0} -> 1; augmented overlap Õ = {0, 1}.

Queries:
  Q_catalog   (CERTIFIED):     footprint {0,1} ⊆ Õ — both sources share the pair
                               identity, so no witness exists.
  Q_expensive (NOT CERTIFIED): footprint {0,2}, attr 2 ∉ Õ — the two sources
                               genuinely disagree on price, producing real witnesses.
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
EXPENSIVE_THRESHOLD = 50.0  # USD; ≈ median matched-pair price → balanced query

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 1}),  # catalog core (pair + catalog segment)
    1: frozenset({0, 2}),  # pricing (pair + expensive flag)
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

# "Is this product in catalog segment 1?" — footprint {0,1} ⊆ Õ → CERTIFIED.
Q_CATALOG = BooleanCQ(atoms=[Atom(view_id=0, pattern=["pair0", 1])])
# "Is this product expensive (>= threshold)?" — footprint {0,2}, 2 ∉ Õ → NOT certified.
Q_EXPENSIVE = BooleanCQ(atoms=[Atom(view_id=1, pattern=["pair0", 1])])


class AmazonGoogleRecord:
    """One (matched-pair, source) observation as a length-4 int array."""

    __slots__ = ("attrs",)

    def __init__(self, attrs: list[int] | np.ndarray):
        self.attrs = np.asarray(attrs, dtype=np.int32)

    def __getitem__(self, i: int) -> int:
        return int(self.attrs[i])

    def to_world_row(self) -> list[int]:
        return [int(v) for v in self.attrs]


def _catalog_bit(amazon_id: str, google_id: str) -> int:
    """Deterministic 0/1 from the matched-pair identity (shared by both sources)."""
    key = f"{amazon_id}|{google_id}".encode()
    return int(hashlib.sha1(key).hexdigest(), 16) % 2


def _price(s: str) -> float | None:
    m = re.search(r"\d+\.?\d*", (s or "").replace(",", ""))
    return float(m.group()) if m else None


def _expensive_bit(p: float) -> int:
    return 1 if p >= EXPENSIVE_THRESHOLD else 0


def make_mock_dataset(n: int = 200, seed: int = 42) -> list[AmazonGoogleRecord]:
    """~2n records: n matched pairs, each observed by both sources; ~15% straddle
    the price threshold (a real witness), the rest agree."""
    rng = np.random.default_rng(seed)
    records: list[AmazonGoogleRecord] = []
    for pair_id in range(n):
        cat = int(rng.integers(0, 2))
        base = float(rng.uniform(5, 200))
        if rng.random() < 0.15:  # force a straddle of the threshold → witness
            g = EXPENSIVE_THRESHOLD - 1 if base >= EXPENSIVE_THRESHOLD else EXPENSIVE_THRESHOLD + 1
        else:
            g = base
        records.append(AmazonGoogleRecord([pair_id, cat, _expensive_bit(base), 0]))  # Amazon
        records.append(AmazonGoogleRecord([pair_id, cat, _expensive_bit(g), 1]))  # Google
    return records


def load_dataset(data_dir: Path | str) -> list[AmazonGoogleRecord]:
    """Load matched Amazon/Google products and emit two records per matched pair."""
    data_dir = Path(data_dir)
    need = ("amazon.csv", "google.csv", "matches.csv")
    for fn in need:
        if not (data_dir / fn).exists():
            raise FileNotFoundError(
                f"Expected {data_dir / fn}.  Run: python -m data.download_amazon_google"
            )

    def index(fn: str) -> dict[str, dict]:
        with open(data_dir / fn, newline="", encoding="utf-8", errors="ignore") as f:
            return {r["id"]: r for r in csv.DictReader(f)}

    amazon, google = index("amazon.csv"), index("google.csv")
    records: list[AmazonGoogleRecord] = []
    with open(data_dir / "matches.csv", newline="", encoding="utf-8") as f:
        for pair_id, m in enumerate(csv.DictReader(f)):
            a, g = amazon.get(m["amazon_id"]), google.get(m["google_id"])
            if a is None or g is None:
                continue
            pa, pg = _price(a["price"]), _price(g["price"])
            if pa is None or pg is None:  # need both prices to compare
                continue
            cat = _catalog_bit(m["amazon_id"], m["google_id"])
            records.append(AmazonGoogleRecord([pair_id, cat, _expensive_bit(pa), 0]))
            records.append(AmazonGoogleRecord([pair_id, cat, _expensive_bit(pg), 1]))
    return records


def records_to_worlds(records: list[AmazonGoogleRecord]) -> list[list[list[int]]]:
    """Convert flat records to the 1-tuple-world format expected by the scan."""
    return [[r.to_world_row()] for r in records]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Amazon-Google dataset utility")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/amazon_google"))
    p.add_argument("--info", action="store_true", help="Print schema and FD summary")
    p.add_argument("--mock", action="store_true", help="Summarize the mock dataset")
    args = p.parse_args(argv)

    if args.info:
        from data.utils import augmented_overlap

        aug = augmented_overlap(OVERLAP_SCHEMAS[0], FDS)
        print(f"Amazon-Google: {N_ATTRS} attributes, {len(VIEW_SCHEMAS)} views")
        print(f"Views:    {VIEW_SCHEMAS}")
        print(f"Overlap:  {OVERLAP_SCHEMAS}")
        print(f"FDs:      {FDS}")
        print(f"Õ (augmented overlap): {sorted(aug)}")
        print(f"Q_catalog   certified: {Q_CATALOG.is_certified(CONFIG)}")
        print(f"Q_expensive certified: {Q_EXPENSIVE.is_certified(CONFIG)}")
        return

    if args.mock:
        recs = make_mock_dataset()
        print(f"Mock Amazon-Google: {len(recs)} records ({len(recs) // 2} pairs × 2 sources)")
        print(f"  Example: {recs[0].to_world_row()} / {recs[1].to_world_row()}")
        return

    records = load_dataset(args.data_dir)
    print(f"Loaded {len(records)} records ({len(records) // 2} matched pairs × 2 sources)")


if __name__ == "__main__":
    main()

"""
WDC-Product dataset loader — Amazon + Walmart + BestBuy product listings.

Schema (3 views, 8 attributes):
  attr 0: brand_hash      (int, domain=2^12)
  attr 1: model_hash      (int, domain=2^14)
  attr 2: category_id     (int, 0..63)
  attr 3: price_bucket    (int, 0..15; log-scale price range)
  attr 4: rating_bucket   (int, 0..4; 1-star bins, 0=no rating)
  attr 5: n_reviews_log   (int, 0..10; floor(log2(n_reviews+1)))
  attr 6: source_id       (int, 0=Amazon, 1=Walmart, 2=BestBuy)
  attr 7: in_stock        (int, 0/1)

Views (relation = one retail source):
  V0 = Amazon:   attrs {0, 1, 2, 3, 4}
  V1 = Walmart:  attrs {0, 1, 2, 3, 7}
  V2 = BestBuy:  attrs {0, 1, 2, 5, 6}

Overlaps:
  O_01 = {0, 1, 2}   (brand + model + category)
  O_02 = {0, 1, 2}
  O_12 = {0, 1, 2}

FDs (cross-world interface laws):
  {0, 1} → 2   (brand+model determines category)
  {0, 1} → 3   (brand+model determines price bucket, from Amazon)
  {0, 1} → 7   (brand+model determines in-stock, from Walmart)

Query examples:
  Q_available:    ∃ t in V1 s.t. t.in_stock = 1
  Q_cheap:        ∃ t in V1 s.t. t.price_bucket = 3  (Walmart, exact bucket)
  Q_highly_rated: ∃ t in V0 s.t. t.rating_bucket >= 4

Download:
  Raw CSVs are NOT bundled.  Run:
    python -m data.wdc --download --data-dir data/raw/wdc
  which fetches the pre-processed snapshots from the project data share.
  (Requires VPN / institutional access.)

  For local unit-testing and CI, use `make_mock_dataset()` instead.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np

from data.synthetic import Atom, BooleanCQ, Config

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

N_ATTRS = 8
DOMAIN_SIZES = [2**12, 2**14, 64, 16, 5, 11, 3, 2]  # per-attribute

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 1, 2, 3, 4}),  # Amazon
    1: frozenset({0, 1, 2, 3, 7}),  # Walmart
    2: frozenset({0, 1, 2, 5, 6}),  # BestBuy
}

OVERLAP_SCHEMAS: list[frozenset[int]] = [
    frozenset({0, 1, 2}),  # brand + model + category
]

FDS: list[tuple[frozenset[int], int]] = [
    (frozenset({0, 1}), 2),
    (frozenset({0, 1}), 3),
    (frozenset({0, 1}), 7),
]

CONFIG = Config(
    n_attrs=N_ATTRS,
    domain_size=max(DOMAIN_SIZES),
    fds=FDS,
    view_schemas=VIEW_SCHEMAS,
    overlap_schemas=OVERLAP_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Canonical queries
# ---------------------------------------------------------------------------

# "Is product (brand b, model m) in stock at Walmart?" — footprint {0,1,2,7}
# Certified: {0,1}→7 and {0,1}→2, so 7 and 2 both in closure({0,1,2}) = Õ
Q_AVAILABLE = BooleanCQ(atoms=[Atom(view_id=1, pattern=["b0", "m0", "c0", "p0", 1])])

# "Is brand b / model m priced at bucket 3 at Walmart?" — footprint {0,1,2,3,7} = Õ
# Certified: {0,1}→3 and {0,1}→7 both in Õ; view footprint equals Õ exactly.
Q_CHEAP = BooleanCQ(atoms=[Atom(view_id=1, pattern=["b0", "m0", "c0", 3, "s0"])])

# "Is model (b,m) rated ≥ 4 stars on Amazon?" — footprint {0,1,2,4}
# NOT certified: attr 4 (rating_bucket) not functionally determined by overlap attrs
Q_HIGHLY_RATED = BooleanCQ(atoms=[Atom(view_id=0, pattern=["b0", "m0", "c0", "p0", 4])])

# "Does product (b,m) have ~32+ user reviews on BestBuy?" — footprint {0,1,2,5,6}
# NOT certified: attr 5 (n_reviews_log) outside Σ-closure of overlap {0,1,2,3,7}
# Witness structure: same cluster → same obs key; Amazon/Walmart/BestBuy records have
# independently varying review counts, so straddling witnesses appear within clusters.
Q_REVIEWED = BooleanCQ(atoms=[Atom(view_id=2, pattern=["b0", "m0", "c0", 5, "src0"])])

# "Is product (b,m) both 4-star on Amazon AND well-reviewed on BestBuy?" — footprint {0,1,2,3,4,5,6}
# NOT certified: attrs 4 (rating_bucket) and 5 (n_reviews_log) both outside aug_overlap.
# Multi-atom query: join on brand+model+category across V0 and V2.
Q_POPULAR = BooleanCQ(
    atoms=[
        Atom(view_id=0, pattern=["b0", "m0", "c0", "p0", 4]),  # Amazon: rating_bucket = 4
        Atom(view_id=2, pattern=["b0", "m0", "c0", 5, "src0"]),  # BestBuy: n_reviews_log = 5
    ]
)


# ---------------------------------------------------------------------------
# Data record type
# ---------------------------------------------------------------------------


class ProductRecord:
    """One product listing as a length-8 int array."""

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


def make_mock_dataset(n: int = 200, seed: int = 42) -> list[ProductRecord]:
    """
    Return n synthetic ProductRecords respecting the FDs, structured as product clusters.

    Each cluster represents one product (fixed brand+model) listed by all three sources
    (Amazon=0, Walmart=1, BestBuy=2).  Within a cluster, brand+model → category,
    price_bucket, in_stock are identical (FD-determined via hash); rating_bucket,
    n_reviews_log, and source_id vary independently across sources.

    This mirrors the real WDC-Product structure where each product cluster has one
    listing per retailer, giving multiple records sharing the same observation key.
    Witness discovery for non-certified queries (e.g. Q_HIGHLY_RATED, Q_REVIEWED)
    requires at least two records with the same obs key but different query answers,
    which this clustered structure reliably produces.
    """
    rng = np.random.default_rng(seed)
    n_sources = len(VIEW_SCHEMAS)  # 3: Amazon, Walmart, BestBuy
    n_clusters = max(1, n // n_sources)

    records: list[ProductRecord] = []
    for _ in range(n_clusters):
        brand = int(rng.integers(0, 2**12))
        model = int(rng.integers(0, 2**14))

        key = f"{brand},{model}".encode()
        h = int(hashlib.md5(key).hexdigest(), 16)
        category = h % 64
        price_bucket = (h >> 6) % 16
        in_stock = (h >> 10) % 2

        for source_id in range(n_sources):
            rating_bucket = int(rng.integers(0, 5))
            n_reviews_log = int(rng.integers(0, 11))
            records.append(
                ProductRecord(
                    [
                        brand,
                        model,
                        category,
                        price_bucket,
                        rating_bucket,
                        n_reviews_log,
                        source_id,
                        in_stock,
                    ]
                )
            )

    return records


# ---------------------------------------------------------------------------
# CSV-based loader (requires downloaded data)
# ---------------------------------------------------------------------------


def _hash_str(s: str, bits: int = 12) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % (2**bits)


def _price_to_bucket(price_str: str) -> int:
    """Parse WDC price strings like 'usd 9 98', '4 095 10 zar', '79 99 usd'."""
    import math
    import re

    s = re.sub(r"[a-zA-Z]", "", price_str).strip()
    parts = s.split()
    try:
        if len(parts) >= 2:
            integer_part = "".join(parts[:-1])
            decimal_part = parts[-1]
            p = float(f"{integer_part}.{decimal_part}")
        elif len(parts) == 1:
            p = float(parts[0])
        else:
            return 0
        if p <= 0:
            return 0
        return min(15, int(math.log2(p + 1)))
    except (ValueError, TypeError):
        return 0


def _load_csv(path: Path) -> Iterator[ProductRecord]:
    """Yield ProductRecords from a pre-processed CSV with columns:
    brand, model, category, price, rating, n_reviews, in_stock, source
    Missing in_stock/rating/price values are filled via brand+model hash so
    the FDs {brand,model}→{price_bucket, in_stock} hold by construction.
    """
    import csv
    import math

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            brand = _hash_str(row.get("brand", ""), 12)
            model = _hash_str(row.get("model", ""), 14)

            key = f"{brand},{model}".encode()
            h = int(hashlib.md5(key).hexdigest(), 16)
            category = h % 64

            price_str = row.get("price", "").strip()
            price_bucket = _price_to_bucket(price_str) if price_str else 0
            if price_bucket == 0:
                price_bucket = (h >> 6) % 16

            in_stock_raw = row.get("in_stock", "").strip()
            if in_stock_raw in ("1", "true", "True", "yes"):
                in_stock = 1
            else:
                in_stock = int((h >> 10) % 2)

            try:
                rating = float(row.get("rating", 0) or 0)
                rating_bucket = max(0, min(4, int(rating) - 1))
            except ValueError:
                rating_bucket = 0
            if rating_bucket == 0 and not row.get("rating", "").strip():
                # Rating absent from corpus: use row-index hash (independent of brand+model)
                # so FD {brand,model}→rating does NOT hold → Q_highly_rated stays uncertified.
                r_seed = int(hashlib.md5(f"rating_row_{row_idx}".encode()).hexdigest(), 16)
                rating_bucket = r_seed % 5

            try:
                n_rev = max(0, int(row.get("n_reviews", 0) or 0))
                n_reviews_log = min(10, int(math.log2(n_rev + 1)))
            except ValueError:
                n_reviews_log = 0
            if n_reviews_log == 0 and not row.get("n_reviews", "").strip():
                # Review count absent from corpus: use row-index hash (independent of brand+model)
                # so FD {brand,model}→n_reviews_log does NOT hold → Q_REVIEWED stays uncertified.
                nr_seed = int(hashlib.md5(f"nreviews_row_{row_idx}".encode()).hexdigest(), 16)
                n_reviews_log = nr_seed % 11

            source_map = {"amazon": 0, "walmart": 1, "bestbuy": 2}
            source_id = source_map.get(str(row.get("source", "")).lower(), 0)

            yield ProductRecord(
                [
                    brand,
                    model,
                    category,
                    price_bucket,
                    rating_bucket,
                    n_reviews_log,
                    source_id,
                    in_stock,
                ]
            )


def load_dataset(data_dir: Path | str) -> list[ProductRecord]:
    """
    Load WDC-Product from pre-processed CSVs in data_dir.
    Expects: amazon.csv, walmart.csv, bestbuy.csv
    Each CSV must have columns: brand, model, category, price, rating, n_reviews, in_stock, source.
    """
    data_dir = Path(data_dir)
    records: list[ProductRecord] = []
    for fname in ("amazon.csv", "walmart.csv", "bestbuy.csv"):
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {p}.  Run: python -m data.wdc --download --data-dir {data_dir}"
            )
        records.extend(_load_csv(p))
    return records


def records_to_worlds(records: list[ProductRecord]) -> list[list[list[int]]]:
    """Convert flat ProductRecords to world format expected by SyntheticBenchmark."""
    return [[r.to_world_row()] for r in records]


def records_to_cluster_worlds(records: list[ProductRecord]) -> list[list[list[int]]]:
    """Group records by (brand_hash, model_hash) into multi-tuple worlds (m=3 per cluster).

    Each cluster world contains all retailer listings for one product, giving
    each world m=3 tuples (Amazon + Walmart + BestBuy).  The source_id at attr 6
    distinguishes which row belongs to which view.

    Use _RealWorldBench with source_col=6 so that view-specific atoms are
    evaluated only against rows from the corresponding retailer.
    """
    from collections import defaultdict

    clusters: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    for r in records:
        clusters[(int(r[0]), int(r[1]))].append(r.to_world_row())
    return list(clusters.values())


# ---------------------------------------------------------------------------
# Download stub
# ---------------------------------------------------------------------------


def _download(data_dir: Path) -> None:
    raise NotImplementedError(
        "WDC-Product download requires institutional VPN access.\n"
        "Place amazon.csv, walmart.csv, bestbuy.csv in:\n"
        f"  {data_dir}\n"
        "then re-run without --download."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="WDC-Product dataset utility")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/wdc"))
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
        print(f"WDC-Product schema: {N_ATTRS} attributes, {len(VIEW_SCHEMAS)} views")
        print(f"Views:    {VIEW_SCHEMAS}")
        print(f"Overlaps: {OVERLAP_SCHEMAS}")
        print(f"FDs:      {FDS}")
        from data.utils import augmented_overlap

        aug = augmented_overlap(OVERLAP_SCHEMAS[0], FDS)
        print(f"Õ (augmented overlap): {sorted(aug)}")
        return

    if args.mock:
        recs = make_mock_dataset()
        print(f"Mock WDC-Product: {len(recs)} records")
        r = recs[0]
        print(f"  Example: {r.to_world_row()}")
        return

    records = load_dataset(args.data_dir)
    print(f"Loaded {len(records)} WDC-Product records from {args.data_dir}")


if __name__ == "__main__":
    main()

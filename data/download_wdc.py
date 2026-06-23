"""
Download WDC-Product source data from the WDC Large-Scale Product Corpus v2.

Reference:
  Primpeli et al., "The WDC Training Dataset and Gold Standard for Large-Scale
  Product Matching", VLDB 2019.  https://webdatacommons.org/largescaleproductcorpus/

The corpus groups product offers by cluster_id (same product, different shops).
We reconstruct three retailer views by rank within each cluster:
  rank 0 → amazon.csv    (first  offer per cluster)
  rank 1 → walmart.csv   (second offer per cluster)
  rank 2 → bestbuy.csv   (third  offer per cluster)

Only clusters with at least three offers are used, so every record has a
natural counterpart in each view.  Singleton/pair clusters are discarded.

Output (written to --data-dir, default data/raw/wdc/):
  amazon.csv    columns: brand, model, category, price, rating, n_reviews, in_stock
  walmart.csv   same columns
  bestbuy.csv   same columns

The `model` column is derived from the offer title (stripped of brand prefix).
The `rating`, `n_reviews`, `in_stock` columns are empty strings where not
present in the corpus; data/wdc.py treats missing values as 0.

Usage:
  python -m data.download_wdc
  python -m data.download_wdc --n-clusters 5000 --data-dir data/raw/wdc
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

CORPUS_URL = (
    "https://data.dws.informatik.uni-mannheim.de/"
    "largescaleproductcorpus/data/v2/offers_corpus_all_v2.json.gz"
)
SAMPLE_URL = (
    "https://data.dws.informatik.uni-mannheim.de/"
    "largescaleproductcorpus/data/v2/samples/offers_corpus_all_v2_sample.json"
)

FIELDNAMES = ["brand", "model", "category", "price", "rating", "n_reviews", "in_stock", "source"]
VIEW_NAMES = ["amazon", "walmart", "bestbuy"]


def _extract_model(title: str, brand: str) -> str:
    """Derive a model identifier by stripping the brand prefix from the title."""
    t = title.strip()
    b = brand.strip()
    if b and t.lower().startswith(b.lower()):
        t = t[len(b) :].lstrip(" -–—:").strip()
    return t[:120]  # cap length


def _parse_offer(rec: dict) -> dict:
    brand = (rec.get("brand") or "").strip()
    title = (rec.get("title") or "").strip()
    model = _extract_model(title, brand)
    category = (rec.get("category") or "").strip()
    price = (rec.get("price") or "").strip()

    # WDC v2 corpus does not carry rating/n_reviews/in_stock; downstream
    # loader (data/wdc.py) treats missing values as 0.
    return {
        "brand": brand,
        "model": model,
        "category": category,
        "price": price,
        "rating": "",
        "n_reviews": "",
        "in_stock": "",
    }


def _stream_jsonl(url: str, use_sample: bool = False):
    """Yield parsed JSON objects, streaming line-by-line (gzip-aware)."""
    target = SAMPLE_URL if use_sample else url
    print(f"Streaming from {target} …")
    req = urllib.request.Request(target)
    with urllib.request.urlopen(req, timeout=120) as resp:
        fobj = gzip.GzipFile(fileobj=resp) if target.endswith(".gz") else resp
        for raw_line in fobj:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def download(n_clusters: int, data_dir: Path, use_sample: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- stream and group offers by cluster_id; stop once we have enough ---
    print(f"Collecting clusters (target: {n_clusters} with ≥3 offers) …")
    clusters: dict[str, list[dict]] = defaultdict(list)
    n_valid = 0
    n_lines = 0

    for rec in _stream_jsonl(CORPUS_URL, use_sample=use_sample):
        n_lines += 1
        cid = str(rec.get("cluster_id") or rec.get("label") or "")
        if not cid:
            continue
        clusters[cid].append(_parse_offer(rec))
        if len(clusters[cid]) == 3:
            n_valid += 1
            if n_valid % 500 == 0:
                print(f"  lines read: {n_lines:>8,}  valid clusters: {n_valid:>6,}")
        if n_valid >= n_clusters:
            print(f"  Reached {n_clusters} valid clusters after {n_lines:,} lines.")
            break

    # keep only clusters with ≥ 3 offers (one per view)
    valid = {cid: offers for cid, offers in clusters.items() if len(offers) >= 3}
    print(f"Clusters with ≥3 offers: {len(valid)}")

    if len(valid) < n_clusters:
        print(
            f"Warning: only {len(valid)} valid clusters available "
            f"(requested {n_clusters}); using all."
        )
        n_clusters = len(valid)

    selected = dict(list(valid.items())[:n_clusters])

    # --- write three view CSVs ---
    writers: dict[str, csv.DictWriter] = {}
    files = {}
    for name in VIEW_NAMES:
        path = data_dir / f"{name}.csv"
        files[name] = open(path, "w", newline="", encoding="utf-8")
        writers[name] = csv.DictWriter(files[name], fieldnames=FIELDNAMES)
        writers[name].writeheader()

    for offers in selected.values():
        for rank, name in enumerate(VIEW_NAMES):
            row = dict(offers[rank])
            row["source"] = name
            writers[name].writerow(row)

    for f in files.values():
        f.close()

    for name in VIEW_NAMES:
        path = data_dir / f"{name}.csv"
        print(f"Wrote {path} ({n_clusters} rows)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download WDC-Product data")
    p.add_argument(
        "--n-clusters",
        type=int,
        default=5_000,
        help="Number of product clusters to keep (default: 5000)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/wdc"),
        help="Output directory (default: data/raw/wdc)",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="Use small sample file instead of full corpus (for testing)",
    )
    args = p.parse_args(argv)
    download(args.n_clusters, args.data_dir, use_sample=args.sample)


if __name__ == "__main__":
    main()

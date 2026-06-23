"""
Download BibInteg source data from the OpenAlex API.

OpenAlex (https://openalex.org) aggregates papers from Crossref, PubMed,
arXiv and other sources.  No API key is required; we identify ourselves
via the mailto parameter as required by the polite pool.

Output (written to --data-dir, default data/raw/bibinteg/):
  dblp.csv              papers with venue info   → DBLP view
  acm.csv               papers with DOI          → ACM view
  semantic_scholar.csv  papers with author count → SemanticScholar view

All three CSVs use the same record set (the same papers projected onto
different attribute subsets), so the FDs title+author+year → venue/doi/n_authors
hold by construction.

Usage:
  python -m data.download_openalex
  python -m data.download_openalex --n-records 10000 --data-dir data/raw/bibinteg
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAILTO = "query-identifiability@example.org"
BASE_URL = "https://api.openalex.org/works"

# CS / database / AI venues by OpenAlex concept ID
# concept: "Computer Science" C41008148, selects a broad but focused corpus
FILTER = (
    "concepts.id:C41008148,"  # Computer Science
    "has_doi:true,"
    "publication_year:2015-2024,"
    "primary_location.source.type:journal|conference"
)
FIELDS = "title,authorships,publication_year,primary_location,doi"
PER_PAGE = 200


def _fetch_page(cursor: str, per_page: int = PER_PAGE) -> tuple[list[dict], str | None]:
    params = {
        "filter": FILTER,
        "select": FIELDS,
        "per-page": per_page,
        "cursor": cursor,
        "mailto": MAILTO,
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"mailto:{MAILTO}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    results = data.get("results", [])
    next_cursor = data.get("meta", {}).get("next_cursor")
    return results, next_cursor


def _parse(work: dict) -> dict | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None
    year = work.get("publication_year")
    if not year:
        return None
    authors_list = work.get("authorships") or []
    authors = "; ".join(
        a.get("author", {}).get("display_name", "") or "" for a in authors_list
    ).strip()
    n_authors = len(authors_list)
    doi = (work.get("doi") or "").replace("https://doi.org/", "").strip()
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    venue = (src.get("display_name") or "").strip()

    if not authors or n_authors == 0:
        return None

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "n_authors": n_authors,
    }


def download(n_records: int, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    cursor = "*"
    page = 0

    print(f"Fetching up to {n_records} records from OpenAlex …")
    while len(records) < n_records:
        try:
            results, next_cursor = _fetch_page(cursor)
        except Exception as exc:
            print(f"  Warning: page {page} failed ({exc}); retrying in 5 s")
            time.sleep(5)
            continue

        for work in results:
            parsed = _parse(work)
            if parsed:
                records.append(parsed)

        page += 1
        fetched = len(records)
        print(f"  page {page:4d} | records so far: {fetched}")

        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

        # polite pool: 10 req/s max; we stay well below that
        time.sleep(0.15)

    records = records[:n_records]
    print(f"Downloaded {len(records)} records.")

    # --- write three view CSVs ---

    # DBLP view: title, authors, year, venue
    dblp_path = data_dir / "dblp.csv"
    with open(dblp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "authors", "year", "venue"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in ["title", "authors", "year", "venue"]})
    print(f"Wrote {dblp_path} ({len(records)} rows)")

    # ACM view: title, authors, year, doi
    acm_path = data_dir / "acm.csv"
    with open(acm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "authors", "year", "doi"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in ["title", "authors", "year", "doi"]})
    print(f"Wrote {acm_path} ({len(records)} rows)")

    # SemanticScholar view: title, authors, year, n_authors
    ss_path = data_dir / "semantic_scholar.csv"
    with open(ss_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "authors", "year", "n_authors"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in ["title", "authors", "year", "n_authors"]})
    print(f"Wrote {ss_path} ({len(records)} rows)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download BibInteg data from OpenAlex")
    p.add_argument(
        "--n-records",
        type=int,
        default=10_000,
        help="Number of records to download (default: 10000)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bibinteg"),
        help="Output directory (default: data/raw/bibinteg)",
    )
    args = p.parse_args(argv)
    download(args.n_records, args.data_dir)


if __name__ == "__main__":
    main()

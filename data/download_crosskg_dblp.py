"""
Download the OpenAlex × DBLP cross-source probe corpus.

Two genuinely independent CS bibliographic sources describing the same papers:
  - DBLP  (curated CS bibliography; venue/author data hand-normalized)
  - OpenAlex (aggregated; messier venue strings, independent author extraction)

We join on DOI (a strong shared key, so disagreements are real semantic conflicts,
not entity-resolution noise) and later (experiments/e_crosskg_probe.py) test which
attributes are functionally determined by the DOI *across the two sources*:
  {doi} -> title / year      expected to (nearly) hold  -> certified queries
  {doi} -> venue / n_authors expected to genuinely vary -> real witnesses

Output (written to --out-dir, default data/raw/crosskg_dblp/):
  dblp.csv      columns: doi, title, venue, year, n_authors
  openalex.csv  columns: doi, title, venue, year, n_authors   (same DOIs)

Usage:
  python -m data.download_crosskg_dblp
  python -m data.download_crosskg_dblp --max-dois 2500 --out-dir data/raw/crosskg_dblp
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
DBLP_API = "https://dblp.org/search/publ/api"
OPENALEX_API = "https://api.openalex.org/works"

# Broad CS topic/venue queries; deduped by DOI afterwards. The exact mix does not
# matter for the probe — we just need real CS papers present in both sources.
DEFAULT_QUERIES = [
    "Proc. VLDB Endow.",
    "SIGMOD Conference",
    "ICDE",
    "VLDB Journal",
    "query optimization",
    "data integration",
    "knowledge graph",
    "machine learning",
    "neural network",
    "transformer",
    "database system",
]


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": f"query-identifiability/0.1 (mailto:{MAILTO})"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _n_authors(info: dict) -> int:
    a = (info.get("authors") or {}).get("author")
    if isinstance(a, list):
        return len(a)
    return 1 if a else 0


def _venue_str(info: dict) -> str:
    v = info.get("venue")
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return str(v) if v else ""


def fetch_dblp(queries: list[str], max_dois: int, per_query: int) -> dict[str, dict]:
    """Return {doi_lower: {doi,title,venue,year,n_authors}} from DBLP search."""
    out: dict[str, dict] = {}
    for q in queries:
        if len(out) >= max_dois:
            break
        params = {"q": q, "format": "json", "h": per_query, "f": 0}
        url = DBLP_API + "?" + urllib.parse.urlencode(params)
        try:
            data = _get_json(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  DBLP query {q!r} failed ({exc}); skipping")
            time.sleep(2)
            continue
        hits = (data.get("result", {}).get("hits", {}) or {}).get("hit", []) or []
        added = 0
        for h in hits:
            info = h.get("info", {})
            doi = (info.get("doi") or "").strip().lower()
            if not doi or doi in out:
                continue
            out[doi] = {
                "doi": doi,
                "title": (str(info.get("title", "")) or "").strip(),
                "venue": _venue_str(info),
                "year": str(info.get("year", "")).strip(),
                "n_authors": _n_authors(info),
                "type": (info.get("type") or "").strip(),
            }
            added += 1
            if len(out) >= max_dois:
                break
        print(f"  DBLP {q!r}: {len(hits)} hits, +{added} new DOIs (total {len(out)})")
        time.sleep(1.0)  # be polite to DBLP
    return out


def fetch_openalex(dois: list[str], batch: int = 50) -> dict[str, dict]:
    """Return {doi_lower: {...}} from OpenAlex, looked up by DOI in batches."""
    out: dict[str, dict] = {}
    for i in range(0, len(dois), batch):
        chunk = dois[i : i + batch]
        filt = "doi:" + "|".join(chunk)
        params = {
            "filter": filt,
            "select": "doi,title,publication_year,primary_location,authorships,type",
            "per-page": batch,
            "mailto": MAILTO,
        }
        url = OPENALEX_API + "?" + urllib.parse.urlencode(params)
        try:
            data = _get_json(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  OpenAlex batch {i // batch} failed ({exc}); retrying in 3 s")
            time.sleep(3)
            continue
        for w in data.get("results", []):
            doi = (w.get("doi") or "").replace("https://doi.org/", "").strip().lower()
            if not doi:
                continue
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            out[doi] = {
                "doi": doi,
                "title": (w.get("title") or "").strip(),
                "venue": (src.get("display_name") or "").strip(),
                "year": str(w.get("publication_year") or "").strip(),
                "n_authors": len(w.get("authorships") or []),
                "type": (w.get("type") or "").strip(),
            }
        print(f"  OpenAlex batch {i // batch + 1}: matched {len(out)}/{len(dois)} so far")
        time.sleep(0.2)
    return out


def _write(path: Path, rows: list[dict]) -> None:
    cols = ["doi", "title", "venue", "year", "n_authors", "type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"Wrote {path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download OpenAlex x DBLP cross-source probe corpus")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/crosskg_dblp"))
    p.add_argument("--max-dois", type=int, default=2500)
    p.add_argument("--per-query", type=int, default=1000)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching DBLP (target {args.max_dois} DOIs across {len(DEFAULT_QUERIES)} queries) …")
    dblp = fetch_dblp(DEFAULT_QUERIES, args.max_dois, args.per_query)
    print(f"DBLP: {len(dblp)} unique DOIs with metadata.")

    print("Looking up the same DOIs in OpenAlex …")
    oa = fetch_openalex(list(dblp.keys()))
    print(f"OpenAlex: matched {len(oa)} of {len(dblp)} DOIs.")

    _write(args.out_dir / "dblp.csv", list(dblp.values()))
    _write(args.out_dir / "openalex.csv", list(oa.values()))
    print(f"\nJoin coverage (DOIs in both): {len(set(dblp) & set(oa))} / {len(dblp)}")


if __name__ == "__main__":
    main()

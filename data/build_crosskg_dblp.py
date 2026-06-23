"""
Build the DBLP side of the OpenAlex x DBLP probe corpus from the bulk dump.

The DBLP search API rate-limits hard (429 after a few queries), so for a corpus
of a few thousand papers we stream-parse the official dump instead:
    https://dblp.org/xml/dblp.xml.gz   (~1 GB gzipped, ISO-8859-1)

We keep only real papers (<article>, <inproceedings>) that carry a DOI (in <ee>
as a doi.org URL), reservoir-sample a uniform subset, and write dblp.csv.  Then
data.download_crosskg_dblp.fetch_openalex looks the same DOIs up in OpenAlex.

The dump uses ~250 custom HTML entities (accented chars) defined in dblp.dtd.
Without lxml we cannot resolve them, so we strip undefined named entities from
the text stream before parsing.  This only affects author-name *text*; DOIs,
years, and author *counts* (one <author> element each) are unaffected.

Usage (after downloading the dump to data/raw/crosskg_dblp/dblp.xml.gz):
    python -m data.build_crosskg_dblp --sample 6000
    python -m data.build_crosskg_dblp --sample 6000 --no-openalex   # DBLP only
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import cast

RAW = Path("data/raw/crosskg_dblp")
DUMP = RAW / "dblp.xml.gz"
KEEP_TYPES = {"article", "inproceedings"}  # real papers; drop proceedings/editor entries
# undefined named entities = &foo; that are not the 5 predefined XML ones
_ENT = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)[a-zA-Z][a-zA-Z0-9]*;")


def _sanitize(line: str) -> str:
    # DBLP's custom entities are standard HTML named entities (&uuml; etc.).
    # Resolve them to real characters (html.unescape) while leaving the five
    # predefined XML entities intact so the stream stays valid XML.
    return _ENT.sub(lambda m: html.unescape(m.group(0)), line)


def _doi_from(elem: ET.Element) -> str:
    for ee in elem.findall("ee"):
        t = (ee.text or "").strip()
        if "doi.org/" in t:
            return t.split("doi.org/", 1)[1].strip().lower()
    return ""


def _venue(elem: ET.Element) -> str:
    for tag in ("journal", "booktitle"):
        v = elem.find(tag)
        if v is not None and v.text:
            return v.text.strip()
    return ""


def parse_dump(dump: Path, sample: int, seed: int = 0) -> list[dict]:
    """Reservoir-sample `sample` DOI-bearing article/inproceedings records."""
    rng = random.Random(seed)
    parser = ET.XMLPullParser(events=("end",))
    reservoir: list[dict] = []
    n_seen = 0  # number of qualifying (typed + DOI) records seen
    n_records = 0

    with gzip.open(dump, "rt", encoding="iso-8859-1") as f:
        for line in f:
            parser.feed(_sanitize(line))
            # read_events() yields (event, Element); cast tightens the loose stub type.
            events = cast("Iterator[tuple[str, ET.Element]]", parser.read_events())
            for _event, elem in events:
                if elem.tag not in KEEP_TYPES:
                    continue
                n_records += 1
                doi = _doi_from(elem)
                if doi:
                    title_el = elem.find("title")
                    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
                    rec = {
                        "doi": doi,
                        "title": title,
                        "venue": _venue(elem),
                        "year": (elem.findtext("year") or "").strip(),
                        "n_authors": len(elem.findall("author")),
                        "type": elem.tag,
                    }
                    # reservoir sampling (uniform without knowing the total)
                    if len(reservoir) < sample:
                        reservoir.append(rec)
                    else:
                        j = rng.randint(0, n_seen)
                        if j < sample:
                            reservoir[j] = rec
                    n_seen += 1
                elem.clear()  # free memory; do not retain the parsed subtree
            if n_records % 500_000 == 0 and n_records:
                print(
                    f"  parsed {n_records:,} papers; {n_seen:,} with DOI; reservoir {len(reservoir)}"
                )

    print(f"Done: {n_records:,} article/inproceedings records, {n_seen:,} with a DOI.")
    return reservoir


def _write(path: Path, rows: list[dict]) -> None:
    cols = ["doi", "title", "venue", "year", "n_authors", "type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"Wrote {path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build DBLP side of crosskg probe from the dump")
    p.add_argument("--sample", type=int, default=6000, help="reservoir sample size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-openalex", action="store_true", help="skip the OpenAlex lookup")
    args = p.parse_args(argv)

    if not DUMP.exists():
        raise FileNotFoundError(
            f"{DUMP} not found. Download it first:\n"
            f"  curl -o {DUMP} https://dblp.org/xml/dblp.xml.gz"
        )

    print(f"Parsing {DUMP} (reservoir sample {args.sample}) …")
    dblp = parse_dump(DUMP, args.sample, args.seed)
    _write(RAW / "dblp.csv", dblp)

    if args.no_openalex:
        return

    from data.download_crosskg_dblp import fetch_openalex

    print("Looking up the same DOIs in OpenAlex …")
    oa = fetch_openalex([r["doi"] for r in dblp])
    _write(RAW / "openalex.csv", list(oa.values()))
    print(
        f"\nJoin coverage (DOIs in both): {len(set(r['doi'] for r in dblp) & set(oa))} / {len(dblp)}"
    )


if __name__ == "__main__":
    main()

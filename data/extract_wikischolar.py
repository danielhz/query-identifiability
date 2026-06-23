"""
Stream a Wikidata JSON dump on stdin and emit three projected CSVs for the
WikiScholar benchmark (loaded by data/wikischolar.py).

Pipeline (full run on ki-kg1):
    lbzip2 -dc /data/ki_nfs_data0/ac/wikidata-dump-2026-04/latest-all.json.bz2 \\
      | python -m data.extract_wikischolar \\
          --out-dir data/raw/wikischolar --target 20000

Dev mode (fast iteration; ~75 s for 300 matches on ki-kg1):
    Scholarly-article Q-IDs cluster in the *interior* of the dump (the first
    ~30 GB of decompressed output is early entities — countries, concepts —
    almost no articles). Skip past that region for dev runs:

    lbzip2 -dc /data/ki_nfs_data0/ac/wikidata-dump-2026-04/latest-all.json.bz2 \\
      | dd bs=1M skip=30720 count=2048 iflag=fullblock 2>/dev/null \\
      | python -m data.extract_wikischolar --dev --out-dir /tmp/wikischolar_dev

Output (three CSVs with the exact column names the loader expects):
    <out-dir>/crossref.csv    columns: title, author, year, venue
    <out-dir>/arxiv.csv       columns: title, author, year, subject_category
    <out-dir>/citations.csv   columns: title, author, year, n_citations

All three files have the same number of rows in the same order; row i in each
corresponds to the same article. The shared (title, author, year) overlap is
unique per row (extractor deduplicates on it) so the FDs
    {title, author, year} -> {venue, subject_category, n_citations}
hold by construction.

Notes on extraction
-------------------
* "author" / "venue" / "subject_category" are emitted as Wikidata Q-IDs
  (stable string identifiers). The downstream loader hashes them, so resolving
  them to English labels is unnecessary and would require a second pass.
* "title" is the literal monolingualtext from P1476, falling back to the
  English label if P1476 is absent. Whitespace is normalized.
* "n_citations" is the count of outgoing P2860 (cites) statements on the
  article entity at dump time.
* We require *all* fields to be present and year ∈ [2010, 2024] (matching the
  loader's year filter). Records missing any field are skipped.

The cheap byte-substring prefilter on Q13442814 (scholarly article class)
avoids json.loads on the ~95% of Wikidata entities that aren't articles, so
end-to-end runtime on the full dump is dominated by lbzip2 decompression
(~45 min on ki-kg1's 16 cores).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterator

# ---------------------------------------------------------------------------
# Wikidata constants
# ---------------------------------------------------------------------------

SCHOLARLY_ARTICLE_QID = "Q13442814"
SCHOLARLY_ARTICLE_QID_BYTES = b'"Q13442814"'

P_INSTANCE_OF = "P31"
P_TITLE = "P1476"
P_AUTHOR_QID = "P50"
P_AUTHOR_STR = "P2093"  # "author name string", used when no item exists
P_PUB_DATE = "P577"
P_VENUE = "P1433"
P_SUBJECT = "P921"
P_CITES = "P2860"

YEAR_MIN, YEAR_MAX = 2010, 2024  # matches data/wikischolar.py loader filter


# ---------------------------------------------------------------------------
# Claim helpers
# ---------------------------------------------------------------------------


def _claim_value_ids(claims: dict, prop: str) -> list[str]:
    """Return Q-IDs from a wikibase-entityid-typed claim."""
    out: list[str] = []
    for s in claims.get(prop, []):
        ms = s.get("mainsnak") or {}
        if ms.get("snaktype") != "value":
            continue
        dv = ms.get("datavalue") or {}
        v = dv.get("value")
        if isinstance(v, dict):
            qid = v.get("id")
            if isinstance(qid, str):
                out.append(qid)
    return out


def _claim_value_strings(claims: dict, prop: str) -> list[str]:
    """Return literal strings from a string- or monolingualtext-typed claim."""
    out: list[str] = []
    for s in claims.get(prop, []):
        ms = s.get("mainsnak") or {}
        if ms.get("snaktype") != "value":
            continue
        dv = ms.get("datavalue") or {}
        v = dv.get("value")
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict) and isinstance(v.get("text"), str):
            out.append(v["text"])
    return out


def _claim_first_year(claims: dict, prop: str) -> int | None:
    """First 4-digit year from a time-typed claim, or None."""
    for s in claims.get(prop, []):
        ms = s.get("mainsnak") or {}
        if ms.get("snaktype") != "value":
            continue
        dv = ms.get("datavalue") or {}
        v = dv.get("value")
        if isinstance(v, dict):
            t = v.get("time")
            if isinstance(t, str) and len(t) >= 5:
                try:
                    return int(t.lstrip("+-")[:4])
                except ValueError:
                    continue
    return None


def _english_label(item: dict) -> str:
    labels = item.get("labels") or {}
    en = labels.get("en") or {}
    val = en.get("value")
    return val if isinstance(val, str) else ""


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------


def extract_record(item: dict) -> dict | None:
    """Return a record dict if `item` is a usable scholarly article, else None.

    A record requires *every* field present and year within the loader's window.
    """
    claims = item.get("claims") or {}

    if SCHOLARLY_ARTICLE_QID not in _claim_value_ids(claims, P_INSTANCE_OF):
        return None

    titles = _claim_value_strings(claims, P_TITLE)
    title = titles[0] if titles else _english_label(item)
    if not title:
        return None
    title = " ".join(title.split())  # normalize whitespace for CSV safety

    author_qids = _claim_value_ids(claims, P_AUTHOR_QID)
    if author_qids:
        author = author_qids[0]
    else:
        author_strs = _claim_value_strings(claims, P_AUTHOR_STR)
        if not author_strs:
            return None
        author = author_strs[0].strip()
        if not author:
            return None

    year = _claim_first_year(claims, P_PUB_DATE)
    if year is None or not (YEAR_MIN <= year <= YEAR_MAX):
        return None

    venues = _claim_value_ids(claims, P_VENUE)
    if not venues:
        return None
    venue = venues[0]

    subjects = _claim_value_ids(claims, P_SUBJECT)
    if not subjects:
        return None
    subject = subjects[0]

    n_citations = len(claims.get(P_CITES, []))

    return {
        "title": title,
        "author": author,
        "year": year,
        "venue": venue,
        "subject_category": subject,
        "n_citations": n_citations,
    }


# ---------------------------------------------------------------------------
# Stream parser
# ---------------------------------------------------------------------------


def iter_candidate_lines(fp_bin) -> Iterator[bytes]:
    """Yield raw JSON-object bytes for lines that mention the scholarly-article QID.

    Reads bytes from `fp_bin` (a binary file-like). The dump format is a JSON
    array, one entity per line, with leading '[' / trailing ']'. We tolerate
    trailing commas and leading whitespace.
    """
    for raw in fp_bin:
        if SCHOLARLY_ARTICLE_QID_BYTES not in raw:
            continue
        line = raw.strip()
        if not line or line[:1] != b"{":
            continue
        if line.endswith(b","):
            line = line[:-1]
        if not line.endswith(b"}"):
            continue
        yield line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Extract WikiScholar CSVs from a Wikidata JSON dump on stdin"
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for the 3 CSVs")
    p.add_argument("--target", type=int, default=20_000, help="Stop after this many records")
    p.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: target=300, report every 50k candidates",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=200_000,
        help="Log progress every N candidate lines scanned",
    )
    args = p.parse_args(argv)

    if args.dev:
        args.target = 300
        args.report_every = 50_000

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cr_path = args.out_dir / "crossref.csv"
    ax_path = args.out_dir / "arxiv.csv"
    ci_path = args.out_dir / "citations.csv"

    cr_f = open(cr_path, "w", newline="", encoding="utf-8")
    ax_f = open(ax_path, "w", newline="", encoding="utf-8")
    ci_f = open(ci_path, "w", newline="", encoding="utf-8")

    cr_w = csv.writer(cr_f)
    ax_w = csv.writer(ax_f)
    ci_w = csv.writer(ci_f)
    cr_w.writerow(["title", "author", "year", "venue"])
    ax_w.writerow(["title", "author", "year", "subject_category"])
    ci_w.writerow(["title", "author", "year", "n_citations"])

    seen: set[tuple[str, str, int]] = set()
    written = 0
    candidates = 0
    matched = 0
    parse_errors = 0
    t0 = perf_counter()
    last_t = t0
    last_candidates = 0

    try:
        for line in iter_candidate_lines(sys.stdin.buffer):
            candidates += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            rec = extract_record(item)
            if rec is None:
                if candidates % args.report_every == 0:
                    _report(
                        t0, last_t, candidates, last_candidates, matched, written, parse_errors
                    )
                    last_t = perf_counter()
                    last_candidates = candidates
                continue

            matched += 1
            key = (rec["title"], rec["author"], rec["year"])
            if key in seen:
                continue
            seen.add(key)

            cr_w.writerow([rec["title"], rec["author"], rec["year"], rec["venue"]])
            ax_w.writerow([rec["title"], rec["author"], rec["year"], rec["subject_category"]])
            ci_w.writerow([rec["title"], rec["author"], rec["year"], rec["n_citations"]])
            written += 1

            if candidates % args.report_every == 0:
                _report(t0, last_t, candidates, last_candidates, matched, written, parse_errors)
                last_t = perf_counter()
                last_candidates = candidates

            if written >= args.target:
                break
    finally:
        cr_f.close()
        ax_f.close()
        ci_f.close()

    elapsed = perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s. "
        f"candidates={candidates:,}  matched={matched:,}  written={written:,}  "
        f"parse_errors={parse_errors:,}\n"
        f"Output → {args.out_dir}/{{crossref,arxiv,citations}}.csv",
        file=sys.stderr,
    )


def _report(
    t0: float,
    last_t: float,
    candidates: int,
    last_candidates: int,
    matched: int,
    written: int,
    parse_errors: int,
) -> None:
    now = perf_counter()
    dt = now - last_t
    delta = candidates - last_candidates
    rate = delta / dt if dt > 0 else 0.0
    print(
        f"[{int(now - t0):>5}s] candidates={candidates:>10,}  "
        f"matched={matched:>8,}  written={written:>6,}  "
        f"errs={parse_errors:>4}  ({rate:>8,.0f} cand/s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
e_crosskg_probe.py — fail-fast probe: does OpenAlex x DBLP have a NATURAL
certified/uncertified split on real values?

Two independent CS bibliographic sources (DBLP, OpenAlex) describe the same
papers; we join on DOI and test, per attribute, whether the two sources agree.
For each candidate FD {doi} -> X:
    a joined DOI whose two sources DISAGREE on X is a real, in-the-wild
    non-identifiability witness (two records sharing the overlap key {doi}
    but differing on the determined attribute X).

go-gate (printed at the end):
  (a) join coverage large enough,
  (b) >= 1 attribute agrees ~always  -> a real CERTIFIED query,
  (c) >= 1 attribute genuinely conflicts at a meaningful rate, with the
      conflict NOT explained away by string normalization -> real WITNESSES.

Run (after data.download_crosskg_dblp):
    python -m experiments.e_crosskg_probe
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "crosskg_dblp"
OUT = Path(__file__).resolve().parent.parent / "results" / "e_crosskg_probe_dblp.json"

_VENUE_ABBR = {
    "proc": "proceedings",
    "endow": "endowment",
    "j": "journal",
    "conf": "conference",
    "trans": "transactions",
    "int": "international",
    "intl": "international",
    "ieee": "",
    "acm": "",
    "the": "",
    "of": "",
    "on": "",
}


def load(name: str) -> dict[str, dict]:
    with open(RAW / f"{name}.csv", newline="", encoding="utf-8") as f:
        return {r["doi"].strip().lower(): r for r in csv.DictReader(f)}


def norm_title(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", s)  # fold unicode dashes / ligatures
    s = re.sub(r"<[^>]+>", "", s)  # strip HTML/MathML markup (e.g. <scp>, <sub>)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def norm_venue(s: str) -> frozenset[str]:
    toks = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
    out = []
    for t in toks:
        t = _VENUE_ABBR.get(t, t)
        if t:
            out.append(t)
    return frozenset(out)


def venue_match(a: str, b: str) -> bool:
    """Loose venue agreement: one normalized token set contains the other."""
    na, nb = norm_venue(a), norm_venue(b)
    if not na or not nb:
        return False
    return na <= nb or nb <= na


def check(joined: list[tuple[dict, dict]], target: str, agree) -> dict[str, Any]:
    n = len(joined)
    witnesses: list[dict] = []
    n_agree = 0
    for d, o in joined:
        if agree(d.get(target, ""), o.get(target, "")):
            n_agree += 1
        elif len(witnesses) < 8:
            witnesses.append(
                {"doi": d["doi"], "dblp": d.get(target, ""), "openalex": o.get(target, "")}
            )
    return {
        "n": n,
        "n_agree": n_agree,
        "n_conflict": n - n_agree,
        "hold_rate": n_agree / n if n else 1.0,
        "witnesses": witnesses,
    }


def main() -> None:
    dblp, oa = load("dblp"), load("openalex")
    keys = sorted(set(dblp) & set(oa))
    joined_all = [(dblp[k], oa[k]) for k in keys]

    # Mandatory cleaning: drop front-matter / proceedings entries. OpenAlex flags
    # them type=paratext (0 authors); they inflated venue/title/n_authors conflicts
    # without being genuine witnesses (verified 2026-06-10).
    keep_dblp = {"article", "inproceedings"}
    joined = [
        (d, o)
        for (d, o) in joined_all
        if o.get("type", "") != "paratext"
        and (d.get("type", "") in keep_dblp or not d.get("type"))
    ]
    n_paratext = len(joined_all) - len(joined)

    report: dict[str, Any] = {
        "dataset": "OpenAlex x DBLP (cross-source, join on DOI)",
        "n_dblp": len(dblp),
        "n_openalex": len(oa),
        "n_joined_raw": len(joined_all),
        "n_paratext_dropped": n_paratext,
        "n_joined": len(joined),
        "join_coverage": len(joined_all) / len(dblp) if dblp else 0.0,
        "attributes": {},
    }
    print(f"Dropped {n_paratext} paratext/front-matter rows; {len(joined)} clean papers remain.")

    def yr_eq(a, b):
        return a.strip() == b.strip()

    def yr_eq1(a, b):  # within one year (preprint vs published)
        try:
            return abs(int(a) - int(b)) <= 1
        except ValueError:
            return a.strip() == b.strip()

    checks = {
        "title (normalized)": lambda a, b: norm_title(a) == norm_title(b),
        "year (exact)": yr_eq,
        "year (±1)": yr_eq1,
        "venue (raw string)": lambda a, b: a.strip() == b.strip(),
        "venue (normalized)": venue_match,
        "n_authors (exact)": lambda a, b: str(a).strip() == str(b).strip(),
    }
    target_col = {
        "title (normalized)": "title",
        "year (exact)": "year",
        "year (±1)": "year",
        "venue (raw string)": "venue",
        "venue (normalized)": "venue",
        "n_authors (exact)": "n_authors",
    }

    print(f"Joined {len(joined)} DOIs (coverage {100 * report['join_coverage']:.1f}% of DBLP).\n")
    for label, agree in checks.items():
        res = check(joined, target_col[label], agree)
        report["attributes"][label] = res
        print(
            f"  {label:22s}: holds {100 * res['hold_rate']:6.2f}%  ({res['n_conflict']} conflicts)"
        )
        for w in res["witnesses"][:1]:
            print(f"       e.g. {w['dblp']!r}  vs  {w['openalex']!r}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

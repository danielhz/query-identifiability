#!/usr/bin/env python3
"""
e_fd_validation.py — empirically validate the BibInteg interface laws on raw data.

The BibInteg interface posits the functional dependencies
    (title, authors, year) -> venue        [DBLP]
    (title, authors, year) -> doi           [ACM]
    (title, authors, year) -> n_authors     [SemanticScholar]
as deterministic interface laws.  Unlike the WDC determined/target attributes
(which data/wdc.py synthesizes), venue/doi/n_authors are present in the raw
OpenAlex-derived CSVs, so these FDs are directly measurable on real data.

For each FD we group records by the antecedent key and report how many keys map
to a single consequent value (FD holds) versus several (FD violated).  A violating
group is a genuine real-data non-identifiability witness: two records sharing the
overlap key but disagreeing on the determined attribute.

Run from the code repo root:
    python -m experiments.e_fd_validation
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "bibinteg"
KEY = ("title", "authors", "year")
FDS = [("dblp", "venue"), ("acm", "doi"), ("semantic_scholar", "n_authors")]


def load(name: str) -> list[dict]:
    with open(RAW / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def check_fd(rows: list[dict], key_cols: tuple[str, ...], target: str):
    groups: dict[tuple, set] = defaultdict(set)
    examples: dict[tuple, list] = defaultdict(list)
    for r in rows:
        k = tuple(r[c].strip() for c in key_cols)
        v = r[target].strip()
        groups[k].add(v)
        if len(examples[k]) < 8:
            examples[k].append(v)
    violating = {k: sorted(vs) for k, vs in groups.items() if len(vs) > 1}
    n = len(groups)
    return {
        "n_keys": n,
        "n_violating": len(violating),
        "hold_rate": (n - len(violating)) / n if n else 1.0,
        "witnesses": [
            {"key": dict(zip(key_cols, k)), "values": vs} for k, vs in list(violating.items())[:5]
        ],
    }


def main() -> None:
    report: dict[str, Any] = {"dataset": "BibInteg", "key": list(KEY), "fds": {}}
    for source, target in FDS:
        res = check_fd(load(source), KEY, target)
        report["fds"][f"{'/'.join(KEY)}->{target}"] = {"source": source, **res}
        print(
            f"({'/'.join(KEY)}) -> {target:10s} [{source}]: "
            f"{res['n_keys']} keys, {res['n_violating']} violating, "
            f"holds on {100 * res['hold_rate']:.2f}% of keys"
        )
        for w in res["witnesses"][:1]:
            print(f"    e.g. witness: {w['key']['title'][:55]}... -> {w['values']}")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "e_fd_validation_bibinteg.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

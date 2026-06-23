"""
Merge multiple per-seed experiment JSON files into one combined file.

Each input file must have the structure {"meta": {...}, "results": [...]}.
The combined file keeps all individual metas in a list under "sources" and
concatenates the result records.

Usage
-----
    # Merge all E1 seed files (excluding already-combined files):
    python -m analysis.combine_results \\
        --prefix e1_error_floor \\
        --output results/e1_error_floor_combined.json

    # Merge E2 seeds 0-2 by explicit glob:
    python -m analysis.combine_results \\
        --inputs results/e2_capability_jump_202605*.json \\
        --output results/e2_capability_jump_combined.json

    # Merge specific files:
    python -m analysis.combine_results \\
        --inputs results/e2_capability_jump_20260523_114219.json \\
                 results/e2_capability_jump_20260523_114232.json \\
                 results/e2_capability_jump_20260523_114250.json \\
        --output results/e2_capability_jump_combined.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def combine(input_paths: list[Path], output: Path) -> None:
    all_results: list[dict] = []
    all_metas: list[dict] = []

    for p in sorted(input_paths):
        with open(p) as f:
            data = json.load(f)
        all_results.extend(data.get("results", []))
        all_metas.append({"source_file": str(p), **data.get("meta", {})})
        print(f"  Loaded {len(data.get('results', []))} records from {p.name}")

    payload = {
        "meta": {
            "combined": True,
            "n_sources": len(input_paths),
            "total_records": len(all_results),
            "combined_at": datetime.now().isoformat(),
            "sources": all_metas,
        },
        "results": all_results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Combined {len(all_results)} records from {len(input_paths)} files → {output}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Merge per-seed experiment result files")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        help="Explicit list of input JSON files",
    )
    src.add_argument(
        "--prefix",
        type=str,
        help="Prefix pattern: combines all <results-dir>/<prefix>_*.json "
        "that do NOT already contain 'combined' in the filename",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to search when using --prefix (default: results/)",
    )
    p.add_argument("--output", type=Path, required=True, help="Output combined JSON file")
    args = p.parse_args(argv)

    if args.inputs:
        inputs = [Path(i) for i in args.inputs]
    else:
        inputs = [
            f
            for f in sorted(args.results_dir.glob(f"{args.prefix}_*.json"))
            if "combined" not in f.name
        ]
        if not inputs:
            raise FileNotFoundError(
                f"No {args.prefix}_*.json (excluding combined) found in {args.results_dir}"
            )

    combine(inputs, args.output)


if __name__ == "__main__":
    main()

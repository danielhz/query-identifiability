#!/usr/bin/env python3
"""
Anonymize experiment result files for double-blind artifact submission.

Reads every e*.json file from --input-dir, replaces identifying values
(hostnames, SLURM job IDs) with stable pseudonyms, and writes the
sanitized files to --output-dir.

A deanonymization key is written to --key-file (default: anon_key.json)
in the *source* directory so the original information is never lost.
The key file must NOT be included in the submitted artifact.

Usage
-----
    python anonymize_artifacts.py                       # defaults
    python anonymize_artifacts.py --input-dir results --output-dir results_anon

Anonymized fields (meta.system)
--------------------------------
  hostname          → "compute-node-<N>"
  slurm_job_id      → "job-<N>"
  os                → strip domain from kernel string (kept for hardware context)

Preserved fields
----------------
  git_commit        — kept: it is a content hash, not an institution identifier
  cuda_version      — kept: hardware context relevant to reviewers
  gpu_name          — kept: hardware context relevant to reviewers
  torch_version     — kept: software environment
  slurm_array_task_id — kept: integer 0-N, not identifying
  python_version    — kept: software environment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _make_pseudonym(real: str, registry: dict[str, str], prefix: str) -> str:
    """Return a stable pseudonym for *real*, creating one if needed."""
    if real not in registry:
        registry[real] = f"{prefix}-{len(registry) + 1}"
    return registry[real]


def anonymize_system(system: dict, state: dict) -> dict:
    """Return a copy of *system* with identifying fields replaced."""
    anon = dict(system)

    # hostname: most identifying field — maps to "compute-node-N"
    if system.get("hostname"):
        anon["hostname"] = _make_pseudonym(
            system["hostname"], state["hostname_map"], "compute-node"
        )

    # SLURM job ID: links back to cluster account logs
    if system.get("slurm_job_id"):
        anon["slurm_job_id"] = _make_pseudonym(system["slurm_job_id"], state["job_map"], "job")

    # OS string: strip the full kernel release if it contains the hostname,
    # but keep the OS family (e.g. "Linux 5.14.0") for hardware context.
    if system.get("os"):
        # Drop everything after the second space-separated token
        parts = system["os"].split()
        anon["os"] = " ".join(parts[:2]) if len(parts) >= 2 else system["os"]

    return anon


def anonymize_file(src: Path, dst: Path, state: dict) -> None:
    with open(src) as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    if "system" in meta:
        meta = {**meta, "system": anonymize_system(meta["system"], state)}
        payload = {**payload, "meta": meta}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("results"))
    p.add_argument("--output-dir", type=Path, default=Path("results_anon"))
    p.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Where to write the deanonymization key (default: <input-dir>/../anon_key.json)",
    )
    args = p.parse_args(argv)

    key_file = args.key_file or args.input_dir.parent / "anon_key.json"

    # Shared mutable state for stable cross-file pseudonyms
    state: dict = {"hostname_map": {}, "job_map": {}}

    result_files = sorted(args.input_dir.glob("e*.json"))
    if not result_files:
        print(f"No result files found in {args.input_dir}")
        return

    for src in result_files:
        dst = args.output_dir / src.name
        anonymize_file(src, dst, state)
        print(f"  {src.name}  →  {dst}")

    # Write the deanonymization key — keep this out of the artifact
    key = {
        "WARNING": "Do NOT include this file in the submitted artifact.",
        "hostname_map": {v: k for k, v in state["hostname_map"].items()},
        "job_map": {v: k for k, v in state["job_map"].items()},
    }
    with open(key_file, "w") as f:
        json.dump(key, f, indent=2)

    print(f"\nAnonymized {len(result_files)} file(s) → {args.output_dir}/")
    print(f"Deanonymization key saved to: {key_file}")
    print(f"  *** Do NOT submit {key_file} with the artifact ***")


if __name__ == "__main__":
    main()

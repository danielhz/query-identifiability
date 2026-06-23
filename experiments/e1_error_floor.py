"""
E1 — Error Floor and Certificate Accuracy  (RQ1 + RQ2)

Empirically verifies two theorems on a synthetic benchmark:

  Theorem 1 (certificate): certified queries (footprint ⊆ Õ) are identifiable;
    all architectures should achieve high accuracy with enough training worlds.

  Theorem 5 (error floor): non-certified queries have error ≥ 1/2 for EVERY
    evidence-measurable estimator, regardless of architecture or capacity.

Protocol
--------
1. Build a SyntheticBenchmark from the CRM-mini config (or a custom one).
2. Sample n_cert certified and n_noncert non-certified single-atom Boolean CQs
   with random constant patterns.
3. For each query × architecture × seed: train and record test accuracy.
4. Write results as JSON to --output-dir.

Quick test (< 5 s on CPU):
    python -m experiments.e1_error_floor --mini

Full sweep:
    python -m experiments.e1_error_floor \\
        --n-train 5000 --n-val 500 --n-test 1000 \\
        --n-tuples 20 --n-cert 10 --n-noncert 10 \\
        --epochs 300 --patience 30 --hidden-dim 128 \\
        --seeds 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from data import crm_mini as crm
from data.synthetic import Atom, BooleanCQ, Config
from data.utils import augmented_overlap, build_overlap_graph
from experiments.runner import ARCHITECTURES, run_trial, save_results
from models.train import TrainConfig

# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


def _certified_view_ids(config: Config) -> list[int]:
    augs = [augmented_overlap(o, config.fds) for o in config.overlap_schemas]
    return [vid for vid, attrs in config.view_schemas.items() if any(attrs <= aug for aug in augs)]


def _noncertified_view_ids(config: Config) -> list[int]:
    augs = [augmented_overlap(o, config.fds) for o in config.overlap_schemas]
    return [
        vid for vid, attrs in config.view_schemas.items() if not any(attrs <= aug for aug in augs)
    ]


def generate_queries(
    config: Config,
    n_cert: int,
    n_noncert: int,
    rng: np.random.Generator,
) -> tuple[list[BooleanCQ], list[BooleanCQ]]:
    """
    Sample certified and non-certified single-atom Boolean CQs with random
    constant patterns.  Raises ValueError if the schema has no views of the
    required type.
    """
    cert_vids = _certified_view_ids(config)
    noncert_vids = _noncertified_view_ids(config)

    if not cert_vids and n_cert > 0:
        raise ValueError("Schema has no view with a certified footprint.")
    if not noncert_vids and n_noncert > 0:
        raise ValueError("Schema has no view with a non-certified footprint.")

    def _sample(vids: list[int], n: int) -> list[BooleanCQ]:
        queries = []
        for _ in range(n):
            vid = vids[rng.integers(len(vids))]
            attrs = sorted(config.view_schemas[vid])
            pattern: list[int | str] = [
                int(v) for v in rng.integers(0, config.domain_size, size=len(attrs))
            ]
            queries.append(BooleanCQ(atoms=[Atom(view_id=vid, pattern=pattern)]))
        return queries

    return _sample(cert_vids, n_cert), _sample(noncert_vids, n_noncert)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="E1: Error floor and certificate accuracy")
    p.add_argument(
        "--mini", action="store_true", help="Fast smoke-test mode: tiny dataset, few epochs"
    )
    # Dataset
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-tuples", type=int, default=10, help="Tuples per generated world")
    p.add_argument("--n-cert", type=int, default=5, help="Number of certified queries to sample")
    p.add_argument(
        "--n-noncert", type=int, default=5, help="Number of non-certified queries to sample"
    )
    # Model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--architectures", nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    # Replication
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    # Output
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    if args.mini:
        args.n_train = 80
        args.n_val = 20
        args.n_test = 50
        args.n_tuples = 5
        args.n_cert = 2
        args.n_noncert = 2
        args.epochs = 15
        args.patience = 15
        args.seeds = [0]
        args.hidden_dim = 16

    config = crm.make_config()
    rng = np.random.default_rng(args.seeds[0])
    bench = crm.make_benchmark(seed=args.seeds[0])
    adj = build_overlap_graph(config.n_attrs, config.overlap_schemas, config.fds)

    cert_queries, noncert_queries = generate_queries(config, args.n_cert, args.n_noncert, rng)
    all_queries = [("certified", q) for q in cert_queries] + [
        ("noncertified", q) for q in noncert_queries
    ]

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

    records: list[dict] = []
    total = len(all_queries) * len(args.architectures) * len(args.seeds)
    done = 0

    for qtype, query in all_queries:
        for arch in args.architectures:
            for seed in args.seeds:
                rec = run_trial(
                    model_name=arch,
                    bench=bench,
                    adj=adj,
                    query=query,
                    train_cfg=train_cfg,
                    seed=seed,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    n_test=args.n_test,
                    n_tuples=args.n_tuples,
                    hidden_dim=args.hidden_dim,
                )
                rec["query_type"] = qtype
                records.append(rec)
                done += 1
                cert_flag = "✓" if rec["certified"] else "✗"
                bal = rec["test_balanced_accuracy"]
                bal_str = f"{bal:.3f}" if bal == bal else " nan"  # nan check
                print(
                    f"[{done}/{total}] {qtype:12s} | {arch:18s} | seed={seed} "
                    f"| acc={rec['test_accuracy']:.3f} bal={bal_str} "
                    f"pos={rec['test_positive_rate']:.2f} {cert_flag}"
                )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e1_error_floor_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e1_error_floor",
            "timestamp": ts,
            "n_cert_queries": args.n_cert,
            "n_noncert_queries": args.n_noncert,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "n_tuples": args.n_tuples,
            "architectures": args.architectures,
            "seeds": args.seeds,
            "train_config": {
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": args.lr,
                "batch_size": 256,
                "weight_decay": 1e-5,
                "device": args.device,
            },
        },
    )

    # Print summary
    from collections import defaultdict

    by_type_arch: dict = defaultdict(list)
    for r in records:
        by_type_arch[(r["query_type"], r["model"])].append(r["test_accuracy"])
    print("\n=== Summary (mean test accuracy) ===")
    for (qtype, arch), accs in sorted(by_type_arch.items()):
        print(f"  {qtype:12s} | {arch:18s} | {np.mean(accs):.3f} ± {np.std(accs):.3f}")


if __name__ == "__main__":
    main()

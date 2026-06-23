"""
E2 — Capability Jumps Under Interface Augmentation  (RQ3)

Empirically verifies Theorem 6: adding a single FD law (or overlap) can flip
a query from non-identifiable to identifiable in one step, causing an abrupt
jump in predictor accuracy.

Schema
------
5 attributes {0,1,2,3,4}, domain {0,1}.
Views: V0 = {0,1}, V1 = {0,2,3,4}.
Overlap: {0}.
Query Q: ∃x0. R_{V1}(x0, 0, 0, 0)  (constants 0 on non-overlap attrs 2,3,4; variable on overlap attr 0)
Footprint of Q = {0,2,3,4}.

Why non-overlap constants: attrs 2,3,4 are NOT observable in the overlap {0}, so their
values cannot be inferred without FDs. With n_tuples=5 and binary domain the positive rate
≈ 0.49 (balanced), enabling meaningful balanced-accuracy measurement.

Augmentation sequence (FDs added cumulatively):
  step 0: no FDs         → Õ={0},           footprint ⊄ Õ  → NOT CERTIFIED
  step 1: +FD {0}→1      → Õ={0,1},         footprint ⊄ Õ  → NOT CERTIFIED
  step 2: +FD {0}→2      → Õ={0,1,2},       footprint ⊄ Õ  → NOT CERTIFIED
  step 3: +FD {0}→3      → Õ={0,1,2,3},     footprint ⊄ Õ  → NOT CERTIFIED
  step 4: +FD {0}→4      → Õ={0,1,2,3,4},   footprint ⊆ Õ  → CERTIFIED  ← JUMP

Expected result: balanced accuracy ≈ 0.5 for steps 0-3, then a sharp jump at step 4.

Quick test (< 10 s on CPU):
    python -m experiments.e2_capability_jump --mini

Full sweep:
    python -m experiments.e2_capability_jump \\
        --n-train 5000 --n-val 500 --n-test 1000 --n-tuples 20 \\
        --epochs 300 --patience 30 --hidden-dim 128 \\
        --seeds 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from data.synthetic import Atom, BooleanCQ, Config, SyntheticBenchmark
from data.utils import augmented_overlap, build_overlap_graph
from experiments.runner import ARCHITECTURES, run_trial, save_results
from models.train import TrainConfig

# ---------------------------------------------------------------------------
# Augmentation sequence
# ---------------------------------------------------------------------------

_N_ATTRS = 5
_DOMAIN = 2
_VIEW_SCHEMAS = {
    0: frozenset({0, 1}),
    1: frozenset({0, 2, 3, 4}),
}
_OVERLAP_SCHEMAS = [frozenset({0})]

# Cumulative FD sets: each step adds one FD {0}→k
_STEPS: list[tuple[str, list[tuple[frozenset, int]]]] = [
    ("no FDs", []),
    ("+FD 0→1", [(frozenset({0}), 1)]),
    ("+FD 0→2", [(frozenset({0}), 1), (frozenset({0}), 2)]),
    ("+FD 0→3", [(frozenset({0}), 1), (frozenset({0}), 2), (frozenset({0}), 3)]),
    (
        "+FD 0→4",
        [(frozenset({0}), 1), (frozenset({0}), 2), (frozenset({0}), 3), (frozenset({0}), 4)],
    ),
]

# Query: R_{V1}(x0, 0, 0, 0) — variables on overlap attr 0, constants on non-overlap attrs 2,3,4.
# Constants on non-overlap attrs are NOT determinable from the overlap {0} alone, making the
# query non-identifiable until FDs {0}→2, {0}→3, {0}→4 are all active (step 4).
# With n_tuples=5 and binary domain, positive rate ≈ 0.49 (near-balanced for balanced_accuracy).
QUERY = BooleanCQ(atoms=[Atom(view_id=1, pattern=["x0", 0, 0, 0])])


def _make_step_config(fds) -> Config:
    return Config(
        n_attrs=_N_ATTRS,
        domain_size=_DOMAIN,
        fds=list(fds),
        view_schemas=_VIEW_SCHEMAS,
        overlap_schemas=_OVERLAP_SCHEMAS,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="E2: Capability jump under FD augmentation")
    p.add_argument(
        "--mini", action="store_true", help="Fast smoke-test: tiny dataset, steps 0 and 4 only"
    )
    # Dataset
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-tuples", type=int, default=5)
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
        args.epochs = 10
        args.patience = 10
        args.seeds = [0]
        args.hidden_dim = 16

    steps = _STEPS if not args.mini else [_STEPS[0], _STEPS[-1]]

    train_cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device
    )

    records: list[dict] = []
    total = len(steps) * len(args.architectures) * len(args.seeds)
    done = 0

    for step_idx, (step_label, fds) in enumerate(steps):
        config = _make_step_config(fds)
        bench = SyntheticBenchmark(config, seed=0)
        adj = build_overlap_graph(_N_ATTRS, _OVERLAP_SCHEMAS, fds)
        cert = QUERY.is_certified(config)

        aug_overlap = augmented_overlap(_OVERLAP_SCHEMAS[0], fds)
        feature_dim = bench.feature_dim()

        for arch in args.architectures:
            for seed in args.seeds:
                rec = run_trial(
                    model_name=arch,
                    bench=bench,
                    adj=adj,
                    query=QUERY,
                    train_cfg=train_cfg,
                    seed=seed,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    n_test=args.n_test,
                    n_tuples=args.n_tuples,
                    hidden_dim=args.hidden_dim,
                )
                rec["step"] = step_idx
                rec["step_label"] = step_label
                rec["aug_overlap"] = sorted(aug_overlap)
                rec["aug_overlap_size"] = len(aug_overlap)
                rec["feature_dim"] = feature_dim
                records.append(rec)
                done += 1
                cert_flag = "CERTIFIED ✓" if cert else "not certified"
                bal = rec["test_balanced_accuracy"]
                bal_str = f"{bal:.3f}" if bal == bal else " nan"
                print(
                    f"[{done}/{total}] step={step_idx} ({step_label:12s}) | "
                    f"{arch:18s} | seed={seed} | {cert_flag} | "
                    f"acc={rec['test_accuracy']:.3f} bal={bal_str} "
                    f"pos={rec['test_positive_rate']:.2f}"
                )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e2_capability_jump_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e2_capability_jump",
            "timestamp": ts,
            "n_attrs": _N_ATTRS,
            "domain_size": _DOMAIN,
            "steps": [s for s, _ in steps],
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

    # Print jump summary
    from collections import defaultdict

    by_step_arch: dict = defaultdict(list)
    for r in records:
        by_step_arch[(r["step"], r["step_label"], r["model"])].append(r["test_accuracy"])
    print("\n=== Accuracy by step (mean over seeds) ===")
    prev_arch = None
    for (step, label, arch), accs in sorted(by_step_arch.items()):
        if arch != prev_arch:
            print(f"\n  {arch}")
            prev_arch = arch
        cert_str = (
            "✓"
            if any(r["certified"] for r in records if r["step"] == step and r["model"] == arch)
            else "✗"
        )
        print(f"    step {step} {label:12s} {cert_str}  acc={np.mean(accs):.3f}")


if __name__ == "__main__":
    main()

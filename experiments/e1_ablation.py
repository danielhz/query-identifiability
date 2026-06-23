"""
E1 Ablation — FD completeness  (ρ sweep)

Isolates the contribution of FD knowledge to certificate quality.

World generation: always uses the FULL FD set Σ.
Predictor knowledge: only ρ·|Σ| FDs (a randomly chosen prefix of Σ).

As ρ increases from 0 to 1, the predictor's augmented overlap Õ_ρ grows.
The certificate flips from non-certified to certified at the critical ρ where
Õ_ρ first covers the query footprint.  Before that point, CA incorrectly
abstains (outputs 0.5) even though the query IS identifiable; at ρ=1 CA uses
the correct certificate and achieves high accuracy.

Schema (same as E2, five attributes):
  n_attrs=5, domain=2, FDs: {0}→1, {0}→2, {0}→3, {0}→4
  Overlap: {0}
  Query: R_{V1}(0, x₂, x₃, x₄)  — footprint {0,2,3,4}

  ρ = 0/4 = 0.00 → Σ_ρ = []            Õ = {0}           not certified
  ρ = 1/4 = 0.25 → Σ_ρ = [{0}→1]       Õ = {0,1}         not certified
  ρ = 2/4 = 0.50 → Σ_ρ adds {0}→2      Õ = {0,1,2}       not certified
  ρ = 3/4 = 0.75 → Σ_ρ adds {0}→3      Õ = {0,1,2,3}     not certified
  ρ = 4/4 = 1.00 → Σ_ρ = full Σ        Õ = {0,1,2,3,4}   CERTIFIED  ← jump

Key prediction (from Theorems 1 + 5):
  All architectures:  accuracy ≈ 0.5 for ρ < 1
  All architectures:  accuracy → high for ρ = 1
  CA specifically:    accuracy = 0.5 exactly for ρ < 1 (by design — it abstains)
                      accuracy → high for ρ = 1 (certificate is correct)

Quick test (< 15 s on CPU):
    python -m experiments.e1_ablation --mini

Full sweep:
    python -m experiments.e1_ablation \\
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
from experiments.runner import ARCHITECTURES, build_model, save_results
from models.train import TrainConfig, evaluate, train

# ---------------------------------------------------------------------------
# Schema constants  (identical to E2)
# ---------------------------------------------------------------------------

_N_ATTRS = 5
_DOMAIN = 2
_VIEW_SCHEMAS = {0: frozenset({0, 1}), 1: frozenset({0, 2, 3, 4})}
_OVERLAP_SCHEMAS = [frozenset({0})]

# Full FD set, ordered so each prefix gives a larger Õ
_ALL_FDS: list[tuple[frozenset, int]] = [
    (frozenset({0}), 1),
    (frozenset({0}), 2),
    (frozenset({0}), 3),
    (frozenset({0}), 4),
]

# Query: constant 0 on attr_0 in V1={0,2,3,4}; footprint = {0,2,3,4}
QUERY = BooleanCQ(atoms=[Atom(view_id=1, pattern=[0, "x2", "x3", "x4"])])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rho_levels(n_fds_total: int) -> list[tuple[float, list]]:
    """Return [(rho, prefix_fds), ...] for all k=0..n_fds_total."""
    return [(k / n_fds_total, list(_ALL_FDS[:k])) for k in range(n_fds_total + 1)]


def _make_config(fds) -> Config:
    return Config(
        n_attrs=_N_ATTRS,
        domain_size=_DOMAIN,
        fds=list(fds),
        view_schemas=_VIEW_SCHEMAS,
        overlap_schemas=_OVERLAP_SCHEMAS,
    )


def _run_rho_trial(
    *,
    full_bench: SyntheticBenchmark,
    rho: float,
    predictor_fds: list,
    arch: str,
    train_cfg: TrainConfig,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_tuples: int,
    hidden_dim: int,
) -> dict:
    """
    Generate worlds with full_bench (full Σ), extract features using predictor_fds,
    build a model that "knows" only predictor_fds, train, and evaluate.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    N = n_train + n_val + n_test

    # --- World generation uses the full FD bench ---
    worlds = full_bench.generate_worlds(N, n_tuples)
    labels = np.array(
        [float(full_bench.evaluate(QUERY, w)) for w in worlds],
        dtype=np.float32,
    )

    # --- Feature extraction uses the reduced-FD bench ---
    feat_config = _make_config(predictor_fds)
    feat_bench = SyntheticBenchmark(feat_config, seed=seed)
    X = np.stack([feat_bench.overlap_feature(w) for w in worlds])

    X_tr, y_tr = X[:n_train], labels[:n_train]
    X_v = X[n_train : n_train + n_val] if n_val > 0 else None
    y_v = labels[n_train : n_train + n_val] if n_val > 0 else None
    X_te, y_te = X[n_train + n_val :], labels[n_train + n_val :]

    adj = build_overlap_graph(_N_ATTRS, _OVERLAP_SCHEMAS, predictor_fds)
    model = build_model(arch, feat_bench, adj, QUERY, hidden_dim=hidden_dim)

    tr = train(model, X_tr, y_tr, X_v, y_v, cfg=train_cfg)
    metrics = evaluate(model, X_te, y_te, device=train_cfg.device)

    aug_ov = augmented_overlap(_OVERLAP_SCHEMAS[0], predictor_fds)

    return {
        "model": arch,
        "rho": round(rho, 4),
        "n_fds_revealed": len(predictor_fds),
        "n_fds_total": len(_ALL_FDS),
        "certified": QUERY.is_certified(feat_config),
        "aug_overlap": sorted(aug_ov),
        "aug_overlap_size": len(aug_ov),
        "feature_dim": feat_bench.feature_dim(),
        "seed": seed,
        "test_accuracy": round(metrics["accuracy"], 6),
        "test_loss": round(metrics["loss"], 6),
        "n_epochs": len(tr.train_loss),
        "stopped_early": tr.stopped_early,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="E1 ablation: FD completeness ρ sweep")
    p.add_argument(
        "--mini",
        action="store_true",
        help="Fast smoke-test: tiny dataset, ρ=0 and ρ=1 only, one seed",
    )
    # Dataset
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-tuples", type=int, default=10)
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

    rho_levels = _rho_levels(len(_ALL_FDS))
    if args.mini:
        # Keep only ρ=0 and ρ=1 to stay fast
        rho_levels = [rho_levels[0], rho_levels[-1]]

    full_config = _make_config(_ALL_FDS)
    full_bench = SyntheticBenchmark(full_config, seed=0)

    train_cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device
    )

    records: list[dict] = []
    total = len(rho_levels) * len(args.architectures) * len(args.seeds)
    done = 0

    for rho, pred_fds in rho_levels:
        cert = QUERY.is_certified(_make_config(pred_fds))
        aug = augmented_overlap(_OVERLAP_SCHEMAS[0], pred_fds)

        for arch in args.architectures:
            for seed in args.seeds:
                rec = _run_rho_trial(
                    full_bench=full_bench,
                    rho=rho,
                    predictor_fds=pred_fds,
                    arch=arch,
                    train_cfg=train_cfg,
                    seed=seed,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    n_test=args.n_test,
                    n_tuples=args.n_tuples,
                    hidden_dim=args.hidden_dim,
                )
                records.append(rec)
                done += 1
                cert_flag = "CERTIFIED ✓" if cert else "not certified"
                print(
                    f"[{done}/{total}] ρ={rho:.2f} |Õ|={len(aug)} "
                    f"({cert_flag}) | {arch:18s} | seed={seed} "
                    f"| acc={rec['test_accuracy']:.3f}"
                )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e1_ablation_{ts}.json"
    save_results(
        records,
        out,
        meta={
            "experiment": "e1_ablation",
            "timestamp": ts,
            "n_attrs": _N_ATTRS,
            "domain_size": _DOMAIN,
            "n_fds_total": len(_ALL_FDS),
            "rho_levels": [r for r, _ in rho_levels],
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

    # Summary
    from collections import defaultdict

    by_rho_arch: dict = defaultdict(list)
    for r in records:
        by_rho_arch[(r["rho"], r["model"])].append(r["test_accuracy"])
    print("\n=== Mean test accuracy by ρ ===")
    prev_arch = None
    for (rho, arch), accs in sorted(by_rho_arch.items()):
        if arch != prev_arch:
            print(f"\n  {arch}")
            prev_arch = arch
        cert = any(
            r["certified"] for r in records if abs(r["rho"] - rho) < 1e-9 and r["model"] == arch
        )
        print(f"    ρ={rho:.2f}  {'✓' if cert else '✗'}  acc={np.mean(accs):.3f}")


if __name__ == "__main__":
    main()

"""
Shared utilities for experiment scripts.

Every experiment follows the same trial pattern:
    model = build_model(name, bench, adj, query)
    result = run_trial(...)      # train + eval on held-out test set
    save_results(records, path)  # JSON output
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from data.synthetic import BooleanCQ, Config
from models.baselines import MajorityVote, VanillaOverlap
from models.closure_aware import ClosureAwarePredictor
from models.gnn_og import gnnog_from_benchmark
from models.mlp import MLP
from models.set_transformer import SetTransformer
from models.train import TrainConfig, evaluate, train


@runtime_checkable
class BenchLike(Protocol):
    """Minimal interface required by build_model (feature extraction + model sizing)."""

    config: Config
    _augmented_overlaps: list[frozenset[int]]

    def feature_dim(self) -> int: ...


@runtime_checkable
class GenerativeBench(BenchLike, Protocol):
    """BenchLike extended with dataset generation, required by run_trial."""

    def make_dataset(self, cq: BooleanCQ, N: int, n_tuples: int) -> tuple[Any, Any]: ...


ARCHITECTURES = [
    "mlp",
    "set_transformer",
    "gnn_og",
    "closure_aware",
    "vanilla_overlap",
    "majority_vote",
]


def build_model(
    name: str,
    bench: BenchLike,
    adj: list[list[int]],
    query: BooleanCQ | None = None,
    hidden_dim: int = 64,
) -> torch.nn.Module:
    d = bench.config.domain_size
    overlap_dims = [d ** len(aug) for aug in bench._augmented_overlaps]
    in_dim = bench.feature_dim()
    # n_heads: largest power of 2 that divides hidden_dim and is ≤ 4
    n_heads = max(1, min(4, hidden_dim // 16))

    if name == "mlp":
        return MLP(in_dim=in_dim, hidden_dim=hidden_dim)
    elif name == "set_transformer":
        return SetTransformer(overlap_dims=overlap_dims, hidden_dim=hidden_dim, n_heads=n_heads)
    elif name == "gnn_og":
        return gnnog_from_benchmark(bench, adj, hidden_dim=hidden_dim)
    elif name == "closure_aware":
        assert query is not None, "closure_aware requires a query"
        base = gnnog_from_benchmark(bench, adj, hidden_dim=hidden_dim)
        return ClosureAwarePredictor.from_query(query, bench.config, base)
    elif name == "vanilla_overlap":
        return VanillaOverlap(in_dim=in_dim)
    elif name == "majority_vote":
        return MajorityVote()
    else:
        raise ValueError(f"Unknown architecture: {name!r}")


def run_trial(
    model_name: str,
    bench: GenerativeBench,
    adj: list[list[int]],
    query: BooleanCQ,
    train_cfg: TrainConfig,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_tuples: int,
    hidden_dim: int = 64,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = bench.make_dataset(query, N=n_train + n_val + n_test, n_tuples=n_tuples)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_v = X[n_train : n_train + n_val] if n_val > 0 else None
    y_v = y[n_train : n_train + n_val] if n_val > 0 else None
    X_te, y_te = X[n_train + n_val :], y[n_train + n_val :]

    model = build_model(model_name, bench, adj, query, hidden_dim=hidden_dim)

    t0 = time.perf_counter()
    tr = train(model, X_tr, y_tr, X_v, y_v, cfg=train_cfg)
    elapsed = time.perf_counter() - t0

    metrics = evaluate(model, X_te, y_te, device=train_cfg.device)

    return {
        "model": model_name,
        "certified": query.is_certified(bench.config),
        "seed": seed,
        "test_accuracy": round(metrics["accuracy"], 6),
        "test_balanced_accuracy": round(metrics["balanced_accuracy"], 6),
        "test_positive_rate": round(metrics["positive_rate"], 6),
        "test_loss": round(metrics["loss"], 6),
        "n_epochs": len(tr.train_loss),
        "stopped_early": tr.stopped_early,
        "train_time_s": round(elapsed, 3),
    }


def _collect_system_info() -> dict[str, Any]:
    """Collect environment provenance required for FAIR-compliant result files."""

    def _git_commit() -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    def _pkg_version(name: str) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    gpu_name: str | None = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": _pkg_version("torch-geometric"),
        "numpy_version": _pkg_version("numpy"),
        "scipy_version": _pkg_version("scipy"),
        "sklearn_version": _pkg_version("scikit-learn"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def save_results(results: list[dict], path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {**(meta or {}), "system": _collect_system_info()}, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(results)} records → {path}")

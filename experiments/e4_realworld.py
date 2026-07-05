"""
E4 Real-world Benchmark — BibInteg + WDC-Product.

Tests the framework on two multi-source real-world datasets where the schema,
FDs, and overlap structure are derived from domain knowledge (not generated).

For each dataset we run all 6 architectures across multiple seeds and report:
  - Test accuracy on certified queries  (CA should ≈ match best learned)
  - Test accuracy on non-certified queries  (all should ≈ 0.5 by Theorem 5)
  - Certification status per query

Since downloading the actual data requires institutional access, the experiment
has a --mock flag (default: True if data files are absent) that substitutes
make_mock_dataset() from each loader.  The experimental logic is identical;
mock mode simply tests that the code runs end-to-end.

Schemas
-------
BibInteg  — 7 attrs, 3 views (DBLP, ACM, SemanticScholar), 4 FDs
  Certified queries: Q_VENUE, Q_DOI, Q_LARGE_TEAM
WDC-Product — 8 attrs, 3 views (Amazon, Walmart, BestBuy), 3 FDs
  Certified queries:    Q_AVAILABLE, Q_CHEAP
  Non-certified:        Q_HIGHLY_RATED, Q_REVIEWED, Q_POPULAR

Datasets
--------
  wdc           — single-tuple worlds (m=1, one record per world)
  wdc_cluster   — cluster worlds (m=3, all three retailer records per product)
  bibinteg      — single-tuple worlds (m=1)

For wdc_cluster, --n-train/val/test refer to number of cluster worlds (not records).
Recommended: --n-train 1000 --n-val 100 --n-test 300

Quick test (< 30 s on CPU, mock data):
    python -m experiments.e4_realworld --mock --mini --output-dir results

Full run (needs data in data/raw/):
    python -m experiments.e4_realworld \\
        --n-train 5000 --n-val 500 --n-test 1000 \\
        --epochs 300 --patience 30 --hidden-dim 128 \\
        --seeds 0 1 2 --device cuda

Cluster-world run:
    python -m experiments.e4_realworld --datasets wdc_cluster \\
        --n-train 1000 --n-val 100 --n-test 300 \\
        --epochs 300 --patience 30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np

import data.amazon_google as ag
import data.bibinteg as bib
import data.crosskg_dblp as ckg
import data.fodors_zagat as fz
import data.wdc as wdc
import data.wikischolar as wiki
from data.synthetic import BooleanCQ, Config
from data.utils import augmented_overlap, build_overlap_graph
from experiments.runner import ARCHITECTURES, build_model, save_results
from models.train import TrainConfig, evaluate, train


class _DatasetDesc(TypedDict):
    config: Config
    queries: list[BooleanCQ]
    q_names: list[str]
    mock_fn: Callable[..., Any]
    load_fn: Callable[..., Any]
    to_worlds: Callable[..., Any]
    data_dir: Path
    source_col: "int | None"  # if set: filter world rows by source_id when evaluating atoms
    records_per_world: int  # used to scale mock generation (1 for single-tuple, 3 for cluster)


# ---------------------------------------------------------------------------
# Dataset descriptors
# ---------------------------------------------------------------------------

_DATASETS: dict[str, _DatasetDesc] = {
    "bibinteg": {
        "config": bib.CONFIG,
        "queries": [bib.Q_VENUE, bib.Q_DOI, bib.Q_LARGE_TEAM],
        "q_names": ["Q_venue", "Q_doi", "Q_large_team"],
        "mock_fn": bib.make_mock_dataset,
        "load_fn": bib.load_dataset,
        "to_worlds": bib.records_to_worlds,
        "data_dir": Path("data/raw/bibinteg"),
        "source_col": None,
        "records_per_world": 1,
    },
    "wdc": {
        "config": wdc.CONFIG,
        "queries": [
            wdc.Q_AVAILABLE,
            wdc.Q_CHEAP,
            wdc.Q_HIGHLY_RATED,
            wdc.Q_REVIEWED,
            wdc.Q_POPULAR,
        ],
        "q_names": ["Q_available", "Q_cheap", "Q_highly_rated", "Q_reviewed", "Q_popular"],
        "mock_fn": wdc.make_mock_dataset,
        "load_fn": wdc.load_dataset,
        "to_worlds": wdc.records_to_worlds,
        "data_dir": Path("data/raw/wdc"),
        "source_col": None,
        "records_per_world": 1,
    },
    "wdc_cluster": {
        "config": wdc.CONFIG,
        "queries": [
            wdc.Q_AVAILABLE,
            wdc.Q_CHEAP,
            wdc.Q_HIGHLY_RATED,
            wdc.Q_REVIEWED,
            wdc.Q_POPULAR,
        ],
        "q_names": ["Q_available", "Q_cheap", "Q_highly_rated", "Q_reviewed", "Q_popular"],
        "mock_fn": wdc.make_mock_dataset,
        "load_fn": wdc.load_dataset,
        "to_worlds": wdc.records_to_cluster_worlds,
        "data_dir": Path("data/raw/wdc"),
        "source_col": 6,  # attr 6 = source_id (0=Amazon,1=Walmart,2=BestBuy)
        "records_per_world": 3,  # each cluster world has 3 rows
    },
    "wikischolar": {
        "config": wiki.CONFIG,
        "queries": [wiki.Q_VENUE, wiki.Q_SUBJECT, wiki.Q_HIGHLY_CITED],
        "q_names": ["Q_venue", "Q_subject", "Q_highly_cited"],
        "mock_fn": wiki.make_mock_dataset,
        "load_fn": wiki.load_dataset,
        "to_worlds": wiki.records_to_worlds,
        "data_dir": Path("data/raw/wikischolar"),
        "source_col": None,
        "records_per_world": 1,
    },
    # Cross-source matched-pair datasets (one record per source per matched pair).
    # The overlap is the matched-pair identity (a near-unique id quantized into
    # feature_domain bins), so the certified query reads its answer from an overlap
    # bit (→ ~1.0) while the uncertified query's contested attribute is outside the
    # overlap and unseen at test time (→ ~0.5 error floor).
    "crosskg_dblp": {
        "config": ckg.CONFIG,
        "queries": [ckg.Q_PUBLISHER, ckg.Q_LARGE_TEAM],
        "q_names": ["Q_publisher", "Q_large_team"],
        "mock_fn": ckg.make_mock_dataset,
        "load_fn": ckg.load_dataset,
        "to_worlds": ckg.records_to_worlds,
        "data_dir": Path("data/raw/crosskg_dblp"),
        "source_col": None,
        "records_per_world": 1,
    },
    "amazon_google": {
        "config": ag.CONFIG,
        "queries": [ag.Q_CATALOG, ag.Q_EXPENSIVE],
        "q_names": ["Q_catalog", "Q_expensive"],
        "mock_fn": ag.make_mock_dataset,
        "load_fn": ag.load_dataset,
        "to_worlds": ag.records_to_worlds,
        "data_dir": Path("data/raw/amazon_google"),
        "source_col": None,
        "records_per_world": 1,
    },
    "fodors_zagat": {
        "config": fz.CONFIG,
        "queries": [fz.Q_SEGMENT, fz.Q_CUISINE],
        "q_names": ["Q_segment", "Q_cuisine"],
        "mock_fn": fz.make_mock_dataset,
        "load_fn": fz.load_dataset,
        "to_worlds": fz.records_to_worlds,
        "data_dir": Path("data/raw/fodors_zagat"),
        "source_col": None,
        "records_per_world": 1,
    },
}


# ---------------------------------------------------------------------------
# Feature extraction from real records
# ---------------------------------------------------------------------------
# Real-world schemas often have large augmented overlaps (|Õ| = 5..8 attrs),
# making joint-distribution features (domain^|Õ| entries) impractically large.
# We therefore use per-attribute marginals: for each attr in Õ, compute the
# empirical frequency vector of size `feature_domain`, then concatenate.
# Total feature dim = feature_domain × |Õ|.  This remains tractable for any
# overlap size and still carries the information needed by all architectures.
# ---------------------------------------------------------------------------


class _RealWorldBench:
    """
    Thin benchmark wrapper for real-world data.
    Exposes the interface expected by build_model / run_trial without
    allocating joint-distribution feature tables.

    Features are per-attribute marginals: for each attribute a in Õ, a
    feature_domain-dim histogram.  Total dim = feature_domain × |Õ|.

    _augmented_overlaps is stored as a list of single-element frozensets
    (one per Õ attribute) so that runner.build_model computes
    overlap_dims = [d^1, d^1, ...] = [d, d, ...] — consistent with the
    per-attribute feature layout.
    """

    def __init__(
        self,
        config: Config,
        feature_domain: int,
        seed: int = 0,
        source_col: "int | None" = None,
    ):
        self.config = Config(
            n_attrs=config.n_attrs,
            domain_size=feature_domain,
            fds=config.fds,
            view_schemas=config.view_schemas,
            overlap_schemas=config.overlap_schemas,
        )
        # True augmented overlaps (for query evaluation / adj)
        self._true_augmented: list[frozenset[int]] = [
            augmented_overlap(o, config.fds) for o in config.overlap_schemas
        ]
        # Single-element overlaps: one per Õ attribute, for build_model compat
        self._augmented_overlaps: list[frozenset[int]] = [
            frozenset({a}) for aug in self._true_augmented for a in sorted(aug)
        ]
        self._feature_domain = feature_domain
        self._feat_dim = feature_domain * len(self._augmented_overlaps)
        # For cluster worlds: source_id column used to filter rows per atom's view
        self._source_col = source_col

    def feature_dim(self) -> int:
        return max(self._feat_dim, 1)

    def overlap_feature(self, world: np.ndarray) -> np.ndarray:
        """Per-attribute marginals: a feature_domain-dim histogram per Õ attribute."""
        d = self._feature_domain
        arr = np.asarray(world, dtype=np.int32)
        n = max(len(arr), 1)
        parts: list[np.ndarray] = []
        for single_aug in self._augmented_overlaps:
            a = next(iter(single_aug))
            counts = np.zeros(d, dtype=np.float32)
            vals = arr[:, a] % d
            for v in vals:
                counts[int(v)] += 1.0
            parts.append(counts / n)
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

    def evaluate(self, query: BooleanCQ, world: np.ndarray) -> bool:
        """Evaluate query on a world using the feature-domain config.

        For cluster worlds (source_col set), filters world rows to those from
        the atom's view before evaluating, ensuring view-specific attribute
        values (rating_bucket, n_reviews_log) are read from the correct source.
        """
        from data.synthetic import _atom_assignments, _merge

        if not query.atoms:
            return True

        def _rows_for_atom(atom: Any) -> np.ndarray:
            if self._source_col is not None:
                mask = world[:, self._source_col] == atom.view_id
                return world[mask] if mask.any() else world[:0]
            return world

        assignments = _atom_assignments(
            query.atoms[0], _rows_for_atom(query.atoms[0]), self.config
        )
        for atom in query.atoms[1:]:
            if not assignments:
                return False
            next_assigns = _atom_assignments(atom, _rows_for_atom(atom), self.config)
            merged: list[dict] = []
            for a1 in assignments:
                for a2 in next_assigns:
                    m = _merge(a1, a2)
                    if m is not None:
                        merged.append(m)
                        break
            assignments = merged
        return len(assignments) > 0


def _quantize(worlds, feature_domain: int) -> list[np.ndarray]:
    """Hash-bucket each attribute value into [0, feature_domain) via modulo."""
    return [np.asarray(w, dtype=np.int32) % feature_domain for w in worlds]


def _records_to_features(
    records,
    config: Config,
    to_worlds,
    seed: int,
    feature_domain: int = 16,
    source_col: "int | None" = None,
) -> tuple[np.ndarray, "_RealWorldBench"]:
    """
    Convert raw records into the overlap-feature matrix used by all models.
    Uses per-attribute marginals (size = feature_domain × |Õ|) so the feature
    dimension is bounded regardless of the original domain or overlap size.
    """
    bench = _RealWorldBench(config, feature_domain, seed, source_col=source_col)
    worlds = _quantize(to_worlds(records), feature_domain)
    X = np.stack([bench.overlap_feature(w) for w in worlds])
    return X, bench


def _labels_from_query(
    worlds: list[np.ndarray],
    query: BooleanCQ,
    bench: "_RealWorldBench",
) -> np.ndarray:
    return np.array(
        [float(bench.evaluate(query, w)) for w in worlds],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def _run_trial(
    *,
    dataset_name: str,
    query_name: str,
    query: BooleanCQ,
    config: Config,
    X: np.ndarray,
    y: np.ndarray,
    bench: _RealWorldBench,
    adj: list[list[int]],
    arch: str,
    train_cfg: TrainConfig,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    hidden_dim: int,
    feature_domain: int = 16,
) -> dict:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use the same seed-shuffled split every time for reproducibility
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    total = n_train + n_val + n_test
    if len(X) < total:
        raise ValueError(
            f"Dataset too small: need {total} records, got {len(X)}. "
            "Use --mock for smoke testing or provide larger data."
        )

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_v = X[n_train : n_train + n_val] if n_val > 0 else None
    y_v = y[n_train : n_train + n_val] if n_val > 0 else None
    X_te, y_te = X[n_train + n_val : total], y[n_train + n_val : total]

    model = build_model(arch, bench, adj, query, hidden_dim=hidden_dim)
    tr = train(model, X_tr, y_tr, X_v, y_v, cfg=train_cfg)
    metrics = evaluate(model, X_te, y_te, device=train_cfg.device)

    aug = augmented_overlap(config.overlap_schemas[0], config.fds)

    return {
        "dataset": dataset_name,
        "query": query_name,
        "model": arch,
        "certified": query.is_certified(config),
        "aug_overlap": sorted(aug),
        "aug_overlap_size": len(aug),
        "feature_dim": bench.feature_dim(),
        "feature_domain": feature_domain,
        "world_mode": "cluster" if bench._source_col is not None else "single",
        "seed": seed,
        "test_accuracy": round(metrics["accuracy"], 6),
        "test_balanced_accuracy": round(metrics["balanced_accuracy"], 6),
        "test_positive_rate": round(metrics["positive_rate"], 6),
        "test_loss": round(metrics["loss"], 6),
        "n_epochs": len(tr.train_loss),
        "stopped_early": tr.stopped_early,
        "n_train": n_train,
        "n_test": len(X_te),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="E4 real-world benchmark")
    p.add_argument(
        "--mock", action="store_true", help="Use mock (generated) data instead of real CSVs"
    )
    p.add_argument(
        "--mini",
        action="store_true",
        help="Fast smoke-test: tiny split, one seed, one dataset, one query",
    )
    p.add_argument(
        "--datasets", nargs="+", default=list(_DATASETS.keys()), choices=list(_DATASETS.keys())
    )
    p.add_argument("--architectures", nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    # Dataset sizes
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=500)
    # Model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument(
        "--feature-domain",
        type=int,
        default=16,
        help="Quantize attribute values into this many bins for feature extraction",
    )
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
        args.n_train = 60
        args.n_val = 20
        args.n_test = 30
        args.epochs = 5
        args.patience = 5
        args.seeds = [0]
        args.hidden_dim = 16
        args.mock = True

    train_cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device
    )

    records_out: list[dict] = []

    for ds_name in args.datasets:
        ds = _DATASETS[ds_name]
        config: Config = ds["config"]

        # Load or generate records
        source_col: "int | None" = ds["source_col"]
        records_per_world: int = ds["records_per_world"]

        use_mock = args.mock
        if not use_mock:
            try:
                ds["load_fn"](ds["data_dir"])  # probe — raises if missing
            except FileNotFoundError:
                print(f"[{ds_name}] Data not found at {ds['data_dir']}; falling back to mock.")
                use_mock = True

        if use_mock:
            n_worlds_needed = args.n_train + args.n_val + args.n_test
            n_records_needed = n_worlds_needed * records_per_world
            # +2 ensures WDC-type generators (which snap to multiples of 3)
            # always return at least n_records_needed records.
            raw_records = ds["mock_fn"](n=max(n_records_needed + 2, 300), seed=0)
            print(
                f"[{ds_name}] Using mock dataset ({len(raw_records)} records, "
                f"{len(raw_records) // records_per_world} worlds)"
            )
        else:
            raw_records = ds["load_fn"](ds["data_dir"])
            print(f"[{ds_name}] Loaded {len(raw_records)} records from {ds['data_dir']}")

        worlds = ds["to_worlds"](raw_records)

        adj = build_overlap_graph(config.n_attrs, config.overlap_schemas, config.fds)

        queries = ds["queries"] if not args.mini else [ds["queries"][0]]
        q_names = ds["q_names"] if not args.mini else [ds["q_names"][0]]

        for query, q_name in zip(queries, q_names):
            certified = query.is_certified(config)
            print(f"\n  [{ds_name}] {q_name}  {'CERTIFIED ✓' if certified else 'not certified ✗'}")

            # Build feature matrix + labels once per (dataset, query)
            X_all, bench = _records_to_features(
                raw_records,
                config,
                ds["to_worlds"],
                seed=0,
                feature_domain=args.feature_domain,
                source_col=source_col,
            )
            q_worlds = _quantize(worlds, args.feature_domain)
            y_all = _labels_from_query(q_worlds, query, bench)
            pos_rate = float(y_all.mean())
            print(f"           positive rate = {pos_rate:.3f}")

            total = len(args.architectures) * len(args.seeds)
            done = 0

            for arch in args.architectures:
                for seed in args.seeds:
                    try:
                        rec = _run_trial(
                            dataset_name=ds_name,
                            query_name=q_name,
                            query=query,
                            config=config,
                            X=X_all,
                            y=y_all,
                            bench=bench,
                            adj=adj,
                            arch=arch,
                            train_cfg=train_cfg,
                            seed=seed,
                            n_train=args.n_train,
                            n_val=args.n_val,
                            n_test=args.n_test,
                            hidden_dim=args.hidden_dim,
                            feature_domain=args.feature_domain,
                        )
                        records_out.append(rec)
                        done += 1
                        print(
                            f"    [{done}/{total}] {arch:18s} seed={seed} "
                            f"acc={rec['test_accuracy']:.3f}"
                        )
                    except ValueError as e:
                        print(f"    SKIP {arch} seed={seed}: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"e4_realworld_{ts}.json"
    save_results(
        records_out,
        out,
        meta={
            "experiment": "e4_realworld",
            "timestamp": ts,
            "datasets": args.datasets,
            "architectures": args.architectures,
            "seeds": args.seeds,
            "mock": args.mock or (not bool(records_out)),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "feature_domain": args.feature_domain,
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

    by_ds_q_arch: dict = defaultdict(list)
    for r in records_out:
        by_ds_q_arch[(r["dataset"], r["query"], r["model"])].append(r["test_accuracy"])
    print("\n=== Mean test accuracy ===")
    prev = None
    for (ds, q, arch), accs in sorted(by_ds_q_arch.items()):
        tag = (ds, q)
        if tag != prev:
            cert_flag = next(
                (r["certified"] for r in records_out if r["dataset"] == ds and r["query"] == q),
                False,
            )
            print(f"\n  {ds} / {q}  {'✓' if cert_flag else '✗'}")
            prev = tag
        print(f"    {arch:18s}  acc={np.mean(accs):.3f}")


if __name__ == "__main__":
    main()

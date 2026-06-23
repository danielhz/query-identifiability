"""
Tests for E4 (real-world benchmark) and the analysis script.
All tests use --mini --mock so no data download is required.
"""

import json

import numpy as np
import pytest

import data.bibinteg as bib
import data.wdc as wdc
from analysis.utils import latest_result, load_results
from experiments.e4_realworld import (
    _quantize,
    _RealWorldBench,
    _records_to_features,
)
from experiments.e4_realworld import (
    main as e4_main,
)

# ---------------------------------------------------------------------------
# Unit: _RealWorldBench
# ---------------------------------------------------------------------------


class TestRealWorldBench:
    @pytest.fixture(scope="class")
    def bench(self):
        return _RealWorldBench(bib.CONFIG, feature_domain=8)

    def test_feature_dim(self, bench):
        aug_total = sum(len(a) for a in bench._true_augmented)
        assert bench.feature_dim() == 8 * aug_total

    def test_augmented_overlaps_are_singletons(self, bench):
        for aug in bench._augmented_overlaps:
            assert len(aug) == 1

    def test_overlap_feature_shape(self, bench):
        recs = bib.make_mock_dataset(n=5, seed=0)
        worlds = _quantize(bib.records_to_worlds(recs), 8)
        for w in worlds:
            f = bench.overlap_feature(w)
            assert f.shape == (bench.feature_dim(),)

    def test_overlap_feature_sums_to_one_per_attr(self, bench):
        recs = bib.make_mock_dataset(n=5, seed=0)
        worlds = _quantize(bib.records_to_worlds(recs), 8)
        d = bench._feature_domain
        for w in worlds:
            f = bench.overlap_feature(w)
            n_attrs = len(bench._augmented_overlaps)
            for i in range(n_attrs):
                chunk = f[i * d : (i + 1) * d]
                assert abs(chunk.sum() - 1.0) < 1e-5, f"attr {i} sum != 1"

    def test_feature_nonnegative(self, bench):
        recs = bib.make_mock_dataset(n=5, seed=0)
        worlds = _quantize(bib.records_to_worlds(recs), 8)
        for w in worlds:
            assert (bench.overlap_feature(w) >= 0).all()

    def test_overlap_feature_shape_wdc(self):
        bench_wdc = _RealWorldBench(wdc.CONFIG, feature_domain=8)
        recs = wdc.make_mock_dataset(n=5, seed=0)
        worlds = _quantize(wdc.records_to_worlds(recs), 8)
        for w in worlds:
            f = bench_wdc.overlap_feature(w)
            assert f.shape == (bench_wdc.feature_dim(),)


class TestQuantize:
    def test_values_in_range(self):
        recs = bib.make_mock_dataset(n=20, seed=0)
        worlds = _quantize(bib.records_to_worlds(recs), 16)
        for w in worlds:
            assert (w >= 0).all()
            assert (w < 16).all()

    def test_shape_preserved(self):
        recs = bib.make_mock_dataset(n=5, seed=0)
        worlds = bib.records_to_worlds(recs)
        qworlds = _quantize(worlds, 16)
        assert len(qworlds) == len(worlds)
        for q, o in zip(qworlds, worlds):
            assert q.shape == np.asarray(o).shape


class TestRecordsToFeatures:
    def test_shape(self):
        recs = bib.make_mock_dataset(n=30, seed=0)
        X, bench = _records_to_features(
            recs, bib.CONFIG, bib.records_to_worlds, seed=0, feature_domain=8
        )
        assert X.shape == (30, bench.feature_dim())

    def test_finite(self):
        recs = bib.make_mock_dataset(n=30, seed=0)
        X, _ = _records_to_features(
            recs, bib.CONFIG, bib.records_to_worlds, seed=0, feature_domain=8
        )
        assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# End-to-end: E4 produces valid JSON
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e4_results(tmp_path_factory) -> tuple[list[dict], dict]:
    d = tmp_path_factory.mktemp("e4")
    e4_main(["--mini", "--output-dir", str(d)])
    src = latest_result(d, "e4_realworld")
    return load_results(src)


class TestE4Output:
    def test_json_written(self, tmp_path):
        e4_main(["--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e4_realworld_*.json"))
        assert len(files) == 1

    def test_meta_fields(self, e4_results):
        _, meta = e4_results
        assert meta["experiment"] == "e4_realworld"
        assert "datasets" in meta
        assert meta["mock"] is True

    def test_result_keys(self, e4_results):
        records, _ = e4_results
        required = {
            "dataset",
            "query",
            "model",
            "certified",
            "aug_overlap_size",
            "feature_dim",
            "seed",
            "test_accuracy",
        }
        for r in records:
            assert required <= r.keys()

    def test_certified_query_present(self, e4_results):
        records, _ = e4_results
        assert any(r["certified"] for r in records)

    def test_accuracy_in_range(self, e4_results):
        records, _ = e4_results
        for r in records:
            assert 0.0 <= r["test_accuracy"] <= 1.0

    def test_all_architectures_run(self, e4_results):
        records, _ = e4_results
        from experiments.runner import ARCHITECTURES

        models = {r["model"] for r in records}
        assert models == set(ARCHITECTURES)

    def test_feature_domain_recorded(self, e4_results):
        records, _ = e4_results
        for r in records:
            assert "feature_domain" in r

    def test_single_dataset_flag(self, tmp_path):
        e4_main(["--mini", "--datasets", "wdc", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e4_realworld_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        datasets = {r["dataset"] for r in data["results"]}
        assert datasets == {"wdc"} or len(datasets) == 0  # wdc mini skips if too small

    def test_arch_subset_flag(self, tmp_path):
        e4_main(
            ["--mini", "--architectures", "mlp", "majority_vote", "--output-dir", str(tmp_path)]
        )
        data = json.loads(list(tmp_path.glob("e4_realworld_*.json"))[0].read_text())
        models = {r["model"] for r in data["results"]}
        assert models <= {"mlp", "majority_vote"}

    def test_world_mode_field_present(self, e4_results):
        records, _ = e4_results
        for r in records:
            assert "world_mode" in r
            assert r["world_mode"] in ("single", "cluster")


# ---------------------------------------------------------------------------
# Cluster-world mode (wdc_cluster)
# ---------------------------------------------------------------------------


class TestClusterWorldBench:
    @pytest.fixture(scope="class")
    def bench_cluster(self):
        from experiments.e4_realworld import _RealWorldBench

        return _RealWorldBench(wdc.CONFIG, feature_domain=8, source_col=6)

    def test_source_col_stored(self, bench_cluster):
        assert bench_cluster._source_col == 6

    def test_feature_dim_same_as_single(self):
        bench_single = _RealWorldBench(wdc.CONFIG, feature_domain=8)
        bench_cluster = _RealWorldBench(wdc.CONFIG, feature_domain=8, source_col=6)
        assert bench_single.feature_dim() == bench_cluster.feature_dim()

    def test_evaluate_certified_query_cluster_world(self, bench_cluster):
        """Q_AVAILABLE on a cluster world: result should match the Walmart row's in_stock."""
        import numpy as np

        # Build a cluster world: brand=1, model=1, category=0, price_bucket=3, in_stock=1
        # Amazon row (source_id=0), Walmart row (source_id=1), BestBuy row (source_id=2)
        # in_stock is FD-determined so all rows share in_stock=1
        amazon_row = [1, 1, 0, 3, 4, 5, 0, 1]
        walmart_row = [1, 1, 0, 3, 2, 3, 1, 1]
        bestbuy_row = [1, 1, 0, 3, 0, 7, 2, 1]
        world = np.array([amazon_row, walmart_row, bestbuy_row], dtype=np.int32) % 8
        result = bench_cluster.evaluate(wdc.Q_AVAILABLE, world)
        assert result is True

    def test_evaluate_uncertified_query_filters_by_source(self, bench_cluster):
        """Q_HIGHLY_RATED (view_id=0, Amazon) only checks the Amazon row."""
        import numpy as np

        # Amazon row: rating_bucket=0 (does NOT satisfy Q_HIGHLY_RATED ≥4, pattern=4)
        # Walmart/BestBuy rows: col 4 = 4 (would match if not filtered)
        amazon_row = [1, 1, 0, 3, 0, 5, 0, 1]  # col 4 = 0, source_id = 0
        walmart_row = [1, 1, 0, 3, 4, 3, 1, 1]  # col 4 = 4, source_id = 1 (should be ignored)
        bestbuy_row = [1, 1, 0, 3, 4, 7, 2, 1]  # col 4 = 4, source_id = 2 (should be ignored)
        world = np.array([amazon_row, walmart_row, bestbuy_row], dtype=np.int32) % 8
        result = bench_cluster.evaluate(wdc.Q_HIGHLY_RATED, world)
        # With source filtering: only checks Amazon row, col 4 = 0 ≠ 4 → False
        assert result is False

    def test_evaluate_uncertified_query_amazon_positive(self, bench_cluster):
        """Q_HIGHLY_RATED returns True when Amazon row has rating_bucket=4."""
        import numpy as np

        amazon_row = [1, 1, 0, 3, 4, 5, 0, 1]  # col 4 = 4, source_id = 0 → match
        walmart_row = [1, 1, 0, 3, 0, 3, 1, 1]
        bestbuy_row = [1, 1, 0, 3, 0, 7, 2, 1]
        world = np.array([amazon_row, walmart_row, bestbuy_row], dtype=np.int32) % 8
        result = bench_cluster.evaluate(wdc.Q_HIGHLY_RATED, world)
        assert result is True

    def test_evaluate_popular_join_query(self, bench_cluster):
        """Q_POPULAR requires both Amazon rating_bucket=4 and BestBuy n_reviews_log=5."""
        import numpy as np

        # pattern for Q_POPULAR atom 0: rating_bucket = 4 (constant 4 in attr pos)
        # pattern for Q_POPULAR atom 1: n_reviews_log = 5 (constant 5 in attr pos)
        # Amazon attr schema: {0,1,2,3,4} → col 4 = rating_bucket; pattern constant = 4
        # BestBuy attr schema: {0,1,2,5,6} → col 5 = n_reviews_log; pattern constant = 5
        amazon_row = [1, 1, 0, 3, 4, 0, 0, 1]  # rating_bucket=4 ✓
        walmart_row = [1, 1, 0, 3, 0, 0, 1, 1]
        bestbuy_row = [1, 1, 0, 3, 0, 5, 2, 1]  # n_reviews_log=5 ✓
        world = np.array([amazon_row, walmart_row, bestbuy_row], dtype=np.int32) % 8
        result = bench_cluster.evaluate(wdc.Q_POPULAR, world)
        assert result is True

    def test_evaluate_popular_join_query_negative(self, bench_cluster):
        """Q_POPULAR is False when Amazon rating_bucket ≠ 4."""
        import numpy as np

        amazon_row = [1, 1, 0, 3, 2, 0, 0, 1]  # rating_bucket=2 ✗
        walmart_row = [1, 1, 0, 3, 0, 0, 1, 1]
        bestbuy_row = [1, 1, 0, 3, 0, 5, 2, 1]  # n_reviews_log=5 ✓
        world = np.array([amazon_row, walmart_row, bestbuy_row], dtype=np.int32) % 8
        result = bench_cluster.evaluate(wdc.Q_POPULAR, world)
        assert result is False


class TestE4ClusterMode:
    def test_wdc_cluster_mini_runs(self, tmp_path):
        e4_main(
            [
                "--mini",
                "--datasets",
                "wdc_cluster",
                "--output-dir",
                str(tmp_path),
            ]
        )
        files = list(tmp_path.glob("e4_realworld_*.json"))
        assert len(files) == 1

    def test_wdc_cluster_world_mode_field(self, tmp_path):
        e4_main(
            [
                "--mini",
                "--datasets",
                "wdc_cluster",
                "--architectures",
                "mlp",
                "--output-dir",
                str(tmp_path),
            ]
        )
        data = json.loads(list(tmp_path.glob("e4_realworld_*.json"))[0].read_text())
        for r in data["results"]:
            assert r["world_mode"] == "cluster"

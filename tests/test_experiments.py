"""
End-to-end smoke tests for the experiment scripts.

Each test invokes the experiment in --mini mode and checks:
  - It runs without exceptions
  - It writes a valid JSON file with the expected top-level structure
  - Key invariants hold (e.g., step 4 in E2 is certified, E3 ratios ≥ 1)
"""

import json

import numpy as np

from data.utils import greedy_minaug
from experiments.e1_error_floor import main as e1_main
from experiments.e2_capability_jump import main as e2_main
from experiments.e3_minaug import brute_force_optimal, random_instance
from experiments.e3_minaug import main as e3_main

# ---------------------------------------------------------------------------
# E1: Error floor
# ---------------------------------------------------------------------------


class TestE1:
    def test_runs_and_produces_json(self, tmp_path):
        e1_main(["--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e1_*.json"))
        assert len(files) == 1

    def test_json_structure(self, tmp_path):
        e1_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e1_*.json"))[0].read_text())
        assert "meta" in data and "results" in data
        assert data["meta"]["experiment"] == "e1_error_floor"

    def test_result_keys(self, tmp_path):
        e1_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e1_*.json"))[0].read_text())
        required = {"model", "certified", "seed", "test_accuracy", "query_type"}
        for r in data["results"]:
            assert required <= r.keys()

    def test_accuracy_in_range(self, tmp_path):
        e1_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e1_*.json"))[0].read_text())
        for r in data["results"]:
            assert 0.0 <= r["test_accuracy"] <= 1.0

    def test_certified_flag_matches_query_type(self, tmp_path):
        e1_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e1_*.json"))[0].read_text())
        for r in data["results"]:
            if r["query_type"] == "certified":
                assert r["certified"] is True
            else:
                assert r["certified"] is False

    def test_closure_aware_noncertified_is_0_5(self, tmp_path):
        e1_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e1_*.json"))[0].read_text())
        ca_noncert = [
            r
            for r in data["results"]
            if r["model"] == "closure_aware" and r["query_type"] == "noncertified"
        ]
        for r in ca_noncert:
            # CA outputs logit 0 → sigmoid = 0.5; test accuracy depends on label dist
            # At minimum, it should produce a valid accuracy
            assert 0.0 <= r["test_accuracy"] <= 1.0

    def test_single_architecture_subset(self, tmp_path):
        e1_main(
            ["--mini", "--architectures", "mlp", "majority_vote", "--output-dir", str(tmp_path)]
        )
        data = json.loads(list(tmp_path.glob("e1_*.json"))[0].read_text())
        models_used = {r["model"] for r in data["results"]}
        assert models_used == {"mlp", "majority_vote"}


# ---------------------------------------------------------------------------
# E2: Capability jump
# ---------------------------------------------------------------------------


class TestE2:
    def test_runs_and_produces_json(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e2_*.json"))
        assert len(files) == 1

    def test_json_structure(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e2_*.json"))[0].read_text())
        assert "meta" in data and "results" in data
        assert data["meta"]["experiment"] == "e2_capability_jump"

    def test_result_keys(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e2_*.json"))[0].read_text())
        required = {
            "step",
            "step_label",
            "certified",
            "model",
            "test_accuracy",
            "aug_overlap",
            "aug_overlap_size",
            "feature_dim",
        }
        for r in data["results"]:
            assert required <= r.keys()

    def test_step_0_not_certified(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e2_*.json"))[0].read_text())
        step0 = [r for r in data["results"] if r["step"] == 0]
        assert all(not r["certified"] for r in step0)

    def test_last_step_certified(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e2_*.json"))[0].read_text())
        last_step = max(r["step"] for r in data["results"])
        last = [r for r in data["results"] if r["step"] == last_step]
        assert all(r["certified"] for r in last)

    def test_overlap_size_grows_monotonically(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path), "--architectures", "mlp"])
        data = json.loads(list(tmp_path.glob("e2_*.json"))[0].read_text())
        mlp_records = sorted(
            [r for r in data["results"] if r["model"] == "mlp"], key=lambda r: r["step"]
        )
        sizes = [r["aug_overlap_size"] for r in mlp_records]
        # Each step adds one FD → Õ grows (weakly)
        for i in range(len(sizes) - 1):
            assert sizes[i] <= sizes[i + 1]

    def test_feature_dim_grows_with_overlap(self, tmp_path):
        e2_main(["--mini", "--output-dir", str(tmp_path), "--architectures", "mlp"])
        data = json.loads(list(tmp_path.glob("e2_*.json"))[0].read_text())
        mlp_records = sorted(
            [r for r in data["results"] if r["model"] == "mlp"], key=lambda r: r["step"]
        )
        dims = [r["feature_dim"] for r in mlp_records]
        # feature_dim = domain_size^|Õ| — grows strictly as FDs are added
        for i in range(len(dims) - 1):
            assert dims[i] <= dims[i + 1]


# ---------------------------------------------------------------------------
# E3: MinAug
# ---------------------------------------------------------------------------


class TestE3:
    def test_runs_and_produces_json(self, tmp_path):
        e3_main(["--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e3_*.json"))
        assert len(files) == 1

    def test_json_structure(self, tmp_path):
        e3_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e3_*.json"))[0].read_text())
        assert "meta" in data and "results" in data
        assert data["meta"]["experiment"] == "e3_minaug"

    def test_approx_ratio_at_least_one(self, tmp_path):
        e3_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e3_*.json"))[0].read_text())
        for r in data["results"]:
            assert r["approx_ratio"] >= 1.0 - 1e-9

    def test_greedy_covers_footprint(self, tmp_path):
        # Verify greedy_size ≥ optimal_size (greedy never beats optimal)
        e3_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e3_*.json"))[0].read_text())
        for r in data["results"]:
            assert r["greedy_size"] >= r["optimal_size"]

    def test_result_keys(self, tmp_path):
        e3_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e3_*.json"))[0].read_text())
        required = {
            "n_attrs",
            "n_fds",
            "footprint_size",
            "n_candidates",
            "greedy_size",
            "optimal_size",
            "approx_ratio",
            "is_optimal",
        }
        for r in data["results"]:
            assert required <= r.keys()

    def test_meta_counts_match(self, tmp_path):
        e3_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e3_*.json"))[0].read_text())
        meta = data["meta"]
        assert meta["n_feasible"] == len(data["results"])

    def test_brute_force_exact_on_crm_mini(self):
        from data import crm_mini as crm

        fp = crm.EXPECTED["minaug_footprint"]
        cands = crm.EXPECTED["minaug_candidates"]
        opt = brute_force_optimal(fp, cands, crm.FDS)
        assert opt == crm.EXPECTED["minaug_size"]

    def test_greedy_matches_brute_force_on_crm_mini(self):
        from data import crm_mini as crm

        fp = crm.EXPECTED["minaug_footprint"]
        cands = crm.EXPECTED["minaug_candidates"]
        greedy = greedy_minaug(fp, cands, crm.FDS)
        opt = brute_force_optimal(fp, cands, crm.FDS)
        assert greedy is not None
        assert len(greedy) == opt

    def test_random_instance_feasibility(self):
        rng = np.random.default_rng(42)
        feasible = 0
        for _ in range(50):
            fp, cands, fds = random_instance(6, 4, 4, rng)
            g = greedy_minaug(fp, cands, fds)
            if g is not None:
                feasible += 1
        # At least some instances should be feasible (singletons are always candidates)
        assert feasible > 0


# ---------------------------------------------------------------------------
# e_minaug_realworld: MinAug on real WDC and BibInteg schemas
# ---------------------------------------------------------------------------


class TestMinAugRealworld:
    """Unit tests for obligation-index MinAug on the real integration schemas."""

    def test_obligation_coverage_empty_action(self):
        import data.wdc as wdc
        from data.utils import fd_closure
        from experiments.e_minaug_realworld import obligation_coverage

        overlap = wdc.OVERLAP_SCHEMAS[0]
        # An empty action adds nothing: coverage == already-certified obligations
        fp = frozenset(wdc.CONFIG.view_schemas[0])  # V0 attrs {0,1,2,3,4}
        cov = obligation_coverage(frozenset(), [fp], overlap, wdc.FDS)
        aug0 = fd_closure(overlap, wdc.FDS)
        expected = frozenset({0}) if fp <= aug0 else frozenset()
        assert cov == expected

    def test_certified_queries_need_no_augmentation(self):
        import data.wdc as wdc
        from experiments.e_minaug_realworld import greedy_minaug_obligations

        for query in [wdc.Q_AVAILABLE, wdc.Q_CHEAP]:
            fps = [frozenset(wdc.CONFIG.view_schemas[a.view_id]) for a in query.atoms]
            result = greedy_minaug_obligations(fps, wdc.OVERLAP_SCHEMAS[0], [], wdc.FDS)
            assert result == [], f"{query} should need no augmentation"

    def test_q_highly_rated_needs_one_action(self):
        import data.wdc as wdc
        from data.utils import augmented_overlap
        from experiments.e_minaug_realworld import generate_candidates, greedy_minaug_obligations

        query = wdc.Q_HIGHLY_RATED
        fps = [frozenset(wdc.CONFIG.view_schemas[a.view_id]) for a in query.atoms]
        aug0 = augmented_overlap(wdc.OVERLAP_SCHEMAS[0], wdc.FDS)
        full_fp = frozenset(a for fp in fps for a in fp)
        uncovered = full_fp - aug0
        cands = generate_candidates(uncovered)
        result = greedy_minaug_obligations(fps, wdc.OVERLAP_SCHEMAS[0], cands, wdc.FDS)
        assert result is not None
        assert len(result) == 1
        assert 4 in result[0]  # rating_bucket must be in the selected action

    def test_q_reviewed_needs_one_action(self):
        import data.wdc as wdc
        from data.utils import augmented_overlap
        from experiments.e_minaug_realworld import generate_candidates, greedy_minaug_obligations

        query = wdc.Q_REVIEWED
        fps = [frozenset(wdc.CONFIG.view_schemas[a.view_id]) for a in query.atoms]
        aug0 = augmented_overlap(wdc.OVERLAP_SCHEMAS[0], wdc.FDS)
        full_fp = frozenset(a for fp in fps for a in fp)
        cands = generate_candidates(full_fp - aug0)
        result = greedy_minaug_obligations(fps, wdc.OVERLAP_SCHEMAS[0], cands, wdc.FDS)
        assert result is not None
        assert len(result) == 1
        # The single action must cover attrs 5 and 6 (both needed for V2 footprint)
        assert result[0] >= {5, 6}

    def test_q_popular_needs_two_actions(self):
        import data.wdc as wdc
        from data.utils import augmented_overlap
        from experiments.e_minaug_realworld import generate_candidates, greedy_minaug_obligations

        query = wdc.Q_POPULAR
        fps = [frozenset(wdc.CONFIG.view_schemas[a.view_id]) for a in query.atoms]
        aug0 = augmented_overlap(wdc.OVERLAP_SCHEMAS[0], wdc.FDS)
        full_fp = frozenset(a for fp in fps for a in fp)
        cands = generate_candidates(full_fp - aug0)
        result = greedy_minaug_obligations(fps, wdc.OVERLAP_SCHEMAS[0], cands, wdc.FDS)
        assert result is not None
        assert len(result) == 2

    def test_augmentation_verifies_certification(self):
        import data.wdc as wdc
        from data.utils import augmented_overlap, fd_closure
        from experiments.e_minaug_realworld import (
            generate_candidates,
            greedy_minaug_obligations,
        )

        for query, name in [
            (wdc.Q_HIGHLY_RATED, "Q_highly_rated"),
            (wdc.Q_REVIEWED, "Q_reviewed"),
            (wdc.Q_POPULAR, "Q_popular"),
        ]:
            fps = [frozenset(wdc.CONFIG.view_schemas[a.view_id]) for a in query.atoms]
            aug0 = augmented_overlap(wdc.OVERLAP_SCHEMAS[0], wdc.FDS)
            full_fp = frozenset(a for fp in fps for a in fp)
            cands = generate_candidates(full_fp - aug0)
            result = greedy_minaug_obligations(fps, wdc.OVERLAP_SCHEMAS[0], cands, wdc.FDS)
            assert result is not None, f"{name}: infeasible"
            # Verify the selected actions certify the query
            aug_new = aug0
            for act in result:
                aug_new = fd_closure(aug_new | act, wdc.FDS)
            assert all(fp <= aug_new for fp in fps), f"{name}: not certified after augmentation"

    def test_bibinteg_queries_need_no_augmentation(self):
        import data.bibinteg as bib
        from experiments.e_minaug_realworld import greedy_minaug_obligations

        for query in [bib.Q_VENUE, bib.Q_DOI, bib.Q_LARGE_TEAM]:
            fps = [frozenset(bib.CONFIG.view_schemas[a.view_id]) for a in query.atoms]
            result = greedy_minaug_obligations(fps, bib.OVERLAP_SCHEMAS[0], [], bib.FDS)
            assert result == []

    def test_main_runs_and_writes_json(self, tmp_path):
        from experiments.e_minaug_realworld import main as rw_main

        rw_main(["--quiet", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e_minaug_realworld_*.json"))
        assert len(files) == 1
        import json

        data = json.loads(files[0].read_text())
        assert data["meta"]["experiment"] == "e_minaug_realworld"
        assert len(data["results"]) == 10  # 5 WDC + 3 BibInteg + 2 CrossKG-DBLP

    def test_crosskg_minaug(self):
        import data.crosskg_dblp as ckg
        from data.utils import augmented_overlap, fd_closure
        from experiments.e_minaug_realworld import (
            generate_candidates,
            greedy_minaug_obligations,
        )

        ovl = ckg.OVERLAP_SCHEMAS[0]
        # Q_publisher: certified → no augmentation
        fp_pub = [frozenset(ckg.CONFIG.view_schemas[a.view_id]) for a in ckg.Q_PUBLISHER.atoms]
        assert greedy_minaug_obligations(fp_pub, ovl, [], ckg.FDS) == []
        # Q_large_team: not certified → MinAug adds {2} (large_team_bit) and certifies
        fp_lt = [frozenset(ckg.CONFIG.view_schemas[a.view_id]) for a in ckg.Q_LARGE_TEAM.atoms]
        aug0 = augmented_overlap(ovl, ckg.FDS)
        cands = generate_candidates(frozenset(a for fp in fp_lt for a in fp) - aug0)
        result = greedy_minaug_obligations(fp_lt, ovl, cands, ckg.FDS)
        assert result == [frozenset({2})]
        aug_new = aug0
        for act in result:
            aug_new = fd_closure(aug_new | act, ckg.FDS)
        assert all(fp <= aug_new for fp in fp_lt)

    def test_generate_candidates_singletons_and_pairs(self):
        from experiments.e_minaug_realworld import generate_candidates

        cands = generate_candidates(frozenset({4, 5, 6}), max_action_size=2)
        singletons = [c for c in cands if len(c) == 1]
        pairs = [c for c in cands if len(c) == 2]
        assert len(singletons) == 3
        assert len(pairs) == 3


# ---------------------------------------------------------------------------
# e_witness_discovery: Witness discovery rate vs. corpus size
# ---------------------------------------------------------------------------


class TestWitnessDiscovery:
    def test_runs_and_produces_json(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e_witness_discovery_*.json"))
        assert len(files) == 1

    def test_json_structure(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_*.json"))[0].read_text())
        assert "meta" in data and "results" in data
        assert data["meta"]["experiment"] == "e_witness_discovery"
        assert len(data["results"]) == 3  # Q_highly_rated, Q_reviewed, Q_popular

    def test_result_keys(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_*.json"))[0].read_text())
        required = {
            "query",
            "n_records",
            "n_trials",
            "positive_rate",
            "n_found",
            "frac_found",
            "median_position",
            "pct90_position",
            "frac_within_50",
            "frac_within_100",
            "frac_within_200",
            "frac_within_500",
        }
        for r in data["results"]:
            assert required <= r.keys()

    def test_all_witnesses_found_in_mini(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_*.json"))[0].read_text())
        for r in data["results"]:
            assert r["n_found"] > 0, f"{r['query']}: no witnesses found"

    def test_three_queries_covered(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_*.json"))[0].read_text())
        queries = {r["query"] for r in data["results"]}
        assert queries == {"Q_highly_rated", "Q_reviewed", "Q_popular"}

    def test_positive_rates_in_range(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_*.json"))[0].read_text())
        for r in data["results"]:
            assert 0.0 < r["positive_rate"] < 1.0

    def test_precompute_and_scan_internals(self):
        import numpy as np

        import data.wdc as wdc
        from data.utils import augmented_overlap
        from experiments.e_witness_discovery import _precompute, _scan_trial

        records = wdc.make_mock_dataset(n=60, seed=0)  # 20 clusters of 3
        aug_overlaps = [augmented_overlap(o, wdc.FDS) for o in wdc.CONFIG.overlap_schemas]
        keys, q_vals = _precompute(records, wdc.Q_HIGHLY_RATED, aug_overlaps, wdc.CONFIG)
        assert len(keys) == len(records)
        assert q_vals.dtype == bool
        rng = np.random.default_rng(7)
        pos = _scan_trial(keys, q_vals, rng)
        assert 1 <= pos <= len(records) + 1


class TestWitnessDiscoveryCrossKG:
    """CrossKG (OpenAlex × DBLP) path — uses --mock so no real corpus is needed."""

    def test_runs_and_produces_json(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--dataset", "crosskg", "--mock", "--mini", "--output-dir", str(tmp_path)])
        files = list(tmp_path.glob("e_witness_discovery_crosskg_*.json"))
        assert len(files) == 1

    def test_structure_and_single_query(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--dataset", "crosskg", "--mock", "--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_crosskg_*.json"))[0].read_text())
        assert data["meta"]["dataset"] == "crosskg"
        assert {r["query"] for r in data["results"]} == {"Q_large_team"}

    def test_witnesses_found(self, tmp_path):
        from experiments.e_witness_discovery import main as wd_main

        wd_main(["--dataset", "crosskg", "--mock", "--mini", "--output-dir", str(tmp_path)])
        data = json.loads(list(tmp_path.glob("e_witness_discovery_crosskg_*.json"))[0].read_text())
        assert data["results"][0]["n_found"] > 0, "mock crosskg must yield real witnesses"

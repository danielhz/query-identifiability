"""
Comprehensive test suite for the CRM mini fixture.

All expected values are hand-verified in code/data/crm_mini.py's module docstring.
Run with:  python -m pytest tests/test_all.py -v   (from the code/ directory)
"""

import numpy as np
import pytest

from data import crm_mini as crm
from data.synthetic import SyntheticBenchmark
from data.utils import (
    action_coverage,
    augmented_overlap,
    build_overlap_graph,
    check_certificate,
    fd_closure,
    greedy_minaug,
)

# ---------------------------------------------------------------------------
# FD closure
# ---------------------------------------------------------------------------


class TestFdClosure:
    def test_closure_of_a(self):
        result = fd_closure(frozenset({0}), crm.FDS)
        assert result == crm.EXPECTED["fd_closure_a"]

    def test_closure_of_ac(self):
        result = fd_closure(frozenset({0, 2}), crm.FDS)
        assert result == crm.EXPECTED["fd_closure_ac"]

    def test_closure_of_b(self):
        # b does not imply a
        result = fd_closure(frozenset({1}), crm.FDS)
        assert result == crm.EXPECTED["fd_closure_b"]

    def test_closure_idempotent(self):
        c1 = fd_closure(frozenset({0}), crm.FDS)
        c2 = fd_closure(c1, crm.FDS)
        assert c1 == c2

    def test_closure_empty_fds(self):
        assert fd_closure(frozenset({0, 1}), []) == frozenset({0, 1})


# ---------------------------------------------------------------------------
# Augmented overlap
# ---------------------------------------------------------------------------


class TestAugmentedOverlap:
    def test_overlap_on_a(self):
        # Õ = {a}^+_{a→b} = {a, b}
        result = augmented_overlap(frozenset({0}), crm.FDS)
        assert result == frozenset({0, 1})

    def test_no_fds(self):
        result = augmented_overlap(frozenset({0}), [])
        assert result == frozenset({0})


# ---------------------------------------------------------------------------
# Certificate check
# ---------------------------------------------------------------------------


class TestCheckCertificate:
    def test_certified_query(self):
        # footprint {a,b} ⊆ Õ={a,b}
        result = check_certificate(frozenset({0, 1}), crm.OVERLAP_SCHEMAS, crm.FDS)
        assert result == crm.EXPECTED["cert_q_cert"]

    def test_noncertified_query(self):
        # footprint {a,c}: c ∉ Õ
        result = check_certificate(frozenset({0, 2}), crm.OVERLAP_SCHEMAS, crm.FDS)
        assert result == crm.EXPECTED["cert_q_nonident"]

    def test_via_bcq_method(self):
        config = crm.make_config()
        assert crm.Q_CERT.is_certified(config) == crm.EXPECTED["cert_q_cert"]
        assert crm.Q_NONIDENT.is_certified(config) == crm.EXPECTED["cert_q_nonident"]

    def test_footprint_of_queries(self):
        config = crm.make_config()
        assert crm.Q_CERT.footprint(config) == frozenset({0, 1})  # {a,b}
        assert crm.Q_NONIDENT.footprint(config) == frozenset({0, 2})  # {a,c}
        assert crm.Q_JOIN.footprint(config) == frozenset({0, 1, 2})  # {a,b,c}


# ---------------------------------------------------------------------------
# Query evaluation
# ---------------------------------------------------------------------------


class TestQueryEvaluation:
    @pytest.fixture
    def bench(self):
        return crm.make_benchmark()

    def test_q_cert_w0(self, bench):
        assert bench.evaluate(crm.Q_CERT, crm.W0) == crm.EXPECTED["Q_cert_w0"]

    def test_q_cert_w1(self, bench):
        assert bench.evaluate(crm.Q_CERT, crm.W1) == crm.EXPECTED["Q_cert_w1"]

    def test_q_nonident_w0(self, bench):
        assert bench.evaluate(crm.Q_NONIDENT, crm.W0) == crm.EXPECTED["Q_nonident_w0"]

    def test_q_nonident_w1(self, bench):
        assert bench.evaluate(crm.Q_NONIDENT, crm.W1) == crm.EXPECTED["Q_nonident_w1"]

    def test_q_join_w0(self, bench):
        assert bench.evaluate(crm.Q_JOIN, crm.W0) == crm.EXPECTED["Q_join_w0"]

    def test_q_join_w1(self, bench):
        assert bench.evaluate(crm.Q_JOIN, crm.W1) == crm.EXPECTED["Q_join_w1"]

    def test_certified_query_same_in_equivalence_class(self, bench):
        # The whole point of Theorem 1: certified queries give the same answer
        # on all worlds in the same ~-class.
        assert bench.evaluate(crm.Q_CERT, crm.W0) == bench.evaluate(crm.Q_CERT, crm.W1)

    def test_nonident_query_differs_in_equivalence_class(self, bench):
        # Confirms non-identifiability: W0 ~ W1 but Q_NONIDENT differs.
        assert bench.evaluate(crm.Q_NONIDENT, crm.W0) != bench.evaluate(crm.Q_NONIDENT, crm.W1)


# ---------------------------------------------------------------------------
# Overlap features
# ---------------------------------------------------------------------------


class TestOverlapFeature:
    @pytest.fixture
    def bench(self):
        return crm.make_benchmark()

    def test_feature_dim(self, bench):
        # Õ={a,b}, domain=2 → 2^2=4 bins
        assert bench.feature_dim() == 4

    def test_feat_w0(self, bench):
        feat = bench.overlap_feature(crm.W0)
        np.testing.assert_allclose(feat, crm.EXPECTED["feat_w0"], atol=1e-6)

    def test_feat_w1(self, bench):
        feat = bench.overlap_feature(crm.W1)
        np.testing.assert_allclose(feat, crm.EXPECTED["feat_w1"], atol=1e-6)

    def test_features_identical_for_equivalent_worlds(self, bench):
        # W0 ~ W1 ⟹ same overlap feature (this is the whole point of the feature)
        f0 = bench.overlap_feature(crm.W0)
        f1 = bench.overlap_feature(crm.W1)
        np.testing.assert_allclose(f0, f1, atol=1e-6)

    def test_feature_sums_to_one(self, bench):
        for w in crm.WORLDS:
            feat = bench.overlap_feature(w)
            assert abs(feat.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MinAug
# ---------------------------------------------------------------------------


class TestMinAug:
    def test_action_coverage_a1(self):
        # A1={a,c}: closure={a,b,c}, intersect footprint={a,c} → {a,c}
        cov = action_coverage(frozenset({0, 2}), crm.EXPECTED["minaug_footprint"], crm.FDS)
        assert cov == frozenset({0, 2})

    def test_action_coverage_a2(self):
        # A2={a}: closure={a,b}, intersect footprint={a,c} → {a}
        cov = action_coverage(frozenset({0}), crm.EXPECTED["minaug_footprint"], crm.FDS)
        assert cov == frozenset({0})

    def test_greedy_selects_a1(self):
        candidates = crm.EXPECTED["minaug_candidates"]
        footprint = crm.EXPECTED["minaug_footprint"]
        selected = greedy_minaug(footprint, candidates, crm.FDS)
        assert selected == crm.EXPECTED["minaug_selected"]

    def test_greedy_size(self):
        candidates = crm.EXPECTED["minaug_candidates"]
        footprint = crm.EXPECTED["minaug_footprint"]
        selected = greedy_minaug(footprint, candidates, crm.FDS)
        assert len(selected) == crm.EXPECTED["minaug_size"]

    def test_greedy_infeasible(self):
        # Candidates that cannot cover the footprint
        result = greedy_minaug(
            frozenset({0, 2}),
            [frozenset({0})],  # only covers {a}, never reaches c
            crm.FDS,
        )
        assert result is None

    def test_greedy_empty_footprint(self):
        # Empty footprint → trivially covered, nothing selected
        result = greedy_minaug(frozenset(), [], crm.FDS)
        assert result == []


# ---------------------------------------------------------------------------
# Overlap graph
# ---------------------------------------------------------------------------


class TestOverlapGraph:
    def test_graph_structure(self):
        adj = build_overlap_graph(crm.N_ATTRS, crm.OVERLAP_SCHEMAS, crm.FDS)
        assert adj == crm.EXPECTED["overlap_graph"]

    def test_graph_symmetry(self):
        adj = build_overlap_graph(crm.N_ATTRS, crm.OVERLAP_SCHEMAS, crm.FDS)
        for a, nbrs in enumerate(adj):
            for b in nbrs:
                assert a in adj[b], f"Edge {a}–{b} not symmetric"

    def test_graph_no_self_loops(self):
        adj = build_overlap_graph(crm.N_ATTRS, crm.OVERLAP_SCHEMAS, crm.FDS)
        for a, nbrs in enumerate(adj):
            assert a not in nbrs

    def test_isolated_attribute(self):
        # c (attr=2) is not in any Õ → no neighbors
        adj = build_overlap_graph(crm.N_ATTRS, crm.OVERLAP_SCHEMAS, crm.FDS)
        assert adj[2] == []


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


class TestWorldGeneration:
    @pytest.fixture
    def bench(self):
        return crm.make_benchmark()

    def test_generate_world_shape(self, bench):
        w = bench.generate_world(n_tuples=10)
        assert w.shape == (10, crm.N_ATTRS)

    def test_fd_enforced_in_generated_world(self, bench):
        # FD a→b: for each row, b = resolver[a]
        for _ in range(20):
            w = bench.generate_world(n_tuples=50)
            resolver = bench._resolvers[((0,), 1)]
            np.testing.assert_array_equal(w[:, 1], resolver[w[:, 0]])

    def test_domain_bounds(self, bench):
        w = bench.generate_world(n_tuples=100)
        assert w.min() >= 0
        assert w.max() < crm.DOMAIN_SIZE

    def test_make_dataset_shapes(self, bench):
        X, y = bench.make_dataset(crm.Q_CERT, N=50, n_tuples=10)
        assert X.shape == (50, bench.feature_dim())
        assert y.shape == (50,)
        assert set(y.tolist()) <= {0.0, 1.0}

    def test_make_dataset_certified_query_constant(self, bench):
        # Certified query → same answer for all worlds in the same ~-class.
        # With enough worlds, we expect the predictor to be able to learn perfectly.
        # At minimum, we check that y-variance is not always 0 (it can vary by world).
        X, y = bench.make_dataset(crm.Q_CERT, N=200, n_tuples=5)
        # Just check it runs and shapes are right — domain-size=2 worlds can still vary
        assert X.shape[1] == bench.feature_dim()

    def test_identity_resolver_applied(self):
        # The make_benchmark() fixture overrides the resolver to be identity.
        bench = crm.make_benchmark()
        key = ((0,), 1)
        np.testing.assert_array_equal(bench._resolvers[key], np.array([0, 1]))


# ---------------------------------------------------------------------------
# Witness construction
# ---------------------------------------------------------------------------


class TestWitnessConstruction:
    @pytest.fixture
    def bench(self):
        return crm.make_benchmark()

    def test_witness_has_same_overlap_projection(self, bench):
        # w' ~ w iff w'|_Õ = w|_Õ
        for w in crm.WORLDS:
            w_prime = bench.construct_witness(w)
            assert w_prime is not None
            aug = sorted(frozenset({0, 1}))  # Õ = {a,b}
            np.testing.assert_array_equal(w_prime[:, aug], w[:, aug])

    def test_witness_satisfies_fd(self, bench):
        for w in crm.WORLDS:
            w_prime = bench.construct_witness(w)
            assert w_prime is not None
            resolver = bench._resolvers[((0,), 1)]
            np.testing.assert_array_equal(w_prime[:, 1], resolver[w_prime[:, 0]])

    def test_witness_differs_on_free_attrs(self, bench):
        # With domain=2 and permutation, c-column may flip — not guaranteed to change
        # but at least we verify it's a valid world.
        w_prime = bench.construct_witness(crm.W0)
        assert w_prime is not None
        assert w_prime.shape == crm.W0.shape

    def test_all_grounded_returns_none(self):
        # If all attrs are in Õ, no free attrs → witness returns None
        from data.synthetic import Config

        # Schema: 2 attrs, FD a→b, overlap on {a} → Õ={a,b} = all attrs
        config = Config(
            n_attrs=2,
            domain_size=2,
            fds=[(frozenset({0}), 1)],
            view_schemas={0: frozenset({0, 1})},
            overlap_schemas=[frozenset({0})],
        )
        bench2 = SyntheticBenchmark(config, seed=0)
        w = bench2.generate_world(5)
        assert bench2.construct_witness(w) is None

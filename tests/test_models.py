"""
Model smoke-tests: forward pass shapes, training loop, and CA predictor logic.

These run on CPU without a GPU.  They don't test convergence (that needs real
training runs), but they verify that every forward pass compiles and produces
the right output shape, and that the training loop reduces loss on a trivial
dataset.
"""

import numpy as np
import pytest
import torch

from data import crm_mini as crm
from data.utils import build_overlap_graph
from models.baselines import MajorityVote, VanillaOverlap
from models.closure_aware import ClosureAwarePredictor
from models.gnn_og import GNNOG, gnnog_from_benchmark
from models.mlp import MLP
from models.set_transformer import SetTransformer
from models.train import TrainConfig, evaluate, train

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BATCH = 16
FEATURE_DIM = 4  # CRM mini: Õ={a,b}, d=2 → 2^2=4


@pytest.fixture
def bench():
    return crm.make_benchmark()


@pytest.fixture
def adj():
    return build_overlap_graph(crm.N_ATTRS, crm.OVERLAP_SCHEMAS, crm.FDS)


@pytest.fixture
def overlap_dims(bench):
    d = bench.config.domain_size
    return [d ** len(aug) for aug in bench._augmented_overlaps]


@pytest.fixture
def overlap_attrs(bench):
    return [sorted(aug) for aug in bench._augmented_overlaps]


@pytest.fixture
def dummy_batch():
    torch.manual_seed(0)
    return torch.rand(BATCH, FEATURE_DIM)


@pytest.fixture
def tiny_dataset(bench):
    """100 worlds for a trivial training/eval dataset."""
    X, y = bench.make_dataset(crm.Q_CERT, N=100, n_tuples=5)
    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_output_shape(self, dummy_batch):
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=32, n_layers=2)
        out = model(dummy_batch)
        assert out.shape == (BATCH,)

    def test_output_scalar_per_sample(self, dummy_batch):
        model = MLP(in_dim=FEATURE_DIM)
        assert model(dummy_batch).ndim == 1

    @pytest.mark.parametrize("hidden", [64, 256])
    def test_hidden_dim_variants(self, dummy_batch, hidden):
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=hidden)
        assert model(dummy_batch).shape == (BATCH,)

    def test_dropout_does_not_crash(self, dummy_batch):
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=32, dropout=0.5)
        model.train()
        assert model(dummy_batch).shape == (BATCH,)


# ---------------------------------------------------------------------------
# VanillaOverlap & MajorityVote baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_vanilla_overlap_shape(self, dummy_batch):
        model = VanillaOverlap(in_dim=FEATURE_DIM)
        assert model(dummy_batch).shape == (BATCH,)

    def test_majority_vote_shape(self, dummy_batch):
        model = MajorityVote()
        assert model(dummy_batch).shape == (BATCH,)

    def test_majority_vote_constant_output(self, dummy_batch):
        model = MajorityVote()
        out = model(dummy_batch)
        # All outputs should be identical (same learned constant)
        assert torch.allclose(out, out[0].expand(BATCH))


# ---------------------------------------------------------------------------
# SetTransformer
# ---------------------------------------------------------------------------


class TestSetTransformer:
    def test_single_overlap_shape(self, dummy_batch, overlap_dims):
        model = SetTransformer(overlap_dims=overlap_dims, hidden_dim=16, n_heads=2)
        assert model(dummy_batch).shape == (BATCH,)

    def test_multi_overlap_shape(self):
        # Two overlaps: sizes 4 and 8
        dims = [4, 8]
        x = torch.rand(BATCH, 12)
        model = SetTransformer(overlap_dims=dims, hidden_dim=16, n_heads=2, n_layers=1)
        assert model(x).shape == (BATCH,)

    def test_gradients_flow(self, dummy_batch, overlap_dims):
        model = SetTransformer(overlap_dims=overlap_dims, hidden_dim=16, n_heads=2)
        loss = model(dummy_batch).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# GNN-OG
# ---------------------------------------------------------------------------


class TestGNNOG:
    def test_output_shape(self, bench, adj, dummy_batch, overlap_dims, overlap_attrs):
        model = GNNOG(
            overlap_dims=overlap_dims,
            overlap_attrs=overlap_attrs,
            adj=adj,
            n_attrs=crm.N_ATTRS,
            domain_size=crm.DOMAIN_SIZE,
            hidden_dim=16,
            n_layers=2,
        )
        assert model(dummy_batch).shape == (BATCH,)

    def test_factory_constructor(self, bench, adj, dummy_batch):
        model = gnnog_from_benchmark(bench, adj, hidden_dim=16)
        assert model(dummy_batch).shape == (BATCH,)

    def test_gradients_flow(self, bench, adj, dummy_batch):
        model = gnnog_from_benchmark(bench, adj, hidden_dim=16)
        loss = model(dummy_batch).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_isolated_node_handled(self, bench, adj, dummy_batch):
        # Attribute c (idx=2) has no neighbors; should not crash or produce NaN
        model = gnnog_from_benchmark(bench, adj, hidden_dim=16)
        out = model(dummy_batch)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# ClosureAwarePredictor
# ---------------------------------------------------------------------------


class TestClosureAwarePredictor:
    def test_certified_delegates_to_base(self, dummy_batch):
        base = MLP(in_dim=FEATURE_DIM, hidden_dim=16)
        ca = ClosureAwarePredictor(base=base, certified=True)
        expected = base(dummy_batch)
        actual = ca(dummy_batch)
        assert torch.allclose(actual, expected)

    def test_noncertified_outputs_zero(self, dummy_batch):
        ca = ClosureAwarePredictor(base=None, certified=False)
        out = ca(dummy_batch)
        assert torch.allclose(out, torch.zeros(BATCH))

    def test_noncertified_sigmoid_is_half(self, dummy_batch):
        ca = ClosureAwarePredictor(base=None, certified=False)
        probs = torch.sigmoid(ca(dummy_batch))
        assert torch.allclose(probs, torch.full((BATCH,), 0.5))

    def test_from_query_certified(self, bench):
        config = crm.make_config()
        base = MLP(in_dim=FEATURE_DIM, hidden_dim=16)
        ca = ClosureAwarePredictor.from_query(crm.Q_CERT, config, base)
        assert ca.certified is True
        assert not ca.is_oracle_abstaining()

    def test_from_query_noncertified(self, bench):
        config = crm.make_config()
        ca = ClosureAwarePredictor.from_query(crm.Q_NONIDENT, config, base=None)
        assert ca.certified is False
        assert ca.is_oracle_abstaining()

    def test_no_base_required_when_not_certified(self):
        # Should not raise
        ca = ClosureAwarePredictor(base=None, certified=False)
        x = torch.rand(5, FEATURE_DIM)
        assert ca(x).shape == (5,)

    def test_base_required_when_certified(self):
        with pytest.raises(AssertionError):
            ClosureAwarePredictor(base=None, certified=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    def test_train_reduces_loss(self, tiny_dataset):
        X, y = tiny_dataset
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=32)
        cfg = TrainConfig(epochs=20, lr=1e-2, batch_size=32, patience=20, device="cpu")
        result = train(model, X, y, cfg=cfg)
        assert result.train_loss[-1] < result.train_loss[0]

    def test_train_with_val_set(self, tiny_dataset):
        X, y = tiny_dataset
        split = len(X) // 5
        X_tr, y_tr = X[split:], y[split:]
        X_v, y_v = X[:split], y[:split]
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=32)
        cfg = TrainConfig(epochs=30, lr=1e-2, patience=10, device="cpu")
        result = train(model, X_tr, y_tr, X_v, y_v, cfg=cfg)
        assert len(result.val_loss) > 0

    def test_early_stopping_triggers(self, tiny_dataset):
        X, y = tiny_dataset
        split = len(X) // 5
        X_tr, y_tr = X[split:], y[split:]
        X_v, y_v = X[:split], y[:split]
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=32)
        # Very short patience → should stop early before 500 epochs
        cfg = TrainConfig(epochs=500, lr=1e-2, patience=5, device="cpu")
        result = train(model, X_tr, y_tr, X_v, y_v, cfg=cfg)
        assert result.stopped_early
        assert len(result.train_loss) < 500

    def test_evaluate_returns_dict(self, tiny_dataset):
        X, y = tiny_dataset
        model = MLP(in_dim=FEATURE_DIM, hidden_dim=32)
        metrics = evaluate(model, X, y, device="cpu")
        assert "loss" in metrics and "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_all_model_types_trainable(self, tiny_dataset, overlap_dims, overlap_attrs, adj):
        X, y = tiny_dataset
        cfg = TrainConfig(epochs=5, device="cpu")
        models = [
            MLP(in_dim=FEATURE_DIM, hidden_dim=16),
            VanillaOverlap(in_dim=FEATURE_DIM),
            MajorityVote(),
            SetTransformer(overlap_dims=overlap_dims, hidden_dim=16, n_heads=2, n_layers=1),
            GNNOG(
                overlap_dims=overlap_dims,
                overlap_attrs=overlap_attrs,
                adj=adj,
                n_attrs=crm.N_ATTRS,
                domain_size=crm.DOMAIN_SIZE,
                hidden_dim=16,
                n_layers=1,
            ),
        ]
        for m in models:
            result = train(m, X, y, cfg=cfg)
            assert len(result.train_loss) > 0, f"{type(m).__name__} produced no loss history"

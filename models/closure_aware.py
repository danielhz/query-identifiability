"""
Closure-Aware (CA) predictor: theory-guided wrapper.

If the query is certified (Theorem 1), the query answer IS determined by the
overlap features → delegate to a trained base model.

If the query is not certified (error floor ≥ 1/2 by Theorem 5), no
evidence-measurable estimator can do better than chance → output logit 0
(= probability 0.5 after sigmoid), reflecting maximum uncertainty.

This is the theoretically optimal strategy under the identifiability framework:
it is right when the query is identifiable, and refuses to guess when it is not.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data.synthetic import BooleanCQ, Config


class ClosureAwarePredictor(nn.Module):
    """
    Wraps any base model with a certificate gate.

    certified : pre-computed result of query.is_certified(config).
                Pass it at construction time so the forward pass is fast.
    base      : a trained nn.Module with the standard forward(x) → logits interface.
                Ignored (and may be None) when certified=False.
    """

    def __init__(self, base: nn.Module | None, certified: bool):
        super().__init__()
        self.certified = certified
        self.base: nn.Module | None = None
        # Register base as a submodule only when it is used
        if certified:
            assert base is not None, "base model required for a certified query"
            self.base = base

    @classmethod
    def from_query(
        cls,
        query: BooleanCQ,
        config: Config,
        base: nn.Module | None,
    ) -> "ClosureAwarePredictor":
        """Build from a BooleanCQ without pre-computing the certificate manually."""
        return cls(base=base, certified=query.is_certified(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, feature_dim) → logits: (B,)"""
        if self.certified:
            assert self.base is not None
            return self.base(x)
        # Non-identifiable: return 0 (= 0.5 prob) for every sample
        return torch.zeros(x.shape[0], device=x.device)

    def is_oracle_abstaining(self) -> bool:
        """True when the predictor will always output 0.5 (non-identifiable query)."""
        return not self.certified

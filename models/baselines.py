"""
Baselines for the identifiability experiments.

VanillaOverlap : logistic regression on the raw overlap feature vector.
                 Has FD information only insofar as it is encoded in the feature
                 (it is not — the features are the same for ~-equivalent worlds).
                 Serves as the weakest learned baseline.

MajorityVote   : learns a single scalar logit (= log-odds of the majority class).
                 Equivalent to always predicting the training-set majority.
                 Expected error ≈ min(p, 1-p) where p = base rate; useful as a
                 sanity check that no model does worse.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VanillaOverlap(nn.Module):
    """
    Logistic regression on the raw overlap feature.

    No hidden layers, no FD-aware structure.  This is the 'flat' baseline that
    uses the same input as MLP but with zero capacity to exploit relational
    structure.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → logits: (B,)"""
        return self.linear(x).squeeze(-1)


class MajorityVote(nn.Module):
    """
    Learns a single bias term (constant logit), ignoring all features.

    After training with BCE loss, this converges to logit = log(p/(1-p))
    where p is the training-set positive rate.  Equivalent to always
    predicting the majority class.
    """

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, *) → logits: (B,)  — x is ignored"""
        return self.logit.expand(x.shape[0])

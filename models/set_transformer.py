"""
SetTransformer predictor: cross-overlap attention over per-overlap feature vectors.

Architecture (Lee et al. 2019, ISAB variant):
  1. Project each overlap schema's probability vector to a shared hidden dim.
  2. Apply L rounds of multi-head self-attention across the K overlap tokens.
  3. Pool (mean) → MLP head → scalar logit.

When K=1 (single overlap schema, as in the CRM mini fixture) this degenerates
to a linear projection + MLP, which is fine — the architecture only contributes
additional inductive bias when multiple overlaps are present.

Input schema: the SyntheticBenchmark feature vector is the concatenation of
per-overlap probability vectors:
    x = [f_0 | f_1 | ... | f_{K-1}]
where f_k has length domain_size ** |Õ_k|.

The constructor receives `overlap_dims: list[int]` (sizes of each f_k) so it
can split x back into K tokens before attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MAB(nn.Module):
    """Multihead Attention Block: MAB(Q, K) = LayerNorm(H + rFF(H)), H = Attn(Q,K,K)."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        H, _ = self.attn(Q, K, K)
        H = self.ln1(Q + H)
        return self.ln2(H + self.ff(H))


class _SAB(nn.Module):
    """Self-Attention Block: SAB(X) = MAB(X, X)."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.mab = _MAB(dim, n_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class SetTransformer(nn.Module):
    """
    overlap_dims : sizes of each per-overlap feature slice (sum = in_dim)
    hidden_dim   : common token embedding dimension
    n_heads      : attention heads (must divide hidden_dim)
    n_layers     : SAB rounds
    dropout      : applied inside attention and feed-forward
    """

    def __init__(
        self,
        overlap_dims: list[int],
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.overlap_dims = overlap_dims

        # Per-overlap input projections (handle variable-size tokens)
        self.proj = nn.ModuleList([nn.Linear(d, hidden_dim) for d in overlap_dims])

        self.encoder = nn.Sequential(
            *[_SAB(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, sum(overlap_dims)) → logits: (B,)"""
        # Split flat feature into K per-overlap slices
        slices = torch.split(x, self.overlap_dims, dim=-1)
        # Project each slice to hidden_dim → (B, K, hidden_dim)
        tokens = torch.stack([p(s) for p, s in zip(self.proj, slices)], dim=1)
        # Self-attention across overlaps
        for layer in self.encoder:
            tokens = layer(tokens)
        # Mean-pool over K tokens → (B, hidden_dim)
        pooled = tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)

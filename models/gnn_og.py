"""
GNN-OG: message-passing GNN over the constraint-closed overlap graph G_{Σ,Ω}.

Node features: per-attribute marginal probability vectors extracted from the
overlap feature.  For attribute a in Õ_k, the marginal p(a=v) is obtained by
summing the Õ_k-probability tensor over all dimensions except a's.  When a
appears in multiple overlaps the marginals are averaged.  Attributes outside
every Õ get a uniform zero vector.

Message passing: L rounds of mean-aggregation + linear transform + ReLU
(GraphSAGE-style, no torch_geometric dependency).

Global readout: mean of final node representations → MLP head → scalar logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _extract_marginals(
    x: torch.Tensor,  # (B, feature_dim)
    overlap_dims: list[int],  # len = K, each = domain_size ** |Õ_k|
    overlap_attrs: list[list[int]],  # sorted attribute indices per Õ_k
    n_attrs: int,
    domain_size: int,
) -> torch.Tensor:
    """
    Returns node feature matrix of shape (B, n_attrs, domain_size).
    Entry [b, a, v] = marginal probability P(a = v) in world b.
    """
    B = x.shape[0]
    d = domain_size
    accum = torch.zeros(B, n_attrs, d, device=x.device, dtype=x.float().dtype)
    count = torch.zeros(n_attrs, device=x.device)

    offset = 0
    for k, (dim_k, attrs_k) in enumerate(zip(overlap_dims, overlap_attrs)):
        f_k = x[:, offset : offset + dim_k]  # (B, d^|Õ_k|)
        offset += dim_k
        shape_k = [d] * len(attrs_k)
        # Reshape to (B, d, d, ...) with one dim per attr in Õ_k
        prob_k = f_k.view(B, *shape_k)  # (B, d, ..., d)

        for local_i, global_a in enumerate(attrs_k):
            # Sum over all dims except local_i → marginal of attr global_a
            sum_dims = [1 + j for j in range(len(attrs_k)) if j != local_i]
            marg = prob_k.sum(dim=sum_dims)  # (B, d)
            accum[:, global_a, :] += marg
            count[global_a] += 1.0

    # Average contributions from multiple overlaps; leave zero for isolated attrs
    mask = count > 0
    accum[:, mask, :] /= count[mask].unsqueeze(0).unsqueeze(-1)
    return accum  # (B, n_attrs, domain_size)


class GNNLayer(nn.Module):
    """One round of mean-aggregation message passing (GraphSAGE concat variant)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Concat self + mean-neighbor → linear
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(
        self,
        h: torch.Tensor,  # (B, n_attrs, in_dim)
        adj: list[list[int]],  # adjacency list (n_attrs entries)
        n_attrs: int,
    ) -> torch.Tensor:  # (B, n_attrs, out_dim)
        # Build mean neighbor messages
        agg = torch.zeros_like(h)
        deg = torch.zeros(n_attrs, device=h.device)
        for a, nbrs in enumerate(adj):
            if nbrs:
                agg[:, a, :] = h[:, nbrs, :].mean(dim=1)
                deg[a] = 1.0  # mark as having neighbors
        # For isolated nodes, agg stays zero
        cat = torch.cat([h, agg], dim=-1)  # (B, n_attrs, 2*in_dim)
        return torch.relu(self.lin(cat))


class GNNOG(nn.Module):
    """
    GNN over the constraint-closed overlap graph.

    overlap_dims   : per-overlap feature sizes (same as SetTransformer)
    overlap_attrs  : sorted attribute lists for each Õ_k
    adj            : adjacency list from build_overlap_graph()
    n_attrs        : total number of attributes
    domain_size    : d
    hidden_dim     : node embedding size
    n_layers       : message-passing rounds
    """

    def __init__(
        self,
        overlap_dims: list[int],
        overlap_attrs: list[list[int]],
        adj: list[list[int]],
        n_attrs: int,
        domain_size: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.overlap_dims = overlap_dims
        self.overlap_attrs = overlap_attrs
        self.adj = adj
        self.n_attrs = n_attrs
        self.domain_size = domain_size

        # Input projection: domain_size → hidden_dim
        self.input_proj = nn.Linear(domain_size, hidden_dim)

        self.layers = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, feature_dim) → logits: (B,)"""
        # Node features: (B, n_attrs, domain_size)
        node_feat = _extract_marginals(
            x,
            self.overlap_dims,
            self.overlap_attrs,
            self.n_attrs,
            self.domain_size,
        )
        # Project to hidden dim
        h = torch.relu(self.input_proj(node_feat))  # (B, n_attrs, hidden_dim)
        # Message passing
        for layer in self.layers:
            h = layer(h, self.adj, self.n_attrs)
        # Global mean pooling → (B, hidden_dim)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory: build GNNOG from a SyntheticBenchmark instance + overlap graph
# ---------------------------------------------------------------------------


def gnnog_from_benchmark(bench, adj: list[list[int]], **kwargs) -> "GNNOG":
    """
    Convenience constructor.  bench is a SyntheticBenchmark; adj comes from
    build_overlap_graph().
    """
    cfg = bench.config
    d = cfg.domain_size
    overlap_dims = [d ** len(aug) for aug in bench._augmented_overlaps]
    overlap_attrs = [sorted(aug) for aug in bench._augmented_overlaps]
    return GNNOG(
        overlap_dims=overlap_dims,
        overlap_attrs=overlap_attrs,
        adj=adj,
        n_attrs=cfg.n_attrs,
        domain_size=d,
        **kwargs,
    )

"""
Core algorithms: FD closure, identifiability certificate, and Greedy-MinAug.

Attribute sets are frozensets of ints.
An FD is a (frozenset[int], int) pair representing X → {b}.
"""

from __future__ import annotations

from typing import Sequence

FD = tuple[frozenset[int], int]


def fd_closure(seed: frozenset[int], fds: Sequence[FD]) -> frozenset[int]:
    """Return X^+_Σ: attribute closure of seed under FD set (Theorem 1 machinery)."""
    closure = set(seed)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs <= closure and rhs not in closure:
                closure.add(rhs)
                changed = True
    return frozenset(closure)


def augmented_overlap(overlap_attrs: frozenset[int], fds: Sequence[FD]) -> frozenset[int]:
    """Return Õ = att(O)^+_Σ for an overlap with attribute set overlap_attrs."""
    return fd_closure(overlap_attrs, fds)


def check_certificate(
    footprint: frozenset[int],
    overlap_schemas: Sequence[frozenset[int]],
    fds: Sequence[FD],
) -> bool:
    """
    Sufficient identifiability certificate (Theorem 1).
    Returns True iff ∃ overlap O such that footprint ⊆ att(O)^+_Σ.
    """
    for o_attrs in overlap_schemas:
        if footprint <= augmented_overlap(o_attrs, fds):
            return True
    return False


def action_coverage(
    action: frozenset[int],
    footprint: frozenset[int],
    fds: Sequence[FD],
) -> frozenset[int]:
    """Σ-coverage of an action w.r.t. footprint S: A^+_Σ ∩ S (Definition MinAug)."""
    return fd_closure(action, fds) & footprint


def greedy_minaug(
    footprint: frozenset[int],
    candidate_actions: Sequence[frozenset[int]],
    fds: Sequence[FD],
    costs: Sequence[float] | None = None,
) -> list[frozenset[int]] | None:
    """
    Algorithm 1 (Greedy-MinAug): greedy H(|S|)-approximation for minimum interface
    augmentation.  Returns selected actions or None if no feasible cover exists.
    """
    if costs is None:
        costs = [1.0] * len(candidate_actions)

    coverages = [action_coverage(a, footprint, fds) for a in candidate_actions]
    covered: frozenset[int] = frozenset()
    selected: list[frozenset[int]] = []
    remaining = list(range(len(candidate_actions)))

    while covered < footprint:
        best_i, best_ratio = None, -1.0
        for i in remaining:
            marginal = len(coverages[i] - covered)
            if marginal == 0:
                continue
            ratio = marginal / costs[i]
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i
        if best_i is None:
            return None  # infeasible
        selected.append(candidate_actions[best_i])
        covered = covered | coverages[best_i]
        remaining.remove(best_i)

    return selected


def build_overlap_graph(
    n_attrs: int,
    overlap_schemas: Sequence[frozenset[int]],
    fds: Sequence[FD],
) -> list[list[int]]:
    """
    Build the constraint-closed overlap graph G_{Σ,Ω} (Definition 3.3).
    Returns adj[a] = sorted list of attributes in the same clique as a.
    """
    adj: list[set[int]] = [set() for _ in range(n_attrs)]
    for o_attrs in overlap_schemas:
        clique = augmented_overlap(o_attrs, fds)
        for a in clique:
            adj[a].update(clique - {a})
    return [sorted(nbrs) for nbrs in adj]

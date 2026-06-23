"""
Minimal hand-crafted test fixture derived from the paper's CRM running example.

=== Schema ===
3 attributes: a=0, b=1, c=2   (domain {0, 1})
FD: a → b  (cross-world resolver: a=0 → b=0, a=1 → b=1, i.e. identity)
Views:
  U (view_id=0): {a=0, c=2}   (like a billing view missing the join key)
  V (view_id=1): {a=0, b=1}   (like a CRM view exposing the derived key)
Designated overlap: {a=0}
Augmented overlap Õ = {a}^+_{a→b} = {a, b} = {0, 1}

=== Hand-verified expected values ===

fd_closure({a}, [a→b])    = {a, b}             = {0, 1}
fd_closure({a,c}, [a→b])  = {a, b, c}          = {0, 1, 2}
fd_closure({b}, [a→b])    = {b}                = {1}      (b doesn't imply a)

check_certificate footprint={a,b}, overlaps=[{a}], fds=[a→b]:
  Õ = {a,b}.  {a,b} ⊆ {a,b} ✓  → True

check_certificate footprint={a,c}, overlaps=[{a}], fds=[a→b]:
  Õ = {a,b}.  {a,c} ⊆ {a,b}? No (c ∉ {a,b}) → False

=== Worlds ===

W0 = [(a=0,b=0,c=1), (a=1,b=1,c=0)]
  W0|_{a,b} = {(0,0),(1,1)}   ← Õ projection
  W0|_{a,c} = {(0,1),(1,0)}   ← view U
  W0|_{a,b} = {(0,0),(1,1)}   ← view V  (same as Õ here since V={a,b})

W1 = [(a=0,b=0,c=0), (a=1,b=1,c=1)]
  W1|_{a,b} = {(0,0),(1,1)}   ← same Õ projection as W0 → W0 ~ W1
  W1|_{a,c} = {(0,0),(1,1)}   ← view U  (different from W0)

=== Overlap feature vectors ===
Õ = {a,b}, domain 2 → 4 possible tuples indexed (a*2+b):
  (0,0)→idx0, (0,1)→idx1, (1,0)→idx2, (1,1)→idx3

W0 overlap feature: (0,0) once, (1,1) once → [0.5, 0.0, 0.0, 0.5]
W1 overlap feature: (0,0) once, (1,1) once → [0.5, 0.0, 0.0, 0.5]  ← identical ✓

=== Queries ===

Q_CERT    = ∃a,b. R_V(a,b) ∧ a=0 ∧ b=0
  Atom over V={a=0,b=1}, sorted attrs=[a,b], pattern=[0,0] (constants a=0,b=0)
  footprint = att(V) = {a,b} = {0,1}
  Õ = {a,b}.  {0,1} ⊆ {0,1} ✓ → CERTIFIED

  Q_CERT(W0): (a=0,b=0) ∈ W0|_V = {(0,0),(1,1)} ✓ → True
  Q_CERT(W1): (a=0,b=0) ∈ W1|_V = {(0,0),(1,1)} ✓ → True
  (same answer for both worlds in class → consistent with identifiability)

Q_NONIDENT = ∃a,c. R_U(a,c) ∧ a=0 ∧ c=0
  Atom over U={a=0,c=2}, sorted attrs=[a,c], pattern=[0,0] (constants a=0,c=0)
  footprint = att(U) = {a,c} = {0,2}
  Õ = {a,b}.  {0,2} ⊆ {0,1}? No → NOT CERTIFIED

  Q_NONIDENT(W0): (a=0,c=0) ∈ W0|_U = {(0,1),(1,0)} → (0,0) absent → False
  Q_NONIDENT(W1): (a=0,c=0) ∈ W1|_U = {(0,0),(1,1)} → (0,0) present → True
  W0 ~ W1 but answers differ → confirms non-identifiability ✓

=== MinAug ===

footprint = {a,c} = {0,2}
candidate actions: A1={a,c}={0,2},  A2={a}={0}

coverage(A1={a,c}, {a,c}, [a→b]):
  {a,c}^+_{a→b} = {a,b,c}.  {a,b,c} ∩ {a,c} = {a,c} → covers full footprint

coverage(A2={a}, {a,c}, [a→b]):
  {a}^+_{a→b} = {a,b}.  {a,b} ∩ {a,c} = {a} → only covers {a}

greedy picks A1 (ratio 2/1 > A2's ratio 1/1) → selected=[A1], size=1
optimal is also 1 (A1 alone covers everything) → approx ratio = 1.0

=== build_overlap_graph ===

n_attrs=3, overlap_schemas=[{a}={0}], fds=[a→b]:
  Clique on Õ={a,b}={0,1}: adj[0]=[1], adj[1]=[0], adj[2]=[]
"""

import numpy as np

from .synthetic import Atom, BooleanCQ, Config, SyntheticBenchmark
from .utils import FD

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

N_ATTRS = 3
DOMAIN_SIZE = 2

FDS: list[FD] = [(frozenset({0}), 1)]  # a → b

VIEW_SCHEMAS: dict[int, frozenset[int]] = {
    0: frozenset({0, 2}),  # U: {a, c}
    1: frozenset({0, 1}),  # V: {a, b}
}

OVERLAP_SCHEMAS: list[frozenset[int]] = [frozenset({0})]  # overlap on {a}

# ---------------------------------------------------------------------------
# Config & benchmark factory
# ---------------------------------------------------------------------------


def make_config() -> Config:
    return Config(
        n_attrs=N_ATTRS,
        domain_size=DOMAIN_SIZE,
        fds=FDS,
        view_schemas=VIEW_SCHEMAS,
        overlap_schemas=OVERLAP_SCHEMAS,
    )


def make_benchmark(seed: int = 0) -> SyntheticBenchmark:
    """
    Return a benchmark whose resolver implements the identity mapping a → b = a.
    This is set explicitly so the fixture is deterministic regardless of seed.
    """
    bench = SyntheticBenchmark(make_config(), seed=seed)
    # Override: resolver for (lhs=(0,), rhs=1): a=0→b=0, a=1→b=1
    bench._resolvers[((0,), 1)] = np.array([0, 1], dtype=np.int32)
    return bench


# ---------------------------------------------------------------------------
# Hand-crafted worlds (satisfy FD a→b via identity resolver)
# ---------------------------------------------------------------------------

# (a=0,b=0,c=1) and (a=1,b=1,c=0)
W0 = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int32)

# (a=0,b=0,c=0) and (a=1,b=1,c=1)
W1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)

WORLDS = [W0, W1]

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

# Q_CERT: ∃a,b. R_V(a,b) ∧ a=0 ∧ b=0   (CERTIFIED — footprint {a,b} ⊆ Õ)
Q_CERT = BooleanCQ(
    atoms=[
        Atom(view_id=1, pattern=[0, 0])  # sorted attrs of V=[a,b]: a=0, b=0
    ]
)

# Q_NONIDENT: ∃a,c. R_U(a,c) ∧ a=0 ∧ c=0  (NOT CERTIFIED — c not in Õ)
Q_NONIDENT = BooleanCQ(
    atoms=[
        Atom(view_id=0, pattern=[0, 0])  # sorted attrs of U=[a,c]: a=0, c=0
    ]
)

# Two-atom join query for variable binding test
# ∃x. R_U(x, "z") ∧ R_V(x, "w")   — join on variable x=0 (constant)
# Asks: is there a row in U with a=0 AND a row in V with a=0?
Q_JOIN = BooleanCQ(
    atoms=[
        Atom(view_id=0, pattern=[0, "z"]),  # U: a=0, c=z (free)
        Atom(view_id=1, pattern=[0, "w"]),  # V: a=0, b=w (free)
    ]
)

# ---------------------------------------------------------------------------
# Expected values  (verified by hand in the module docstring above)
# ---------------------------------------------------------------------------

EXPECTED = {
    # FD closure
    "fd_closure_a": frozenset({0, 1}),  # {a}^+ = {a,b}
    "fd_closure_ac": frozenset({0, 1, 2}),  # {a,c}^+ = {a,b,c}
    "fd_closure_b": frozenset({1}),  # {b}^+ = {b}  (b doesn't imply a)
    # Certificate
    "cert_q_cert": True,
    "cert_q_nonident": False,
    # Query answers
    "Q_cert_w0": True,
    "Q_cert_w1": True,
    "Q_nonident_w0": False,
    "Q_nonident_w1": True,
    "Q_join_w0": True,  # a=0 present in both views of W0
    "Q_join_w1": True,  # a=0 present in both views of W1
    # Overlap features: Õ={a,b}, d=2, idx = a*2+b
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    "feat_w0": np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32),
    "feat_w1": np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32),  # identical to W0
    # MinAug
    "minaug_candidates": [frozenset({0, 2}), frozenset({0})],  # A1={a,c}, A2={a}
    "minaug_footprint": frozenset({0, 2}),
    "minaug_selected": [frozenset({0, 2})],  # greedy picks A1 first
    "minaug_size": 1,
    # Overlap graph: clique on Õ={a,b}={0,1}
    "overlap_graph": [[1], [0], []],  # adj[a], adj[b], adj[c]
}

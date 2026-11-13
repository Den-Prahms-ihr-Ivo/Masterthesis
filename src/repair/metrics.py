"""Metrics computed from stored schedules and instances (exposé 4.6).

For now only the distance H. Churn is C_t = H(S_t, S_{t-1}) / J with
J = instance.n_cells; the slice test computes it directly from distance.
"""

from repair.model import Schedule


def distance(a: Schedule, b: Schedule) -> int:
    """H(a, b) = |a xor b|: cells assigned in exactly one of the two schedules.

    Symmetric, and 0 iff a == b. Example: H({(0,0),(1,1)}, {(1,1),(2,0)}) = 2.
    """
    return len(a ^ b)

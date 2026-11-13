"""Metrics computed from stored schedules and instances (exposé 4.6).

So far: the distance H, and the per-volunteer burden HWF needs at run time.
Churn is C_t = H(S_t, S_{t-1}) / J with J = instance.n_cells; the slice test computes
it directly from distance.
"""

from collections.abc import Iterable

from repair.model import Schedule, Trajectory, VolunteerId


def distance(a: Schedule, b: Schedule) -> int:
    """H(a, b) = |a xor b|: cells assigned in exactly one of the two schedules.

    Symmetric, and 0 iff a == b. Example: H({(0,0),(1,1)}, {(1,1),(2,0)}) = 2.
    """
    return len(a ^ b)


def affected(
    previous: Schedule,
    current: Schedule,
) -> frozenset[VolunteerId]:
    """Volunteers with at least one changed assignment cell."""
    return frozenset(w for w, _ in previous ^ current)


def burden(
    trajectory: Trajectory,
    s0: Schedule,
    volunteers: Iterable[VolunteerId],
    until: int,
) -> dict[VolunteerId, int]:
    """Count affected repairs per volunteer through step `until`.

    Include every supplied volunteer, including those with zero burden.
    `volunteers` must include all volunteer IDs appearing in the schedules.
    """
    if until < 0 or until > len(trajectory.steps):
        raise ValueError(
            f"until must be between 0 and {len(trajectory.steps)}, " f"got {until}"
        )

    counts = {w: 0 for w in volunteers}
    previous = s0

    for step in trajectory.steps[:until]:
        for w in affected(previous, step.schedule):
            counts[w] += 1

        previous = step.schedule

    return counts

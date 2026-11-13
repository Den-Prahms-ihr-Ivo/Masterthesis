"""metrics.py: distance H (hand values of repair_slice), affected volunteers, burden."""

import pytest

from repair.expected import REPAIR_SLICE_DISTANCE, REPAIR_SLICE_S0
from repair.metrics import affected, burden, distance
from repair.model import Arm, PolicyName, ReferenceMode, Step, Trajectory


@pytest.mark.parametrize("schedule, expected", list(REPAIR_SLICE_DISTANCE.items()))
def test_distance_symmetric_difference(schedule, expected):
    """H(S, S0) = |S xor S0| from the value sheet; symmetric in its arguments."""
    assert distance(schedule, REPAIR_SLICE_S0) == expected
    assert distance(REPAIR_SLICE_S0, schedule) == expected


def test_distance_counts_added_and_removed_cells():
    """One removed cell and one added cell: H = 2, not 1 (unlike a plain set difference)."""
    a = frozenset({(0, 0), (1, 1)})
    b = frozenset({(1, 1), (2, 0)})

    assert distance(a, b) == 2
    assert distance(a, a) == 0


# --- affected and burden (HWF needs them at run time) ------------------------------

_ARM = Arm(PolicyName.SF, ReferenceMode.CHAINED)


def _trajectory(*schedules):
    """A trajectory whose step t has schedule schedules[t - 1] (stages do not matter)."""
    return Trajectory(
        _ARM, tuple(Step(t, s, ()) for t, s in enumerate(schedules, start=1))
    )


def test_affected_volunteers_with_a_changed_cell():
    """v0 moved (removed), v2 added; v1 kept its cell."""
    assert affected(frozenset({(0, 0), (1, 1)}), frozenset({(1, 1), (2, 0)})) == {0, 2}
    assert affected(frozenset({(1, 1)}), frozenset({(1, 1)})) == frozenset()


def test_affected_counts_a_swap_once():
    """v0 moves from s0 to s2: two changed cells, one affected volunteer."""
    assert affected(frozenset({(0, 0)}), frozenset({(0, 2)})) == {0}


def test_burden_counts_affected_repairs():
    """S0 -> S1 affects v0, v2; S1 -> S2 changes nothing; S2 -> S3 affects v0."""
    s0 = frozenset({(0, 0), (1, 1)})
    s1 = frozenset({(1, 1), (2, 0)})
    s3 = frozenset({(0, 2), (1, 1), (2, 0)})
    trajectory = _trajectory(s1, s1, s3)

    assert burden(trajectory, s0, (0, 1, 2), until=0) == {0: 0, 1: 0, 2: 0}
    assert burden(trajectory, s0, (0, 1, 2), until=1) == {0: 1, 1: 0, 2: 1}
    assert burden(trajectory, s0, (0, 1, 2), until=2) == {0: 1, 1: 0, 2: 1}
    assert burden(trajectory, s0, (0, 1, 2), until=3) == {0: 2, 1: 0, 2: 1}


@pytest.mark.parametrize("until", [-1, 4])
def test_burden_rejects_until_outside_the_trajectory(until):
    with pytest.raises(ValueError):
        burden(_trajectory(frozenset(), frozenset(), frozenset()), frozenset(), (0,), until)

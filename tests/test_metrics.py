"""metrics.py: distance H, checked against the hand values of repair_slice."""

import pytest

from repair.expected import REPAIR_SLICE_DISTANCE, REPAIR_SLICE_S0
from repair.metrics import distance


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

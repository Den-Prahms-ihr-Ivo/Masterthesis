"""perturbations.py: events produce a new instance; the old one is untouched."""

import pytest

from repair.expected import REPAIR_SLICE_EVENT
from repair.model import DemandChange
from repair.perturbations import apply_event


def test_apply_removal_returns_new_instance(repair_slice):
    """A loses period 0; nothing else changes, and repair_slice itself is unchanged."""
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)

    assert after is not repair_slice
    assert after.volunteers[0].available == frozenset({1})
    assert repair_slice.volunteers[0].available == frozenset({0, 1})  # not mutated

    # Everything else is identical
    assert after.volunteers[1:] == repair_slice.volunteers[1:]
    assert after.shifts == repair_slice.shifts
    assert after.volunteers[0].skill == repair_slice.volunteers[0].skill


def test_other_events_not_supported_yet(repair_slice):
    """Until they are implemented, other event types fail loudly instead of doing nothing."""
    with pytest.raises(NotImplementedError):
        apply_event(repair_slice, DemandChange(shift=0, delta=1))

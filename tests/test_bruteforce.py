"""bruteforce.py

The brute force is the reference for the MILP, so it is tested against the hand values first.
"""

import pytest

from repair.expected import (
    EXPERT_SHIFT_OPTIMUM,
    EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY,
    MIN_CREW_BOTH_ROWS,
    NONE,
    SINGLE_SHIFT_ROWS,
)
from repair.model import StageResult, StageStatus
from repair.objectives import MAX_QUALITY, MIN_SHORTFALL
from tests.bruteforce import all_schedules, solve_stage
from tests.builders import Grid


def _feasible(rows):
    """The schedules a value sheet lists as feasible."""
    return {schedule for schedule, row in rows.items() if row.violations == NONE}


# --- all_schedules -------------------------------------------------------------


def test_single_shift_has_four_feasible_schedules(single_shift):
    """All four subsets of the two cells are feasible."""
    found = list(all_schedules(single_shift))

    assert len(found) == 4  # each schedule exactly once
    assert set(found) == _feasible(SINGLE_SHIFT_ROWS)


def test_min_crew_excludes_partial_staffing(min_crew_both_available):
    """d = 3 needs 2 people: one person alone is not feasible, so only {} and the pair remain."""
    found = set(all_schedules(min_crew_both_available))

    assert found == _feasible(MIN_CREW_BOTH_ROWS)
    assert found == {frozenset(), frozenset({(0, 0), (1, 0)})}


def test_all_schedules_rejects_large_instances():
    """More than MAX_CELLS cells would take too long to enumerate."""
    grid = Grid(n_periods=1, n_roles=1)
    instance = grid.instance(grid.volunteers(21), grid.shifts(1))

    with pytest.raises(ValueError):
        next(all_schedules(instance))


# --- solve_stage ---------------------------------------------------------------


def test_maxquality_single_shift(single_shift):
    """[MaxQuality] on single_shift: {(0, 0)}, U = 6."""
    assert solve_stage(single_shift, MAX_QUALITY) == StageResult(frozenset({(0, 0)}), 6)


def test_fixed_and_cap_are_respected(expert_shift):
    """MinShortfall, then MaxQuality with that value fixed.

    Alone, MaxQuality leaves the expert shift cancelled (Lambda = 1, U = 7).
    With Lambda held at 0 it must staff both shifts (U = 6).
    """
    coverage = solve_stage(expert_shift, MIN_SHORTFALL)
    assert coverage.value == 0

    assert solve_stage(expert_shift, MAX_QUALITY) == StageResult(
        EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY, 7
    )

    fixed = solve_stage(
        expert_shift, MAX_QUALITY, fixed={MIN_SHORTFALL: coverage.value}
    )
    assert fixed == StageResult(EXPERT_SHIFT_OPTIMUM, 6)

    # A cap is an upper bound: Lambda <= 0 gives the same result here ...
    capped = solve_stage(expert_shift, MAX_QUALITY, cap={MIN_SHORTFALL: 0})
    assert capped == StageResult(EXPERT_SHIFT_OPTIMUM, 6)

    # ... and Lambda <= 1 lets the quality-only optimum through again
    loose = solve_stage(expert_shift, MAX_QUALITY, cap={MIN_SHORTFALL: 1})
    assert loose == StageResult(EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY, 7)


def test_negative_cap_gives_no_solution(single_shift):
    """Lambda >= 0, so Lambda <= -1 leaves no schedule, not even the empty one."""
    assert solve_stage(single_shift, MAX_QUALITY, cap={MIN_SHORTFALL: -1}) == StageResult(
        None, None, StageStatus.NO_SOLUTION
    )


def test_ties_return_smallest_sorted_schedule():
    """Two identical volunteers on one d = 1 shift: {(0, 0)} and {(1, 0)} tie; (0, 0) sorts first."""
    grid = Grid(n_periods=1, n_roles=1)
    instance = grid.instance(grid.volunteers(2), grid.shifts(1))

    assert solve_stage(instance, MAX_QUALITY) == StageResult(frozenset({(0, 0)}), 1)

    # MinShortfall ties three ways (Lambda = 0); the empty schedule is not among them
    assert solve_stage(instance, MIN_SHORTFALL) == StageResult(frozenset({(0, 0)}), 0)
